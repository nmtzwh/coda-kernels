import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from kernels.benchmarks.llm_inference import FusedLayerWeights, apply_rotary_pos_emb, repeat_kv
from kernels.cpu.providers import select_provider
from models.ops import gemm_residual_rmsnorm_gemm_fwd

def debug():
    torch.set_grad_enabled(False)
    
    model_path = "/home/tom/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    
    select_provider("aten-vec")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    fused_weights = [
        FusedLayerWeights(layer, is_qwen3=True)
        for layer in model.model.layers
    ]
    
    prompt = "Explain gravity."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    B, T = input_ids.shape
    device = input_ids.device
    dtype = model.dtype
    
    ref_layer_0 = model.model.layers[0]
    
    # Layer 0 inputs
    x = model.model.embed_tokens(input_ids)
    h = ref_layer_0.input_layernorm(x)
    
    # 1. QKV projection
    l0_weights = fused_weights[0]
    coda_qkv = h.matmul(l0_weights.w3)
    
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    coda_q, coda_k, coda_v = coda_qkv.split([q_dim, k_dim, k_dim], dim=-1)
    
    # QK-norm
    ref_hidden_shape = (B, T, -1, head_dim)
    ref_q_norm = ref_layer_0.self_attn.q_norm(ref_layer_0.self_attn.q_proj(h).view(ref_hidden_shape)).transpose(1, 2)
    ref_k_norm = ref_layer_0.self_attn.k_norm(ref_layer_0.self_attn.k_proj(h).view(ref_hidden_shape)).transpose(1, 2)
    ref_v_proj = ref_layer_0.self_attn.v_proj(h).view(ref_hidden_shape).transpose(1, 2)
    
    coda_q_view = coda_q.view(B, T, num_heads, head_dim)
    coda_k_view = coda_k.view(B, T, num_kv_heads, head_dim)
    coda_v_view = coda_v.view(B, T, num_kv_heads, head_dim)
    
    coda_q_normed = l0_weights.q_norm(coda_q_view).transpose(1, 2)
    coda_k_normed = l0_weights.k_norm(coda_k_view).transpose(1, 2)
    coda_v_transposed = coda_v_view.transpose(1, 2)
    
    print("5. Q_normed max diff:", (ref_q_norm - coda_q_normed).abs().max().item())
    print("6. K_normed max diff:", (ref_k_norm - coda_k_normed).abs().max().item())
    print("7. V_transposed max diff:", (ref_v_proj - coda_v_transposed).abs().max().item())
    
    # RoPE
    position_ids = torch.arange(T, device=device).unsqueeze(0)
    dummy_x = torch.empty((1,), dtype=dtype, device=device)
    cos, sin = model.model.rotary_emb(x=dummy_x, position_ids=position_ids)
    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)
    
    ref_q_rope, ref_k_rope = apply_rotary_pos_emb(ref_q_norm, ref_k_norm, cos, sin)
    coda_q_rope, coda_k_rope = apply_rotary_pos_emb(coda_q_normed, coda_k_normed, cos, sin)
    
    print("8. Q_rope max diff:", (ref_q_rope - coda_q_rope).abs().max().item())
    print("9. K_rope max diff:", (ref_k_rope - coda_k_rope).abs().max().item())
    
    # SDPA
    ref_k_full = repeat_kv(ref_k_rope, num_heads // num_kv_heads)
    ref_v_full = repeat_kv(ref_v_proj, num_heads // num_kv_heads)
    ref_attn_out = torch.nn.functional.scaled_dot_product_attention(
        ref_q_rope, ref_k_full, ref_v_full, is_causal=(T > 1)
    )
    ref_attn_out = ref_attn_out.transpose(1, 2).reshape(B, T, -1).contiguous()
    
    coda_k_full = repeat_kv(coda_k_rope, num_heads // num_kv_heads)
    coda_v_full = repeat_kv(coda_v_transposed, num_heads // num_kv_heads)
    coda_attn_out = torch.nn.functional.scaled_dot_product_attention(
        coda_q_rope, coda_k_full, coda_v_full, is_causal=(T > 1)
    )
    coda_attn_out = coda_attn_out.transpose(1, 2).reshape(B, T, -1).contiguous()
    
    print("10. Attn_out max diff:", (ref_attn_out - coda_attn_out).abs().max().item())
    
    # Fused MLP input block
    ref_attn_output = ref_layer_0.self_attn.o_proj(ref_attn_out)
    ref_x_mlp_res = x + ref_attn_output
    ref_h_mlp = ref_layer_0.post_attention_layernorm(ref_x_mlp_res)
    ref_y_gate = ref_layer_0.mlp.gate_proj(ref_h_mlp)
    ref_y_up = ref_layer_0.mlp.up_proj(ref_h_mlp)
    ref_y_swiglu = torch.nn.functional.silu(ref_y_gate) * ref_y_up
    
    coda_x_mlp_res, coda_y_swiglu, _, _ = gemm_residual_rmsnorm_gemm_fwd(
        x=x,
        y=coda_attn_out,
        w_a=l0_weights.w0,
        w_b=l0_weights.w1,
        w_n=l0_weights.wn0,
        block_size_norm=128,
        block_size_loss=None,
        cos_sin=None,
        targets=None,
        eps=config.rms_norm_eps,
        epilogue="swiglu",
        backend="cpu",
        use_quack=False,
    )
    
    print("11. Fused MLP res max diff:", (ref_x_mlp_res - coda_x_mlp_res).abs().max().item())
    print("12. Fused SwiGLU max diff:", (ref_y_swiglu - coda_y_swiglu).abs().max().item())

if __name__ == "__main__":
    debug()
