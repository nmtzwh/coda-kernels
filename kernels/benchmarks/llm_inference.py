import os
# Set thread affinity variables if not already set before importing torch
os.environ.setdefault("OMP_PLACES", "cores")
os.environ.setdefault("OMP_PROC_BIND", "close")

import argparse
import time
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from models.ops import gemm_residual_rmsnorm_gemm_fwd, BlockSizeConfig2

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV states from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    if cos.ndim < 4:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SimpleKVCache:
    def __init__(self, batch_size, max_seq_len, num_kv_heads, head_dim, num_layers, dtype, device):
        self.k_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        self.seq_len = 0

    def update(self, k, v, layer_idx):
        # k: (B, num_kv_heads, T, head_dim)
        # v: (B, num_kv_heads, T, head_dim)
        T = k.shape[2]
        self.k_cache[layer_idx, :, :, self.seq_len : self.seq_len + T, :] = k
        self.v_cache[layer_idx, :, :, self.seq_len : self.seq_len + T, :] = v

    def increment_seq_len(self, T):
        self.seq_len += T

    def get_k(self, layer_idx, T=0):
        return self.k_cache[layer_idx, :, :, :self.seq_len + T, :]

    def get_v(self, layer_idx, T=0):
        return self.v_cache[layer_idx, :, :, :self.seq_len + T, :]

class FusedLayerWeights:
    def __init__(self, layer, is_qwen3=False):
        self.wn0 = layer.post_attention_layernorm.weight
        self.wn1 = layer.input_layernorm.weight
        
        # Interleave and transpose gate_proj and up_proj
        gate = layer.mlp.gate_proj.weight
        up = layer.mlp.up_proj.weight
        gate_up = torch.stack([gate, up], dim=1).flatten(0, 1)
        self.w1 = gate_up.mT.contiguous()
        
        self.w2 = layer.mlp.down_proj.weight.mT.contiguous()
        self.w0 = layer.self_attn.o_proj.weight.mT.contiguous()
        
        # Concatenate and transpose q_proj, k_proj, v_proj
        qkv = torch.cat([
            layer.self_attn.q_proj.weight,
            layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight
        ], dim=0)
        self.w3 = qkv.mT.contiguous()
        
        self.is_qwen3 = is_qwen3
        if is_qwen3:
            self.q_norm = layer.self_attn.q_norm
            self.k_norm = layer.self_attn.k_norm

        # Check if QKV bias exists
        self.has_qkv_bias = (layer.self_attn.q_proj.bias is not None)
        if self.has_qkv_bias:
            self.qkv_bias = torch.cat([
                layer.self_attn.q_proj.bias,
                layer.self_attn.k_proj.bias,
                layer.self_attn.v_proj.bias
            ], dim=0)
        else:
            self.qkv_bias = None

        # Pre-pack the weights for the aten-vec backend
        from kernels.cpu.providers import prepack_weight
        prepack_weight(self.w0)
        prepack_weight(self.w1)
        prepack_weight(self.w2)
        prepack_weight(self.w3)

def coda_forward(
    model,
    fused_weights,
    input_ids,
    position_ids,
    past_key_values,
    cos,
    sin,
    num_heads,
    num_kv_heads,
    head_dim,
    rms_norm_eps,
    is_qwen3
):
    B, T = input_ids.shape
    x = model.model.embed_tokens(input_ids) # shape (B, T, hidden_size)
    
    # Pre-unsqueeze cos and sin once outside the layers loop
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # Layer 0 Attention input
    l0_weights = fused_weights[0]
    h = model.model.layers[0].input_layernorm(x)
    qkv = h.matmul(l0_weights.w3) # shape (B, T, qkv_dim)
    if l0_weights.has_qkv_bias:
        qkv = qkv + l0_weights.qkv_bias
    
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    
    for l in range(len(fused_weights)):
        l_weights = fused_weights[l]
        
        # Split QKV
        q, k, v = qkv.split([q_dim, k_dim, k_dim], dim=-1)
        
        # Reshape Q, K, V for attention
        q = q.view(B, T, num_heads, head_dim)
        k = k.view(B, T, num_kv_heads, head_dim)
        v = v.view(B, T, num_kv_heads, head_dim)
        
        if is_qwen3:
            q = l_weights.q_norm(q)
            k = l_weights.k_norm(k)
            
        # Transpose for SDPA & RoPE
        q = q.transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2) # (B, num_kv_heads, T, head_dim)
        v = v.transpose(1, 2) # (B, num_kv_heads, T, head_dim)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if past_key_values is not None:
            past_key_values.update(k, v, l)
            k_full = past_key_values.get_k(l, T)
            v_full = past_key_values.get_v(l, T)
        else:
            k_full = k
            v_full = v
            
        if num_heads != num_kv_heads:
            # 5D broadcasting for GQA to avoid repeat_kv memory copy
            q_5d = q.view(B, num_kv_heads, num_heads // num_kv_heads, T, head_dim)
            k_5d = k_full.unsqueeze(2) # (B, num_kv_heads, 1, T_seq, head_dim)
            v_5d = v_full.unsqueeze(2)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q_5d, k_5d, v_5d, is_causal=(T > 1)
            )
            attn_out = attn_out.view(B, num_heads, T, head_dim)
        else:
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k_full, v_full, is_causal=(T > 1)
            )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1).contiguous()
        
        # Fused MLP input block
        x_mlp_res, y_swiglu, _, _ = gemm_residual_rmsnorm_gemm_fwd(
            x=x,
            y=attn_out,
            w_a=l_weights.w0,
            w_b=l_weights.w1,
            w_n=l_weights.wn0,
            block_size_norm=128,
            block_size_loss=None,
            cos_sin=None,
            targets=None,
            eps=rms_norm_eps,
            epilogue="swiglu",
            backend="cpu",
            use_quack=False,
        )
        
        if l < len(fused_weights) - 1:
            next_l_weights = fused_weights[l + 1]
            # Fused MLP output & next block input
            x, _, qkv, _ = gemm_residual_rmsnorm_gemm_fwd(
                x=x_mlp_res,
                y=y_swiglu,
                w_a=l_weights.w2,
                w_b=next_l_weights.w3,
                w_n=next_l_weights.wn1,
                block_size_norm=128,
                block_size_loss=None,
                cos_sin=None,
                targets=None,
                eps=rms_norm_eps,
                epilogue=None,
                backend="cpu",
                use_quack=False,
            )
            if next_l_weights.has_qkv_bias:
                qkv = qkv + next_l_weights.qkv_bias
        else:
            # Last layer post-MLP down-proj & final norm
            x_final = x_mlp_res + y_swiglu.matmul(l_weights.w2)
            h_final = model.model.norm(x_final)
            logits = model.lm_head(h_final)
            
    if past_key_values is not None:
        past_key_values.increment_seq_len(T)
        
    return logits

def run_decode_loop(
    model,
    fused_weights,
    input_ids,
    max_new_tokens,
    is_qwen3,
    config,
    use_coda=True
):
    B, T = input_ids.shape
    device = input_ids.device
    dtype = model.dtype
    
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    rms_norm_eps = config.rms_norm_eps
    
    # 1. Prefill phase
    start_prefill = time.perf_counter()
    past_key_values = None
    if use_coda:
        # Pre-allocate cache
        past_key_values = SimpleKVCache(
            batch_size=B,
            max_seq_len=T + max_new_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=config.num_hidden_layers,
            dtype=dtype,
            device=device
        )
        
        position_ids = torch.arange(T, device=device).unsqueeze(0)
        dummy_x = torch.empty((1,), dtype=dtype, device=device)
        cos, sin = model.model.rotary_emb(x=dummy_x, position_ids=position_ids)
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        
        logits = coda_forward(
            model=model,
            fused_weights=fused_weights,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cos=cos,
            sin=sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            is_qwen3=is_qwen3
        )
    else:
        # Standard PyTorch model forward with HF Cache
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token]
    prefill_time = time.perf_counter() - start_prefill
    
    # 2. Decode phase
    start_decode = time.perf_counter()
    for i in range(max_new_tokens - 1):
        curr_token = tokens[-1]
        if use_coda:
            T_past = past_key_values.seq_len
            position_ids = torch.tensor([[T_past]], device=device)
            dummy_x = torch.empty((1,), dtype=dtype, device=device)
            cos, sin = model.model.rotary_emb(x=dummy_x, position_ids=position_ids)
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
            
            logits = coda_forward(
                model=model,
                fused_weights=fused_weights,
                input_ids=curr_token,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cos=cos,
                sin=sin,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                is_qwen3=is_qwen3
            )
        else:
            outputs = model(curr_token, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token)
        
    decode_time = time.perf_counter() - start_decode
    return torch.cat(tokens, dim=-1), prefill_time, decode_time

def main():
    parser = argparse.ArgumentParser(description="E2E LLM Inference benchmark using aten-vec fused operators.")
    parser.add_argument("--model-path", type=str, required=True, help="HF Cache snapshot path of the model.")
    parser.add_argument("--prompt", type=str, default="Explain the theory of general relativity in simple terms.", help="Input prompt.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads.")
    parser.add_argument("--skip-compiled", action="store_true", help="Skip torch.compile benchmarking.")
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16"], default="bfloat16", help="Precision (float32 or bfloat16).")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    
    print(f"Loading tokenizer & configuration from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    
    is_qwen3 = "qwen3" in config.model_type.lower()
    print(f"Model Type: {config.model_type} (is_qwen3={is_qwen3})")
    
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"Loading weights in {args.dtype} on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    print("Preparing fused layer weights...")
    fused_weights = [
        FusedLayerWeights(layer, is_qwen3=is_qwen3)
        for layer in model.model.layers
    ]
    
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    print(f"Prompt tokens: {input_ids.shape[1]}")
    
    # 1. Warmup & validation of correctness
    print("\n--- Correctness Validation ---")
    
    # Run torch/native
    ref_tokens, ref_pref_t, ref_dec_t = run_decode_loop(
        model=model,
        fused_weights=fused_weights,
        input_ids=input_ids.clone(),
        max_new_tokens=args.gen_len,
        is_qwen3=is_qwen3,
        config=config,
        use_coda=False
    )
    
    # Run cpu/aten-vec
    coda_tokens, coda_pref_t, coda_dec_t = run_decode_loop(
        model=model,
        fused_weights=fused_weights,
        input_ids=input_ids.clone(),
        max_new_tokens=args.gen_len,
        is_qwen3=is_qwen3,
        config=config,
        use_coda=True
    )
    
    ref_text = tokenizer.decode(ref_tokens[0], skip_special_tokens=True)
    coda_text = tokenizer.decode(coda_tokens[0], skip_special_tokens=True)
    
    print(f"torch/native generated tokens: {ref_tokens[0].tolist()}")
    print(f"cpu/aten-vec generated tokens: {coda_tokens[0].tolist()}")
    
    matching_tokens = (ref_tokens == coda_tokens).all().item()
    print(f"Tokens match: {matching_tokens}")
    
    # 2. Benchmarking latency & throughput
    print("\n--- Latency & Throughput Benchmarking ---")
    prompt_len = input_ids.shape[1]
    
    # Benchmark torch/native
    _, pref_native, dec_native = run_decode_loop(
        model=model,
        fused_weights=fused_weights,
        input_ids=input_ids.clone(),
        max_new_tokens=args.gen_len,
        is_qwen3=is_qwen3,
        config=config,
        use_coda=False
    )
    print(f"torch/native:")
    print(f"  Prefill: elapsed={pref_native:.3f} s, throughput={prompt_len / pref_native:.2f} tok/s")
    print(f"  Decode:  elapsed={dec_native:.3f} s, throughput={(args.gen_len - 1) / dec_native:.2f} tok/s")
    
    # Benchmark cpu/aten-vec
    # Warmup first
    run_decode_loop(
        model=model,
        fused_weights=fused_weights,
        input_ids=input_ids.clone(),
        max_new_tokens=args.gen_len,
        is_qwen3=is_qwen3,
        config=config,
        use_coda=True
    )
    
    _, pref_coda, dec_coda = run_decode_loop(
        model=model,
        fused_weights=fused_weights,
        input_ids=input_ids.clone(),
        max_new_tokens=args.gen_len,
        is_qwen3=is_qwen3,
        config=config,
        use_coda=True
    )
    print(f"cpu/aten-vec:")
    print(f"  Prefill: elapsed={pref_coda:.3f} s, throughput={prompt_len / pref_coda:.2f} tok/s")
    print(f"  Decode:  elapsed={dec_coda:.3f} s, throughput={(args.gen_len - 1) / dec_coda:.2f} tok/s")
    
    if not args.skip_compiled:
        try:
            print("Compiling model (this may take a few minutes)...")
            compiled_model = torch.compile(model, fullgraph=False)
            # Warmup compilation
            run_decode_loop(
                model=compiled_model,
                fused_weights=fused_weights,
                input_ids=input_ids.clone(),
                max_new_tokens=args.gen_len,
                is_qwen3=is_qwen3,
                config=config,
                use_coda=False
            )
            _, pref_compiled, dec_compiled = run_decode_loop(
                model=compiled_model,
                fused_weights=fused_weights,
                input_ids=input_ids.clone(),
                max_new_tokens=args.gen_len,
                is_qwen3=is_qwen3,
                config=config,
                use_coda=False
            )
            print(f"torch/compiled:")
            print(f"  Prefill: elapsed={pref_compiled:.3f} s, throughput={prompt_len / pref_compiled:.2f} tok/s")
            print(f"  Decode:  elapsed={dec_compiled:.3f} s, throughput={(args.gen_len - 1) / dec_compiled:.2f} tok/s")
        except Exception as exc:
            print(f"torch/compiled skipped: {exc}")

if __name__ == "__main__":
    main()
