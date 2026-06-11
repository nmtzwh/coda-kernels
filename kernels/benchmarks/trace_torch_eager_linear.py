import argparse
import os
import sys


def _shape(tensor):
    return tuple(tensor.shape) if hasattr(tensor, "shape") else None


def _dtype(tensor):
    return str(tensor.dtype) if hasattr(tensor, "dtype") else type(tensor).__name__


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Trace torch eager linear/mm/addmm dispatch and oneDNN primitive selection."
    )
    parser.add_argument("--op", choices=["linear", "mm", "addmm"], default="linear")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--no-onednn-verbose", action="store_true")
    args = parser.parse_args()

    if not args.no_onednn_verbose:
        os.environ.setdefault("ONEDNN_VERBOSE", "all")
        os.environ.setdefault("DNNL_VERBOSE", "1")

    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch.utils._python_dispatch import TorchDispatchMode

    class AtenTraceMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
            shapes = ", ".join(f"{_dtype(arg)}{_shape(arg)}" for arg in tensor_args)
            print(f"ATen: {func}({shapes})")
            return func(*args, **kwargs)

    if args.disable_mkldnn:
        torch.backends.mkldnn.enabled = False
    torch.set_num_threads(args.threads)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    x = torch.randn(args.m, args.k, dtype=dtype)
    w = torch.randn(args.n, args.k, dtype=dtype)
    b = None if args.no_bias else torch.randn(args.n, dtype=dtype)
    mm_rhs = w.t().contiguous()
    addmm_self = torch.zeros(args.m, args.n, dtype=dtype) if b is None else b.expand(args.m, args.n).contiguous()

    def run_once():
        if args.op == "linear":
            return F.linear(x, w, b)
        if args.op == "mm":
            return torch.mm(x, mm_rhs)
        return torch.addmm(addmm_self, x, mm_rhs)

    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("torch file:", torch.__file__)
    print("machine tensors:", args.op, args.dtype, f"M={args.m}", f"K={args.k}", f"N={args.n}")
    print("mkldnn available:", torch.backends.mkldnn.is_available())
    print("mkldnn enabled:", torch.backends.mkldnn.enabled)
    print("torch config:")
    print(torch.__config__.show())

    print("ATen operator trace:")
    with torch.no_grad(), AtenTraceMode():
        y = run_once()
    print("output:", y.dtype, tuple(y.shape), float(y.float().sum()))

    with torch.no_grad():
        for _ in range(2):
            run_once()

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(args.iters):
                with record_function(f"trace_{args.op}"):
                    run_once()

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
