# CODA CPU Native Provider

This directory contains the native implementation target for
`OneDnnX64BrgemmProvider`.

The provider shape is:

```text
oneDNN BRGeMM tile mainloop -> f32 accumulator tile -> CodaPostOpChain -> stores
```

`coda_post_ops.*` owns the transformer post-op library. oneDNN native post-ops
are intentionally not the primary abstraction because CODA needs multi-output
stores, block reductions, RoPE, and cross-entropy auxiliaries that are outside
oneDNN primitive post-op coverage.

Build locally with:

```powershell
$env:CODA_CPU_JIT_BUILD = "1"
$env:CODA_CPU_WITH_ONEDNN = "1"
$env:DNNL_INCLUDE_DIR = "C:\path\to\oneDNN\include"
$env:DNNL_LIBRARY_DIR = "C:\path\to\oneDNN\lib"
python -c "from kernels.cpu.native import load_native_extension; print(load_native_extension())"
```

Without `CODA_CPU_WITH_ONEDNN=1`, the extension can still compile the native
post-op chain against ATen. The high-performance oneDNN route is available only
when linked with oneDNN ukernels.

For AVX2-only hosts, build oneDNN with AVX2 GEMM kernels enabled and do not add
`-march=native` or AVX512/AMX-only compiler flags to this extension. oneDNN's
JIT dispatch owns ISA-specific kernel generation; CODA's native layer should
stay portable C++ around the ukernel boundary and post-op chain.

Validation commands used by the CPU backend:

```bash
uv venv --python python3.12 --clear .venv
. .venv/bin/activate
uv pip install pytest einops 'torch==2.4.1+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
pytest kernels/tests/test_cpu_gpt.py -q
CODA_CPU_JIT_BUILD=1 pytest kernels/tests/test_cpu_gpt.py -q
```
