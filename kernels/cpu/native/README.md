# CODA CPU Native Provider

This directory contains the native implementation target for
`AtenVecProvider`, `OneDnnX64BrgemmProvider`, and `LibxsmmProvider`.

The provider shapes are:

```text
ATen Vectorized f32 mainloop -> transformer epilogues in the row/block loop -> stores
oneDNN BRGeMM tile mainloop -> f32 accumulator tile -> CodaPostOpChain -> stores
LIBXSMM JIT SMM tile mainloop -> f32 accumulator tile -> CodaPostOpChain -> stores
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
post-op chain against ATen. Enable the opt-in ATen vector path with
`CODA_CPU_WITH_ATEN_VEC=1` and select it with `CODA_CPU_PROVIDER=aten-vec`.
The ATen vector path uses `ATen/cpu/vec/vec.h`, compiles against the PyTorch
2.12 CPU headers, and supports forward contiguous float32 transformer GEMM
epilogues first. Unsupported dtype/layout cases fall back to the Python ATen
provider unless `CODA_ATEN_VEC_STRICT=1` is set. Set
`CODA_CPU_ATEN_VEC_ISA=generic` to avoid AVX2 flags, or
`CODA_CPU_ATEN_VEC_ISA=avx2` to require AVX2/FMA flags.
The GEMM loop packs RHS register panels, keeps `Kc x Nr` in 80% of L1D, keeps
the active LHS `Mc x K` panel in 75% of L2, and uses named `4 x 24` ATen-vector
register tiles before applying vectorized transformer epilogues. Cache sizes
and the ATen parallel backend are detected from the installed PyTorch 2.12 CPU
wheel. OpenMP wheels compile and link the extension with OpenMP; native-thread-
pool wheels use ATen's native parallel backend.

PyTorch 2.12 native extensions require C++20 headers. On older POSIX compilers
that implement C++20 under `-std=c++2a`, the loader creates a temporary
compatibility wrapper that translates PyTorch's `-std=c++20` flag. Optional
native providers report unavailable when neither spelling is accepted.

The high-performance oneDNN route is available only
when linked with oneDNN ukernels. The JIT extension defines
`DNNL_EXPERIMENTAL_UKERNEL=1` when `CODA_CPU_WITH_ONEDNN=1` is enabled because
the BRGeMM ukernel C++ API is guarded as experimental in oneDNN headers.

Enable the optional LIBXSMM bridge with:

```bash
export CODA_CPU_JIT_BUILD=1
export CODA_CPU_WITH_LIBXSMM=1
export LIBXSMM_INCLUDE_DIR=/path/to/libxsmm/include
export LIBXSMM_LIBRARY_DIR=/path/to/libxsmm/lib
export LIBXSMM_LIBS="xsmm"  # optional; can also contain raw linker flags
python -c "from kernels.cpu.native import load_native_extension; n = load_native_extension(); print(n, n.has_libxsmm() if n else False)"
```

If `CODA_CPU_WITH_LIBXSMM=1` is set but `libxsmm.h` or a linkable LIBXSMM
library cannot be found, the JIT build compiles the native extension without
LIBXSMM and reports `has_libxsmm() == False`. This keeps the CPU backend usable
on systems where LIBXSMM is not installed.

For large dense Transformer projections, LIBXSMM main no longer exposes the old
parallel dense GEMM helper. The LIBXSMM provider stays on native LIBXSMM paths:
the default is a tiled strided batch-reduce GEMM mainloop with
`CODA_LIBXSMM_M_TILE=128`, `CODA_LIBXSMM_N_TILE=32`, and
`CODA_LIBXSMM_BR_K_TILE=512` on AVX2. `CODA_LIBXSMM_DENSE_SGEMM=1` enables the
native `libxsmm_sgemm` row-block path for comparison, and
`CODA_LIBXSMM_USE_BRGEMM=0` forces the tiled JIT SMM path.

For AVX2-only hosts, build oneDNN with AVX2 GEMM kernels enabled and do not add
`-march=native` or AVX512/AMX-only compiler flags to this extension. oneDNN's
JIT dispatch owns ISA-specific kernel generation; CODA's native layer should
stay portable C++ around the ukernel boundary and post-op chain.

Validation commands used by the CPU backend:

```bash
uv venv --python python3.12 --clear .venv
. .venv/bin/activate
uv pip install pytest einops ninja packaging
uv pip install 'torch==2.12.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__); assert torch.__version__.startswith('2.12.0')"
pytest kernels/tests/test_cpu_gpt.py -q
CODA_CPU_JIT_BUILD=1 pytest kernels/tests/test_cpu_gpt.py -q
CODA_CPU_JIT_BUILD=1 CODA_CPU_WITH_ATEN_VEC=1 CODA_CPU_PROVIDER=aten-vec pytest kernels/tests/test_cpu_gpt.py -q
CODA_CPU_JIT_BUILD=1 CODA_CPU_WITH_ATEN_VEC=1 python kernels/benchmarks/cpu_transformer.py --threads 8
CODA_CPU_JIT_BUILD=1 CODA_CPU_WITH_LIBXSMM=1 pytest kernels/tests/test_cpu_gpt.py -q
```
