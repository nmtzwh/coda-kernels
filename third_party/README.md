# Third-Party CPU Kernel Dependencies

CODA's CPU backend can use these optional native GEMM providers:

| Dependency | Path | Pinned release | Purpose |
| --- | --- | --- | --- |
| oneDNN | `third_party/onednn` | `v3.12` | x86 BRGeMM provider |
| LIBXSMM | `third_party/libxsmm` | `1.17` | portable JIT SMM/BRGEMM provider |

The dependencies are recorded as shallow Git submodules. After cloning CODA,
initialize them with:

```bash
git submodule update --init --depth 1 third_party/onednn third_party/libxsmm
```

The native JIT build also accepts external installs via `DNNL_INCLUDE_DIR`,
`DNNL_LIBRARY_DIR`, `LIBXSMM_INCLUDE_DIR`, `LIBXSMM_LIBRARY_DIR`, and
`LIBXSMM_LIBS`. If these variables are not set, it looks for libraries built
under `third_party/onednn/build/install` and `third_party/libxsmm`.
