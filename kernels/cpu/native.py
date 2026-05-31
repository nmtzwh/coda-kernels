from __future__ import annotations

import importlib
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_native_extension():
    """Load the optional native CPU extension.

    The extension is deliberately optional so `backend="cpu"` keeps working on
    machines without oneDNN or a compiler. Set `CODA_CPU_JIT_BUILD=1` to build it
    in-place during local development.
    """

    for module_name in ("kernels.cpu._coda_cpu_native", "_coda_cpu_native"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass

    if os.environ.get("CODA_CPU_JIT_BUILD") != "1":
        return None

    try:
        from torch.utils.cpp_extension import load
    except Exception:
        return None

    native_dir = Path(__file__).with_name("native")
    extra_cflags = ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]
    extra_include_paths = [str(native_dir)]
    extra_ldflags: list[str] = []

    if os.environ.get("CODA_CPU_WITH_ONEDNN") == "1":
        include_dir = os.environ.get("DNNL_INCLUDE_DIR")
        library_dir = os.environ.get("DNNL_LIBRARY_DIR")
        if include_dir:
            extra_include_paths.append(include_dir)
        if library_dir:
            extra_ldflags.append(f"/LIBPATH:{library_dir}" if os.name == "nt" else f"-L{library_dir}")
        extra_cflags.append("/DCODA_CPU_WITH_ONEDNN=1" if os.name == "nt" else "-DCODA_CPU_WITH_ONEDNN=1")
        extra_ldflags.append("dnnl.lib" if os.name == "nt" else "-ldnnl")

    return load(
        name="_coda_cpu_native",
        sources=[
            str(native_dir / "bindings.cpp"),
            str(native_dir / "coda_post_ops.cpp"),
            str(native_dir / "coda_brgemm.cpp"),
        ],
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
        verbose=bool(int(os.environ.get("CODA_CPU_JIT_VERBOSE", "0"))),
    )
