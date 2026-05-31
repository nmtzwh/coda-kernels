from __future__ import annotations

import importlib
import shlex
import os
from ctypes.util import find_library
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

    if os.environ.get("CODA_CPU_WITH_ONEDNN") == "1" and _can_enable_onednn():
        include_dir, library_dir = _onednn_paths()
        if include_dir:
            extra_include_paths.append(str(include_dir))
        if library_dir:
            extra_ldflags.append(f"/LIBPATH:{library_dir}" if os.name == "nt" else f"-L{library_dir}")
            if os.name != "nt":
                extra_ldflags.append(f"-Wl,-rpath,{library_dir}")
        extra_cflags.extend(
            ["/DCODA_CPU_WITH_ONEDNN=1", "/DDNNL_EXPERIMENTAL_UKERNEL=1"]
            if os.name == "nt"
            else ["-DCODA_CPU_WITH_ONEDNN=1", "-DDNNL_EXPERIMENTAL_UKERNEL=1"]
        )
        extra_ldflags.append("dnnl.lib" if os.name == "nt" else "-ldnnl")

    if os.environ.get("CODA_CPU_WITH_LIBXSMM") == "1" and _can_enable_libxsmm():
        include_dir, library_dir = _libxsmm_paths()
        libs = _split_libs(os.environ.get("LIBXSMM_LIBS"))
        if include_dir:
            extra_include_paths.append(str(include_dir))
        if library_dir:
            extra_ldflags.append(f"/LIBPATH:{library_dir}" if os.name == "nt" else f"-L{library_dir}")
            if os.name != "nt":
                extra_ldflags.append(f"-Wl,-rpath,{library_dir}")
        extra_cflags.append("/DCODA_CPU_WITH_LIBXSMM=1" if os.name == "nt" else "-DCODA_CPU_WITH_LIBXSMM=1")
        extra_ldflags.extend(_normalize_libs(libs or ["xsmm"]))

    return load(
        name="_coda_cpu_native",
        sources=[
            str(native_dir / "bindings.cpp"),
            str(native_dir / "coda_post_ops.cpp"),
            str(native_dir / "coda_brgemm.cpp"),
            str(native_dir / "coda_libxsmm.cpp"),
        ],
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
        verbose=bool(int(os.environ.get("CODA_CPU_JIT_VERBOSE", "0"))),
    )


def _split_libs(value: str | None) -> list[str]:
    if not value:
        return []
    return shlex.split(value)


def _normalize_libs(libs: list[str]) -> list[str]:
    normalized = []
    for lib in libs:
        if os.name == "nt":
            normalized.append(lib if lib.endswith(".lib") or lib.startswith("/") else f"{lib}.lib")
        elif lib.startswith("-") or "/" in lib or lib.endswith((".a", ".so", ".dylib")):
            normalized.append(lib)
        elif lib.startswith("lib") and len(lib) > 3:
            normalized.append(f"-l{lib[3:]}")
        else:
            normalized.append(f"-l{lib}")
    return normalized


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _first_existing(paths: list[Path], marker: str | None = None) -> Path | None:
    for path in paths:
        if path.exists() and (marker is None or (path / marker).exists()):
            return path
    return None


def _onednn_paths() -> tuple[Path | None, Path | None]:
    root = _repo_root()
    include_dir = os.environ.get("DNNL_INCLUDE_DIR")
    library_dir = os.environ.get("DNNL_LIBRARY_DIR")
    include_path = Path(include_dir) if include_dir else _first_existing(
        [
            root / "third_party" / "onednn" / "build" / "install" / "include",
            root / "third_party" / "onednn" / "include",
            Path("/usr/include"),
            Path("/usr/local/include"),
        ],
        "oneapi/dnnl/dnnl.hpp",
    )
    library_path = Path(library_dir) if library_dir else _first_existing(
        [
            root / "third_party" / "onednn" / "build" / "install" / "lib",
            root / "third_party" / "onednn" / "build" / "install" / "lib64",
            root / "third_party" / "onednn" / "build" / "src",
            Path("/usr/lib"),
            Path("/usr/local/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
        ],
    )
    return include_path, library_path


def _can_enable_onednn() -> bool:
    include_dir, library_dir = _onednn_paths()
    if include_dir is None or not (include_dir / "oneapi/dnnl/dnnl.hpp").exists():
        return False

    if os.environ.get("DNNL_LIBRARY_DIR"):
        return True
    if library_dir is not None:
        candidates = (
            ("dnnl.lib",)
            if os.name == "nt"
            else ("libdnnl.so", "libdnnl.a", "libdnnl.dylib")
        )
        if any((library_dir / candidate).exists() for candidate in candidates):
            return True

    return find_library("dnnl") is not None


def _libxsmm_paths() -> tuple[Path | None, Path | None]:
    root = _repo_root()
    include_dir = os.environ.get("LIBXSMM_INCLUDE_DIR")
    library_dir = os.environ.get("LIBXSMM_LIBRARY_DIR")
    include_path = Path(include_dir) if include_dir else _first_existing(
        [
            root / "third_party" / "libxsmm" / "include",
            Path("/usr/include"),
            Path("/usr/local/include"),
            Path("/opt/homebrew/include"),
        ],
        "libxsmm.h",
    )
    library_path = Path(library_dir) if library_dir else _first_existing(
        [
            root / "third_party" / "libxsmm" / "lib",
            root / "third_party" / "libxsmm",
            Path("/usr/lib"),
            Path("/usr/local/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/opt/homebrew/lib"),
        ],
    )
    return include_path, library_path


def _can_enable_libxsmm() -> bool:
    include_dir, library_dir = _libxsmm_paths()
    libs = _split_libs(os.environ.get("LIBXSMM_LIBS"))

    include_paths = [include_dir] if include_dir else [
        Path("/usr/include"),
        Path("/usr/local/include"),
        Path("/opt/homebrew/include"),
    ]
    if not any(path is not None and (path / "libxsmm.h").exists() for path in include_paths):
        return False

    if libs:
        return True

    if library_dir is not None:
        candidates = (
            ("libxsmm.lib", "xsmm.lib")
            if os.name == "nt"
            else ("libxsmm.so", "libxsmm.a", "libxsmm.dylib")
        )
        return any((library_dir / candidate).exists() for candidate in candidates)

    return find_library("xsmm") is not None
