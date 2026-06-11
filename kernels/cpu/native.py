from __future__ import annotations

import importlib
import shlex
import os
import platform
import subprocess
import tempfile
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

    compiler_env = _torch_extension_compiler_env()
    if compiler_env is None:
        return None

    try:
        from torch.utils.cpp_extension import load
    except Exception:
        return None

    native_dir = Path(__file__).with_name("native")
    extra_cflags = ["/O2", "/DNDEBUG"] if os.name == "nt" else ["-O3", "-DNDEBUG"]
    extra_include_paths = [str(native_dir)]
    extra_ldflags: list[str] = []

    if os.environ.get("CODA_CPU_WITH_ATEN_VEC") == "1":
        extra_cflags.extend(_aten_vec_cflags())
        extra_cflags.extend(_aten_vec_cache_cflags())
        extra_cflags.extend(_aten_vec_parallel_cflags())
        extra_ldflags.extend(_aten_vec_parallel_ldflags())

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
        libxsmm_libs = libs or _default_libxsmm_libs(library_dir)
        extra_ldflags.extend(_normalize_libs(libxsmm_libs))

    previous_env = {key: os.environ.get(key) for key in compiler_env}
    os.environ.update(compiler_env)
    try:
        return load(
            name="_coda_cpu_native",
            sources=[
                str(native_dir / "bindings.cpp"),
                str(native_dir / "coda_post_ops.cpp"),
                str(native_dir / "coda_aten_vec.cpp"),
                str(native_dir / "coda_brgemm.cpp"),
                str(native_dir / "coda_libxsmm.cpp"),
            ],
            extra_cflags=extra_cflags,
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
            verbose=bool(int(os.environ.get("CODA_CPU_JIT_VERBOSE", "0"))),
        )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def _compiler_accepts(compiler: list[str], standard: str) -> bool:
    try:
        result = subprocess.run(
            [*compiler, standard, "-x", "c++", "-", "-fsyntax-only"],
            input="int main() { return 0; }\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except (OSError, ValueError):
        return False
    return result.returncode == 0


def _torch_extension_compiler_env() -> dict[str, str] | None:
    if os.name != "posix":
        return {}

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if _compiler_accepts(compiler, "-std=c++20"):
        return {}
    if not _compiler_accepts(compiler, "-std=c++2a"):
        return None

    wrapper = _cxx2a_compat_wrapper(compiler)
    return {"CXX": str(wrapper)}


def _cxx2a_compat_wrapper(compiler: list[str]) -> Path:
    wrapper_dir = Path(tempfile.gettempdir()) / "coda-cxx20-compat"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper = wrapper_dir / "cxx"
    content = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -e",
            "args=()",
            'for arg in "$@"; do',
            '  if [ "$arg" = "-std=c++20" ]; then',
            '    args+=("-std=c++2a")',
            "  else",
            '    args+=("$arg")',
            "  fi",
            "done",
            f'exec {shlex.join(compiler)} "${{args[@]}}"',
            "",
        ]
    )
    if not wrapper.exists() or wrapper.read_text(encoding="utf-8") != content:
        wrapper.write_text(content, encoding="utf-8", newline="\n")
        wrapper.chmod(0o755)
    return wrapper


def _torch_cpu_capability() -> str:
    try:
        import torch

        get_capability = getattr(getattr(torch, "backends", None), "cpu", None)
        if get_capability is None:
            return ""
        capability_fn = getattr(get_capability, "get_cpu_capability", None)
        if capability_fn is None:
            return ""
        return str(capability_fn()).upper()
    except Exception:
        return ""


def _supports_avx2() -> bool:
    capability = _torch_cpu_capability()
    if "AVX512" in capability or "AVX2" in capability:
        return True
    if os.name == "posix":
        try:
            flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            return "avx2" in flags
        except OSError:
            return False
    return False


def _supports_avx512() -> bool:
    capability = _torch_cpu_capability()
    if "AVX512" in capability:
        return True
    if os.name == "posix":
        try:
            flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            return "avx512" in flags
        except OSError:
            return False
    return False


def _supports_sve256() -> bool:
    return "SVE256" in _torch_cpu_capability()


@lru_cache(maxsize=1)
def _cpu_flags() -> frozenset[str]:
    if os.name != "posix":
        return frozenset()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return frozenset()

    flags: set[str] = set()
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"flags", "features"}:
            flags.update(value.lower().split())
    return frozenset(flags)


def _supports_sve2_bf16() -> bool:
    flags = _cpu_flags()
    has_sve_bf16 = "svebf16" in flags or "bf16" in flags
    return _supports_sve256() and "sve2" in flags and has_sve_bf16


def _select_aten_vec_isa(requested_isa: str) -> str:
    if requested_isa not in {"auto", "generic", "avx2", "avx512", "sve256", "sve2-bf16"}:
        raise ValueError(f"unknown CODA_CPU_ATEN_VEC_ISA={requested_isa!r}")
    if requested_isa != "auto":
        return requested_isa
    if _supports_avx512():
        return "avx512"
    if _supports_avx2():
        return "avx2"
    if _supports_sve2_bf16():
        return "sve2-bf16"
    if _supports_sve256():
        return "sve256"
    return "generic"


def _aten_vec_cflags() -> list[str]:
    requested_isa = os.environ.get("CODA_CPU_ATEN_VEC_ISA", "auto").lower()
    selected_isa = _select_aten_vec_isa(requested_isa)

    if selected_isa == "avx512":
        if os.name == "nt":
            return [
                "/DCODA_CPU_WITH_ATEN_VEC=1",
                "/DCPU_CAPABILITY=AVX512",
                "/DCPU_CAPABILITY_AVX512=1",
                "/DCODA_CPU_ATEN_VEC_ISA_AVX512=1",
                "/arch:AVX512",
            ]
        return [
            "-DCODA_CPU_WITH_ATEN_VEC=1",
            "-DCPU_CAPABILITY=AVX512",
            "-DCPU_CAPABILITY_AVX512=1",
            "-DCODA_CPU_ATEN_VEC_ISA_AVX512=1",
            "-mavx512f",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512bw",
            "-mfma",
            "-march=native",
        ]

    if selected_isa == "avx2":
        if os.name == "nt":
            return [
                "/DCODA_CPU_WITH_ATEN_VEC=1",
                "/DCPU_CAPABILITY=AVX2",
                "/DCPU_CAPABILITY_AVX2=1",
                "/DCODA_CPU_ATEN_VEC_ISA_AVX2=1",
                "/arch:AVX2",
            ]
        return [
            "-DCODA_CPU_WITH_ATEN_VEC=1",
            "-DCPU_CAPABILITY=AVX2",
            "-DCPU_CAPABILITY_AVX2=1",
            "-DCODA_CPU_ATEN_VEC_ISA_AVX2=1",
            "-mavx2",
            "-mfma",
            "-march=native",
        ]

    if selected_isa == "sve256":
        if os.name == "nt":
            raise ValueError("CODA_CPU_ATEN_VEC_ISA=sve256 requires a POSIX AArch64 toolchain")
        return [
            "-DCODA_CPU_WITH_ATEN_VEC=1",
            "-DCPU_CAPABILITY=SVE",
            "-DCPU_CAPABILITY_SVE=1",
            "-DCPU_CAPABILITY_SVE256=1",
            "-DCODA_CPU_ATEN_VEC_ISA_SVE256=1",
            "-DAT_BUILD_ARM_VEC256_WITH_SLEEF=1",
            "-march=armv8-a+sve+bf16",
            "-msve-vector-bits=256",
        ]

    if selected_isa == "sve2-bf16":
        if os.name == "nt":
            raise ValueError("CODA_CPU_ATEN_VEC_ISA=sve2-bf16 requires a POSIX AArch64 toolchain")
        return [
            "-DCODA_CPU_WITH_ATEN_VEC=1",
            "-DCPU_CAPABILITY=SVE",
            "-DCPU_CAPABILITY_SVE=1",
            "-DCPU_CAPABILITY_SVE256=1",
            "-DCODA_CPU_ATEN_VEC_ISA_SVE2_BF16=1",
            "-DAT_BUILD_ARM_VEC256_WITH_SLEEF=1",
            "-march=armv8.6-a+sve2+bf16",
            "-msve-vector-bits=256",
        ]

    if os.name == "nt":
        return [
            "/DCODA_CPU_WITH_ATEN_VEC=1",
            "/DCPU_CAPABILITY=DEFAULT",
            "/DCODA_CPU_ATEN_VEC_ISA_GENERIC=1",
        ]
    return [
        "-DCODA_CPU_WITH_ATEN_VEC=1",
        "-DCPU_CAPABILITY=DEFAULT",
        "-DCODA_CPU_ATEN_VEC_ISA_GENERIC=1",
    ]


def _torch_parallel_backend() -> str:
    try:
        import torch

        info = torch.__config__.parallel_info().lower()
    except Exception:
        return ""
    if "aten parallel backend: openmp" in info:
        return "openmp"
    if "aten parallel backend: native" in info:
        return "native"
    return ""


def _torch_cpu_cache_sizes() -> tuple[int, int]:
    default = (32 * 1024, 512 * 1024)
    try:
        import torch

        get_capabilities = getattr(getattr(torch, "cpu", None), "get_capabilities", None)
        if get_capabilities is None:
            return default
        capabilities = get_capabilities()
        l1 = int(capabilities.get("l1d_cache_size", default[0]))
        l2 = int(capabilities.get("l2_cache_size", default[1]))
        if l1 <= 0 or l2 <= 0:
            return default
        return l1, l2
    except Exception:
        return default


def _aten_vec_cache_cflags() -> list[str]:
    l1, l2 = _torch_cpu_cache_sizes()
    if os.name == "nt":
        return [
            f"/DCODA_CPU_ATEN_VEC_L1_BYTES={l1}",
            f"/DCODA_CPU_ATEN_VEC_L2_BYTES={l2}",
        ]
    return [
        f"-DCODA_CPU_ATEN_VEC_L1_BYTES={l1}",
        f"-DCODA_CPU_ATEN_VEC_L2_BYTES={l2}",
    ]


def _aten_vec_parallel_cflags() -> list[str]:
    backend = _torch_parallel_backend()
    if backend == "openmp":
        if os.name == "nt":
            return ["/DINTRA_OP_PARALLEL=1", "/DAT_PARALLEL_OPENMP=1", "/openmp"]
        return ["-DINTRA_OP_PARALLEL=1", "-DAT_PARALLEL_OPENMP=1", "-fopenmp"]
    if backend == "native":
        if os.name == "nt":
            return ["/DINTRA_OP_PARALLEL=1", "/DAT_PARALLEL_NATIVE=1"]
        return ["-DINTRA_OP_PARALLEL=1", "-DAT_PARALLEL_NATIVE=1"]
    return []


def _aten_vec_parallel_ldflags() -> list[str]:
    if _torch_parallel_backend() == "openmp" and os.name != "nt":
        return ["-fopenmp"]
    return []


def _first_existing(paths: list[Path], marker: str | None = None) -> Path | None:
    for path in paths:
        if path.exists() and (marker is None or (path / marker).exists()):
            return path
    return None


def _linux_multiarch_library_dirs() -> list[Path]:
    if os.name != "posix":
        return []
    machine = platform.machine().lower()
    triples = {
        "aarch64": "aarch64-linux-gnu",
        "arm64": "aarch64-linux-gnu",
        "x86_64": "x86_64-linux-gnu",
        "amd64": "x86_64-linux-gnu",
        "s390x": "s390x-linux-gnu",
    }
    triple = triples.get(machine)
    return [Path("/usr/lib") / triple] if triple else []


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
            *_linux_multiarch_library_dirs(),
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
            *_linux_multiarch_library_dirs(),
            Path("/opt/homebrew/lib"),
        ],
    )
    return include_path, library_path


def _default_libxsmm_libs(library_dir: Path | None) -> list[str]:
    return ["xsmm"]


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
