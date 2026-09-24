#!/usr/bin/env python3
"""Construct a private CPython prefix and patch only its own native executables.

Run this with the existing working lerobot310 Python. Never edits that Python,
its site-packages, the uv cache, the host loader, or an existing foreign venv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile


NVIDIA = ("cublas", "cuda_cupti", "cuda_nvrtc", "cuda_runtime", "cudnn", "cufft",
          "curand", "cusolver", "cusparse", "cusparselt", "nccl", "nvjitlink", "nvtx", "npp", "cufile")


def run(*args: str, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def capture(*args: str) -> str:
    return run(*args, stdout=subprocess.PIPE).stdout.strip()


def pathvar(key: str, fallback: str | Path) -> Path:
    return Path(os.environ.get(key) or fallback).expanduser().absolute()


def elf_interpreter(path: Path) -> str | None:
    """Inspect PT_INTERP without executing ldd or untrusted native binaries."""
    try:
        with path.open("rb") as f:
            header = f.read(64)
            if header[:6] != b"\x7fELF\x02\x01":  # Linux x86_64 ELF64 little endian
                return None
            offset = struct.unpack_from("<Q", header, 32)[0]
            size, count = struct.unpack_from("<HH", header, 54)
            for i in range(count):
                f.seek(offset + i * size)
                ph = f.read(size)
                if struct.unpack_from("<I", ph)[0] == 3:
                    start, length = struct.unpack_from("<Q", ph, 8)[0], struct.unpack_from("<Q", ph, 32)[0]
                    f.seek(start)
                    return f.read(length).rstrip(b"\0").decode()
    except (OSError, struct.error, UnicodeError):
        pass
    return None


def patch(path: Path, loader: str, paths: list[str], patchelf: str) -> None:
    old = capture(patchelf, "--print-rpath", str(path))
    # Keep $ORIGIN in the object's context. New libc is first, bundled libs next.
    origins = [p for p in old.split(":") if "$ORIGIN" in p or "${ORIGIN}" in p]
    absolute = [p for p in old.split(":") if p and p not in origins]
    combined = paths[:2] + origins + paths[2:] + absolute
    rpath = ":".join(dict.fromkeys(p for p in combined if p))
    # Avoid rewriting an ELF that this adapter has already patched identically.
    if old == rpath and elf_interpreter(path) == loader:
        return
    # Atomic copy-on-write, even if a previous uv invocation used hardlinks.
    fd, temp = tempfile.mkstemp(prefix=path.name + ".patch-", dir=path.parent)
    os.close(fd)
    try:
        shutil.copy2(path, temp)
        os.chmod(temp, path.stat().st_mode | 0o200)
        run(patchelf, "--set-interpreter", loader, "--force-rpath", "--set-rpath", rpath, temp)
        os.chmod(temp, path.stat().st_mode)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_shell(path: Path, text: str, mode: int = 0o644) -> None:
    if path.parent.resolve() != path.parent or path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"Refusing unexpected shell-file target: {path}")
    fd, name = tempfile.mkstemp(prefix=path.name + ".new-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def make_launcher(path: Path, command: list[str], env: dict[str, str], prefix_path: list[str]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             "unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH", "export PYTHONNOUSERSITE=1"]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in env.items()]
    lines += ["export PATH=" + shlex.quote(":".join(prefix_path)) + ':"${PATH}"',
              "exec " + " ".join(shlex.quote(arg) for arg in command) + ' "$@"', ""]
    write_shell(path, "\n".join(lines), 0o755)


def activation_script(env: dict[str, str], prefix_path: list[str], profile: str) -> str:
    # Do not source venv/activate: that would deactivate an outer virtualenv.
    # Save declarations so unset, empty and unexported values round-trip too.
    managed = sorted(set(env) | {"PATH", "PS1", "VIRTUAL_ENV_PROMPT",
                                "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"})
    script = r'''# Source this file from Bash (including CentOS 7 Bash 4.2).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file; executing it cannot activate your parent shell." >&2
    exit 2
fi
_fasterwam_activate() {
    local _fw_name _fw_decl _fw_flags
    if declare -F _fasterwam_deactivate >/dev/null; then
        _fasterwam_deactivate
    fi
    # Check before changing anything: readonly variables cannot be restored.
    for _fw_name in @MANAGED@; do
        _fw_decl=$(declare -p "$_fw_name" 2>/dev/null) || _fw_decl=
        _fw_flags=${_fw_decl#declare -}; _fw_flags=${_fw_flags%% *}
        if [[ $_fw_flags == *r* || $_fw_flags == *a* || $_fw_flags == *A* || $_fw_flags == *n* ]]; then
            echo "Cannot activate with readonly, array or nameref variable: $_fw_name" >&2
            return 1
        fi
    done
    declare -gA _FASTERWAM_SAVED_DECL=()
    for _fw_name in @MANAGED@; do
        _FASTERWAM_SAVED_DECL[$_fw_name]=$(declare -p "$_fw_name" 2>/dev/null) || _FASTERWAM_SAVED_DECL[$_fw_name]=
    done
    _FASTERWAM_OLD_DEACTIVATE=$(declare -f deactivate) || _FASTERWAM_OLD_DEACTIVATE=
    _fasterwam_deactivate() {
        local _fw_name _fw_decl _fw_flags _fw_old_function=$_FASTERWAM_OLD_DEACTIVATE
        for _fw_name in "${!_FASTERWAM_SAVED_DECL[@]}"; do
            _fw_decl=${_FASTERWAM_SAVED_DECL[$_fw_name]}
            unset "$_fw_name"
            if [[ -n $_fw_decl ]]; then
                eval "${_fw_decl/declare /declare -g }"
                _fw_flags=${_fw_decl#declare -}; _fw_flags=${_fw_flags%% *}
                if [[ $_fw_flags != *x* ]]; then export -n "$_fw_name"; fi
            fi
        done
        unset _FASTERWAM_SAVED_DECL _FASTERWAM_OLD_DEACTIVATE
        unset -f deactivate _fasterwam_deactivate
        if [[ -n $_fw_old_function ]]; then eval "$_fw_old_function"; fi
        hash -r 2>/dev/null || true
    }
    deactivate() { _fasterwam_deactivate "$@"; }
    unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
@EXPORTS@
    export PATH=@PATH_PREFIX@:"${PATH:-}"
    export VIRTUAL_ENV_PROMPT=@PROMPT@
    if [[ -z ${VIRTUAL_ENV_DISABLE_PROMPT:-} ]]; then
        export PS1="${VIRTUAL_ENV_PROMPT}${PS1:-}"
    fi
    hash -r 2>/dev/null || true
}
if _fasterwam_activate; then
    unset -f _fasterwam_activate
else
    unset -f _fasterwam_activate
    return 1
fi
'''
    return (script.replace("@MANAGED@", " ".join(managed))
            .replace("@EXPORTS@", "\n".join(f"    export {k}={shlex.quote(v)}" for k, v in env.items()))
            .replace("@PATH_PREFIX@", shlex.quote(":".join(prefix_path)))
            .replace("@PROMPT@", shlex.quote(f"(FasterWAM-{profile}) ")))


def write_shell_files(state: dict) -> None:
    root, venv = Path(state["root"]), Path(state["venv"])
    profile = state["profile"]
    state_dir = root / ".runtime/centos7" / profile
    env = {"VIRTUAL_ENV": str(venv), "UV_PROJECT_ENVIRONMENT": str(venv),
           "UV_PYTHON": str(venv / "bin/python"), "UV_PYTHON_DOWNLOADS": "never",
           "UV_LINK_MODE": "copy", "PYTHONNOUSERSITE": "1", "FASTERWAM_GLIBC_ROOT": state["glibc"]}
    # Match the upstream lock's index spelling without re-locking.
    project = root if profile == "core" else root / "environments" / profile
    lock_text = (project / "uv.lock").read_text()
    env["UV_INDEX"] = "pypi=https://pypi.org/simple" + ("/" if 'registry = "https://pypi.org/simple/"' in lock_text else "")
    prefix_path = [str(state_dir / "bin"), str(venv / "bin")]
    gcc_root = pathvar("FASTERWAM_GCC_ROOT", Path.home() / "depends/gcc-12.1.0")
    if (gcc_root / "bin/gcc").is_file():
        prefix_path.append(str(gcc_root / "bin"))
        env.update(CC=str(gcc_root / "bin/gcc"), CXX=str(gcc_root / "bin/g++"))
    cuda = os.environ.get("FASTERWAM_CUDA_HOME") or os.environ.get("CUDA_HOME")
    if cuda:
        env["CUDA_HOME"] = cuda
        prefix_path.append(str(Path(cuda) / "bin"))
    for name in ("MAGICK_HOME", "MUJOCO_GL", "PYOPENGL_PLATFORM", "VK_ICD_FILENAMES"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    (state_dir / "bin").mkdir(exist_ok=True)
    write_shell(state_dir / "activate.sh", activation_script(env, prefix_path, profile))
    # uv pip's --python means install target, so use the venv here, not its base.
    make_launcher(state_dir / "bin/uv", [state["uv"]], env, prefix_path[1:])
    make_launcher(state_dir / "run", [], env, prefix_path)


def refresh_shell(args) -> None:
    state_path = Path(args.state).absolute()
    state = json.loads(state_path.read_text())
    root = Path(state["root"])
    profile = state["profile"]
    if profile not in ("core", "libero", "libero-plus", "robotwin") or not root.is_absolute() or root.resolve() != root:
        raise RuntimeError("Unexpected runtime profile or repository path.")
    state_dir = root / ".runtime/centos7" / profile
    private = state_dir / "python-base"
    venv = root / (".venv" if profile == "core" else f".venvs/{profile}")
    expected = {"private": str(private), "python": str(private / "bin/python3.10"), "venv": str(venv)}
    if state_path != state_dir / "state.json" or any(state.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Runtime state does not belong to the expected profile paths.")
    for path in (state_path, private, Path(state["python"]), venv, state_dir / "bin",
                 state_dir / "activate.sh", state_dir / "bin/uv", state_dir / "run"):
        if path.resolve() != path:
            raise RuntimeError(f"Refusing a symlink in an owned runtime path: {path}")
    info = json.loads(capture(str(venv / "bin/python"), "-I", "-B", "-c",
                             "import sys,json; print(json.dumps([sys.prefix,sys.base_prefix]))"))
    if info != [str(venv), str(private)]:
        raise RuntimeError(f"Refusing an unrelated Python environment: {info}")
    uv = Path(state["uv"])
    launchers = [root / ".runtime/centos7" / p / "bin/uv" for p in ("core", "libero", "libero-plus", "robotwin")]
    if not uv.is_absolute() or not uv.is_file() or not os.access(uv, os.X_OK) or uv.resolve() in launchers:
        raise RuntimeError(f"Invalid underlying uv executable: {uv}")
    write_shell_files(state)
    print(f"Refreshed activate.sh, bin/uv and run for {profile}; Python and packages were not changed.")


def prepare(args) -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Bootstrap must use your working Python 3.10 ELF, not a shell wrapper.")
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("This adapter targets Linux x86_64.")
    root = Path(args.root).resolve()
    base = pathvar("FASTERWAM_BASE_PREFIX", sys.base_prefix).resolve()
    stdlib = base / "lib/python3.10"
    if not (stdlib / "os.py").is_file():
        raise RuntimeError(f"Cannot find Python 3.10 standard library: {stdlib}")
    source = Path(sys.executable).resolve()
    if elf_interpreter(source) is None:
        raise RuntimeError(f"Not a dynamically linked x86_64 ELF Python: {source}")
    glibc = pathvar("FASTERWAM_GLIBC_ROOT", Path.home() / "depends/glibc-2.35").resolve()
    candidates = [glibc / "lib/ld-linux-x86-64.so.2", glibc / "lib64/ld-linux-x86-64.so.2"]
    loader = next((p for p in candidates if p.is_file()), None)
    if loader is None:
        raise RuntimeError(f"Cannot find private glibc loader under {glibc}")
    if not loader.resolve().is_relative_to(glibc):
        raise RuntimeError("Private loader points outside the configured glibc prefix.")
    glibc_lib = loader.parent
    if not (glibc_lib / "libc.so.6").is_file():
        raise RuntimeError(f"libc.so.6 must be next to {loader}")
    state_dir = root / ".runtime/centos7" / args.profile
    private = state_dir / "python-base"
    venv = root / (".venv" if args.profile == "core" else f".venvs/{args.profile}")
    state_path = state_dir / "state.json"
    for owned in (root / ".runtime", root / ".runtime/centos7", root / ".venvs", state_dir, private, venv):
        if owned.is_symlink():
            raise RuntimeError(f"Refusing a symlink in a runtime-owned location: {owned}")
    if venv.exists() and not state_path.is_file():
        raise RuntimeError(f"Existing unmanaged environment: {venv}. Rename it manually first; it will not be deleted.")
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    marker_text = json.dumps({"prefix": str(base), "python_sha256": fingerprint}, sort_keys=True) + "\n"
    if (private / ".stdlib-ready").is_file() and (private / ".stdlib-ready").read_text() != marker_text:
        raise RuntimeError("Private standard library belongs to a different base Python. Rename this profile's runtime and venv directories, then recreate.")
    if state_path.exists():
        old = json.loads(state_path.read_text())
        expected = (str(root), str(base), fingerprint, str(glibc), str(venv))
        found = tuple(old.get(k) for k in ("root", "source_prefix", "source_sha256", "glibc", "venv"))
        if expected != found:
            raise RuntimeError("Base Python, glibc or project location changed. Rename this profile's venv AND runtime directory, then recreate.")
    state_dir.mkdir(parents=True, exist_ok=True)
    if not (private / ".stdlib-ready").exists():
        if private.exists():
            raise RuntimeError(f"Incomplete private Python at {private}. Rename that directory and retry.")
        (private / "bin").mkdir(parents=True)
        # Copies keep imports / bytecode writes out of the original Conda env.
        shutil.copytree(stdlib, private / "lib/python3.10",
                        ignore=shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc"))
        (private / "lib/python3.10/site-packages").mkdir(exist_ok=True)
        if (base / "include").is_dir():
            (private / "include").symlink_to(base / "include", target_is_directory=True)
        (private / ".stdlib-ready").write_text(marker_text)
    native_libs = state_dir / "native-libs"
    native_libs.mkdir(exist_ok=True)
    gcc_root = pathvar("FASTERWAM_GCC_ROOT", Path.home() / "depends/gcc-12.1.0")
    explicit_cxx = os.environ.get("FASTERWAM_CXX_LIBDIR")
    cxx_candidates = [Path(explicit_cxx)] if explicit_cxx else [base / "lib", gcc_root / "lib64", gcc_root / "lib"]
    cxx = next((p for p in cxx_candidates if (p / "libstdc++.so.6").is_file()
                and b"GLIBCXX_3.4.30" in (p / "libstdc++.so.6").read_bytes()), None)
    if cxx is None:
        raise RuntimeError("Need libstdc++.so.6 with GLIBCXX_3.4.30 (e.g. GCC 12). Set FASTERWAM_CXX_LIBDIR; do not change system libraries.")
    for name in ("libstdc++.so.6", "libgcc_s.so.1"):
        target = cxx / name
        if not target.is_file():
            target = base / "lib" / name
        if not target.is_file():
            raise RuntimeError(f"Missing {name}; set FASTERWAM_CXX_LIBDIR")
        link = native_libs / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"Unexpected non-symlink in owned runtime directory: {link}")
        link.symlink_to(target.resolve())
    site = venv / "lib/python3.10/site-packages"
    paths = [str(glibc_lib), str(native_libs), str(site / "torch/lib")]
    paths += [str(site / "nvidia" / name / "lib") for name in NVIDIA]
    paths += [str(pathvar("FASTERWAM_DRIVER_LIBDIR", Path.home() / "depends/nvidia-driver-libs"))]
    ffmpeg = pathvar("FASTERWAM_FFMPEG_ROOT", Path.home() / "depends/ffmpeg-7.1.1")
    if args.profile == "core":
        if not (ffmpeg / "lib/libavcodec.so.61").exists():
            raise RuntimeError(f"Core torchcodec needs your shared FFmpeg 7 build: {ffmpeg}/lib/libavcodec.so.61. Set FASTERWAM_FFMPEG_ROOT.")
        paths += [str(ffmpeg / "lib")]
        for p in [Path.home() / "depends/dav1d-1.5.3/lib", Path.home() / "depends/x264/lib"]:
            if p.is_dir():
                paths.append(str(p))
    for item in os.environ.get("FASTERWAM_EXTRA_LIB_DIRS", "").split(":"):
        if not item:
            continue
        p = Path(item).expanduser().resolve()
        if not p.is_dir() or (p / "libc.so.6").exists() or "stubs" in p.parts:
            raise RuntimeError(f"Unsafe or missing extra library directory: {p}")
        paths.append(str(p))
    # Low priority: libpython, SSL, libffi etc. Never import old site-packages.
    paths.append(str(base / "lib"))
    if args.profile == "core":
        # Existing FFmpeg may link NPP/NVRTC. Only these fallback CUDA 12.8
        # components are allowed; old cuDNN/cuBLAS/NCCL/torch libs are excluded.
        for name in ("npp", "cuda_nvrtc", "cuda_runtime", "nvjitlink"):
            p = base / "lib/python3.10/site-packages/nvidia" / name / "lib"
            if p.is_dir():
                paths.append(str(p))
    python = private / "bin/python3.10"
    # Discard the source Python's RPATH: it can contain the old env's torch/CUDA.
    temp = python.with_suffix(".new")
    shutil.copy2(source, temp)
    temp.chmod(source.stat().st_mode | 0o200)
    run(args.patchelf, "--set-interpreter", str(loader), "--force-rpath", "--set-rpath", ":".join(paths), str(temp))
    temp.chmod(source.stat().st_mode)
    os.replace(temp, python)
    for name in ("python", "python3"):
        link = private / "bin" / name
        if not link.exists():
            link.symlink_to("python3.10")
    info = json.loads(capture(str(python), "-I", "-c", "import os,sys,json,ssl,sqlite3,bz2,lzma,ctypes; print(json.dumps([sys.prefix,sys.base_prefix,os.confstr('CS_GNU_LIBC_VERSION')]))"))
    if info[0] != str(private) or info[1] != str(private):
        raise RuntimeError(f"CPython prefix relocation failed: {info}")
    if tuple(map(int, info[2].split()[-1].split("."))) < (2, 35):
        raise RuntimeError(f"Patched Python loaded the wrong libc: {info}")
    state = dict(root=str(root), profile=args.profile, source_prefix=str(base), source_sha256=fingerprint,
                 glibc=str(glibc), loader=str(loader), private=str(private), python=str(python), venv=str(venv),
                 rpath=paths, patchelf=args.patchelf, uv=args.uv, cxx_dir=str(cxx), ffmpeg=str(ffmpeg))
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    write_shell_files(state)
    print(json.dumps({"python": str(python), "venv": str(venv), "glibc": info[2], "state": str(state_path)}, indent=2))


def patch_native(args) -> None:
    state = json.loads(Path(args.state).read_text())
    venv = Path(state["venv"])
    private = Path(state["private"])
    # Validate provenance before touching any installed binary.
    info = json.loads(capture(str(venv / "bin/python"), "-I", "-c",
                             "import sys,json; print(json.dumps([sys.prefix,sys.base_prefix]))"))
    if info != [str(venv), str(private)]:
        raise RuntimeError(f"Refusing to patch an unrelated environment: {info}")
    patched = []
    for directory, _, files in os.walk(venv, followlinks=False):
        for name in files:
            p = Path(directory) / name
            if name.startswith(".fasterwam-native-"):
                continue  # pristine payload used by the narrow helper launcher
            if p.is_symlink() or not p.is_file() or not os.access(p, os.X_OK):
                continue
            if state["profile"] == "core" and (
                p == venv / "bin/ninja"
                or p == venv / "lib/python3.10/site-packages/ninja/data/bin/ninja"
                or p == venv / "lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas"
            ):
                continue  # repair_core.py validates/restores these without ELF rewriting
            if elf_interpreter(p) is None:
                continue
            patch(p, state["loader"], state["rpath"], state["patchelf"])
            patched.append(str(p.relative_to(venv)))
    # Core FFmpeg copies are handled by repair_core.py as well. Rewriting their
    # program headers can crash ET_EXEC helpers before their main() starts.
    report = Path(args.state).parent / "patched-native.json"
    report.write_text(json.dumps(patched, indent=2) + "\n")
    print(f"Patched {len(patched)} environment-owned native executables; report: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--root", required=True)
    prep.add_argument("--profile", choices=("core", "libero", "libero-plus", "robotwin"), required=True)
    prep.add_argument("--patchelf", required=True)
    prep.add_argument("--uv", required=True)
    native = subs.add_parser("patch-native")
    native.add_argument("--state", required=True)
    shell = subs.add_parser("refresh-shell")
    shell.add_argument("--state", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "patch-native": patch_native, "refresh-shell": refresh_shell}[args.command](args)


if __name__ == "__main__":
    main()
