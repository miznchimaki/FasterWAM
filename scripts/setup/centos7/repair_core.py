#!/usr/bin/env python3
"""Add the missing NPP runtime and repair only known core native helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    state_dir = root / ".runtime/centos7/core"
    state = json.loads((state_dir / "state.json").read_text())
    venv = root / ".venv"
    if (state.get("root") != str(root) or state.get("profile") != "core"
            or state.get("venv") != str(venv) or Path(sys.prefix) != venv
            or Path(sys.base_prefix) != Path(state["private"])):
        raise RuntimeError("Run with this repository's prepared .venv/bin/python.")
    native_libs = state_dir / "native-libs"
    for p in (root / ".runtime", root / ".runtime/centos7", state_dir,
              venv, state_dir / "native-packages", native_libs):
        if p.is_symlink():
            raise RuntimeError(f"Refusing runtime directory symlink: {p}")
    # This repair targets the versions in the reported core lock, not RoboTwin.
    expected = {"torch": "2.7.1+cu128", "torchcodec": "0.5+cu128",
                "ninja": "1.13.0", "triton": "3.3.1"}
    for name, version in expected.items():
        actual = metadata.version(name)
        if actual != version:
            raise RuntimeError(f"Repair expected {name}=={version}, found {actual}.")
    tracked = [root / "pyproject.toml", root / "uv.lock"]
    before = {p: hashlib.sha256(p.read_bytes()).digest() for p in tracked}
    child_env = os.environ.copy()
    for key in list(child_env):
        if key.startswith("UV_") or key in (
                "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
            child_env.pop(key, None)
    child_env.update(UV_PYTHON_DOWNLOADS="never", UV_LINK_MODE="copy", PYTHONNOUSERSITE="1")
    index = os.environ.get("FASTERWAM_BOOTSTRAP_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple")

    def uv_install(spec: str, target: Path) -> None:
        target = Path(target)
        if not target.absolute().is_relative_to(state_dir) or target.is_symlink():
            raise RuntimeError(f"Refusing target outside owned runtime: {target}")
        for parent in target.parents:
            if parent == state_dir:
                break
            if parent.is_symlink():
                raise RuntimeError(f"Refusing symlink in target: {parent}")
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Native-only package: {spec} -> {target}", flush=True)
        subprocess.run([state["uv"], "--no-config", "pip", "install",
                        "--python", str(venv / "bin/python"), "--target", str(target),
                        "--no-deps", "--only-binary", ":all:", "--link-mode", "copy",
                        "--default-index", index, spec], env=child_env, check=True)

    npp_target = state_dir / "native-packages/npp-cu128"
    npp_lib = npp_target / "nvidia/npp/lib"
    if str(native_libs) not in state["rpath"] or not native_libs.is_dir():
        raise RuntimeError("Expected the prepared core native-libs directory in Python RPATH.")
    # CUDA 12.8 GA: cudart 12.8.57, NVRTC 12.8.61, NPP 12.3.3.65.
    uv_install("nvidia-npp-cu12==12.3.3.65", npp_target)
    if not (npp_lib / "libnppicc.so.12").is_file():
        raise RuntimeError(f"NPP install did not provide {npp_lib}/libnppicc.so.12")
    # Reuse a directory already in the working Python's DT_RPATH; do not
    # rewrite its ELF or export new libc paths into shell subprocesses.
    for source in sorted(npp_lib.glob("libnpp*.so*")):
        if not source.is_file() or not source.resolve().is_relative_to(npp_target):
            raise RuntimeError(f"Unexpected NPP library: {source}")
        dest = native_libs / source.name
        if dest.is_symlink() and dest.resolve() == source.resolve():
            continue
        if dest.exists() or dest.is_symlink():
            raise RuntimeError(f"Refusing to replace an unrelated native library: {dest}")
        dest.symlink_to(source.resolve())
    # Check dynamic loading in a fresh process after installing the libraries.
    check = subprocess.run([str(venv / "bin/python"), "-I", "-c",
                            "import torch, torchcodec; "
                            "print('PASS TorchCodec import:', torchcodec.__version__)"],
                           env=child_env, check=False)
    # Repair helpers even if another FFmpeg shared-library dependency is missing;
    # the full doctor below will retain the precise import error.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repair_native import repair_helpers
    repair_helpers(state, uv_install)
    for p, digest in before.items():
        if hashlib.sha256(p.read_bytes()).digest() != digest:
            raise RuntimeError(f"Unexpected dependency-file change: {p}")
    if check.returncode:
        raise RuntimeError("TorchCodec still cannot import; see the dependency error above.")
    print("Core repair completed; pyproject.toml and uv.lock are unchanged.", flush=True)


if __name__ == "__main__":
    main()
