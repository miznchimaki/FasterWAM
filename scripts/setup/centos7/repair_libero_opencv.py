#!/usr/bin/env python3
"""Keep exactly one headless OpenCV provider in the private LIBERO environment.

Use .runtime/centos7/libero/run python <this-file> --repair after applying the
matching pyproject/lock fix. No GPU or rendering context is created here.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


VERSION = "4.11.0.86"
PROVIDERS = {"opencv-python", "opencv-python-headless"}
CHECK = r'''
import importlib.metadata as md
import re
import cv2
assert md.version("opencv-python-headless") == "4.11.0.86"
for name in ("opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"):
    try:
        md.version(name)
    except md.PackageNotFoundError:
        continue
    raise RuntimeError("Conflicting cv2 provider is still installed: " + name)
assert cv2.__version__ == "4.11.0", cv2.__version__
gui = re.search(r"^\s*GUI:\s*(\S+)", cv2.getBuildInformation(), re.M)
assert gui and gui.group(1) == "NONE", "cv2 is not the headless build"
import numpy as np
assert cv2.resize(np.zeros((2, 2, 3), dtype=np.uint8), (4, 4)).shape == (4, 4, 3)
print("PASS OpenCV: opencv-python-headless==4.11.0.86, GUI=NONE, resize OK")
print("    cv2:", cv2.__file__)
'''


def versions() -> dict[str, str]:
    importlib.invalidate_caches()
    return {d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in importlib.metadata.distributions() if d.metadata["Name"]}


def probe(python: str) -> subprocess.CompletedProcess:
    # A fresh process is essential: repair replaces the cv2 shared object.
    return subprocess.run([python, "-I", "-B", "-c", CHECK], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ensure_headless(python: str, uv: str, before: dict[str, str], repair: bool) -> None:
    result = probe(python)
    if result.returncode == 0:
        print(result.stdout, end="")
        return
    print("DIAGNOSTIC OpenCV before repair:", flush=True)
    print(result.stdout + result.stderr, end="", flush=True)
    if not repair:
        raise RuntimeError("OpenCV check failed. Apply the headless dependency fix, then use --repair.")
    if any(name.startswith("opencv-contrib-") for name in before):
        raise RuntimeError("Unexpected opencv-contrib installation; refusing to remove a custom provider.")
    if before.get("numpy") != "1.26.4":
        raise RuntimeError("Expected the locked numpy==1.26.4; resolve this difference before repair.")
    for name in PROVIDERS:
        if name in before and before[name] != VERSION:
            raise RuntimeError(f"Unexpected {name}=={before[name]}; refusing an implicit version change.")
    # Both distributions own cv2 files. Removing only one can break the other;
    # uninstall both before rebuilding the namespace from a single wheel.
    subprocess.run([uv, "pip", "uninstall", "--python", python,
                    "opencv-python", "opencv-python-headless"], check=True)
    subprocess.run([uv, "pip", "install", "--python", python,
                    "--no-deps", "--only-binary", ":all:",
                    "--index", "pypi=https://pypi.tuna.tsinghua.edu.cn/simple",
                    f"opencv-python-headless=={VERSION}"], check=True)
    after = versions()
    if {k: v for k, v in before.items() if k not in PROVIDERS} != {
            k: v for k, v in after.items() if k not in PROVIDERS}:
        raise RuntimeError("Unexpected change to a non-OpenCV package; inspect the installation log.")
    result = probe(python)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError("Headless OpenCV was reinstalled but its import/check still fails.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", action="store_true", help="Repair the two known conflicting providers")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    state_dir = root / ".runtime/centos7/libero"
    state = json.loads((state_dir / "state.json").read_text())
    venv = root / ".venvs/libero"
    private = state_dir / "python-base"
    if (state.get("root") != str(root) or state.get("profile") != "libero"
            or state.get("venv") != str(venv) or state.get("private") != str(private)
            or Path(sys.prefix) != venv or Path(sys.base_prefix) != private):
        raise RuntimeError("Run this helper with this repository's private LIBERO Python only.")
    uv = state_dir / "bin/uv"
    if not uv.is_file() or not os.access(uv, os.X_OK):
        raise RuntimeError(f"Missing prepared LIBERO uv launcher: {uv}")
    ensure_headless(sys.executable, str(uv), versions(), args.repair)


if __name__ == "__main__":
    main()
