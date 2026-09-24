#!/usr/bin/env python3
"""Check the CentOS 7 private runtime and installed FasterWAM environment.

Run through .runtime/centos7/<profile>/run so the selected profile is respected.
No model downloads, package installation, or dataset changes are performed.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import sysconfig


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def libc_info():
    import ctypes

    version = ctypes.CDLL(None).gnu_get_libc_version
    version.restype = ctypes.c_char_p
    return version().decode("ascii")


def loaded_runtime():
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if len(line.split()) >= 6 and line.split()[-1].startswith("/")}
    return sorted(path for path in paths if re.fullmatch(
        r"(?:libc(?:-[\d.]+)?\.so(?:\.6)?|ld-linux[^/]*|ld-[\d.]+\.so)", Path(path).name))


def elf_interpreter(executable):
    with open(executable, "rb") as stream:
        header = stream.read(64)
        require(header[:5] == b"\x7fELF\x02", f"Expected a 64-bit ELF: {executable}")
        endian = "<" if header[5] == 1 else ">"
        offset = struct.unpack_from(endian + "Q", header, 32)[0]
        entry_size, count = struct.unpack_from(endian + "HH", header, 54)
        for index in range(count):
            stream.seek(offset + index * entry_size)
            entry = stream.read(entry_size)
            if struct.unpack_from(endian + "I", entry)[0] == 3:
                start = struct.unpack_from(endian + "Q", entry, 8)[0]
                size = struct.unpack_from(endian + "Q", entry, 32)[0]
                stream.seek(start)
                return stream.read(size).rstrip(b"\0").decode()
    raise RuntimeError(f"PT_INTERP missing from {executable}")


def spawn_probe(connection):
    try:
        connection.send({"glibc": libc_info(), "executable": sys.executable})
    except Exception as error:
        connection.send({"error": repr(error)})
    finally:
        connection.close()


def runtime_check(runtime_only):
    require(sys.version_info[:2] == (3, 10), f"Python 3.10 required; found {sys.version}")
    current = libc_info()
    expected = os.environ.get("FASTERWAM_EXPECTED_GLIBC", "2.35")
    require(tuple(map(int, current.split("."))) >= tuple(map(int, expected.split("."))),
            f"Loaded glibc {current}; need >= {expected}")
    require(runtime_only or sys.prefix != sys.base_prefix,
            "Full check must run in the selected uv virtual environment")
    mapped = loaded_runtime()
    require(mapped, "No libc/loader mappings found in /proc/self/maps")
    root = os.environ.get("FASTERWAM_GLIBC_ROOT")
    if root:
        root = Path(root).resolve()
        require(all(Path(path).resolve().is_relative_to(root) for path in mapped),
                f"libc/loader loaded outside {root}: {mapped}")
    interpreter = elf_interpreter(sys.executable)
    require(any(Path(interpreter).samefile(path) for path in mapped),
            f"ELF interpreter {interpreter} is absent from loaded runtime: {mapped}")
    require(not os.environ.get("LD_PRELOAD"), "LD_PRELOAD must be unset")
    dangerous = {Path(path).resolve().parent for path in mapped}
    for item in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if item:
            directory = Path(item).resolve()
            require(directory not in dangerous and not (directory / "libc.so.6").exists(),
                    f"LD_LIBRARY_PATH exposes libc to ordinary child programs: {item}")
    for name in ("ssl", "sqlite3", "bz2", "lzma", "ctypes"):
        importlib.import_module(name)
    print(f"    Python: {sys.executable}\n    prefix: {sys.prefix}\n"
          f"    base: {sys.base_prefix}\n    glibc: {current}\n"
          f"    PT_INTERP: {interpreter}\n    mapped: {', '.join(mapped)}")


def subprocess_check():
    code = ("import ctypes,json,sys; f=ctypes.CDLL(None).gnu_get_libc_version; "
            "f.restype=ctypes.c_char_p; "
            "print(json.dumps({'glibc':f().decode(),'executable':sys.executable}))")
    for executable in dict.fromkeys((sys.executable, sys._base_executable)):
        require(bool(executable), "sys._base_executable is missing")
        result = subprocess.run([executable, "-I", "-c", code], check=True,
                                capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        require(info["glibc"] == libc_info(), f"Child runtime differs: {info}")
        require(elf_interpreter(executable) == elf_interpreter(sys.executable),
                f"Child ELF uses a different loader: {executable}")
        print(f"    child: {info['executable']} (glibc {info['glibc']})")
    result = subprocess.run(["/bin/sh", "-c", "printf shell-ok"], check=True,
                            capture_output=True, text=True, timeout=30)
    require(result.stdout == "shell-ok", "System /bin/sh did not execute normally")


def multiprocessing_check():
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=spawn_probe, args=(sender,))
    process.start()
    sender.close()
    try:
        require(receiver.poll(30), "Spawn child timed out before reporting its runtime")
        info = receiver.recv()
        process.join(10)
        require(process.exitcode == 0, f"Spawn child exit code: {process.exitcode}")
        require(info.get("glibc") == libc_info(), f"Spawn child runtime failure: {info}")
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(5)


def package_check(profile):
    robotwin = profile == "robotwin"
    versions = {"torch": "2.4.1+cu121" if robotwin else "2.7.1+cu128",
                "torchvision": "0.19.1+cu121" if robotwin else "0.22.1+cu128",
                "pyarrow": "23.0.0", "av": "16.0.1", "transformers": "4.49.0",
                "datasets": "3.6.0", "accelerate": "1.12.0",
                "h5py": "3.16.0" if robotwin else "3.14.0"}
    if profile == "core":
        versions.update(torchcodec="0.5+cu128", deepspeed="0.18.5")
    elif profile.startswith("libero"):
        versions.update(mujoco="3.3.2", robosuite="1.4.0")
        if profile == "libero-plus":
            versions.update(wand="0.6.13")
    else:
        versions.update(sapien="3.0.0b1", mplib="0.2.1")
    errors = []
    for module, expected in versions.items():
        try:
            actual = importlib.metadata.version(module)
            require(actual == expected, f"{module}: expected {expected}, found {actual}")
            importlib.import_module(module)
            print(f"    {module}=={actual}")
        except Exception as error:
            errors.append(f"{module}: {error}")
    try:
        if profile.startswith("libero"):
            # Importing LIBERO can prompt for paths and change user configuration.
            print(f"    libero=={importlib.metadata.version('libero')} (metadata only)")
            if profile == "libero-plus":
                try:
                    importlib.import_module("wand.api")
                    importlib.import_module("wand.image")
                    print("    Wand / native MagickWand import passed")
                except Exception as error:
                    raise RuntimeError(
                        "Wand cannot load native ImageMagick/MagickWand. Install/use native "
                        "ImageMagick, set MAGICK_HOME and FASTERWAM_EXTRA_LIB_DIRS, then "
                        f"re-run setup. Original error: {error}") from error
        elif robotwin:
            importlib.import_module("curobo")
            print("    curobo import passed")
    except Exception as error:
        errors.append(f"benchmark package: {error}")
    require(not errors, "\n".join(errors))


def native_tools_check(profile):
    site = Path(sysconfig.get_path("purelib"))
    candidates = {Path(sys.prefix) / "bin" / name for name in ("ptxas", "ninja")}
    for package in ("torch", "triton", "ninja", "nvidia"):
        package_root = site / package
        if package_root.is_dir():
            for name in ("ptxas", "ninja"):
                candidates.update(package_root.rglob(name))
    ffmpeg = Path(__file__).resolve().parents[3] / ".runtime" / "centos7" / profile / "bin" / "ffmpeg"
    candidates.add(ffmpeg)
    errors, tested = [], set()
    for executable in sorted(candidates):
        if not executable.is_file() or executable.resolve() in tested:
            continue
        tested.add(executable.resolve())
        flag = "-version" if executable.name == "ffmpeg" else "--version"
        try:
            result = subprocess.run([str(executable), flag], capture_output=True,
                                    text=True, timeout=30)
            require(result.returncode == 0,
                    f"exit {result.returncode}: {(result.stderr or result.stdout).strip()[-2000:]}")
            output = (result.stdout or result.stderr).strip().splitlines()
            print(f"    {executable}: {output[0] if output else 'executed successfully'}")
        except Exception as error:
            errors.append(f"{executable}: {error}")
    if not tested:
        print("    SKIP ptxas/ninja/private FFmpeg: no installed helper found.")
    require(not errors, "Native helper execution failed:\n" + "\n".join(errors))


def shared_memory_check():
    import gc
    import torch
    import torch.multiprocessing as multiprocessing

    previous = multiprocessing.get_sharing_strategy()
    try:
        # file_system starts torch/bin/torch_shm_manager, exercising its own ELF.
        multiprocessing.set_sharing_strategy("file_system")
        tensor = torch.arange(4).share_memory_()
        require(tensor.is_shared() and tensor.tolist() == [0, 1, 2, 3],
                "file_system shared-memory allocation failed")
        del tensor
        gc.collect()
    finally:
        multiprocessing.set_sharing_strategy(previous)


def torch_check(require_cuda):
    import torch
    import torch.nn.functional as functional

    available = torch.cuda.is_available()
    require(not require_cuda or available, "CUDA required, but torch.cuda.is_available() is False")
    device = "cuda" if available else "cpu"
    values = torch.ones((32, 32), device=device)
    require(torch.allclose(values @ values, torch.full_like(values, 32)), "Matmul failed")
    query = torch.ones((1, 2, 8, 32), device=device)
    result = functional.scaled_dot_product_attention(query, query, query)
    require(torch.allclose(result, query), "Scaled dot-product attention failed")
    if available:
        torch.cuda.synchronize()
        print(f"    CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}")
    else:
        print("    SKIP GPU: CUDA unavailable; CPU checks passed. Rerun --require-cuda on a GPU node.")


def video_check(path, profile):
    import av

    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        print(f"    PyAV CPU decode: {frame.width}x{frame.height}")
    if profile == "core":
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path), device="cpu")
        frame = decoder[0]
        require(frame.numel() > 0, "TorchCodec returned an empty frame")
        print(f"    TorchCodec CPU decode: {tuple(frame.shape)}")


def render_check(profile):
    require(profile.startswith("libero"), "--render is implemented only for LIBERO profiles")
    backend = os.environ.setdefault("MUJOCO_GL", "osmesa")
    if backend in ("osmesa", "egl"):
        require(os.environ.setdefault("PYOPENGL_PLATFORM", backend) == backend,
                "PYOPENGL_PLATFORM must match MUJOCO_GL for OSMesa/EGL rendering")
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom type="sphere" size="0.1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.update_scene(data)
        pixels = renderer.render()
        require(pixels.shape == (64, 64, 3), f"Unexpected render shape: {pixels.shape}")
    print(f"    MuJoCo render passed with MUJOCO_GL={os.environ['MUJOCO_GL']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "libero", "libero-plus", "robotwin"), default="core")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--video", type=Path, help="Read and CPU-decode the first frame of a local video")
    parser.add_argument("--render", action="store_true", help="Test MuJoCo offscreen rendering (LIBERO only)")
    args = parser.parse_args()
    if args.runtime_only and (args.require_cuda or args.video or args.render):
        parser.error("--runtime-only cannot be combined with --require-cuda, --video, or --render")
    # MuJoCo reads this setting during its first import in package_check().
    if args.render:
        backend = os.environ.setdefault("MUJOCO_GL", "osmesa")
        if backend in ("osmesa", "egl"):
            os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    checks = [("private runtime / ELF / environment", lambda: runtime_check(args.runtime_only)),
              ("Python and system-shell subprocesses", subprocess_check),
              ("multiprocessing spawn", multiprocessing_check)]
    if not args.runtime_only:
        checks += [("pinned packages / native imports", lambda: package_check(args.profile)),
                   ("native helper execution", lambda: native_tools_check(args.profile)),
                   ("PyTorch execution", lambda: torch_check(args.require_cuda))]
        if args.profile == "core":
            checks.append(("PyTorch shared-memory manager", shared_memory_check))
        if args.video:
            checks.append(("local video decode", lambda: video_check(args.video, args.profile)))
        else:
            print("SKIP actual video decoding: supply --video /path/to/video.mp4 to test it.")
        if args.render:
            checks.append(("MuJoCo rendering", lambda: render_check(args.profile)))
        elif args.profile != "core":
            print("SKIP simulator rendering: imports alone do not validate EGL/Vulkan or an evaluation task.")
    failures = 0
    for label, check in checks:
        try:
            check()
            print(f"PASS {label}", flush=True)
        except Exception as error:
            failures += 1
            print(f"FAIL {label}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    print(f"Doctor finished: {len(checks) - failures} passed, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
