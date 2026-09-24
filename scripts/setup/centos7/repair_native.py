#!/usr/bin/env python3
"""Repair known native helpers using verified pristine executables.

This module does not patch ELF files or export a library path to child shells.
Its caller supplies uv_install(spec, target), which installs exactly one pinned
distribution into a separate target with no dependencies and copy link mode.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import struct
import subprocess
import tempfile


MARKER = "# FasterWAM native helper; managed by repair_native.py"
PAYLOAD_PREFIX = ".fasterwam-native-"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _elf(path):
    try:
        with Path(path).open("rb") as stream:
            header = stream.read(64)
            return (len(header) == 64 and header[:6] == b"\x7fELF\x02\x01"
                    and struct.unpack_from("<H", header, 18)[0] == 62
                    and struct.unpack_from("<H", header, 16)[0] in (2, 3))
    except OSError:
        return False


def _no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _probe(command, cwd=None):
    env = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    try:
        result = subprocess.run([str(arg) for arg in command], cwd=cwd,
                                env=env, capture_output=True, text=True,
                                errors="replace", timeout=10, preexec_fn=_no_core)
        return {"command": list(map(str, command)), "returncode": result.returncode,
                "stdout": result.stdout[-12000:], "stderr": result.stderr[-12000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(map(str, command)), "returncode": None,
                "stdout": "", "stderr": str(error)}


def _version(name):
    return ["-version"] if name in ("ffmpeg", "ffprobe") else ["--version"]


def _smoke(command, name):
    result = _probe(command + _version(name))
    if result["returncode"] or result["returncode"] is None or name == "ffprobe":
        return result
    with tempfile.TemporaryDirectory(prefix="fasterwam-native-check-") as work:
        work = Path(work)
        if name == "ninja":
            (work / "build.ninja").write_text(
                "rule check\n  command = /bin/sh -c 'printf ready > result.txt'\n"
                "build result.txt: check\n")
            result = _probe(command + ["-f", "build.ninja"], cwd=work)
            output = work / "result.txt"
            valid = output.is_file() and output.read_text() == "ready"
        elif name == "ptxas":
            source = work / "check.ptx"
            source.write_text(".version 7.0\n.target sm_80\n.address_size 64\n"
                              ".visible .entry fasterwam_probe() { ret; }\n")
            output = work / "check.cubin"
            result = _probe(command + ["-arch=sm_80", str(source), "-o", str(output)])
            valid = output.is_file() and output.stat().st_size > 0
        elif name == "ffmpeg":
            frame = work / "frame.rgb"
            frame.write_bytes(bytes(range(12)))
            result = _probe(command + ["-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "2x2",
                "-framerate", "1", "-i", str(frame),
                "-frames:v", "1", "-c:v", "rawvideo", "-threads", "1", "-f", "null", "-"])
            valid = True
        else:
            raise ValueError(f"Unknown helper: {name}")
        if result["returncode"] == 0 and not valid:
            result.update(returncode=1, stderr="Native smoke did not produce its expected output")
        return result


def _atomic_copy(source, destination):
    fd, temporary = tempfile.mkstemp(prefix=PAYLOAD_PREFIX + "copy-", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _owned(path, root):
    if not path.is_absolute() or not path.is_relative_to(root) or path.resolve() != path:
        raise RuntimeError(f"Refusing a symlink or unowned repair path: {path}")


def repair_helpers(state: dict, uv_install) -> None:
    """Repair only known core helpers; fail unless real execution checks pass."""
    if state.get("profile") != "core":
        raise RuntimeError("Native helper recovery currently supports the core profile only")
    root, venv, private = (Path(state[key]) for key in ("root", "venv", "private"))
    state_dir = private.parent
    for path in (venv, private, state_dir):
        _owned(path, root)
    identity = _probe([venv / "bin/python", "-I", "-c",
        "import sys,json; print(json.dumps([sys.prefix,sys.base_prefix]))"])
    if identity["returncode"] != 0 or json.loads(identity["stdout"]) != [str(venv), str(private)]:
        raise RuntimeError(f"Refusing an unrelated or broken Python environment: {identity}")
    site = venv / "lib/python3.10/site-packages"
    versions = {d.metadata["Name"].lower(): d.version
                for d in metadata.distributions(path=[str(site)]) if d.metadata["Name"]}
    manifest_path = state_dir / "native-repairs.json"
    _owned(manifest_path, root)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"version": 1, "repairs": {}}
    if manifest.get("version") != 1 or not isinstance(manifest.get("repairs"), dict):
        raise RuntimeError(f"Unrecognized native repair manifest: {manifest_path}")
    failures, staged = [], {}
    targets = [(state_dir / "bin" / name, name, None) for name in ("ffmpeg", "ffprobe")]
    targets += [(site / "ninja/data/bin/ninja", "ninja", "ninja"),
                (venv / "bin/ninja", "ninja", "ninja"),
                (site / "triton/backends/nvidia/bin/ptxas", "ptxas", "triton")]
    diagnostics = state_dir / "native-repair-diagnostics.json"

    def record_failure(target, phase, result):
        failures.append({"target": str(target), "phase": phase, **result})
        minimum = Path("/proc/sys/vm/mmap_min_addr")
        try:
            mmap_min_addr = minimum.read_text().strip()
        except OSError:
            mmap_min_addr = None
        _save(diagnostics, {"uname": list(os.uname()), "mmap_min_addr": mmap_min_addr,
                            "failures": failures})
        print(f"DIAGNOSTIC {phase}: {target} (returncode={result['returncode']})", flush=True)
        for label in ("stdout", "stderr"):
            if result.get(label):
                print(f"  {label}:\n{result[label]}", flush=True)

    def loader_command(binary, target):
        # Include the empty main-object name and both possible object spellings.
        return [state["loader"], "--inhibit-rpath", f":{binary}:{binary.name}",
                "--library-path", ":".join(state["rpath"]), "--argv0", str(target), str(binary)]

    with tempfile.TemporaryDirectory(prefix="native-pristine-", dir=state_dir) as staging:
        for target, name, package in targets:
            _owned(target, root)
            existed = target.exists()
            if not target.exists():
                if target == site / "ninja/data/bin/ninja":
                    continue  # Older Ninja wheels used this additional layout.
                if name == "ffprobe" and not (Path(state["ffmpeg"]) / "bin/ffprobe").exists():
                    print("SKIP optional ffprobe: absent from the custom FFmpeg build", flush=True)
                    continue
                if package:
                    raise RuntimeError(f"Expected installed helper is missing: {target}")
                target.parent.mkdir(exist_ok=True)
            current = _probe([str(target)] + _version(name))
            if current["returncode"] == 0:
                functional = _smoke([str(target)], name)
                if functional["returncode"] != 0:
                    record_failure(target, "functional-smoke-existing", functional)
                    raise RuntimeError(f"Helper version works but functional smoke failed; left unchanged: {target}; see {diagnostics}")
                old = manifest["repairs"].get(str(target), {})
                if old.get("target_sha256") == _sha(target) and old.get("payload"):
                    previous_payload = Path(old["payload"])
                    _owned(previous_payload, root)
                    if not previous_payload.is_file() or _sha(previous_payload) != old.get("payload_sha256"):
                        raise RuntimeError(f"Managed helper payload hash mismatch: {previous_payload}")
                if old.get("target_sha256") != _sha(target):
                    manifest["repairs"][str(target)] = dict(mode="existing", target_sha256=_sha(target),
                        payload=None, payload_sha256=None, source_sha256=None)
                _save(manifest_path, manifest)
                print(f"PASS native helper unchanged: {target}")
                continue
            record_failure(target, "installed", current)
            if existed and not _elf(target) and MARKER not in target.read_text(errors="replace")[:300]:
                raise RuntimeError(f"Refusing to replace an unrecognized non-ELF helper: {target}")
            if package:
                version = versions.get(package)
                if not version:
                    raise RuntimeError(f"Cannot determine installed {package} version")
                if package not in staged:
                    destination = Path(staging) / package
                    uv_install(f"{package}=={version}", destination)
                    staged[package] = destination
                destination = staged[package]
                candidates = ([destination / "bin/ninja", destination / "ninja/data/bin/ninja"]
                              if package == "ninja" else [destination / "triton/backends/nvidia/bin/ptxas"])
            else:
                candidates = [Path(state["ffmpeg"]) / "bin" / name]
            source = next((path.resolve() for path in candidates if _elf(path)), None)
            if source is None or not os.access(source, os.X_OK):
                raise RuntimeError(f"Pristine x86_64 ELF helper not found: {candidates}")
            fingerprint = _sha(source)
            payload = target.with_name(PAYLOAD_PREFIX + name + "-" + fingerprint[:16])
            _owned(payload, root)
            if payload.exists() and _sha(payload) != fingerprint:
                raise RuntimeError(f"Existing pristine payload hash mismatch: {payload}")
            created_payload = not payload.exists()
            if created_payload:
                _atomic_copy(source, payload)
            direct = _smoke([str(payload)], name)
            mode = "direct"
            if direct["returncode"] != 0:
                record_failure(target, "pristine-direct", direct)
                explicit = _smoke(loader_command(payload, target), name)
                if explicit["returncode"] != 0:
                    record_failure(target, "pristine-private-loader", explicit)
                    for label, command in (("readelf", ["readelf", "-h", "-l", "-d", str(target)]),
                            ("loader-list", [state["loader"], "--library-path", ":".join(state["rpath"]), "--list", str(payload)])):
                        record_failure(target, label, _probe(command))
                    if created_payload:
                        payload.unlink()
                    raise RuntimeError(f"Neither pristine execution mode works for {target}; see {diagnostics}")
                mode = "loader-wrapper"
            backups = state_dir / "repair-backups"
            _owned(backups, root)
            backups.mkdir(exist_ok=True)
            backup = None
            if existed:
                backup = backups / (hashlib.sha256(str(target).encode()).hexdigest()[:16] + "-" + _sha(target) + ".bin")
                _owned(backup, root)
                if not backup.exists():
                    _atomic_copy(target, backup)
                if _sha(backup) != _sha(target):
                    raise RuntimeError(f"Backup hash mismatch: {backup}")
            try:
                if mode == "direct":
                    _atomic_copy(payload, target)
                else:
                    fd, temporary = tempfile.mkstemp(prefix=PAYLOAD_PREFIX + "wrapper-", dir=target.parent)
                    try:
                        with os.fdopen(fd, "w") as stream:
                            stream.write("#!/bin/sh\n" + MARKER + "\n"
                                "unset LD_LIBRARY_PATH LD_PRELOAD\nexec "
                                + " ".join(map(shlex.quote, loader_command(payload, target))) + ' "$@"\n')
                        os.chmod(temporary, 0o755)
                        os.replace(temporary, target)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                deployed = _smoke([str(target)], name)
                if deployed["returncode"] != 0:
                    record_failure(target, "deployed", deployed)
                    raise RuntimeError(f"Deployed helper failed; original restored: {target}; see {diagnostics}")
                manifest["repairs"][str(target)] = dict(mode=mode, target_sha256=_sha(target),
                    payload=str(payload) if mode == "loader-wrapper" else None,
                    payload_sha256=fingerprint if mode == "loader-wrapper" else None,
                    source_sha256=fingerprint)
                _save(manifest_path, manifest)
            except BaseException:
                if backup is not None:
                    _atomic_copy(backup, target)
                else:
                    target.unlink(missing_ok=True)
                if created_payload:
                    payload.unlink(missing_ok=True)
                raise
            if mode == "direct" and created_payload:
                payload.unlink()
            print(f"PASS native helper repaired ({mode}): {target}")
