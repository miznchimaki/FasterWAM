"""Plan a private copy of installed EGL libraries; no file mutations or downloads.

The caller owns atomic staging and RPATH repair. System glibc and the CUDA toolkit
are deliberately excluded: the private glibc and existing C++ runtime remain in
charge. Only the selected graphics roots and their ELF dependencies are copied.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import struct
import subprocess


class GraphicsError(RuntimeError):
    def __init__(self, message, *, found=None, missing=(), details=()):
        self.found = dict(found or {})
        self.missing = list(missing)
        self.details = list(details)
        inventory = ["  FOUND {} -> {}".format(k, v) for k, v in sorted(self.found.items())]
        inventory += ["  MISSING " + name for name in self.missing]
        inventory += ["  " + item for item in self.details]
        super().__init__(message + ("\n" + "\n".join(inventory) if inventory else ""))


def _env():
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("LD_") or key in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(key, None)
    env.update(LC_ALL="C", LANG="C")
    return env


def _run(argv, timeout=20):
    return subprocess.run([str(v) for v in argv], env=_env(), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=False)


def _readable_elf(path):
    """Return the real filename only for a readable x86-64 ELF shared object."""
    try:
        real = Path(path).resolve(strict=True)
        if "stubs" in real.parts or not real.is_file():
            return None
        with real.open("rb") as stream:
            header = stream.read(20)
        if (len(header) < 20 or header[:6] != b"\x7fELF\x02\x01"
                or struct.unpack_from("<HH", header, 16) != (3, 62)):
            return None
        return real
    except (OSError, RuntimeError):
        return None


def _cache():
    result = {}
    errors = []
    command = next((p for p in ("/sbin/ldconfig", "/usr/sbin/ldconfig")
                    if os.access(p, os.X_OK)), None)
    if command is None:
        return result, ["Host ldconfig is unavailable; using explicit directories."]
    try:
        proc = _run([command, "-p"])
        if proc.returncode:
            return result, ["Host ldconfig -p failed: " + proc.stderr.strip()]
        for line in proc.stdout.splitlines():
            match = re.match(r"\s*(\S+)\s+\(([^)]*)\)\s+=>\s+(\S+)\s*$", line)
            if match and "x86-64" in match.group(2):
                result.setdefault(match.group(1), []).append(Path(match.group(3)))
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append("Host ldconfig -p failed: " + str(exc))
    return result, errors


def _driver_version():
    try:
        text = Path("/proc/driver/nvidia/version").read_text()
        match = re.search(r"Kernel Module\s+(\d+\.\d+(?:\.\d+)?)", text)
        if match:
            return match.group(1)
    except OSError:
        pass
    command = next((p for p in ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi")
                    if os.access(p, os.X_OK)), None) or shutil.which("nvidia-smi")
    if command:
        try:
            proc = _run([command, "--query-gpu=driver_version", "--format=csv,noheader"], timeout=15)
            versions = set(proc.stdout.strip().splitlines())
            if proc.returncode == 0 and len(versions) == 1:
                version = next(iter(versions)).strip()
                if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
                    return version
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def _basename(name):
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise GraphicsError("Refusing an ELF dependency that is not a plain SONAME: " + repr(name))
    return name


def _glibc_name(name):
    return bool(re.fullmatch(
        r"(?:ld-linux[^/]*\.so(?:\.\d+)*|ld-\d[^/]*\.so|"
        r"lib(?:c|m|mvec|pthread|dl|rt|util|resolv|anl|BrokenLocale|thread_db)\.so(?:\.\d+)*|"
        r"libnss_[A-Za-z0-9_]+\.so(?:\.\d+)*)", name))


PYTHON_SUPPORT_SONAMES = ('libexpat.so.1', 'libz.so.1')


def base_runtime_library(state, name):
    """Preserve the base Python's Expat/zlib ABI when graphics also uses it."""
    if name not in PYTHON_SUPPORT_SONAMES or not state.get('source_prefix'):
        return None
    candidate = Path(state['source_prefix']) / 'lib' / name
    if not candidate.exists() and not candidate.is_symlink():
        return None
    real = _readable_elf(candidate)
    if real is None:
        raise GraphicsError('Base Python support library is unreadable or not x86_64 ELF: ' + str(candidate))
    result = _run([state['patchelf'], '--print-soname', real])
    if result.returncode or result.stdout.strip() != name:
        raise GraphicsError('Unexpected SONAME for base Python support library: ' + str(candidate))
    return real


def plan_graphics(state, extra_dirs=(), glvnd_dir=None):
    """Return ``{load_name: real_source}`` for a host NVIDIA EGL dependency closure.

    ``GraphicsError`` contains any partial inventory and missing libraries. This
    function only reads files and executes ldconfig -p, patchelf inspection and,
    when needed, nvidia-smi's driver-version query. It never creates a GL context.
    The caller must check ownership before replacing existing staging targets.
    """
    glibc = Path(state["glibc"]).resolve()
    glibc_lib = Path(state["loader"]).resolve().parent
    native = (Path(state["private"]).parent / "native-libs").resolve()
    cache, details = _cache()
    driver = _driver_version()
    if driver:
        details.append("Running NVIDIA driver: " + driver)
    else:
        details.append("Driver version unavailable; filename/driver equality could not be checked.")
    conventional = [Path(p) for p in (
        "/usr/lib64", "/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
        "/usr/lib64/nvidia", "/usr/lib64/nvidia-current", "/usr/lib64/nvidia/current",
        "/usr/lib/x86_64-linux-gnu/nvidia/current", "/usr/lib/nvidia-current",
        "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib",
    )]
    # An explicitly supplied private build must win over the host ldconfig
    # cache. CentOS 7 can expose legacy Mesa under the same libEGL.so.1 SONAME.
    dirs = [Path(p).expanduser() for p in extra_dirs] + conventional
    dirs += [Path(p) for p in state["rpath"]]
    dirs = list(dict.fromkeys(p.resolve() for p in dirs
                             if p.is_dir() and p.resolve() != native and "stubs" not in p.parts))
    found = {}
    missing = []
    rejected = []
    vendor_dir = None
    vendor_version = None

    def inspect(source, option):
        try:
            proc = _run([state["patchelf"], option, source])
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GraphicsError("Cannot inspect graphics ELF: " + str(exc), found=found) from exc
        if proc.returncode:
            raise GraphicsError("patchelf {} failed for {}: {}".format(
                option, source, proc.stderr.strip()), found=found)
        return proc.stdout.strip()

    def same_name_files(directory, name):
        # Inspect real versioned files too: an old ldconfig symlink can point
        # at Mesa while an unlinked GLVND libEGL.so.1.1.0 is already installed.
        try:
            values = [directory / name] + sorted(directory.glob(name + ".*"))
        except OSError:
            values = [directory / name]
        return values

    def glvnd_trio(directory):
        chosen = {}
        for name in ("libEGL.so.1", "libOpenGL.so.0", "libGLdispatch.so.0"):
            for candidate in same_name_files(directory, name):
                real = _readable_elf(candidate)
                if real is None or inspect(real, "--print-soname") != name:
                    continue
                if name in ("libEGL.so.1", "libOpenGL.so.0"):
                    if "libGLdispatch.so.0" not in inspect(real, "--print-needed").splitlines():
                        rejected.append("Rejected non-GLVND {}: no libGLdispatch.so.0 dependency".format(real))
                        continue
                if name == "libEGL.so.1":
                    # This string is used by libglvnd's actual vendor loader,
                    # survives stripping, and is absent from legacy Mesa EGL.
                    try:
                        marker = b"__EGL_VENDOR_LIBRARY_FILENAMES" in real.read_bytes()
                    except OSError:
                        marker = False
                    if not marker:
                        rejected.append("Rejected non-GLVND {}: no EGL vendor-loader marker".format(real))
                        continue
                chosen[name] = real
                break
            if name not in chosen:
                return None
        if len({path.parent for path in chosen.values()}) != 1:
            rejected.append("Rejected incoherent GLVND trio in {}: real files are in different directories".format(directory))
            return None
        return chosen

    if glvnd_dir is not None:
        glvnd_dirs = [Path(glvnd_dir).expanduser().resolve()]
        if not glvnd_dirs[0].is_dir() or glvnd_dirs[0] == native:
            raise GraphicsError("--glvnd-dir must be an existing source directory, not native-libs.")
    else:
        # User directories precede cache locations, followed by conventional
        # directories and the runtime's configured search directories.
        glvnd_dirs = [Path(p).expanduser().resolve() for p in extra_dirs]
        glvnd_dirs += [p.parent.resolve() for p in cache.get("libEGL.so.1", [])]
        glvnd_dirs += dirs
        glvnd_dirs = list(dict.fromkeys(p for p in glvnd_dirs
                                      if p.is_dir() and p != native and "stubs" not in p.parts))
    trio = None
    for directory in glvnd_dirs:
        trio = glvnd_trio(directory)
        if trio is not None:
            details.append("GLVND dispatch directory: " + str(directory))
            break

    def locate(name):
        nonlocal vendor_dir, vendor_version
        _basename(name)
        # A copied CentOS 7 libexpat/libz at high priority can break the newer
        # private Python, even though that copy satisfies an old Mesa library.
        # If graphics needs the same SONAME, reuse the base Python's provider.
        support = base_runtime_library(state, name)
        if support is not None:
            return support
        is_nvidia = name.startswith(("libnvidia-", "libEGL_nvidia", "libGLX_nvidia"))
        candidates = ([vendor_dir / name] if is_nvidia and vendor_dir else [])
        candidates += [Path(p).expanduser() / name for p in extra_dirs]
        candidates += cache.get(name, [])
        candidates += [p / name for p in dirs]
        for candidate in dict.fromkeys(candidates):
            real = _readable_elf(candidate)
            if real is None:
                continue
            match = re.search(r"\.so\.(\d{3,}\.\d+(?:\.\d+)?)$", real.name) if is_nvidia else None
            version = match.group(1) if match else None
            expected = driver or vendor_version
            if version and expected and version != expected:
                rejected.append("Rejected {}: version {} differs from {}".format(real, version, expected))
                continue
            if name == "libEGL_nvidia.so.0":
                vendor_dir, vendor_version = real.parent, version
            return real
        return None

    # Choose the NVIDIA vendor first so its own directory is preferred for its
    # companion libraries. The three dispatch roots are a coherent GLVND set;
    # a dlopen-able legacy Mesa library is not sufficient for NVIDIA EGL.
    for name in ("libEGL_nvidia.so.0",):
        path = locate(name)
        if path is None:
            missing.append(name)
        else:
            found[name] = path
    if trio is not None:
        found.update(trio)
    else:
        where = " in " + str(glvnd_dirs[0]) if glvnd_dir is not None else " in searched directories"
        missing.append("coherent GLVND libEGL.so.1 + libOpenGL.so.0 + libGLdispatch.so.0" + where)

    # NVIDIA loads eglcore dynamically, so DT_NEEDED traversal alone misses it.
    # Retain the exact driver's filename; do not substitute a different version.
    selected_version = vendor_version or driver
    if "libEGL_nvidia.so.0" in found:
        if selected_version is None:
            missing.append("NVIDIA vendor version for matching libnvidia-eglcore.so.VERSION")
        else:
            name = "libnvidia-eglcore.so." + selected_version
            path = locate(name)
            if path is None:
                missing.append(name + " (NVIDIA EGL runtime companion)")
            else:
                found[name] = path
            # Some driver versions dlopen this shader compiler at render time.
            # It is optional here because not every EGL driver uses it.
            name = "libnvidia-glvkspirv.so." + selected_version
            path = locate(name)
            if path is not None:
                found[name] = path
    if missing:
        raise GraphicsError("Required host EGL libraries were not found; no files were changed.",
                            found=found, missing=missing, details=details + rejected)

    queue = list(found)
    inspected = set()
    while queue:
        name = queue.pop(0)
        source = found[name]
        if source in inspected:
            continue
        inspected.add(source)
        try:
            proc = _run([state["patchelf"], "--print-needed", source])
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GraphicsError("Cannot inspect graphics dependency: " + str(exc), found=found) from exc
        if proc.returncode:
            raise GraphicsError("patchelf inspection failed for {}: {}".format(source, proc.stderr.strip()),
                                found=found)
        try:
            soname_proc = _run([state["patchelf"], "--print-soname", source])
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GraphicsError("Cannot inspect graphics SONAME: " + str(exc), found=found) from exc
        if soname_proc.returncode:
            raise GraphicsError("patchelf SONAME inspection failed for " + str(source), found=found)
        soname = soname_proc.stdout.strip()
        if soname:
            _basename(soname)
            if _glibc_name(soname) or soname in ("libstdc++.so.6", "libgcc_s.so.1"):
                raise GraphicsError("Refusing a graphics source with reserved SONAME " + soname,
                                    found=found)
            if soname in found and found[soname] != source:
                raise GraphicsError("Conflicting sources for SONAME " + soname, found=found)
            found[soname] = source
        for dep in proc.stdout.splitlines():
            dep = _basename(dep.strip())
            if _glibc_name(dep):
                private = _readable_elf(glibc_lib / dep)
                if private is None or not private.is_relative_to(glibc):
                    missing.append("private glibc: " + str(glibc_lib / dep))
                continue
            if dep in ("libstdc++.so.6", "libgcc_s.so.1"):
                if _readable_elf(native / dep) is None:
                    missing.append("existing C++ runtime: " + str(native / dep))
                continue
            if dep in found:
                continue
            path = locate(dep)
            if path is None:
                missing.append("{} (needed by {})".format(dep, source.name))
            else:
                found[dep] = path
                queue.append(dep)
    if missing:
        raise GraphicsError("Graphics dependency closure is incomplete; no files were changed.",
                            found=found, missing=sorted(set(missing)), details=details + rejected)
    return found
