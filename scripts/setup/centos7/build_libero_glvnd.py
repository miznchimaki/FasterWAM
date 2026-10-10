#!/usr/bin/env python3
"""Build a private, matching GLVND EGL/OpenGL/GLdispatch bundle for LIBERO.

No system packages, driver installation, or venv packages are changed. Downloads
are pinned by SHA256. The existing compiler, C headers and nm must work on
this host. Ninja is bootstrapped privately, independently of the LIBERO lock.
Call build_glvnd from repair_libero_loader.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

SOURCE_URL = 'https://codeload.github.com/NVIDIA/libglvnd/tar.gz/refs/tags/v1.7.0'
SOURCE_SHA256 = '073e7292788d4d3eeb45ea6c7bdcce9bfdb3b3eef8d7dbd47f2f30dce046ef98'
MESON_URL = ('https://files.pythonhosted.org/packages/55/a6/'
             '47b9353c331318a13eb050887eacfd61eb075746285f9baf7ef7de6ae235/'
             'meson-1.5.2-py3-none-any.whl')
MESON_SHA256 = '77706e2368a00d789c097632ccf4fc39251fba56d03e1e1b262559a3c7a08f5b'
NINJA_FILENAME = 'ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl'
NINJA_URL = ('https://files.pythonhosted.org/packages/eb/7a/'
             '455d2877fe6cf99886849c7f9755d897df32eaf3a0fba47b56e615f880f7/' + NINJA_FILENAME)
NINJA_SHA256 = '096487995473320de7f65d622c3f1d16c3ad174797602218ca8c967f51ec38a0'
NINJA_MEMBER = 'ninja-1.11.1.4.data/scripts/ninja'
LIBRARIES = ('libEGL.so.1', 'libOpenGL.so.0', 'libGLdispatch.so.0')
OPTIONS = ('-Dx11=disabled', '-Dglx=disabled', '-Degl=true', '-Dgles1=false',
           '-Dgles2=false', '-Dheaders=false', '-Dhgl=false')


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _owned_directory(path: Path) -> None:
    if path.is_symlink() or path.absolute() != path.resolve():
        raise RuntimeError(f'Refusing symlink build directory: {path}')
    path.mkdir(parents=True, exist_ok=True)


def _artifact(url: str, expected: str, cache: Path, filename: str,
              local: str | Path | None) -> Path:
    path = Path(local).expanduser().resolve() if local else cache / filename
    if not path.exists() and local:
        raise RuntimeError(f'Offline build input does not exist: {path}')
    if not path.exists():
        print(f'DOWNLOAD {url}', flush=True)
        fd, temporary = tempfile.mkstemp(prefix=filename + '.part-', dir=cache)
        try:
            with os.fdopen(fd, 'wb') as output:
                request = urllib.request.Request(url, headers={'User-Agent': 'FasterWAM-LIBERO-GLVND/1'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if not response.url.startswith('https://'):
                        raise RuntimeError('Refusing non-HTTPS download redirect')
                    total = 0
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > 10 * 1024 * 1024:
                            raise RuntimeError(f'Unexpectedly large build input: {url}')
                        output.write(block)
            if _digest(Path(temporary)) != expected:
                raise RuntimeError(f'Download checksum mismatch: {url}')
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if not path.is_file() or path.is_symlink() or _digest(path) != expected:
        raise RuntimeError(f'Build input checksum mismatch: {path}; expected SHA256 {expected}')
    print(f'VERIFIED SHA256 {path}', flush=True)
    return path


def _extract_source(archive: Path, destination: Path) -> Path:
    # The exact authenticated archive contains one internal uthash symlink.
    # Validate all paths and links before extracting, including on Python 3.10.
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        symlinks = set()
        for item in members:
            name = PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0] != 'libglvnd-1.7.0':
                raise RuntimeError(f'Unsafe source archive path: {item.name}')
            if not (item.isfile() or item.isdir() or item.issym()):
                raise RuntimeError(f'Unexpected source archive member: {item.name}')
            if item.issym():
                link = PurePosixPath(item.linkname)
                if link.is_absolute() or '..' in link.parts:
                    raise RuntimeError(f'Unsafe source archive link: {item.name}')
                symlinks.add(name)
        for item in members:
            if any(parent in symlinks for parent in PurePosixPath(item.name).parents):
                raise RuntimeError(f'Source archive writes through a link: {item.name}')
        source.extractall(destination, members=members)
    return destination / 'libglvnd-1.7.0'


def _extract_meson(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as wheel:
        for item in wheel.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or '..' in name.parts:
                raise RuntimeError(f'Unsafe Meson wheel member: {item.filename}')
        wheel.extractall(destination)


def _run(command: list[str], *, env: dict[str, str], cwd: Path | None = None,
         timeout: int = 600) -> None:
    print('BUILD ' + shlex.join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=cwd, timeout=timeout)


def _no_core() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _ninja_smoke(command: list[str], env: dict[str, str], work: Path) -> tuple[bool, str]:
    """Check real command execution, including the system-shell child."""
    output = work / 'result.txt'
    if output.exists():
        output.unlink()
    (work / 'build.ninja').write_text(
        "rule check\n"
        "  command = /bin/sh -c 'test -z \"$${LD_LIBRARY_PATH+x}\" && "
        "test -z \"$${LD_PRELOAD+x}\" && printf ready > result.txt'\n"
        "build result.txt: check\n")
    try:
        version = subprocess.run(command + ['--version'], env=env, cwd=work,
                                 text=True, capture_output=True, timeout=30, preexec_fn=_no_core)
        if version.returncode != 0 or not version.stdout.strip().startswith('1.11.1'):
            return False, f'version returncode={version.returncode}: {version.stdout} {version.stderr}'
        build = subprocess.run(command + ['-f', 'build.ninja'], env=env, cwd=work,
                               text=True, capture_output=True, timeout=30, preexec_fn=_no_core)
        if build.returncode != 0 or not output.is_file() or output.read_text() != 'ready':
            return False, f'build returncode={build.returncode}: {build.stdout} {build.stderr}'
        return True, version.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def _prepare_ninja(state: dict, cache: Path, env: dict[str, str], local=None) -> Path:
    # The LIBERO lock has no Ninja dependency. Keep this build tool outside the
    # venv so uv sync cannot remove it, and avoid mutating any installed helper.
    wheel_path = _artifact(NINJA_URL, NINJA_SHA256, cache, NINJA_FILENAME, local)
    directory = Path(tempfile.mkdtemp(prefix='ninja-', dir=cache))
    payload = directory / 'ninja.bin'
    with zipfile.ZipFile(wheel_path) as wheel:
        data = wheel.read(NINJA_MEMBER)
    if (data[:6] != b'\x7fELF\x02\x01' or data[16:20] != b'\x02\x00\x3e\x00'):
        raise RuntimeError('Pinned Ninja wheel did not contain the expected x86_64 executable.')
    with payload.open('xb') as stream:
        stream.write(data)
    payload.chmod(0o755)
    work = directory / 'smoke'
    work.mkdir()
    command = [str(payload)]
    ready, detail = _ninja_smoke(command, env, work)
    mode = 'direct'
    if not ready:
        print('DIAGNOSTIC private Ninja direct: ' + detail, flush=True)
        # Keep the original ELF intact: some native helpers are sensitive to
        # patchelf layout changes. This loader flag does not leak into children.
        paths = [str(Path(state['loader']).parent), *state.get('rpath', [])]
        command = [state['loader'], '--library-path', ':'.join(dict.fromkeys(paths)), str(payload)]
        ready, detail = _ninja_smoke(command, env, work)
        mode = 'private-loader'
    if not ready:
        raise RuntimeError('Private Ninja could not execute a shell build: ' + detail)
    launcher = directory / 'ninja'
    launcher.write_text('#!/bin/sh\nunset LD_LIBRARY_PATH LD_PRELOAD\nexec '
                        + shlex.join(command) + ' "$@"\n')
    launcher.chmod(0o755)
    ready, detail = _ninja_smoke([str(launcher)], env, work)
    if not ready:
        raise RuntimeError('Private Ninja launcher failed: ' + detail)
    print(f'PASS private Ninja ({mode}): {detail}; {launcher}', flush=True)
    return launcher


def _probe(libdir: Path, env: dict[str, str]) -> None:
    env = env.copy()
    # Test the frontend itself. Host ICDs, or an earlier repair's ICD, must
    # not turn a successful frontend build into a driver/visibility failure.
    env['__EGL_VENDOR_LIBRARY_FILENAMES'] = ''
    code = '''import ctypes, pathlib, sys
p = pathlib.Path(sys.argv[1])
ctypes.CDLL(str(p / 'libGLdispatch.so.0'), mode=ctypes.RTLD_GLOBAL)
egl = ctypes.CDLL(str(p / 'libEGL.so.1'))
ctypes.CDLL(str(p / 'libOpenGL.so.0'))
egl.eglQueryString.argtypes = (ctypes.c_void_p, ctypes.c_int)
egl.eglQueryString.restype = ctypes.c_char_p
version = egl.eglQueryString(None, 0x3054)
if version != b'1.5 libglvnd':
    raise RuntimeError('Unexpected private EGL version: %r' % (version,))
egl.eglGetProcAddress.argtypes = (ctypes.c_char_p,)
egl.eglGetProcAddress.restype = ctypes.c_void_p
if not egl.eglGetProcAddress(b'eglQueryDevicesEXT'):
    raise RuntimeError('Private GLVND has no eglQueryDevicesEXT dispatch address')
print('PASS private GLVND build:', version.decode(), flush=True)
'''
    _run([sys.executable, '-I', '-B', '-u', '-c', code, str(libdir)], env=env, timeout=60)


def _reusable(path: Path, base: Path, env: dict[str, str]) -> Path | None:
    if not path.exists():
        return None
    if path.is_symlink():
        raise RuntimeError(f'Refusing symlink build manifest: {path}')
    record = json.loads(path.read_text())
    prefix = Path(record.get('prefix', ''))
    if (record.get('source_sha256') != SOURCE_SHA256 or record.get('options') != list(OPTIONS)
            or prefix.parent != base or prefix.is_symlink() or prefix.resolve() != prefix):
        raise RuntimeError(f'Unexpected private GLVND build manifest: {path}')
    libdir = prefix / 'lib'
    for name in LIBRARIES:
        target = (libdir / name).resolve()
        if (target.parent != libdir or not target.is_file()
                or _digest(target) != record.get('libraries', {}).get(name)):
            raise RuntimeError(f'Owned private GLVND build was modified or is incomplete: {libdir / name}')
    _probe(libdir, env)
    print(f'REUSE private GLVND: {libdir}', flush=True)
    return libdir


def build_glvnd(state: dict, source_archive: str | Path | None = None,
                *, meson_wheel: str | Path | None = None,
                ninja_wheel: str | Path | None = None) -> Path:
    """Return an owned libdir with matching libEGL/OpenGL/GLdispatch, reusing it safely."""
    if state.get('profile') != 'libero':
        raise RuntimeError('Private GLVND builder is scoped to the LIBERO runtime.')
    runtime = Path(state['root']) / '.runtime/centos7/libero'
    base = runtime / 'private-glvnd'
    cache = runtime / 'glvnd-build'
    for path in (runtime, base, cache):
        _owned_directory(path)
    env = os.environ.copy()
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_DEBUG', 'LD_DEBUG_OUTPUT', 'PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    manifest = base / 'build.json'
    existing = _reusable(manifest, base, env)
    if existing:
        return existing
    compiler = shlex.split(env.get('CC', 'cc'))
    if not compiler or not shutil.which(compiler[0], path=env.get('PATH')):
        raise RuntimeError('C compiler not found; check FASTERWAM_GCC_ROOT and the runtime launcher.')
    _run(compiler + ['--version'], env=env, timeout=30)
    if not shutil.which('nm', path=env.get('PATH')):
        raise RuntimeError('nm is required; make the existing binutils installation available in PATH.')
    jobs = int(env.get('MAX_JOBS', '2'))
    if jobs < 1:
        raise RuntimeError('MAX_JOBS must be a positive integer.')
    ninja = _prepare_ninja(state, cache, env, local=ninja_wheel)
    source_tar = _artifact(SOURCE_URL, SOURCE_SHA256, cache, 'libglvnd-v1.7.0.tar.gz', source_archive)
    meson_archive = _artifact(MESON_URL, MESON_SHA256, cache, 'meson-1.5.2-py3-none-any.whl', meson_wheel)
    work = Path(tempfile.mkdtemp(prefix='work-', dir=cache))
    prefix = base / ('v1.7.0-' + work.name.removeprefix('work-'))
    tools_dir = work / 'tools'
    tools_dir.mkdir()
    _extract_meson(meson_archive, tools_dir)
    source = _extract_source(source_tar, work)
    build_dir, staging = work / 'build', work / 'staging'
    # Running meson.py explicitly makes Meson's generated Python commands use
    # this private interpreter, without modifying PYTHONPATH or the venv.
    meson_script = tools_dir / 'meson.py'
    meson_script.write_text('import sys\nsys.path.insert(0, ' + repr(str(tools_dir)) + ')\n'
                            'from mesonbuild.mesonmain import main\nraise SystemExit(main())\n')
    meson = [sys.executable, '-I', '-B', str(meson_script)]
    env['NINJA'] = str(ninja)
    _run(meson + ['setup', str(build_dir), str(source), '--prefix=' + str(prefix),
                  '--libdir=lib', '--buildtype=release', '--wrap-mode=nodownload', *OPTIONS], env=env)
    _run([str(ninja), '-C', str(build_dir), '-j', str(jobs)], env=env)
    _run(meson + ['install', '-C', str(build_dir), '--no-rebuild', '--destdir', str(staging)], env=env)
    installed = staging / str(prefix).lstrip('/')
    libdir = installed / 'lib'
    hashes = {}
    for name in LIBRARIES:
        library = (libdir / name).resolve()
        if library.parent != libdir or not library.is_file():
            raise RuntimeError(f'Build did not produce the expected library: {libdir / name}')
        # Stage only our own newly built DSOs; the loader and other libraries
        # remain untouched. Copies into native-libs are handled by the caller.
        rpath = ':'.join(dict.fromkeys([str(Path(state['loader']).parent), str(prefix / 'lib')]))
        _run([state['patchelf'], '--force-rpath', '--set-rpath', rpath, str(library)], env=env, timeout=30)
        hashes[name] = _digest(library)
    os.rename(installed, prefix)
    final_libdir = prefix / 'lib'
    _probe(final_libdir, env)
    record = {'source_url': SOURCE_URL, 'source_sha256': SOURCE_SHA256,
              'meson_sha256': MESON_SHA256, 'ninja_sha256': NINJA_SHA256,
              'prefix': str(prefix), 'options': list(OPTIONS),
              'compiler': compiler, 'libraries': hashes}
    fd, temporary = tempfile.mkstemp(prefix='build.json.new-', dir=base)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(record, output, indent=2)
            output.write('\n')
        os.replace(temporary, manifest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'BUILT private GLVND: {final_libdir}', flush=True)
    return final_libdir
