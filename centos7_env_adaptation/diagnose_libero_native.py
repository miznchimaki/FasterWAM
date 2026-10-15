#!/usr/bin/env python3
"""Read-only library diagnostics; run with the private LIBERO Python from repo root.

Only a timestamped diagnostic log is written. No library, permission, package,
environment configuration, or model is changed. No rendering is performed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys


LOAD = """
import ctypes, pathlib, sys
try:
    handle = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
    print('LOADED', sys.argv[1], flush=True)
finally:
    for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
        if any(s in line for s in ('librt', 'libEGL', 'libOpenGL', 'libGL', 'libOSMesa', 'libc.so', 'ld-linux')):
            print('MAP', line, flush=True)
"""

IMPORT = """
import os, pathlib
print('MUJOCO_GL=', os.environ.get('MUJOCO_GL'), flush=True)
print('PYOPENGL_PLATFORM=', os.environ.get('PYOPENGL_PLATFORM'), flush=True)
try:
    import mujoco
    print('mujoco import OK', flush=True)
finally:
    for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
        if any(s in line for s in ('librt', 'libEGL', 'libOpenGL', 'libGL', 'libc.so', 'ld-linux')):
            print('MAP', line, flush=True)
"""


def describe(path: Path, emit) -> None:
    """Report access failures explicitly: Path.exists() can hide some errors."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        emit(f"ABSENT {path}")
        return
    except OSError as error:
        emit(f"STAT ERROR {path}: {error}")
        return
    emit(f"FILE {path}: {stat.filemode(info.st_mode)} uid={info.st_uid} gid={info.st_gid}")
    if stat.S_ISLNK(info.st_mode):
        try:
            emit(f"  link -> {os.readlink(path)}; resolved -> {path.resolve(strict=True)}")
        except (OSError, RuntimeError) as error:
            emit(f"  LINK ERROR: {error}")
    try:
        with path.open('rb') as stream:
            emit(f"  READ OK first4={stream.read(4)!r}")
    except OSError as error:
        emit(f"  READ ERROR: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    state_dir = root / '.runtime/centos7/libero'
    state = json.loads((state_dir / 'state.json').read_text())
    venv = root / '.venvs/libero'
    private = state_dir / 'python-base'
    if (state.get('root') != str(root) or state.get('profile') != 'libero'
            or state.get('venv') != str(venv) or state.get('private') != str(private)
            or Path(sys.prefix) != venv or Path(sys.base_prefix) != private):
        raise RuntimeError('Run from repo root using .runtime/centos7/libero/run python <this-script>.')
    output_dir = state_dir / 'diagnostics'
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = output_dir / f'native-{stamp}-{os.getpid()}.log'
    with output.open('x') as stream:
        def emit(value=''):
            stream.write(str(value) + '\n')
            stream.flush()

        env = os.environ.copy()
        for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_DEBUG', 'LD_DEBUG_OUTPUT', 'PYTHONHOME', 'PYTHONPATH'):
            env.pop(key, None)
        env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONNOUSERSITE='1')

        def run(label, command, *, debug=False, timeout=30):
            print(f'Checking: {label}', flush=True)
            emit('\n===== ' + label + ' =====')
            emit('COMMAND ' + repr(command))
            child_env = env.copy()
            if debug:
                child_env['LD_DEBUG'] = 'libs,files'
            try:
                with subprocess.Popen(command, env=child_env, text=True,
                                      errors='replace', stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, start_new_session=True) as child:
                    try:
                        stdout, stderr = child.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        # Also stop a traced Python child if strace itself hangs.
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        stdout, stderr = child.communicate()
                        emit('TIMEOUT; diagnostic process group was terminated')
                        emit(stdout)
                        emit(stderr)
                        return None
                    result = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
                emit(f'RETURNCODE {result.returncode}')
                emit(result.stdout)
                emit(result.stderr)
                return result
            except OSError as error:
                emit(f'EXEC ERROR: {error}')
            return None

        emit(f'Python: {sys.executable}\nVersion: {sys.version}\nprefix: {sys.prefix}\nbase: {sys.base_prefix}')
        emit(f'uid={os.getuid()} gid={os.getgid()} groups={os.getgroups()}')
        for key in ('loader', 'glibc', 'source_prefix', 'rpath'):
            emit(f'STATE {key}: {state.get(key)}')
        for key in ('MUJOCO_GL', 'PYOPENGL_PLATFORM', 'CUDA_VISIBLE_DEVICES', 'MUJOCO_EGL_DEVICE_ID',
                    '__EGL_VENDOR_LIBRARY_FILENAMES', '__EGL_VENDOR_LIBRARY_DIRS'):
            emit(f'ORIGINAL ENV {key}: {os.environ.get(key)!r}')
        private_rt = Path(state['loader']).parent / 'librt.so.1'
        emit('\n===== Private librt and parent directory access =====')
        describe(private_rt, emit)
        for parent in reversed(private_rt.parents):
            try:
                info = parent.stat()
                emit(f'DIR {parent}: {stat.filemode(info.st_mode)} uid={info.st_uid} gid={info.st_gid} search={os.access(parent, os.X_OK)}')
            except OSError as error:
                emit(f'DIR ERROR {parent}: {error}')
        if shutil.which('namei'):
            run('private librt symlink/permission chain', ['namei', '-l', str(private_rt)])
        if shutil.which('getfacl'):
            run('private librt ACL', ['getfacl', '-p', str(private_rt)])

        site = venv / 'lib/python3.10/site-packages'
        objects = [Path(sys.executable)] + sorted((site / 'mujoco').glob('_specs*.so')) + sorted((site / 'mujoco').glob('libmujoco.so*'))
        for obj in objects:
            describe(obj, emit)
            if shutil.which('readelf'):
                run('ELF dynamic tags: ' + obj.name, ['readelf', '-d', str(obj)])
            elif state.get('patchelf'):
                for flag in ('--print-rpath', '--print-needed'):
                    run('ELF ' + flag + ': ' + obj.name, [state['patchelf'], flag, str(obj)])

        # Each dlopen runs in a FRESH process, so a successful absolute load
        # cannot accidentally repair/mask the next SONAME or MuJoCo test.
        for library in (str(private_rt), 'librt.so.1', 'libEGL.so.1', 'libOpenGL.so.0',
                        'libGL.so.1', 'libGLdispatch.so.0', 'libEGL_nvidia.so.0', 'libOSMesa.so.8'):
            run('fresh dlopen ' + library, [sys.executable, '-I', '-B', '-u', '-c', LOAD, library])
        result = run('fresh MuJoCo import with loader trace',
                     [sys.executable, '-I', '-B', '-u', '-c', IMPORT], debug=True)
        if result:
            candidates = set(re.findall(r'trying file=(\S+)', result.stderr))
            for candidate in sorted(candidates):
                if Path(candidate).name.startswith(('librt.so', 'libEGL.so', 'libOpenGL.so', 'libGL.so')):
                    describe(Path(candidate), emit)

        # This separate, disposable process tests whether the private copy is
        # usable and bypasses the first import failure. It is NOT a repair.
        preload = ('import ctypes, sys; '
                   'ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL); '
                   'print("Private librt preload OK (diagnostic only)", flush=True)\n')
        run('DIAGNOSTIC ONLY: private librt preload then fresh MuJoCo import',
            [sys.executable, '-I', '-B', '-u', '-c', preload + IMPORT, str(private_rt)])

        if shutil.which('strace'):
            run('fresh MuJoCo import with file syscall trace',
                ['strace', '-f', '-s', '256', '-e', 'trace=file', sys.executable,
                 '-I', '-B', '-u', '-c', IMPORT])
        else:
            emit('SKIP strace: not installed. No installation is required to collect the other checks.')
        emit('\n===== EGL vendor manifests =====')
        directories = {Path('/usr/share/glvnd/egl_vendor.d'), Path('/etc/glvnd/egl_vendor.d')}
        for entry in os.environ.get('__EGL_VENDOR_LIBRARY_DIRS', '').split(':'):
            if entry:
                directories.add(Path(entry))
        manifests = {Path(p) for p in os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES', '').split(':') if p}
        for directory in sorted(directories):
            try:
                manifests.update(directory.glob('*.json'))
            except OSError as error:
                emit(f'MANIFEST DIR ERROR {directory}: {error}')
        for manifest in sorted(manifests):
            try:
                emit(f'{manifest}:\n{manifest.read_text()}')
            except OSError as error:
                emit(f'MANIFEST ERROR {manifest}: {error}')
        emit('END OF DIAGNOSTICS')
    print(f'\nDiagnostic report: {output}')
    print('Send the complete report. Failed probes are diagnostic findings, not new installation failures.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
