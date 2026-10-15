#!/usr/bin/env python3
"""Repair PyArrow 23 ELF search paths in one owned CentOS 7 profile.

Run through .runtime/centos7/<profile>/run python. Without --apply this is
read-only. No package installation, downloads, global LD_* changes or GPU use.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

PROFILES = ('core', 'libero', 'libero-plus')
VERSION = '23.0.0'


def clean_env():
    env = os.environ.copy()
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'LD_DEBUG',
                'LD_DEBUG_OUTPUT', 'PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def owned_path(path):
    if path.is_symlink() or path.resolve() != path:
        raise RuntimeError(f'Refusing symlink at owned path: {path}')


def dynamic_tags(path):
    """Return ELF dynamic tags; reject executables or non-x86_64 objects."""
    with path.open('rb') as stream:
        header = stream.read(64)
        if (len(header) != 64 or header[:6] != b'\x7fELF\x02\x01'
                or header[16:20] != b'\x03\x00\x3e\x00'):
            raise RuntimeError(f'Expected x86_64 ELF shared object: {path}')
        offset = struct.unpack_from('<Q', header, 32)[0]
        size, count = struct.unpack_from('<HH', header, 54)
        if size < 56 or count > 4096:
            raise RuntimeError(f'Invalid ELF program headers: {path}')
        tags = set()
        for index in range(count):
            stream.seek(offset + size * index)
            entry = stream.read(size)
            kind = struct.unpack_from('<I', entry)[0]
            if kind == 3:
                raise RuntimeError(f'Refusing a PT_INTERP executable: {path}')
            if kind != 2:
                continue
            start = struct.unpack_from('<Q', entry, 8)[0]
            length = struct.unpack_from('<Q', entry, 32)[0]
            stream.seek(start)
            for tag, value in struct.iter_unpack('<qQ', stream.read(length)):
                if tag == 0:
                    break
                tags.add(tag)
        if not tags:
            raise RuntimeError(f'Missing ELF dynamic section: {path}')
        return tags


def capture(tool, *args):
    return subprocess.check_output([tool, *args], env=clean_env(), text=True,
                                   timeout=60).strip()


def desired_rpath(state, old):
    # PyArrow's sibling libarrow/libparquet must precede broad fallback paths.
    # Private glibc comes first. Keep original $ORIGIN variants and fallbacks.
    existing = [part for part in old.split(':') if part]
    origins = [part for part in existing if '$ORIGIN' in part or '${ORIGIN}' in part]
    paths = [str(Path(state['loader']).parent)]
    paths += origins or ['$ORIGIN']
    # This wheel needs only glibc and the selected C++ runtime outside itself.
    # Do not pull broad torch/NVIDIA/FFmpeg/Conda directories into its search.
    paths += [str(Path(state['private']).parent / 'native-libs')]
    paths += existing
    return ':'.join(dict.fromkeys(paths))


def atomic_json(path, value):
    owned_path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + '.new-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def backup_file(path, directory, sha):
    backup = directory / (sha + '.elf')
    owned_path(backup)
    if backup.exists():
        if digest(backup) != sha:
            raise RuntimeError(f'Backup hash mismatch: {backup}')
        return backup
    fd, name = tempfile.mkstemp(prefix='backup-', dir=directory)
    os.close(fd)
    try:
        shutil.copy2(path, name)
        if digest(Path(name)) != sha:
            raise RuntimeError(f'Source changed while backing up: {path}')
        os.replace(name, backup)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return backup


def patch_object(path, state, backups, record, manifest):
    owned_path(path)
    tags = dynamic_tags(path)
    tool = state['patchelf']
    old = capture(tool, '--print-rpath', str(path))
    wanted = desired_rpath(state, old)
    key = str(path.relative_to(Path(state['root'])))
    before = digest(path)
    if old == wanted and 15 in tags and 29 not in tags:
        print(f'UNCHANGED {path}', flush=True)
        current = record['files'].get(key, {})
        # A completed replacement whose final journal write was interrupted.
        if current.get('patched_sha256') == before:
            current.pop('pending', None)
        else:
            record['files'][key] = dict(patched_sha256=before, rpath=wanted,
                                       already_repaired=True)
        atomic_json(manifest, record)
        return
    backup = backup_file(path, backups, before)
    needed = capture(tool, '--print-needed', str(path))
    mode = path.stat().st_mode & 0o777
    fd, name = tempfile.mkstemp(prefix=path.name + '.new-', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(path, temporary)
        temporary.chmod(mode | 0o200)
        subprocess.run([tool, '--force-rpath', '--set-rpath', wanted, str(temporary)],
                       env=clean_env(), check=True, timeout=60)
        new_tags = dynamic_tags(temporary)
        if (capture(tool, '--print-rpath', str(temporary)) != wanted
                or 15 not in new_tags or 29 in new_tags
                or capture(tool, '--print-needed', str(temporary)) != needed):
            raise RuntimeError(f'Patched ELF validation failed: {path}')
        temporary.chmod(mode)
        patched = digest(temporary)
        entry = dict(before_sha256=before, backup=str(backup), patched_sha256=patched,
                     old_rpath=old, rpath=wanted, needed=needed.splitlines(), pending=True)
        record['files'][key] = entry
        atomic_json(manifest, record)
        owned_path(path)
        if digest(path) != before:
            raise RuntimeError(f'Source changed during repair; stop concurrent uv/pip: {path}')
        # Copy + replace breaks uv hard links and keeps already mapped inodes alive.
        os.replace(temporary, path)
        entry.pop('pending')
        atomic_json(manifest, record)
        print(f'PATCHED {path}', flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()


LIBRT_PROBE = r'''
import ctypes, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
path = pathlib.Path(sys.argv[2])
if not path.resolve().is_relative_to(root):
    raise RuntimeError('Private librt symlink escapes configured glibc: ' + str(path))
ctypes.CDLL(str(path))
paths = {line.split(None, 5)[5].strip() for line in pathlib.Path('/proc/self/maps').read_text().splitlines()
         if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')}
if str(path.resolve()) not in {str(pathlib.Path(p).resolve()) for p in paths}:
    raise RuntimeError('Private librt is absent from process mappings')
print('PASS absolute private librt:', path.resolve(), flush=True)
'''

MAPPING_PROBE = r'''
import pathlib, sys
root = pathlib.Path(sys.argv[2]).resolve()
mapped = set()
for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
    parts = line.split(None, 5)
    if len(parts) != 6 or not parts[5].startswith('/'):
        continue
    path = pathlib.Path(parts[5].strip()).resolve()
    if path.name.startswith(('librt.so', 'librt-', 'libc.so', 'libc-', 'ld-linux', 'ld-',
                             'libpthread.so', 'libpthread-', 'libdl.so', 'libdl-')):
        mapped.add(path)
        if not path.is_relative_to(root):
            raise RuntimeError('glibc component loaded outside private prefix: ' + str(path))
if not any(p.name.startswith(('librt.so', 'librt-')) for p in mapped):
    raise RuntimeError('No librt mapping after fresh PyArrow import')
for path in sorted(mapped):
    print('mapped:', path, flush=True)
'''


def fresh_probe(code, *args, debug=False):
    env = clean_env()
    if debug:
        env['LD_DEBUG'] = 'libs,files'
    return subprocess.run([sys.executable, '-I', '-B', '-u', '-c', code, *args],
                          env=env, capture_output=True, text=True, timeout=180)


def show_result(label, result):
    print(f'{label}: exit={result.returncode}', flush=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), flush=True)


def load_context(root, profile):
    directory = root / '.runtime/centos7' / profile
    state_path = directory / 'state.json'
    venv = root / ('.venv' if profile == 'core' else '.venvs/' + profile)
    private = directory / 'python-base'
    for path in (directory, state_path, venv, private):
        owned_path(path)
    state = json.loads(state_path.read_text())
    expected = dict(root=str(root), profile=profile, venv=str(venv), private=str(private))
    if any(state.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Runtime state does not belong to this repository/profile')
    if Path(sys.prefix) != venv or Path(sys.base_prefix) != private:
        raise RuntimeError(f'Run with .runtime/centos7/{profile}/run python')
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('This repair requires the installed Python 3.10 environment')
    glibc = Path(state['glibc']).resolve()
    loader = Path(state['loader'])
    if not loader.resolve().is_relative_to(glibc):
        raise RuntimeError('Loader is outside the configured private glibc')
    dist = importlib.metadata.distribution('pyarrow')
    if dist.version != VERSION:
        raise RuntimeError(f'Expected locked pyarrow=={VERSION}; found {dist.version}')
    site = venv / 'lib/python3.10/site-packages'
    package = site / 'pyarrow'
    owned_path(package)
    if Path(dist.locate_file('pyarrow')).absolute() != package:
        raise RuntimeError('PyArrow metadata points outside the selected virtual environment')
    objects = sorted(path for path in package.rglob('*')
                     if (path.name.endswith('.so') or '.so.' in path.name) and path.is_file())
    if not objects:
        raise RuntimeError('No installed PyArrow shared objects found')
    for path in objects:
        owned_path(path)
        dynamic_tags(path)
    if not Path(state['patchelf']).is_file():
        raise RuntimeError('Saved patchelf executable is missing: ' + state['patchelf'])
    return state, directory, objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--apply', action='store_true', help='Patch owned PyArrow ELF copies')
    args = parser.parse_args()
    root = args.root.resolve()
    state, directory, objects = load_context(root, args.profile)
    print(f'Profile: {args.profile}; pyarrow=={VERSION}; {len(objects)} shared objects', flush=True)
    # Only stdlib imports in this parent: torch cannot mask the baseline failure.
    spec = importlib.util.spec_from_file_location('fasterwam_doctor', Path(__file__).with_name('doctor.py'))
    doctor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doctor)
    code = doctor.PYARROW_PROBE + '\n' + MAPPING_PROBE
    before = fresh_probe(code, VERSION, state['glibc'])
    show_result('BEFORE fresh PyArrow (diagnostic)', before)
    if not args.apply:
        print('Read-only check. Use --apply to repair ELF search paths.', flush=True)
        return 0 if before.returncode == 0 else 1
    librt = Path(state['loader']).parent / 'librt.so.1'
    result = fresh_probe(LIBRT_PROBE, state['glibc'], str(librt))
    show_result('CHECK private librt before any ELF changes', result)
    if result.returncode:
        raise RuntimeError('Private librt itself cannot be loaded. No PyArrow ELF was changed; '
                           'inspect the exact path/permissions/dependencies above.')
    backups = directory / 'pyarrow-repair-backups'
    manifest = directory / 'pyarrow-repair.json'
    for path in (backups, manifest, directory / 'pyarrow-loader-debug.log'):
        owned_path(path)
    backups.mkdir(exist_ok=True)
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    if previous and (previous.get('root') != str(root) or previous.get('profile') != args.profile):
        raise RuntimeError('PyArrow repair journal belongs to a different profile')
    record = dict(previous)
    record.update(root=str(root), profile=args.profile, version=VERSION, imports_passed=False)
    record.setdefault('files', {})
    atomic_json(manifest, record)
    for path in objects:
        patch_object(path, state, backups, record, manifest)
    result = fresh_probe(code, VERSION, state['glibc'])
    show_result('AFTER fresh PyArrow', result)
    if result.returncode:
        debug = fresh_probe(code, VERSION, state['glibc'], debug=True)
        debug_path = directory / 'pyarrow-loader-debug.log'
        debug_path.write_text(debug.stdout + '\n' + debug.stderr)
        raise RuntimeError(f'Fresh import/roundtrip still fails; loader trace: {debug_path}')
    record['imports_passed'] = True
    atomic_json(manifest, record)
    print(f'PASS PyArrow repair [{args.profile}]', flush=True)
    print(f'Backup directory: {backups}\nRepair journal: {manifest}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
