#!/usr/bin/env python3
"""Repair owned MuJoCo ELF search paths and stage existing host EGL libraries.

Run with the private LIBERO Python from repo root. Without --apply, only print
the plan. --build-glvnd explicitly permits downloading/building a private GLVND
dispatcher. No system-library changes, global LD_LIBRARY_PATH, or GPU rendering.
Rendering is verified separately by doctor.py --render.
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


def capture(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def elf_dso(path: Path) -> bool:
    with path.open('rb') as stream:
        header = stream.read(20)
    return (header[:6] == b'\x7fELF\x02\x01'
            and header[16:18] == b'\x03\x00' and header[18:20] == b'\x3e\x00')


def has_runpath(path: Path) -> bool:
    with path.open('rb') as stream:
        header = stream.read(64)
        offset = struct.unpack_from('<Q', header, 32)[0]
        size, count = struct.unpack_from('<HH', header, 54)
        for index in range(count):
            stream.seek(offset + index * size)
            entry = stream.read(size)
            if struct.unpack_from('<I', entry)[0] == 2:  # PT_DYNAMIC
                start = struct.unpack_from('<Q', entry, 8)[0]
                length = struct.unpack_from('<Q', entry, 32)[0]
                stream.seek(start)
                for tag, value in struct.iter_unpack('<qQ', stream.read(length)):
                    if tag == 0:
                        break
                    if tag == 29:  # DT_RUNPATH
                        return True
    return False


def desired_rpath(state: dict, old: str) -> str:
    existing = [p for p in old.split(':') if p]
    origins = [p for p in existing if '$ORIGIN' in p or '${ORIGIN}' in p]
    values = [str(Path(state['loader']).parent),
              str(Path(state['root']) / '.runtime/centos7/libero/native-libs')]
    values += origins or ['$ORIGIN']
    values += state['rpath']
    values += existing
    return ':'.join(dict.fromkeys(values))


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def patch_copy(source: Path, destination: Path, state: dict, backups: Path, before_publish=None,
               rpath_override=None) -> str:
    """Never mutate source/cache/system ELF in place; publish a patched copy."""
    if destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise RuntimeError(f'Refusing symlink destination: {destination}')
    if not elf_dso(source):
        raise RuntimeError(f'Expected Linux x86_64 shared ELF: {source}')
    tool = state['patchelf']
    old = capture(tool, '--print-rpath', str(source))
    wanted = desired_rpath(state, old) if rpath_override is None else rpath_override
    if source == destination and old == wanted and not has_runpath(source):
        print(f'UNCHANGED {destination}', flush=True)
        return digest(destination)
    if destination.exists():
        key = digest(destination)
        backup = backups / (key + '.elf')
        if not backup.exists():
            shutil.copy2(destination, backup)
    fd, name = tempfile.mkstemp(prefix=destination.name + '.new-', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        original_mode = source.stat().st_mode & 0o777
        temporary.chmod(original_mode | 0o200)
        # A shared object gets only RPATH. Never add PT_INTERP to a DSO.
        subprocess.run([tool, '--force-rpath', '--set-rpath', wanted, str(temporary)], check=True)
        if capture(tool, '--print-rpath', str(temporary)) != wanted:
            raise RuntimeError(f'RPATH verification failed: {temporary}')
        if has_runpath(temporary):
            raise RuntimeError(f'DT_RUNPATH was not converted to DT_RPATH: {temporary}')
        if capture(tool, '--print-needed', str(source)) != capture(tool, '--print-needed', str(temporary)):
            raise RuntimeError(f'DT_NEEDED unexpectedly changed: {source}')
        temporary.chmod(original_mode)
        patched_hash = digest(temporary)
        if before_publish:
            before_publish(patched_hash)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'PATCHED {destination}', flush=True)
    return patched_hash


def probe(label: str, code: str, backend: str, *args: str, vendor_json=None) -> None:
    env = os.environ.copy()
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_DEBUG', 'LD_DEBUG_OUTPUT', 'PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    env.update(MUJOCO_GL=backend, PYOPENGL_PLATFORM='egl', PYTHONNOUSERSITE='1')
    if vendor_json is not None:
        env['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(vendor_json)
    print(f'CHECK {label}', flush=True)
    subprocess.run([sys.executable, '-I', '-B', '-u', '-c', code, *args],
                   env=env, check=True, timeout=60)
    print(f'PASS {label}', flush=True)


def owned_hash(path: Path, known: dict) -> str:
    if path.is_symlink() or path.parent.resolve() != path.parent:
        raise RuntimeError(f'Refusing graphics symlink: {path}')
    actual = digest(path)
    allowed = {known.get('patched_sha256')}
    if known.get('pending'):
        allowed.add(known.get('previous_sha256'))
    if not known or actual not in allowed:
        raise RuntimeError(f'Unmanaged or externally changed graphics file: {path}')
    return actual


def retire_obsolete(native, backups, record, plan, manifest_path):
    """Move only hash-verified, formerly managed copies out of the search path.

    Save the retirement intent first. A crash on either side of os.replace is
    resumable; unrelated native-libs files (including C++ runtimes) stay intact.
    """
    for name in sorted(set(record['graphics']) - set(plan)):
        if Path(name).name != name or name in ('.', '..'):
            raise RuntimeError(f'Invalid managed graphics name: {name!r}')
        path = native / name
        known = record['graphics'][name]
        if path.exists() or path.is_symlink():
            sha = owned_hash(path, known)
            target = backups / ('retired-' + name + '-' + sha + '.elf')
            if target.is_symlink() or (target.exists() and digest(target) != sha):
                raise RuntimeError(f'Unexpected retired-library backup: {target}')
            record.setdefault('retired_graphics', {})[name] = dict(known, backup=str(target), sha256=sha)
            atomic_json(manifest_path, record)
            os.replace(path, target)
            print(f'RETIRED {path} -> {target}', flush=True)
        del record['graphics'][name]
        atomic_json(manifest_path, record)


PYTHON_SUPPORT_PROBE = r'''
import ctypes, json, pathlib, sys
mode, expected = sys.argv[1], json.loads(sys.argv[2])
try:
    if mode == 'preview':
        for name, source in expected.items():
            ctypes.CDLL(source, mode=ctypes.RTLD_GLOBAL)
            print('Preview base provider:', name, source)
    import pyexpat
    import xml.etree.ElementTree as ET
    import zlib
    payload = b'<root><item>fasterwam</item></root>'
    assert ET.fromstring(payload).findtext('item') == 'fasterwam'
    assert zlib.decompress(zlib.compress(payload)) == payload
    print('Expat:', pyexpat.EXPAT_VERSION, '; pyexpat:', getattr(pyexpat, '__file__', '<built-in>'))
    print('zlib:', zlib.ZLIB_RUNTIME_VERSION, '; XML parsing and compression OK')
finally:
    mapped = set()
    for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith('/'):
            path = pathlib.Path(parts[5])
            if path.name.startswith(('libexpat.so.', 'libz.so.')):
                mapped.add(path)
    for path in sorted(mapped):
        print('Python support mapped:', path)
# A statically bundled parser need not map libexpat. For dynamic providers,
# verify the normal (no-preload) child really stopped resolving the stale copy.
for name, source in expected.items():
    for path in mapped:
        if path.name.startswith(name) and path.resolve() != pathlib.Path(source).resolve():
            raise RuntimeError('Unexpected Python support library: ' + str(path))
'''


def patch_python_support_paths(state, backups, record, manifest_path, providers):
    """Repair only relocated private pyexpat/zlib DSOs after a failed probe."""
    root = Path(state['root'])
    directory = Path(state['private']) / 'lib/python3.10/lib-dynload'
    targets = []
    for module_name, soname in (('pyexpat', 'libexpat.so.1'), ('zlib', 'libz.so.1')):
        if soname not in providers:
            continue
        matches = sorted(directory.glob(module_name + '*.so'))
        if len(matches) > 1:
            raise RuntimeError(f'Ambiguous private standard-library extension: {matches}')
        for path in matches:
            if path.resolve() != path or not path.is_relative_to(root) or not elf_dso(path):
                raise RuntimeError(f'Unexpected private standard-library target: {path}')
            targets.append(path)
    if not targets:
        raise RuntimeError('No private pyexpat/zlib ELF found for search-path repair.')
    for path in targets:
        old = capture(state['patchelf'], '--print-rpath', str(path))
        print(f'PYTHON SUPPORT SEARCH PATH {path}: {old!r}; RUNPATH={has_runpath(path)}', flush=True)
        origins = [p for p in old.split(':') if '$ORIGIN' in p or '${ORIGIN}' in p]
        paths = [str(Path(state['loader']).parent), str(Path(state['source_prefix']) / 'lib')]
        paths += origins + state['rpath'] + [p for p in old.split(':') if p]
        wanted = ':'.join(dict.fromkeys(paths))
        relative = str(path.relative_to(root))
        record.setdefault('originals', {})
        if relative not in record['originals']:
            backup = backups / (digest(path) + '.elf')
            if not backup.exists():
                shutil.copy2(path, backup)
            record['originals'][relative] = str(backup.relative_to(root))
            atomic_json(manifest_path, record)
        sha = patch_copy(path, path, state, backups, rpath_override=wanted)
        record.setdefault('python_support', {})[relative] = sha
        atomic_json(manifest_path, record)


def repair_python_support(state, native, backups, record, manifest_path, graphics):
    """Retire verified legacy Expat/zlib copies before XML code generation.

    Validate the base provider in a disposable child first; the final child
    performs a normal import with no preload. Only earlier managed copies move.
    """
    conflicts = {}
    for name in graphics.PYTHON_SUPPORT_SONAMES:
        known = record['graphics'].get(name)
        if not known:
            continue
        source = graphics.base_runtime_library(state, name)
        if source is None:
            continue
        destination = native / name
        if destination.exists() or destination.is_symlink():
            owned_hash(destination, known)
        if known.get('source_sha256') == digest(source) and not known.get('pending'):
            continue
        conflicts[name] = str(source)
        print(f'PYTHON SUPPORT CONFLICT {destination}; base provider: {source}', flush=True)
    if conflicts:
        # This preloading is diagnostic only and never applied to a launcher.
        # Do not move anything unless the base libraries can actually serve
        # this private Python's pyexpat and compression modules.
        probe('base Expat/zlib compatibility preview', PYTHON_SUPPORT_PROBE,
              'disable', 'preview', json.dumps(conflicts))
        keep = set(record['graphics']) - set(conflicts)
        retire_obsolete(native, backups, record, keep, manifest_path)
    try:
        probe('normal Python XML and zlib before graphics build', PYTHON_SUPPORT_PROBE,
              'disable', 'normal', json.dumps(conflicts))
    except subprocess.CalledProcessError:
        # A copied Conda extension can also have RUNPATH=$ORIGIN/../.., which
        # resolves into python-base/lib after relocation. Validate a working
        # base provider before changing only the private extension's RPATH.
        print('DIAGNOSTIC normal XML/zlib import failed; checking relocated extension search paths.', flush=True)
        providers = {}
        for name in graphics.PYTHON_SUPPORT_SONAMES:
            source = graphics.base_runtime_library(state, name)
            if source is not None:
                providers[name] = str(source)
        if not providers:
            raise RuntimeError('Base Python Expat/zlib providers are missing; no extension files were changed.')
        probe('base Expat/zlib compatibility before RPATH repair', PYTHON_SUPPORT_PROBE,
              'disable', 'preview', json.dumps(providers))
        patch_python_support_paths(state, backups, record, manifest_path, providers)
        probe('normal Python XML and zlib after RPATH repair', PYTHON_SUPPORT_PROBE,
              'disable', 'normal', json.dumps(providers))


EGL_PROBE = r'''
import ctypes as C, pathlib, sys
native = pathlib.Path(sys.argv[1]).resolve()
egl = C.CDLL('libEGL.so.1')
egl.eglQueryString.argtypes = [C.c_void_p, C.c_int]
egl.eglQueryString.restype = C.c_char_p
version = egl.eglQueryString(None, 0x3054)
print('EGL dispatcher version:', version)
if not version or b'libglvnd' not in version.lower():
    raise RuntimeError('The loaded libEGL is not GLVND; refusing a legacy Mesa dispatcher.')
print('EGL client extensions:', egl.eglQueryString(None, 0x3055))
egl.eglGetProcAddress.argtypes = [C.c_char_p]
egl.eglGetProcAddress.restype = C.c_void_p
address = egl.eglGetProcAddress(b'eglQueryDevicesEXT')
if not address:
    raise RuntimeError('GLVND did not expose eglQueryDevicesEXT.')
query = C.CFUNCTYPE(C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))(address)
count = C.c_int()
if not query(0, None, C.byref(count)):
    raise RuntimeError('eglQueryDevicesEXT exists, but device enumeration failed.')
print('EGL device count:', count.value)
mapped = set()
for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
    parts = line.split(None, 5)
    if len(parts) == 6 and parts[5].startswith('/'):
        path = pathlib.Path(parts[5])
        if path.name.startswith(('libEGL', 'libOpenGL', 'libGLdispatch', 'libnvidia-eglcore', 'libnvidia-glsi')):
            mapped.add(path)
for path in sorted(mapped):
    print('EGL mapped:', path)
    if path.parent != native:
        raise RuntimeError('Graphics library escaped the private native-libs directory: ' + str(path))
# A driver may defer eglcore until context creation. Its independent dlopen
# probe and the final real-render check cover that without preloading it here.
for prefix in ('libEGL.so.', 'libEGL_nvidia.so.'):
    if not any(path.name.startswith(prefix) for path in mapped):
        raise RuntimeError('NVIDIA vendor runtime was not mapped: ' + prefix)
if count.value == 0:
    raise RuntimeError('GLVND works but NVIDIA reports zero EGL devices. Check GPU allocation, device permissions, and driver visibility on this compute node.')
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply changes to the owned LIBERO environment')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--graphics-dir', action='append', default=[], help='Additional existing host graphics-library source directory; repeatable')
    parser.add_argument('--glvnd-dir', type=Path, help='Existing coherent GLVND source lib directory; never fall back to host Mesa')
    parser.add_argument('--build-glvnd', action='store_true', help='Download and build pinned GLVND privately (requires --apply)')
    parser.add_argument('--glvnd-source', type=Path, help='Verified local v1.7.0 source archive, for --build-glvnd without a GitHub download')
    parser.add_argument('--meson-wheel', type=Path, help='Verified local Meson 1.5.2 wheel for an offline private build')
    parser.add_argument('--ninja-wheel', type=Path, help='Verified local Ninja 1.11.1.4 Linux x86_64 wheel for an offline private build')
    args = parser.parse_args()
    if args.build_glvnd and (not args.apply or args.glvnd_dir):
        parser.error('--build-glvnd requires --apply and cannot be combined with --glvnd-dir')
    if (args.glvnd_source or args.meson_wheel or args.ninja_wheel) and not args.build_glvnd:
        parser.error('--glvnd-source/--meson-wheel/--ninja-wheel requires --build-glvnd')
    root = args.root.resolve()
    state_dir = root / '.runtime/centos7/libero'
    state = json.loads((state_dir / 'state.json').read_text())
    venv = root / '.venvs/libero'
    private = state_dir / 'python-base'
    if (state.get('root') != str(root) or state.get('profile') != 'libero'
            or state.get('venv') != str(venv) or state.get('private') != str(private)
            or Path(sys.prefix) != venv or Path(sys.base_prefix) != private):
        raise RuntimeError('Use this repository\'s .runtime/centos7/libero/run python launcher.')
    if importlib.metadata.version('mujoco') != '3.3.2':
        raise RuntimeError('This repair targets the locked mujoco==3.3.2.')
    native = state_dir / 'native-libs'
    backups = state_dir / 'loader-repair-backups'
    manifest_path = state_dir / 'loader-repair.json'
    for path in (state_dir, venv, native, backups, manifest_path):
        if path.is_symlink():
            raise RuntimeError(f'Refusing symlink at owned repair path: {path}')
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous.get('root') != str(root):
        raise RuntimeError('Repair manifest belongs to another repository.')
    extra_dirs = args.graphics_dir or previous.get('graphics_dirs', [])
    glvnd_dir = args.glvnd_dir or previous.get('glvnd_dir')
    probe('private librt is loadable',
          'import ctypes,sys; ctypes.CDLL(sys.argv[1]); print(sys.argv[1])',
          'disable', str(Path(state['loader']).parent / 'librt.so.1'))

    package = venv / 'lib/python3.10/site-packages/mujoco'
    objects = sorted(p for p in package.rglob('*') if '.so' in p.name and p.is_file())
    if not objects:
        raise RuntimeError('No MuJoCo shared objects found.')
    for obj in objects:
        if obj.is_symlink() or obj.resolve() != obj or not elf_dso(obj):
            raise RuntimeError(f'Unexpected MuJoCo library target: {obj}')
    print(f'MuJoCo shared objects: {len(objects)}', flush=True)
    helper = Path(__file__).resolve().with_name('libero_graphics.py')
    spec = importlib.util.spec_from_file_location('fasterwam_libero_graphics', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Save bookkeeping after each completed replacement, so interruptions can
    # resume safely and later uv reinstallation can be repaired again.
    record = dict(previous)
    record.update(root=str(root), profile='libero', graphics_dirs=extra_dirs)
    record.setdefault('mujoco', {})
    record.setdefault('graphics', {})
    record.setdefault('originals', {})
    if args.apply:
        record['imports_passed'] = False
        native.mkdir(exist_ok=True)
        backups.mkdir(exist_ok=True)
        for obj in objects:
            relative = str(obj.relative_to(root))
            if relative not in record['originals']:
                backup = backups / (digest(obj) + '.elf')
                if not backup.exists():
                    shutil.copy2(obj, backup)
                record['originals'][relative] = str(backup.relative_to(root))
                atomic_json(manifest_path, record)
            record['mujoco'][relative] = patch_copy(obj, obj, state, backups)
            atomic_json(manifest_path, record)
        # Fresh process, no import torch and no preloaded librt workaround.
        probe('fresh MuJoCo import without graphics',
              'import mujoco; print("mujoco", mujoco.__version__)', 'disable')
        # Old Mesa's dependency copies can shadow the base Python's Expat/zlib.
        # Cleanup must precede GLVND's XML code generation, not follow it.
        repair_python_support(state, native, backups, record, manifest_path, module)
    if args.build_glvnd:
        build_spec = importlib.util.spec_from_file_location('fasterwam_build_glvnd', helper.with_name('build_libero_glvnd.py'))
        builder = importlib.util.module_from_spec(build_spec)
        build_spec.loader.exec_module(builder)
        glvnd_dir = builder.build_glvnd(state, source_archive=args.glvnd_source,
                                      meson_wheel=args.meson_wheel, ninja_wheel=args.ninja_wheel)
    try:
        plan = module.plan_graphics(state, extra_dirs, glvnd_dir=glvnd_dir)
    except module.GraphicsError as error:
        print(f'GRAPHICS NOT READY: {error}', file=sys.stderr)
        for detail in getattr(error, 'details', []):
            print(detail, file=sys.stderr)
        print('No system libraries or permissions were changed. For missing GLVND, use --apply --build-glvnd '
              'or --glvnd-dir /path/to/coherent/glvnd/lib. For missing NVIDIA companions, '
              'supply an existing matching driver directory with --graphics-dir.', file=sys.stderr)
        return 2
    for name, source in plan.items():
        print(f'GRAPHICS {name} <- {source}', flush=True)
    if not args.apply:
        print('Plan only. Re-run with --apply to repair this LIBERO environment.')
        return 0
    if glvnd_dir is not None:
        record['glvnd_dir'] = str(Path(glvnd_dir).resolve())
    # Validate *all* existing graphics copies before publishing any new ones.
    for name in set(plan) | set(record['graphics']):
        if Path(name).name != name or name in ('.', '..'):
            raise RuntimeError(f'Invalid graphics name: {name!r}')
        destination = native / name
        if destination.is_symlink():
            raise RuntimeError(f'Refusing to overwrite a symlink: {destination}')
        if destination.exists():
            owned_hash(destination, record['graphics'].get(name, {}))
    for name, source in plan.items():
        destination = native / name
        old = record['graphics'].get(name, {})
        source_hash = digest(source)
        if (destination.exists() and old.get('source_sha256') == source_hash
                and old.get('patched_sha256') == digest(destination)
                and not has_runpath(destination)
                and capture(state['patchelf'], '--print-rpath', str(destination))
                == desired_rpath(state, capture(state['patchelf'], '--print-rpath', str(source)))):
            print(f'UNCHANGED {destination}', flush=True)
        else:
            previous_hash = digest(destination) if destination.exists() else None
            def before_publish(patched_hash):
                record['graphics'][name] = dict(source=str(source), source_sha256=source_hash,
                                               patched_sha256=patched_hash, pending=True,
                                               previous_sha256=previous_hash)
                atomic_json(manifest_path, record)
            patch_copy(source, destination, state, backups, before_publish=before_publish)
        record['graphics'][name].pop('pending', None)
        record['graphics'][name].pop('previous_sha256', None)
        atomic_json(manifest_path, record)
    retire_obsolete(native, backups, record, plan, manifest_path)
    probe('normal Python XML and zlib after graphics staging', PYTHON_SUPPORT_PROBE,
          'disable', 'normal', '{}')
    vendor_dir = state_dir / 'egl-vendor'
    vendor_json = vendor_dir / '10_fasterwam_nvidia.json'
    if vendor_dir.is_symlink() or vendor_json.is_symlink():
        raise RuntimeError('Refusing a symlink at the owned EGL vendor configuration.')
    vendor_value = {'file_format_version': '1.0.0', 'ICD': {'library_path': str(native / 'libEGL_nvidia.so.0')}}
    if vendor_json.exists() and json.loads(vendor_json.read_text()) != vendor_value:
        raise RuntimeError(f'Unexpected existing EGL vendor configuration: {vendor_json}')
    vendor_dir.mkdir(exist_ok=True)
    atomic_json(vendor_json, vendor_value)
    libraries = ['libEGL.so.1', 'libEGL_nvidia.so.0', 'libOpenGL.so.0']
    libraries += sorted(name for name in plan if name.startswith('libnvidia-eglcore.so.'))
    for name in libraries:
        probe('fresh dlopen ' + name,
              'import ctypes,sys; ctypes.CDLL(sys.argv[1]); print("loaded",sys.argv[1])', 'egl', name, vendor_json=vendor_json)
    probe('GLVND dispatcher and NVIDIA EGL devices', EGL_PROBE, 'egl', str(native), vendor_json=vendor_json)
    probe('fresh EGL MuJoCo and robosuite imports',
          'import mujoco; import robosuite; print("mujoco",mujoco.__version__); print("robosuite import OK")', 'egl', vendor_json=vendor_json)
    record['imports_passed'] = True
    record['graphics_mode'] = 'nvidia-glvnd'
    atomic_json(manifest_path, record)
    subprocess.run([sys.executable, '-I', '-B', str(helper.with_name('runtime.py')),
                    'refresh-shell', '--state', str(state_dir / 'state.json')], check=True)
    print('Loader repair completed. Validate actual rendering with doctor.py --profile libero --require-cuda --render under EGL.')
    print(f'Original ELF backups: {backups}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
