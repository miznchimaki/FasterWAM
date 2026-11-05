#!/usr/bin/env python3
"""Build a private MagickWand with PNG support for LIBERO-Plus on CentOS 7.

LIBERO-plus 4976dc3 uses PNG -> MagickMotionBlurImage -> PNG, so a delegate-free
ImageMagick is insufficient. Build private PIC static zlib/libpng, hide their
symbols inside MagickCore, and publish only ImageMagick's shared libraries.
This avoids replacing the base Python's libz/libexpat. No sudo, global loader
variables, venv package installation, or changes to the LIBERO/core profiles.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

def _sibling(name: str):
    # Script directory is deliberately absent from sys.path with python -I.
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load setup helper: {name}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _sibling('build_libero_glvnd')
_artifact, _digest = _helpers._artifact, _helpers._digest
_owned_directory, _run = _helpers._owned_directory, _helpers._run

VERSION = '6.9.13-57'
SOURCE_URL = f'https://download.imagemagick.org/releases/ImageMagick-{VERSION}.tar.xz'
SOURCE_SHA256 = '339e92974a2a7ce5bec231ad0275cec886888b3e5f864ff5563b7eae85dff588'
PNG_URL = 'https://download.sourceforge.net/libpng/libpng-1.6.44.tar.xz'
PNG_SHA256 = '60c4da1d5b7f0aa8d158da48e8f8afa9773c1c8baa5d21974df61f1886b8ce8e'
ZLIB_URL = 'https://zlib.net/fossils/zlib-1.3.1.tar.gz'
ZLIB_SHA256 = '9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23'
LIBRARIES = ('libMagickCore-6.Q16.so', 'libMagickWand-6.Q16.so')
OPTIONS = (
    '--enable-shared', '--disable-static', '--disable-openmp', '--disable-opencl',
    '--disable-hdri', '--with-quantum-depth=16', '--without-modules',
    '--without-magick-plus-plus', '--without-perl', '--disable-docs',
    '--disable-dependency-tracking', '--with-png', '--with-zlib',
    '--without-x', '--without-bzlib', '--without-zstd', '--without-autotrace',
    '--without-dps', '--without-fftw', '--without-flif', '--without-fpx',
    '--without-djvu', '--without-fontconfig', '--without-freetype',
    '--without-raqm', '--without-gdi32', '--without-gslib', '--without-gvc',
    '--without-heic', '--without-jbig', '--without-jpeg', '--without-lcms',
    '--without-openjp2', '--without-lqr', '--without-lzma', '--without-openexr',
    '--without-pango', '--without-raw', '--without-rsvg', '--without-tiff',
    '--without-webp', '--without-wmf', '--without-xml',
)


def _extract(archive: Path, destination: Path, top: str) -> Path:
    # Checksums above authenticate the source; still reject traversal/devices
    # explicitly so this also works safely on Python 3.10 without tar filters.
    with tarfile.open(archive, 'r:*') as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (not path.parts or path.is_absolute() or '..' in path.parts
                    or path.parts[0] != top or not (member.isfile() or member.isdir())):
                raise RuntimeError(f'Unexpected source archive member: {member.name}')
        stream.extractall(destination, members=members)
    return destination / top


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    # Configure tests and make run ordinary host executables: never make those
    # processes consume our private glibc, conda headers or linker flags.
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_DEBUG', 'LD_DEBUG_OUTPUT',
                'LD_RUN_PATH', 'PYTHONHOME', 'PYTHONPATH', 'LIBRARY_PATH',
                'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'CPPFLAGS',
                'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'LIBS', 'PKG_CONFIG_PATH',
                'PKG_CONFIG_LIBDIR', 'LT_SYS_LIBRARY_PATH', 'MAKEFLAGS',
                'MAGICK_HOME', 'MAGICK_CONFIGURE_PATH', 'MAGICK_CODER_MODULE_PATH',
                'MAGICK_FILTER_MODULE_PATH', 'WAND_MAGICK_LIBRARY_SUFFIX'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', CFLAGS='-O2 -fPIC', CXXFLAGS='-O2 -fPIC',
               CONFIG_SITE='/dev/null', LC_ALL='C')
    return env


def _probe(prefix: Path, env: dict[str, str]) -> None:
    env = env.copy()
    env['MAGICK_HOME'] = str(prefix)
    env['WAND_MAGICK_LIBRARY_SUFFIX'] = '-6.Q16'
    code = r'''
import ctypes, io, pathlib, sys, zlib
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
ET.fromstring('<root/>')
assert zlib.decompress(zlib.compress(b'before Wand')) == b'before Wand'
from wand.api import library
from wand.image import Image as WandImage
from wand.version import MAGICK_VERSION
prefix = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(library._name).resolve().parent == prefix / 'lib', library._name
print('MagickWand:', library._name, flush=True)
print(MAGICK_VERSION, flush=True)
pixels = np.zeros((224, 224, 3), dtype=np.uint8)
pixels[40:180, 80:145] = [255, 60, 120]
output = io.BytesIO()
Image.fromarray(pixels).save(output, format='PNG')
library.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double)
with WandImage(blob=output.getvalue()) as image:
    if not library.MagickMotionBlurImage(image.wand, 5.0, 2.0, 20.0):
        image.raise_exception()
        raise RuntimeError('MagickMotionBlurImage returned false')
    result = cv2.imdecode(np.frombuffer(image.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert result is not None and result.shape == pixels.shape, getattr(result, 'shape', None)
    result = result[..., [2, 1, 0]]
    assert np.any(result != pixels), 'Motion blur did not modify test image'
ET.fromstring('<root/>')
assert zlib.decompress(zlib.compress(b'after Wand')) == b'after Wand'
print('PASS private Wand PNG -> motion blur -> PNG/OpenCV; Python XML/zlib intact', flush=True)
'''
    _run([sys.executable, '-I', '-B', '-u', '-c', code, str(prefix)], env=env, timeout=120)


def _manifest_values() -> dict:
    return {'source_sha256': SOURCE_SHA256, 'png_sha256': PNG_SHA256,
            'zlib_sha256': ZLIB_SHA256, 'options': list(OPTIONS),
            'configure_fix': 'preserve-explicit-static-delegate-flags-v1'}


def _reusable(manifest: Path, base: Path, state: dict, env: dict) -> Path | None:
    if not manifest.exists():
        return None
    if manifest.is_symlink():
        raise RuntimeError(f'Refusing symlink manifest: {manifest}')
    record = json.loads(manifest.read_text())
    prefix = Path(record.get('prefix', ''))
    if (record.get('profile') != 'libero-plus' or record.get('root') != state['root']
            or prefix.parent != base or prefix.is_symlink() or prefix.resolve() != prefix
            or any(record.get(k) != v for k, v in _manifest_values().items())):
        raise RuntimeError(f'Unexpected private ImageMagick manifest: {manifest}')
    for name in LIBRARIES:
        target = (prefix / 'lib' / name).resolve()
        if (target.parent != prefix / 'lib' or not target.is_file()
                or _digest(target) != record.get('libraries', {}).get(name)):
            raise RuntimeError(f'Private ImageMagick library changed or is missing: {target}')
    _probe(prefix, env)
    print(f'REUSE private ImageMagick: {prefix}', flush=True)
    return prefix


def build_imagemagick(state: dict, source_archive: str | Path | None = None, *,
                      png_archive: str | Path | None = None,
                      zlib_archive: str | Path | None = None) -> Path:
    """Return a tested private ImageMagick prefix; never change runtime state."""
    root = Path(state['root'])
    runtime = root / '.runtime/centos7/libero-plus'
    venv, private = root / '.venvs/libero-plus', runtime / 'python-base'
    if (state.get('profile') != 'libero-plus' or not root.is_absolute() or root.resolve() != root
            or state.get('venv') != str(venv) or state.get('private') != str(private)
            or Path(sys.prefix) != venv or Path(sys.base_prefix) != private):
        raise RuntimeError('Run this builder with this repository\'s .runtime/centos7/libero-plus/run python.')
    for path in (runtime, venv, private):
        if path.resolve() != path:
            raise RuntimeError(f'Refusing symlink runtime path: {path}')
    base, cache = runtime / 'private-imagemagick', runtime / 'imagemagick-build'
    for path in (runtime, base, cache):
        _owned_directory(path)
    env = _environment()
    manifest = base / 'build.json'
    old = _reusable(manifest, base, state, env)
    if old:
        return old
    compiler = shlex.split(env.get('CC', 'cc'))
    make = shutil.which('make', path=env.get('PATH'))
    if not compiler or not shutil.which(compiler[0], path=env.get('PATH')) or not make:
        raise RuntimeError('Working host C compiler and make required; check FASTERWAM_GCC_ROOT.')
    jobs = int(env.get('MAX_JOBS', '2'))
    if jobs < 1:
        raise RuntimeError('MAX_JOBS must be a positive integer.')
    _run(compiler + ['--version'], env=env, timeout=30)
    source_tar = _artifact(SOURCE_URL, SOURCE_SHA256, cache, f'ImageMagick-{VERSION}.tar.xz', source_archive)
    png_tar = _artifact(PNG_URL, PNG_SHA256, cache, 'libpng-1.6.44.tar.xz', png_archive)
    zlib_tar = _artifact(ZLIB_URL, ZLIB_SHA256, cache, 'zlib-1.3.1.tar.gz', zlib_archive)
    work = Path(tempfile.mkdtemp(prefix='work-', dir=cache))
    prefix = base / ('v' + VERSION + '-' + work.name.removeprefix('work-'))
    dependencies = work / 'static-delegates'
    zlib = _extract(zlib_tar, work, 'zlib-1.3.1')
    png = _extract(png_tar, work, 'libpng-1.6.44')
    source = _extract(source_tar, work, 'ImageMagick-' + VERSION)
    # The release's generated configure advertises these override variables,
    # but clears them immediately before PKG_CHECK_MODULES. Preserve their
    # explicit values so the two private archives work without pkg-config and
    # cannot accidentally resolve to system shared libraries. Only this fresh
    # build-local configure file is changed, never an installed dependency.
    configure = source / 'configure'
    contents = configure.read_text()
    for variable in ('PNG_CFLAGS', 'PNG_LIBS', 'ZLIB_CFLAGS', 'ZLIB_LIBS'):
        old = variable + '=\"\"\n'
        if contents.count(old) != 1:
            raise RuntimeError(f'Unexpected pinned configure layout: {variable}')
        contents = contents.replace(old, ': \"${' + variable + ':=}\"\n')
    configure.write_text(contents)
    _run(['/bin/sh', str(zlib / 'configure'), '--static', '--prefix=' + str(dependencies)],
         env=env, cwd=zlib)
    _run([make, '-j', str(jobs)], env=env, cwd=zlib)
    _run([make, 'install'], env=env, cwd=zlib)
    delegates_env = env.copy()
    delegates_env.update(CPPFLAGS='-I' + str(dependencies / 'include'),
                         LDFLAGS='-L' + str(dependencies / 'lib'),
                         PKG_CONFIG_PATH=str(dependencies / 'lib/pkgconfig'),
                         PKG_CONFIG_LIBDIR=str(dependencies / 'lib/pkgconfig'))
    _run(['/bin/sh', str(png / 'configure'), '--prefix=' + str(dependencies),
          '--libdir=' + str(dependencies / 'lib'), '--disable-shared', '--enable-static',
          '--with-pic', '--disable-dependency-tracking'], env=delegates_env, cwd=png)
    _run([make, '-j', str(jobs)], env=delegates_env, cwd=png)
    _run([make, 'install'], env=delegates_env, cwd=png)
    # Explicit archive filenames prevent accidentally linking system/conda
    # delegates; exclude-libs prevents zlib/png symbols interposing on Python.
    delegates_env.update(
        LDFLAGS='-Wl,--exclude-libs,ALL',
        PNG_CFLAGS='-I' + str(dependencies / 'include'),
        PNG_LIBS=' '.join([str(dependencies / 'lib/libpng16.a'), str(dependencies / 'lib/libz.a'), '-lm']),
        ZLIB_CFLAGS='-I' + str(dependencies / 'include'),
        ZLIB_LIBS=str(dependencies / 'lib/libz.a'))
    _run(['/bin/sh', str(source / 'configure'), '--prefix=' + str(prefix),
          '--libdir=' + str(prefix / 'lib'), *OPTIONS], env=delegates_env, cwd=source)
    config = (source / 'magick/magick-baseconfig.h').read_text()
    for delegate in ('PNG', 'ZLIB'):
        if '#define MAGICKCORE_' + delegate + '_DELEGATE 1' not in config:
            raise RuntimeError(f'ImageMagick configure did not enable {delegate}; inspect config.log.')
    _run([make, '-j', str(jobs)], env=delegates_env, cwd=source, timeout=1800)
    staging = work / 'staging'
    _run([make, 'install', 'DESTDIR=' + str(staging)], env=delegates_env, cwd=source)
    installed = staging / str(prefix).lstrip('/')
    # This prefix provides the C API to Wand, not host-glibc command-line tools.
    # Removing the owned utilities prevents accidentally executing them after
    # the shared objects have been adapted to private glibc.
    if (installed / 'bin').exists():
        shutil.rmtree(installed / 'bin')
    hashes = {}
    for name in LIBRARIES:
        library = (installed / 'lib' / name).resolve()
        if library.parent != installed / 'lib' or not library.is_file():
            raise RuntimeError(f'ImageMagick did not produce required library: {name}')
        needed = subprocess.check_output([state['patchelf'], '--print-needed', str(library)],
                                         env=env, text=True).splitlines()
        allowed = {'libm.so.6', 'libc.so.6', 'libpthread.so.0', 'libdl.so.2',
                   'librt.so.1', 'libgcc_s.so.1', 'libMagickCore-6.Q16.so.7'}
        if set(needed) - allowed:
            raise RuntimeError(f'Unexpected ImageMagick shared dependency: {name}: {needed}')
        rpath = ':'.join(dict.fromkeys([str(Path(state['loader']).parent),
                                       str(prefix / 'lib'), *state['rpath']]))
        _run([state['patchelf'], '--force-rpath', '--set-rpath', rpath, str(library)], env=env)
        hashes[name] = _digest(library)
    os.rename(installed, prefix)
    _probe(prefix, env)
    record = {**_manifest_values(), 'root': state['root'], 'profile': 'libero-plus',
              'prefix': str(prefix), 'source_url': SOURCE_URL, 'png_url': PNG_URL,
              'zlib_url': ZLIB_URL, 'compiler': compiler, 'libraries': hashes,
              'imports_passed': True}
    fd, temporary = tempfile.mkstemp(prefix='build.json.new-', dir=base)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, manifest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'BUILT private ImageMagick: {prefix}', flush=True)
    return prefix


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Download/build or verify/reuse private ImageMagick')
    parser.add_argument('--state', type=Path, default=root / '.runtime/centos7/libero-plus/state.json')
    parser.add_argument('--source', help='Offline pinned ImageMagick source archive')
    parser.add_argument('--png-source', help='Offline pinned libpng source archive')
    parser.add_argument('--zlib-source', help='Offline pinned zlib source archive')
    args = parser.parse_args()
    if not args.apply:
        print('Plan: build private ImageMagick ' + VERSION + ' with static PNG/zlib; use --apply.')
        return 0
    state = json.loads(args.state.read_text())
    prefix = build_imagemagick(state, args.source, png_archive=args.png_source, zlib_archive=args.zlib_source)
    _run([sys.executable, '-I', str(Path(__file__).with_name('runtime.py')),
          'refresh-shell', '--state', str(args.state.resolve())], env=_environment())
    print(f'MAGICK_HOME={prefix}')
    print('Runtime launchers refreshed. Start a fresh run command or re-source activate.sh.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
