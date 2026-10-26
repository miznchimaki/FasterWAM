#!/usr/bin/env python3
"""CPU-only regressions for profile isolation and the LIBERO-Plus resume flow."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


runtime = module('runtime')
loader = module('repair_libero_loader')


class ProfileIsolation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def state(self, profile):
        directory = self.root / '.runtime/centos7' / profile
        directory.mkdir(parents=True)
        project = self.root / 'environments' / profile
        project.mkdir(parents=True)
        (project / 'uv.lock').write_text('registry = "https://pypi.org/simple"\n')
        value = dict(root=str(self.root), profile=profile,
                     venv=str(self.root / '.venvs' / profile),
                     private=str(directory / 'python-base'),
                     loader='/private-glibc/lib/ld-linux-x86-64.so.2',
                     glibc='/private-glibc', uv='/usr/bin/true', rpath=['/private-glibc/lib'])
        (directory / 'state.json').write_text(json.dumps(value))
        return value, directory

    def graphics(self, profile, directory):
        vendor_dir = directory / 'egl-vendor'
        vendor_dir.mkdir()
        vendor = vendor_dir / '10_fasterwam_nvidia.json'
        vendor.write_text(json.dumps({'file_format_version': '1.0.0', 'ICD': {
            'library_path': str(directory / 'native-libs/libEGL_nvidia.so.0')}}))
        (directory / 'loader-repair.json').write_text(json.dumps({
            'root': str(self.root), 'profile': profile, 'graphics_mode': 'nvidia-glvnd'}))
        return vendor

    def test_target_rpath_and_origin_order(self):
        for profile in ('libero', 'libero-plus'):
            state, directory = self.state(profile)
            paths = loader.desired_rpath(state, '$ORIGIN:/old').split(':')
            self.assertEqual(paths[:3], ['/private-glibc/lib', str(directory / 'native-libs'), '$ORIGIN'])
            self.assertEqual(paths.count('/private-glibc/lib'), 1)

    def test_launchers_persist_target_graphics_and_magick(self):
        state, directory = self.state('libero-plus')
        vendor = self.graphics('libero-plus', directory)
        prefix = directory / 'private-imagemagick/v-test'
        (prefix / 'lib').mkdir(parents=True)
        (prefix.parent / 'build.json').write_text(json.dumps({
            'root': str(self.root), 'profile': 'libero-plus',
            'prefix': str(prefix), 'imports_passed': True}))
        with mock.patch.dict(os.environ, {'PATH': '/usr/bin:/bin', 'MUJOCO_GL': 'osmesa',
                                         'PYOPENGL_PLATFORM': 'osmesa'}, clear=True):
            runtime.write_shell_files(state)
        child_env = dict(os.environ, LD_LIBRARY_PATH='/wrong-libs', PYTHONPATH='/wrong-python')
        output = subprocess.check_output(['bash', str(directory / 'run'), '/usr/bin/env'], env=child_env, text=True)
        env = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
        self.assertEqual(env['MUJOCO_GL'], 'egl')
        self.assertEqual(env['PYOPENGL_PLATFORM'], 'egl')
        self.assertEqual(env['__EGL_VENDOR_LIBRARY_FILENAMES'], str(vendor))
        self.assertEqual(env['MAGICK_HOME'], str(prefix))
        self.assertEqual(env['WAND_MAGICK_LIBRARY_SUFFIX'], '-6.Q16')
        self.assertEqual(env['VIRTUAL_ENV'], state['venv'])
        self.assertNotIn('LD_LIBRARY_PATH', env)
        self.assertNotIn('PYTHONPATH', env)

    def test_libero_keeps_its_own_graphics(self):
        state, directory = self.state('libero')
        vendor = self.graphics('libero', directory)
        with mock.patch.dict(os.environ, {'PATH': '/usr/bin:/bin'}, clear=True):
            runtime.write_shell_files(state)
        text = (directory / 'run').read_text()
        self.assertIn(str(vendor), text)
        self.assertNotIn('libero-plus', text)
        self.assertNotIn('MAGICK_HOME', text)

    def test_wrong_magick_profile_is_rejected(self):
        state, directory = self.state('libero-plus')
        prefix = directory / 'private-imagemagick/v-test'
        (prefix / 'lib').mkdir(parents=True)
        (prefix.parent / 'build.json').write_text(json.dumps({
            'root': str(self.root), 'profile': 'libero', 'prefix': str(prefix), 'imports_passed': True}))
        with self.assertRaisesRegex(RuntimeError, 'Invalid managed ImageMagick'):
            runtime.write_shell_files(state)

    def test_wrong_interpreter_rejected_before_repair(self):
        state, directory = self.state('libero-plus')
        argv = ['repair', '--root', str(self.root), '--profile', 'libero-plus', '--apply']
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(sys, 'prefix', '/other-venv'):
            with self.assertRaisesRegex(RuntimeError, 'libero-plus/run'):
                loader.main()
        self.assertFalse((directory / 'loader-repair.json').exists())

    def test_wrong_repair_manifest_profile_is_rejected(self):
        state, directory = self.state('libero-plus')
        (directory / 'loader-repair.json').write_text(json.dumps({'root': str(self.root), 'profile': 'libero'}))
        argv = ['repair', '--root', str(self.root), '--profile', 'libero-plus', '--apply']
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(sys, 'prefix', state['venv']), \
                mock.patch.object(sys, 'base_prefix', state['private']), \
                mock.patch.object(loader.importlib.metadata, 'version', return_value='3.3.2'):
            with self.assertRaisesRegex(RuntimeError, 'another repository or profile'):
                loader.main()

    def donor(self):
        state, directory = self.state('libero')
        native = directory / 'native-libs'
        native.mkdir()
        entries = {}
        for name, origin in (('libEGL_nvidia.so.0', 'libEGL_nvidia.so.525.85.12'),
                             ('libnvidia-eglcore.so.525.85.12', 'libnvidia-eglcore.so.525.85.12')):
            path = native / name
            path.write_bytes(b'\x7fELF\x02\x01' + b'\0' * 10 + b'\x03\0\x3e\0' + name.encode())
            entries[name] = {'source': '/no-longer-present/' + origin,
                             'patched_sha256': loader.digest(path)}
        record = {'root': str(self.root), 'profile': 'libero', 'graphics_mode': 'nvidia-glvnd',
                  'imports_passed': True, 'graphics': entries, 'graphics_dirs': []}
        (directory / 'loader-repair.json').write_text(json.dumps(record))
        return directory, record

    def test_reuse_without_original_paths_and_without_cross_profile_rpath(self):
        directory, record = self.donor()
        before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        sources, versions, dirs = loader.reusable_graphics(self.root, 'libero')
        self.assertEqual(dirs, [])
        self.assertEqual(set(sources), set(record['graphics']))
        self.assertEqual(set(versions.values()), {'525.85.12'})
        state, target = self.state('libero-plus')
        rpath = loader.graphics_rpath(state, sources['libEGL_nvidia.so.0'], sources)
        self.assertIn(str(target / 'native-libs'), rpath)
        self.assertNotIn(str(directory) + '/', rpath)
        self.assertEqual(before, {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()})

    def test_reuse_rejects_changed_donor_library(self):
        directory, record = self.donor()
        with (directory / 'native-libs/libEGL_nvidia.so.0').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'externally changed'):
            loader.reusable_graphics(self.root, 'libero')

    def test_reuse_rejects_unfinished_or_foreign_donor(self):
        directory, record = self.donor()
        for changes in ({'imports_passed': False}, {'profile': 'libero-plus'},
                        {'root': '/another/repository'}, {'graphics_mode': 'mesa'}):
            (directory / 'loader-repair.json').write_text(json.dumps(dict(record, **changes)))
            with self.assertRaisesRegex(RuntimeError, 'no successful matching repair'):
                loader.reusable_graphics(self.root, 'libero')

    def test_reuse_requires_driver_version_provenance(self):
        directory, record = self.donor()
        record['graphics']['libEGL_nvidia.so.0']['source'] = '/unknown/libEGL_nvidia.so.0'
        (directory / 'loader-repair.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, 'Cannot verify NVIDIA donor driver version'):
            loader.reusable_graphics(self.root, 'libero')

    def test_plain_apply_reuses_saved_donor(self):
        directory, record = self.donor()
        state, target = self.state('libero-plus')
        state['patchelf'] = '/unused/patchelf'
        (target / 'state.json').write_text(json.dumps(state))
        package = Path(state['venv']) / 'lib/python3.10/site-packages/mujoco'
        package.mkdir(parents=True)
        obj = package / 'libmujoco.so.3.3.2'
        obj.write_bytes((directory / 'native-libs/libEGL_nvidia.so.0').read_bytes())
        fixture_builder = self.root / 'fake_builder.py'
        fixture_builder.write_text('def _reusable(manifest, base, env):\n    return base / "v-test/lib"\n')
        fixture_graphics = self.root / 'fake_graphics.py'
        fixture_graphics.write_text('class GraphicsError(RuntimeError): pass\n'
                                   'def plan_graphics(state, extra_dirs, **kwargs):\n'
                                   '    assert kwargs["source_versions"]["libEGL_nvidia.so.0"] == "525.85.12"\n'
                                   '    return kwargs["source_libraries"]\n')
        original_spec = importlib.util.spec_from_file_location
        def spec(name, path):
            return original_spec(name, fixture_builder if name == 'fasterwam_build_glvnd' else fixture_graphics)
        def copy(source, destination, state, backups, before_publish=None, rpath_override=None):
            data = source.read_bytes()
            if before_publish:
                before_publish(loader.digest(source))
            destination.write_bytes(data)
            if source != destination:
                self.assertNotIn(str(directory) + '/', rpath_override)
            return loader.digest(destination)
        before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        for extra in (['--reuse-glvnd-from', 'libero'], []):
            argv = ['repair', '--root', str(self.root), '--profile', 'libero-plus', '--apply', *extra]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(sys, 'prefix', state['venv']), \
                    mock.patch.object(sys, 'base_prefix', state['private']), \
                    mock.patch.object(loader.importlib.metadata, 'version', return_value='3.3.2'), \
                    mock.patch.object(loader.importlib.util, 'spec_from_file_location', side_effect=spec), \
                    mock.patch.object(loader, 'patch_copy', side_effect=copy), \
                    mock.patch.object(loader, 'has_runpath', return_value=False), \
                    mock.patch.object(loader, 'capture', return_value='/old-rpath'), \
                    mock.patch.object(loader, 'probe'), mock.patch.object(loader, 'repair_python_support'), \
                    mock.patch.object(loader.subprocess, 'run'):
                self.assertEqual(loader.main(), 0)
            saved = json.loads((target / 'loader-repair.json').read_text())
            self.assertEqual(saved['graphics_reuse_from'], 'libero')
            self.assertEqual(saved['graphics']['libEGL_nvidia.so.0']['nvidia_version'], '525.85.12')
        self.assertEqual(before, {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()})

    def test_resume_uses_only_target_launcher(self):
        state, directory = self.state('libero-plus')
        script = self.root / 'scripts/setup/repair_centos7_libero_plus.sh'
        script.parent.mkdir(parents=True)
        shutil.copyfile(HERE.parent / script.name, script)
        log = self.root / 'calls.txt'
        (directory / 'run').write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_CALLS"\n')
        (directory / 'run').chmod(0o755)
        source = self.root / '.runtime/centos7/libero/private-glvnd'
        source.mkdir(parents=True)
        (source / 'build.json').write_text('{}')
        result = subprocess.run(['bash', str(script)], env=dict(os.environ, TEST_CALLS=str(log)),
                                text=True, capture_output=True, check=True)
        calls = log.read_text().splitlines()
        self.assertEqual(len(calls), 4)
        self.assertIn('--profile libero-plus --repair', calls[0])
        self.assertIn('--profile libero-plus --apply --reuse-glvnd-from libero', calls[1])
        self.assertIn('build_libero_plus_imagemagick.py --apply', calls[2])
        self.assertIn('--profile libero-plus --require-cuda --render', calls[3])
        self.assertEqual((source / 'build.json').read_text(), '{}')
        self.assertIn('Ready:', result.stdout)


if __name__ == '__main__':
    unittest.main()
