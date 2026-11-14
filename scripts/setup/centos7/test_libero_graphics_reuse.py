#!/usr/bin/env python3
"""CPU-only regressions for exact, read-only NVIDIA donor-library discovery.

Fixtures have real ELF headers; the ELF inspector and host inventory are faked
so these tests do not depend on a particular machine's GPU, driver, or patchelf.
Run: python -m unittest discover -s scripts/setup/centos7 -p test_libero_graphics_reuse.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    'fasterwam_graphics_under_test', Path(__file__).with_name('libero_graphics.py'))
graphics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(graphics)


class GraphicsReuseTests(unittest.TestCase):
    VERSION = '525.85.12'

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='fasterwam-graphics-reuse-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dsos = {}
        self.glvnd = self.root / 'verified-glvnd/lib'
        self.donor = self.root / 'libero/native-libs'
        self.state = {
            'glibc': str(self.root / 'glibc'),
            'loader': str(self.root / 'glibc/lib/ld-linux-x86-64.so.2'),
            'private': str(self.root / 'libero-plus/python-base'),
            'rpath': [str(self.root / 'libero-plus/native-libs')],
            'patchelf': 'fixture-patchelf',
        }
        self.elf(self.root / 'glibc/lib/libc.so.6', 'libc.so.6')
        self.elf(self.glvnd / 'libGLdispatch.so.0', 'libGLdispatch.so.0')
        self.elf(self.glvnd / 'libEGL.so.1', 'libEGL.so.1',
                 ['libGLdispatch.so.0'], marker=b'__EGL_VENDOR_LIBRARY_FILENAMES')
        self.elf(self.glvnd / 'libOpenGL.so.0', 'libOpenGL.so.0', ['libGLdispatch.so.0'])
        self.vendor = self.elf(self.donor / 'libEGL_nvidia.so.0', 'libEGL_nvidia.so.0',
                               ['libnvidia-glsi.so.' + self.VERSION, 'libc.so.6'])
        self.core_name = 'libnvidia-eglcore.so.' + self.VERSION
        self.glsi_name = 'libnvidia-glsi.so.' + self.VERSION
        self.core = self.elf(self.donor / self.core_name, self.core_name, ['libc.so.6'])
        self.glsi = self.elf(self.donor / self.glsi_name, self.glsi_name, ['libc.so.6'])
        self.sources = {
            'libEGL_nvidia.so.0': self.vendor,
            self.core_name: self.core,
            self.glsi_name: self.glsi,
        }
        self.versions = {name: self.VERSION for name in self.sources}
        self.cache = {}
        real_reader = graphics._readable_elf

        def fixture_reader(path):
            # Never inspect unrelated host libraries even if the test machine
            # happens to have a working NVIDIA installation.
            path = Path(path).resolve()
            return real_reader(path) if path in self.dsos else None

        self.patches = [
            mock.patch.object(graphics, '_cache', side_effect=lambda: (self.cache, [])),
            mock.patch.object(graphics, '_driver_version', return_value=self.VERSION),
            mock.patch.object(graphics, '_readable_elf', side_effect=fixture_reader),
            mock.patch.object(graphics, '_run', side_effect=self.inspect),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def elf(self, path, soname, needed=(), *, marker=b''):
        path.parent.mkdir(parents=True, exist_ok=True)
        header = bytearray(20)
        header[:6] = b'\x7fELF\x02\x01'
        struct.pack_into('<HH', header, 16, 3, 62)
        path.write_bytes(header + marker)
        self.dsos[path] = {'soname': soname, 'needed': list(needed)}
        return path

    def inspect(self, argv, timeout=20):
        self.assertEqual(argv[0], 'fixture-patchelf')
        record = self.dsos[Path(argv[2])]
        if argv[1] == '--print-soname':
            output = record['soname']
        elif argv[1] == '--print-needed':
            output = '\n'.join(record['needed'])
        else:
            self.fail('Unexpected inspector invocation: ' + repr(argv))
        return subprocess.CompletedProcess(argv, 0, output, '')

    def plan(self, **kwargs):
        return graphics.plan_graphics(
            self.state, glvnd_dir=self.glvnd,
            source_libraries=self.sources, source_versions=self.versions, **kwargs)

    def test_exact_verified_donor_sources_work_without_host_vendor(self):
        before = {path: path.read_bytes() for path in self.sources.values()}
        result = self.plan()
        for name, path in self.sources.items():
            self.assertEqual(result[name], path)
        self.assertEqual(before, {path: path.read_bytes() for path in self.sources.values()})
        self.assertEqual(len(result), 6)

    def test_wrong_running_driver_rejected(self):
        with mock.patch.object(graphics, '_driver_version', return_value='535.104.05'):
            with self.assertRaisesRegex(graphics.GraphicsError, 'differs from running driver'):
                self.plan()

    def test_unknown_running_driver_rejected_for_donor_sources(self):
        with mock.patch.object(graphics, '_driver_version', return_value=None):
            with self.assertRaisesRegex(graphics.GraphicsError, 'without a detected running driver'):
                self.plan()

    def test_missing_and_invalid_version_metadata_rejected(self):
        for value in (None, '', '525', 525, 'not-a-version'):
            with self.subTest(value=value):
                self.versions['libEGL_nvidia.so.0'] = value
                with self.assertRaisesRegex(graphics.GraphicsError, 'Missing or invalid NVIDIA source version'):
                    self.plan()

    def test_wrong_versioned_filename_rejected(self):
        self.versions[self.core_name] = '535.104.05'
        with self.assertRaisesRegex(graphics.GraphicsError, 'disagrees with filename'):
            self.plan()

    def test_wrong_mapped_soname_rejected(self):
        self.dsos[self.vendor]['soname'] = 'libGL.so.1'
        with self.assertRaisesRegex(graphics.GraphicsError, 'Unexpected SONAME'):
            self.plan()

    def test_unreadable_mapped_file_rejected(self):
        self.vendor.unlink()
        with self.assertRaisesRegex(graphics.GraphicsError, 'no longer readable'):
            self.plan()

    def test_unlisted_donor_companion_is_not_discovered_from_its_directory(self):
        del self.sources[self.core_name]
        del self.versions[self.core_name]
        # The file still exists next to the verified donor vendor. It must not
        # enter the plan without independent verification/mapping.
        self.assertTrue(self.core.is_file())
        with self.assertRaisesRegex(graphics.GraphicsError, 'NVIDIA EGL runtime companion'):
            self.plan()

    def test_unlisted_needed_dependency_is_not_discovered_from_donor_directory(self):
        del self.sources[self.glsi_name]
        del self.versions[self.glsi_name]
        self.assertTrue(self.glsi.is_file())
        with self.assertRaisesRegex(graphics.GraphicsError, 'dependency closure is incomplete'):
            self.plan()

    def test_exact_non_nvidia_dependency_mapping_is_supported(self):
        name = 'libfixture-helper.so.1'
        dependency = self.elf(self.donor / name, name, ['libc.so.6'])
        self.sources[name] = dependency
        self.dsos[self.vendor]['needed'].append(name)
        result = self.plan()
        self.assertEqual(result[name], dependency)

    def test_original_host_discovery_api_without_source_maps_is_unchanged(self):
        host = self.root / 'host-driver'
        vendor = self.elf(host / ('libEGL_nvidia.so.' + self.VERSION), 'libEGL_nvidia.so.0',
                          [self.glsi_name, 'libc.so.6'])
        core = self.elf(host / self.core_name, self.core_name, ['libc.so.6'])
        glsi = self.elf(host / self.glsi_name, self.glsi_name, ['libc.so.6'])
        self.cache['libEGL_nvidia.so.0'] = [vendor]
        result = graphics.plan_graphics(self.state, glvnd_dir=self.glvnd)
        self.assertEqual(result['libEGL_nvidia.so.0'], vendor)
        self.assertEqual(result[self.core_name], core)
        self.assertEqual(result[self.glsi_name], glsi)


if __name__ == '__main__':
    unittest.main()
