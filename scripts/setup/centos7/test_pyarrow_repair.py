#!/usr/bin/env python3
"""Real-ELF regression tests for PyArrow repair; no PyArrow/GPU is required."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("repair_pyarrow", Path(__file__).with_name("repair_pyarrow.py"))
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


class PyArrowRepairELFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gcc = shutil.which("gcc")
        cls.patchelf = os.environ.get("PATCHELF") or shutil.which("patchelf")
        if not cls.gcc or not cls.patchelf:
            raise unittest.SkipTest("Requires gcc and patchelf (or PATCHELF=/path/to/patchelf)")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pyarrow-repair-elf-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.provider_dir = self.root / "private-glibc/lib"
        self.package = self.root / ".venv/lib/python3.10/site-packages/pyarrow"
        self.directory = self.root / ".runtime/centos7/core"
        self.backups = self.directory / "pyarrow-repair-backups"
        for directory in (self.provider_dir, self.package, self.backups):
            directory.mkdir(parents=True)
        self.plugin = self.package / "_test_arrow.so"
        self.launcher = self.root / "load-plugin"
        self.manifest = self.directory / "pyarrow-repair.json"
        self.record = {"root": str(self.root), "profile": "core", "files": {}}
        self.state = {"root": str(self.root), "profile": "core",
                      "private": str(self.directory / "python-base"),
                      "loader": str(self.provider_dir / "ld-linux-x86-64.so.2"),
                      "patchelf": self.patchelf}
        provider_source = self.root / "provider.c"
        provider_source.write_text("int private_probe(void) { return 42; }\n")
        plugin_source = self.root / "plugin.c"
        plugin_source.write_text("extern int private_probe(void);\n"
                                 "int plugin_probe(void) { return private_probe(); }\n")
        launcher_source = self.root / "launcher.c"
        launcher_source.write_text(r'''
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) { fprintf(stderr, "%s\n", dlerror()); return 3; }
    int (*probe)(void) = (int (*)(void))dlsym(handle, "plugin_probe");
    if (!probe) { fprintf(stderr, "%s\n", dlerror()); return 4; }
    int result = probe();
    printf("%d\n", result);
    dlclose(handle);
    return result == 42 ? 0 : 5;
}
''')
        commands = (
            [self.gcc, "-shared", "-fPIC", str(provider_source),
             "-Wl,-soname,libprivateprobe.so.1", "-o", str(self.provider_dir / "libprivateprobe.so.1")],
            [self.gcc, "-shared", "-fPIC", str(plugin_source), "-L" + str(self.provider_dir),
             "-l:libprivateprobe.so.1", "-Wl,--enable-new-dtags,-rpath,$ORIGIN", "-o", str(self.plugin)],
            [self.gcc, str(launcher_source), "-ldl",
             "-Wl,--disable-new-dtags,-rpath," + str(self.provider_dir), "-o", str(self.launcher)],
        )
        for command in commands:
            subprocess.run(command, env=repair.clean_env(), check=True, capture_output=True, text=True)

    def run_plugin(self):
        return subprocess.run([str(self.launcher), str(self.plugin)], env=repair.clean_env(),
                              capture_output=True, text=True, timeout=30)

    def patch(self, path=None):
        with contextlib.redirect_stdout(io.StringIO()):
            repair.patch_object(path or self.plugin, self.state, self.backups,
                                self.record, self.manifest)

    def test_loader_repair_preserves_cache_and_backup_and_is_idempotent(self):
        # The executable knows the provider directory, but the plugin's RUNPATH
        # prevents inheriting that RPATH for its own direct dependency.
        before = self.run_plugin()
        self.assertNotEqual(before.returncode, 0)
        self.assertIn("libprivateprobe.so.1", before.stderr)
        self.assertIn(29, repair.dynamic_tags(self.plugin))
        original_hash = repair.digest(self.plugin)
        original_needed = repair.capture(self.patchelf, "--print-needed", str(self.plugin))
        cache = self.root / "uv-cache-copy.so"
        os.link(self.plugin, cache)
        original_inode = self.plugin.stat().st_ino
        self.assertEqual(cache.stat().st_ino, original_inode)

        self.patch()

        after = self.run_plugin()
        self.assertEqual(after.returncode, 0, after.stderr)
        self.assertEqual(after.stdout.strip(), "42")
        self.assertNotEqual(self.plugin.stat().st_ino, original_inode)
        self.assertEqual(cache.stat().st_ino, original_inode)
        self.assertEqual(repair.digest(cache), original_hash)
        backup = self.backups / (original_hash + ".elf")
        self.assertEqual(repair.digest(backup), original_hash)
        self.assertEqual(repair.capture(self.patchelf, "--print-needed", str(self.plugin)), original_needed)
        tags = repair.dynamic_tags(self.plugin)
        self.assertIn(15, tags)
        self.assertNotIn(29, tags)
        rpath = repair.capture(self.patchelf, "--print-rpath", str(self.plugin)).split(":")
        self.assertEqual(rpath[0], str(self.provider_dir))
        self.assertIn("$ORIGIN", rpath)

        repaired_hash = repair.digest(self.plugin)
        repaired_inode = self.plugin.stat().st_ino
        self.patch()
        self.assertEqual(repair.digest(self.plugin), repaired_hash)
        self.assertEqual(self.plugin.stat().st_ino, repaired_inode)
        self.assertEqual(repair.digest(backup), original_hash)
        self.assertEqual(repair.digest(cache), original_hash)
        self.assertEqual(list(self.backups.iterdir()), [backup])
        entry = json.loads(self.manifest.read_text())["files"][str(self.plugin.relative_to(self.root))]
        self.assertEqual(entry["before_sha256"], original_hash)
        self.assertEqual(entry["patched_sha256"], repaired_hash)
        self.assertNotIn("pending", entry)

    def test_symlink_object_is_refused_without_modifying_its_target(self):
        link = self.package / "symlink.so"
        link.symlink_to(self.plugin.name)
        before = repair.digest(self.plugin)
        with self.assertRaisesRegex(RuntimeError, "Refusing symlink"):
            self.patch(link)
        self.assertEqual(repair.digest(self.plugin), before)
        self.assertFalse(self.manifest.exists())
        self.assertEqual(list(self.backups.iterdir()), [])

    def test_symlink_parent_is_refused_without_modifying_its_target(self):
        link = self.root / "linked-package"
        link.symlink_to(self.package, target_is_directory=True)
        before = repair.digest(self.plugin)
        with self.assertRaisesRegex(RuntimeError, "Refusing symlink"):
            self.patch(link / self.plugin.name)
        self.assertEqual(repair.digest(self.plugin), before)
        self.assertFalse(self.manifest.exists())
        self.assertEqual(list(self.backups.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
