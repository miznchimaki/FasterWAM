#!/usr/bin/env python3
"""CPU-only regression tests for standalone PyArrow validation in doctor.py."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("centos7_doctor", Path(__file__).with_name("doctor.py"))
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


class PyArrowDoctorTests(unittest.TestCase):
    def result(self, **changes):
        info = {"version": "23.0.0", "prefix": sys.prefix, "executable": sys.executable}
        info.update(changes)
        return subprocess.CompletedProcess([], 0, "FASTERWAM_PYARROW_PROBE=" + json.dumps(info) + "\n", "")

    def test_fresh_interpreter_and_clean_loader_environment(self):
        polluted = {name: "/unwanted" for name in (
            "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_DEBUG_OUTPUT",
            "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP")}
        polluted["FASTERWAM_GLIBC_ROOT"] = "/private/glibc"
        with mock.patch.dict(os.environ, polluted), \
                mock.patch.object(doctor.subprocess, "run", return_value=self.result()) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            doctor.pyarrow_check()
        arguments, options = run.call_args
        self.assertEqual(arguments[0], [sys.executable, "-I", "-B", "-u", "-c", doctor.PYARROW_PROBE, "23.0.0"])
        self.assertEqual(options["timeout"], 120)
        self.assertEqual(options["env"]["FASTERWAM_GLIBC_ROOT"], "/private/glibc")
        self.assertTrue(all(name not in options["env"] for name in polluted
                            if name != "FASTERWAM_GLIBC_ROOT"))

    def test_native_loader_failure_is_preserved(self):
        failed = subprocess.CompletedProcess([], 1, "", "ImportError: librt.so.1: Permission denied\n")
        with mock.patch.object(doctor.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "librt.so.1: Permission denied"):
                doctor.pyarrow_check()

    def test_signal_failure_is_preserved(self):
        failed = subprocess.CompletedProcess([], -11, "", "")
        with mock.patch.object(doctor.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "exited -11"):
                doctor.pyarrow_check()

    def test_incomplete_or_wrong_environment_results_are_rejected(self):
        results = (
            subprocess.CompletedProcess([], 0, "", ""),
            self.result(prefix="/some/other/environment"),
            self.result(version="22.0.0"),
        )
        for result in results:
            with self.subTest(stdout=result.stdout), \
                    mock.patch.object(doctor.subprocess, "run", return_value=result):
                with self.assertRaises(RuntimeError):
                    doctor.pyarrow_check()

    def test_probe_failure_cannot_be_hidden_by_successful_later_imports(self):
        events = []
        expected = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128",
                    "pyarrow": "23.0.0", "av": "16.0.1", "transformers": "4.49.0",
                    "datasets": "3.6.0", "accelerate": "1.12.0", "h5py": "3.14.0",
                    "torchcodec": "0.5+cu128", "deepspeed": "0.18.5"}

        def probe(version):
            events.append("fresh-pyarrow")
            self.assertEqual(version, "23.0.0")
            raise RuntimeError("librt.so.1 cannot be loaded independently")

        def imported(name):
            events.append(name)

        with mock.patch.object(doctor, "pyarrow_check", side_effect=probe), \
                mock.patch.object(doctor.importlib.metadata, "version", side_effect=expected.__getitem__), \
                mock.patch.object(doctor.importlib, "import_module", side_effect=imported), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "fresh pyarrow.*librt.so.1"):
                doctor.package_check("core")
        self.assertEqual(events[0], "fresh-pyarrow")
        self.assertEqual(events[1], "torch")
        self.assertIn("pyarrow", events[2:])

    def test_real_subprocess_rejects_startup_imported_torch(self):
        # No PyArrow or CUDA installation is needed to exercise this guard.
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c",
             "import sys, types\nsys.modules['torch'] = types.ModuleType('torch')\n" + doctor.PYARROW_PROBE,
             "23.0.0"], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python startup imported torch", result.stderr)
        self.assertNotIn("FASTERWAM_PYARROW_PROBE=", result.stdout)


if __name__ == "__main__":
    unittest.main()
