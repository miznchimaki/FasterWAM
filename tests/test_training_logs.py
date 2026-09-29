"""Exercise launch logging without GPUs, Torch, or an Accelerate installation.

Run with ``python -m unittest discover -s tests -p test_training_logs.py -v``.
Only the project's Hydra/OmegaConf dependencies are needed.
"""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_HYDRA = importlib.util.find_spec("hydra") is not None


@unittest.skipUnless(HAS_HYDRA, "Install hydra-core to run the launch logging tests")
class TrainingLoggingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="fasterwam-log-tests-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.bin_dir = self.root / "bin"
        self.support_dir = self.root / "support"
        self.records_dir = self.root / "records"
        for directory in (self.bin_dir, self.support_dir, self.records_dir):
            directory.mkdir()
        # A symlink outside a virtualenv loses that environment's pyvenv.cfg.
        python_command = self.bin_dir / "python"
        python_command.write_text(
            "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n',
            encoding="utf-8",
        )
        python_command.chmod(0o755)
        self._write_fixture_modules()
        self.env = os.environ.copy()
        for key in (
            "FASTERWAM_RESOLVE_OUTPUT_DIR_FILE",
            "FASTERWAM_RUN_OUTPUT_DIR",
            "FASTERWAM_LOG_CAPTURE",
            "FASTERWAM_TEST_FAIL_RANK",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
        ):
            self.env.pop(key, None)
        self.env.update(
            PATH=str(self.bin_dir) + os.pathsep + self.env.get("PATH", ""),
            PYTHONPATH=os.pathsep.join(
                (str(self.support_dir), str(REPO_ROOT / "src"))
            ),
            PYTHONUNBUFFERED="1",
            HYDRA_FULL_ERROR="1",
            RUN_ID="logging-test",
            NNODES="1",
            NODE_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
            FASTERWAM_TEST_RECORDS=str(self.records_dir),
        )

    def _write_fixture_modules(self):
        # The preview must not import either the runtime or optional media/GPU
        # libraries. Actual workers receive a tiny replacement runtime instead.
        (self.support_dir / "sitecustomize.py").write_text(
            textwrap.dedent(
                r'''
                import importlib.abc
                import json
                import logging
                import os
                from pathlib import Path
                import sys
                import types

                if os.environ.get("FASTERWAM_RESOLVE_OUTPUT_DIR_FILE"):
                    class RejectHeavyImports(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path=None, target=None):
                            if fullname == "fasterwam.runtime" or fullname.split(".")[0] in {
                                "torch", "av", "imageio", "numpy", "PIL", "accelerate"
                            }:
                                raise RuntimeError("Preview imported heavy module: " + fullname)
                    sys.meta_path.insert(0, RejectHeavyImports())
                else:
                    runtime = types.ModuleType("fasterwam.runtime")

                    def run_training(cfg):
                        from omegaconf import OmegaConf

                        rank = os.environ.get("RANK", "0")
                        output_dir = str(Path(str(cfg.output_dir)).resolve())
                        record = {
                            "rank": rank,
                            "output_dir": output_dir,
                            "stored_output_dir": OmegaConf.to_container(cfg, resolve=False)["output_dir"],
                        }
                        records_dir = Path(os.environ["FASTERWAM_TEST_RECORDS"])
                        (records_dir / ("worker-" + rank + ".json")).write_text(json.dumps(record))
                        print("WORKER_STDOUT rank=" + rank, flush=True)
                        print("WORKER_STDERR rank=" + rank, file=sys.stderr, flush=True)
                        os.write(1, ("NATIVE_STDOUT rank=" + rank + "\n").encode())
                        os.write(2, ("NATIVE_STDERR rank=" + rank + "\n").encode())
                        os.write(2, ("PROGRESS rank=" + rank + " 1/2\rPROGRESS rank=" + rank + " 2/2\n").encode())
                        logging.getLogger("training-test").warning("LOGGER_WARNING rank=%s", rank)
                        if os.environ.get("FASTERWAM_TEST_FAIL_RANK") == rank:
                            raise RuntimeError("RUNTIME_FAILURE rank=" + rank)

                    runtime.run_training = run_training
                    sys.modules["fasterwam.runtime"] = runtime
                '''
            ),
            encoding="utf-8",
        )
        launcher = self.bin_dir / "accelerate"
        launcher.write_text(
            "#!/usr/bin/env python\n"
            + textwrap.dedent(
                r'''
                import json
                import os
                from pathlib import Path
                import subprocess
                import sys

                args = sys.argv[1:]
                (Path(os.environ["FASTERWAM_TEST_RECORDS"]) / "launcher.json").write_text(json.dumps(args))
                print("LAUNCHER_STDOUT", flush=True)
                print("LAUNCHER_STDERR", file=sys.stderr, flush=True)
                script_index = args.index("scripts/train.py")
                worker_count = int(args[args.index("--num_processes") + 1])
                workers = []
                for rank in range(worker_count):
                    child_env = os.environ.copy()
                    child_env.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(worker_count))
                    workers.append(subprocess.Popen([sys.executable, *args[script_index:]], env=child_env))
                failed = any(code != 0 for code in [worker.wait() for worker in workers])
                print("LAUNCHER_FINISHED", flush=True)
                sys.exit(37 if failed else 0)
                '''
            ),
            encoding="utf-8",
        )
        launcher.chmod(0o755)

    def _launch(self, output_dir, *, stage=1, overrides=(), fail_rank=None):
        env = self.env.copy()
        if fail_rank is not None:
            env["FASTERWAM_TEST_FAIL_RANK"] = str(fail_rank)
        args = [
            "bash",
            f"scripts/train_zero{stage}.sh",
            "2",
            "task=libero_fasterwam_2cam224_1e-4",
            "output_dir=" + json.dumps(str(output_dir)),
            *overrides,
        ]
        return subprocess.run(
            args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60
        )

    def _assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_both_launchers_capture_launcher_and_all_worker_streams_once(self):
        for stage in (1, 2):
            with self.subTest(stage=stage):
                output_dir = self.root / f"zero{stage}"
                result = self._launch(output_dir, stage=stage)
                self._assert_success(result)
                log = (output_dir / "train.log").read_text(encoding="utf-8")
                markers = ["LAUNCHER_STDOUT", "LAUNCHER_STDERR", "LAUNCHER_FINISHED"]
                for rank in (0, 1):
                    markers.extend(
                        f"{kind} rank={rank}"
                        for kind in (
                            "WORKER_STDOUT", "WORKER_STDERR", "NATIVE_STDOUT",
                            "NATIVE_STDERR", "LOGGER_WARNING",
                        )
                    )
                    self.assertIn(f"PROGRESS rank={rank} 2/2", log)
                for marker in markers:
                    self.assertEqual(log.count(marker), 1, marker + "\n" + log)
                    self.assertIn(marker, result.stdout + result.stderr)
                launch_args = json.loads((self.records_dir / "launcher.json").read_text())
                self.assertEqual(
                    launch_args[launch_args.index("--config_file") + 1],
                    f"scripts/accelerate_configs/accelerate_zero{stage}_ds.yaml",
                )
                for flag, expected in (
                    ("--num_processes", "2"), ("--num_machines", "1"),
                    ("--machine_rank", "0"), ("--main_process_ip", "127.0.0.1"),
                    ("--main_process_port", "29500"),
                    ("--deepspeed_multinode_launcher", "standard"),
                ):
                    self.assertEqual(launch_args[launch_args.index(flag) + 1], expected)

    def test_last_output_override_and_interpolation_are_shared_by_workers(self):
        unused_dir = self.root / "unused"
        actual_dir = self.root / "space,equals=and[brackets]"
        self.env["FASTERWAM_TEST_OUTPUT_ROOT"] = str(actual_dir)
        result = self._launch(
            unused_dir,
            overrides=("output_dir=${oc.env:FASTERWAM_TEST_OUTPUT_ROOT}",),
        )
        self._assert_success(result)
        self.assertTrue((actual_dir / "train.log").is_file())
        self.assertFalse((unused_dir / "train.log").exists())
        for rank in (0, 1):
            record = json.loads((self.records_dir / f"worker-{rank}.json").read_text())
            self.assertEqual(record["output_dir"], str(actual_dir.resolve()))
            self.assertEqual(record["stored_output_dir"], str(actual_dir.resolve()))

    def test_multirun_does_not_start_a_launcher_with_one_shared_log(self):
        for flag in ("-m", "--multirun"):
            with self.subTest(flag=flag):
                result = self._launch(self.root / "sweep", overrides=(flag,))
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse((self.records_dir / "launcher.json").exists())

    def test_reusing_an_output_directory_appends(self):
        output_dir = self.root / "resume"
        for _ in range(2):
            self._assert_success(self._launch(output_dir))
        log = (output_dir / "train.log").read_text(encoding="utf-8")
        self.assertEqual(log.count("LAUNCHER_STDOUT"), 2)
        self.assertEqual(log.count("NATIVE_STDERR rank=1"), 2)

    def test_worker_traceback_is_logged_and_launcher_failure_is_preserved(self):
        output_dir = self.root / "failure"
        result = self._launch(output_dir, fail_rank=1)
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        log = (output_dir / "train.log").read_text(encoding="utf-8")
        self.assertIn("Traceback (most recent call last)", log)
        self.assertIn("RuntimeError: RUNTIME_FAILURE rank=1", log)
        self.assertIn("LAUNCHER_FINISHED", log)

    def test_preview_resolves_only_output_dir_without_heavy_imports(self):
        destination = self.root / "resolved-output.txt"
        env = self.env.copy()
        env["FASTERWAM_RESOLVE_OUTPUT_DIR_FILE"] = str(destination)
        env["FASTERWAM_TEST_OUTPUT_ROOT"] = str(self.root / "preview")
        args = [
            sys.executable, "scripts/train.py",
            "task=libero_fasterwam_2cam224_1e-4",
            "output_dir=${oc.env:FASTERWAM_TEST_OUTPUT_ROOT}/${hydra:runtime.choices.task}/${now:%Y}",
            "+unused_test_value=${oc.env:FASTERWAM_TEST_UNDEFINED_ENV}",
            "hydra/job_logging=stdout",
        ]
        result = subprocess.run(
            args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60
        )
        self._assert_success(result)
        from datetime import datetime

        expected = self.root / "preview" / "libero_fasterwam_2cam224_1e-4" / datetime.now().strftime("%Y")
        self.assertEqual(Path(destination.read_text().strip()), expected.resolve())
        self.assertFalse(list(self.records_dir.glob("worker-*.json")))


if __name__ == "__main__":
    unittest.main()
