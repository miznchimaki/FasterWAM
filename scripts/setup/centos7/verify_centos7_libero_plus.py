#!/usr/bin/env python3
"""One real LIBERO-Plus task, without loading a policy or changing the environment.

Run from the FasterWAM repository root with its private LIBERO-Plus launcher.
API checked against LIBERO-Plus 4976dc3 and FasterWAM codex/zero3-lora.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(args):
    root = args.root.resolve()
    source = root / "third_party/LIBERO-plus"
    config = root / ".runtime/libero-plus/config.yaml"
    expected_venv = root / ".venvs/libero-plus"
    require(Path(sys.prefix).resolve() == expected_venv.resolve(),
            "Run with .runtime/centos7/libero-plus/run python, from repo root.")
    require(config.is_file(), f"Missing existing Plus configuration: {config}")
    require((source / "libero/libero").is_dir(), f"Missing Plus source: {source}")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        require(os.environ.get(key) == "egl", f"Expected {key}=egl from the repaired Plus launcher.")

    # Match eval_fasterwam_libero_plus.sh before importing torch/libero. Do not
    # trigger LIBERO's interactive ~/.libero initialization or modify config.
    os.environ["LIBERO_CONFIG_PATH"] = str(config.parent)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(source))
    import yaml
    settings = yaml.safe_load(config.read_text())
    require(Path(settings["benchmark_root"]).resolve() == (source / "libero/libero").resolve(),
            "config.yaml points to a different benchmark_root; expected third_party/LIBERO-plus.")
    for key in ("bddl_files", "init_states", "assets"):
        directory = Path(settings[key]).expanduser().resolve()
        require(directory.is_dir(), f"Missing {key}: {directory}")
        print(f"CONFIG {key}: {directory}", flush=True)

    print("CHECK actual LIBERO-Plus package and benchmark import", flush=True)
    import numpy as np
    from PIL import Image
    import libero.libero as libero_package
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    package_path = Path(libero_package.__file__).resolve()
    require(package_path.is_relative_to(source.resolve()), f"Wrong LIBERO package: {package_path}")
    print(f"PASS Plus package: {package_path}", flush=True)
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        suite = benchmark.get_benchmark_dict()[args.suite]()
    require(suite.n_tasks > 100, f"Expected an expanded Plus suite, found {suite.n_tasks} tasks.")
    require(0 <= args.task_id < suite.n_tasks, f"task-id must be in [0, {suite.n_tasks - 1}].")
    task = suite.get_task(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # View/noise task names can be virtual paths. The official wrapper resolves
    # them to the underlying BDDL; do not reject them with a plain exists check.
    print(f"TASK {args.suite}/{args.task_id}: {task.name}", flush=True)
    print(f"LANGUAGE {task.language}", flush=True)
    print(f"BDDL (wrapper input): {bddl}", flush=True)
    initial_states = suite.get_task_init_states(args.task_id)
    require(len(initial_states) > 0, "Task has no initial states.")
    print(f"PASS task metadata / initial states: {len(initial_states)} states", flush=True)

    env = None
    stats = {}
    executed = 0
    try:
        print("CHECK scene assets / environment construction", flush=True)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                 camera_heights=224, camera_widths=224)
        env.seed(0)
        print("CHECK reset / initial state / neutral-action steps", flush=True)
        env.reset()
        obs = env.set_init_state(initial_states[0])
        for _ in range(args.steps):
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
            executed += 1
            if done:
                break
        output = args.output
        if output is None:
            output = root / ".runtime/centos7/libero-plus/smoke" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        for key, filename in (("agentview_image", "agentview.png"),
                              ("robot0_eye_in_hand_image", "wrist.png")):
            pixels = np.asarray(obs[key])
            require(pixels.shape == (224, 224, 3), f"Unexpected {key} shape: {pixels.shape}")
            require(pixels.dtype == np.uint8, f"Unexpected {key} dtype: {pixels.dtype}")
            require(int(pixels.max()) > int(pixels.min()), f"Uniform/blank camera frame: {key}")
            stats[key] = {"shape": list(pixels.shape), "dtype": str(pixels.dtype),
                          "min": int(pixels.min()), "max": int(pixels.max())}
            # Same 180-degree rotation as FasterWAM's get_libero_image().
            Image.fromarray(np.ascontiguousarray(pixels[::-1, ::-1])).save(output / filename)
            print(f"PASS camera {key}: {stats[key]}", flush=True)
    finally:
        if env is not None:
            env.close()
    report = {"suite": args.suite, "task_id": args.task_id, "task": task.name,
              "package": str(package_path), "steps_executed": executed, "cameras": stats,
              "scope": "One real task; no model, policy success rate, video encoding, or multi-GPU test."}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"OUTPUT {output}", flush=True)
    print("PASS LIBERO-Plus real task: assets / init state / reset / step / two cameras / close", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--suite", choices=("libero_10", "libero_goal", "libero_spatial", "libero_object"),
                        default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600, help="Whole check timeout in seconds.")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.timeout < 1:
        parser.error("--steps and --timeout must be positive")
    if args.worker:
        verify(args)
        return 0
    # Contain the upstream reset retry loop without changing any vendor code.
    try:
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                 "--worker", *sys.argv[1:]], timeout=args.timeout)
        return result.returncode if result.returncode >= 0 else 128 - result.returncode
    except subprocess.TimeoutExpired:
        print(f"FAIL: real-task check exceeded {args.timeout}s; see the last CHECK line.", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
