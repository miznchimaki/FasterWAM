"""Real CPU optimizer/scheduler round trips for grouped and legacy LR state.

Only Accelerate's distributed orchestration is substituted. Model tensors,
AdamW moments, optimizer groups and scheduler states are serialized by Torch.
These tests do not exercise DeepSpeed's distributed checkpoint format.
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import fasterwam.trainer as trainer_module
from test_lora_training import TinyLoraPolicy, adapter_config
from test_trainer_deepspeed import RecordingAccelerator


class StatefulCPUAccelerator(RecordingAccelerator):
    """Exercise actual Torch checkpoint loading instead of an event-only stub."""

    def prepare(self, model, optimizer, loader, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
        return super().prepare(model, optimizer, loader, scheduler)

    def save_state(self, output_dir):
        self.events.append("save_state")
        torch.save({
            "model": self.engine.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, Path(output_dir, "cpu_state.pt"))

    def load_state(self, input_dir):
        self.events.append("load_state")
        state = torch.load(Path(input_dir, "cpu_state.pt"), weights_only=False)
        self.engine.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


class LearningRateResumeTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="fasterwam-lr-resume-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.events = []
        accelerator_patch = patch.object(
            trainer_module, "Accelerator", side_effect=lambda **kwargs:
            StatefulCPUAccelerator(self.events, stage=2, main=True, **kwargs),
        )
        accelerator_patch.start()
        self.addCleanup(accelerator_patch.stop)

    def config(self, name, *, rates=None, resume=None, max_steps=40):
        return OmegaConf.create({
            "output_dir": str(self.root / name),
            "resume": str(resume) if resume is not None else None,
            "learning_rate": 1e-4,
            "learning_rates": rates or {"video": 1e-4, "action": 1e-4, "fusion": 1e-4},
            "weight_decay": 0.01,
            "batch_size": 1, "num_workers": 0, "num_epochs": 1,
            "max_steps": max_steps, "log_every": 0, "save_every": 0,
            "eval_every": 0, "eval_num_inference_steps": 1,
            "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
            "seed": 42, "mixed_precision": "no",
            "lr_scheduler_type": "cosine", "wandb": {"enabled": False},
        })

    @staticmethod
    def model():
        return TinyLoraPolicy(adapter_config(False), adapter_config(False))

    def trainer(self, name, **config_kwargs):
        return trainer_module.Wan22Trainer(
            self.model(), [{"x": torch.ones(2)}] * 2,
            cfg=self.config(name, **config_kwargs),
        )

    @staticmethod
    def underlying_model(run):
        return getattr(run.model, "module", run.model)

    @classmethod
    def step(cls, run, step_index):
        # Fixed named-parameter gradients make the update independent of group
        # ordering while still populating real AdamW moments and step counters.
        for index, (_, parameter) in enumerate(cls.underlying_model(run).named_parameters()):
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.01 * (index + 1) + 0.001 * step_index)
        run.optimizer.step()
        run.scheduler.step()
        run.optimizer.zero_grad(set_to_none=True)
        run.global_step = step_index + 1

    def assert_same_training_state(self, expected, actual):
        expected_model = self.underlying_model(expected)
        actual_model = self.underlying_model(actual)
        for name, parameter in expected_model.named_parameters():
            other = dict(actual_model.named_parameters())[name]
            torch.testing.assert_close(parameter, other, rtol=0, atol=0, msg=name)
            if parameter.requires_grad:
                expected_state = expected.optimizer.state[parameter]
                actual_state = actual.optimizer.state[other]
                self.assertEqual(set(expected_state), set(actual_state))
                for key in expected_state:
                    torch.testing.assert_close(expected_state[key], actual_state[key], rtol=0, atol=0)
        self.assertEqual(expected.scheduler.last_epoch, actual.scheduler.last_epoch)
        self.assertEqual(
            [group["lr"] for group in expected.optimizer.param_groups],
            [group["lr"] for group in actual.optimizer.param_groups],
        )

    def test_grouped_state_resumes_warmup_and_cosine_with_saved_rates_and_horizon(self):
        for step_count in (1, 11):
            for rates in (
                {"video": 1e-4, "action": 1e-4, "fusion": 1e-4},
                {"video": 1e-4, "action": 3e-4, "fusion": 7e-5},
            ):
                with self.subTest(step_count=step_count, rates=rates):
                    source = self.trainer("source", rates=rates)
                    self.assertFalse(source.legacy_optimizer_group)
                    for step_index in range(step_count):
                        self.step(source, step_index)
                    state_path = source.save_checkpoint()["state_path"]
                    # Full-state restore must retain the saved schedule and LR,
                    # even when constructor hyperparameters differ.
                    restored = self.trainer(
                        "restored", resume=state_path, max_steps=100,
                        rates={"video": 9e-4, "action": 8e-4, "fusion": 6e-4},
                    )
                    self.assertEqual(restored.global_step, step_count)
                    self.assert_same_training_state(source, restored)
                    for step_index in range(step_count, step_count + 5):
                        self.step(source, step_index)
                        self.step(restored, step_index)
                        self.assert_same_training_state(source, restored)

    def old_training_run(self):
        """Build the actual original one-group optimizer/scheduler contract."""
        model = self.model()
        trainer_module.Wan22Trainer._apply_dit_only_train_mode(model)
        named = [("dit." + name, parameter) for name, parameter in model.dit.named_parameters()
                 if parameter.requires_grad]
        named.extend(("proprio_encoder." + name, parameter)
                     for name, parameter in model.proprio_encoder.named_parameters()
                     if parameter.requires_grad)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named], lr=1e-4,
            weight_decay=0.01, betas=(0.9, 0.95),
        )
        main = CosineAnnealingLR(optimizer, T_max=38, eta_min=1e-6)
        warmup = LinearLR(optimizer, start_factor=0.5, end_factor=1.0, total_iters=2)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, main], milestones=[2])
        return SimpleNamespace(
            model=model, optimizer=optimizer, scheduler=scheduler, global_step=0,
            parameter_spec=[{"name": name, "shape": list(parameter.shape)} for name, parameter in named],
        )

    def save_old_checkpoint(self, run, name):
        path = self.root / name
        path.mkdir()
        torch.save({
            "model": run.model.state_dict(), "optimizer": run.optimizer.state_dict(),
            "scheduler": run.scheduler.state_dict(),
        }, path / "cpu_state.pt")
        (path / "trainer_state.json").write_text(json.dumps({
            "global_step": run.global_step, "epoch": 0, "batch_in_epoch": 0,
            "lora": None, "trainable_parameters": run.parameter_spec,
        }))
        return path

    def test_equal_group_rates_preserve_original_schedule_and_updates(self):
        original = self.old_training_run()
        grouped = self.trainer("equal-rates")
        for step_index in range(40):
            for group in grouped.optimizer.param_groups:
                self.assertAlmostEqual(group["lr"], original.optimizer.param_groups[0]["lr"], places=18)
            self.step(original, step_index)
            self.step(grouped, step_index)
            for name, parameter in original.model.named_parameters():
                other = dict(self.underlying_model(grouped).named_parameters())[name]
                torch.testing.assert_close(parameter, other, rtol=0, atol=0, msg=name)
                if parameter.requires_grad:
                    for key, value in original.optimizer.state[parameter].items():
                        torch.testing.assert_close(value, grouped.optimizer.state[other][key], rtol=0, atol=0)
        for group in grouped.optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], 1e-6, places=18)

    def test_distinct_rates_keep_ratios_through_warmup_and_own_one_percent_floor(self):
        rates = {"video": 1e-4, "action": 3e-4, "fusion": 7e-5}
        run = self.trainer("distinct-rates", rates=rates)
        for name, value in run._current_learning_rates().items():
            self.assertAlmostEqual(value, rates[name] / 2, places=18)
        for step_index in range(40):
            self.step(run, step_index)
            current = run._current_learning_rates()
            reference_ratio = current["video"] / rates["video"]
            for name, value in current.items():
                self.assertAlmostEqual(value / rates[name], reference_ratio, places=14)
            if step_index == 1:
                self.assertEqual(current, rates)
        for name, value in run._current_learning_rates().items():
            self.assertAlmostEqual(value, 0.01 * rates[name], places=18)

    def test_actual_legacy_state_resumes_and_remains_resumable_after_resaving(self):
        for step_count in (1, 11):
            with self.subTest(step_count=step_count):
                original = self.old_training_run()
                for step_index in range(step_count):
                    self.step(original, step_index)
                path = self.save_old_checkpoint(original, f"old-{step_count}")
                restored = self.trainer("legacy-restored", resume=path, max_steps=100)
                self.assertTrue(restored.legacy_optimizer_group)
                self.assertEqual(len(restored.optimizer.param_groups), 1)
                self.assert_same_training_state(original, restored)
                for step_index in range(step_count, step_count + 5):
                    self.step(original, step_index)
                    self.step(restored, step_index)
                    self.assert_same_training_state(original, restored)
                    self.assertEqual(set(restored._current_learning_rates()), {"video", "action", "fusion"})
                    self.assertEqual(set(restored._current_learning_rates().values()),
                                     {original.optimizer.param_groups[0]["lr"]})

                resaved = restored.save_checkpoint()["state_path"]
                restored_twice = self.trainer("legacy-restored-twice", resume=resaved, max_steps=100)
                self.assertTrue(restored_twice.legacy_optimizer_group)
                self.assert_same_training_state(restored, restored_twice)
                self.step(restored, step_count + 5)
                self.step(restored_twice, step_count + 5)
                self.assert_same_training_state(restored, restored_twice)

    def test_legacy_state_rejects_distinct_active_rates_before_prepare(self):
        path = self.save_old_checkpoint(self.old_training_run(), "legacy-unequal")
        self.events.clear()
        with self.assertRaisesRegex(ValueError, "[Ll]egacy|learning.rate|single.group"):
            self.trainer("invalid-legacy", resume=path,
                         rates={"video": 1e-4, "action": 2e-4, "fusion": 1e-4})
        self.assertNotIn("prepare", self.events)
        self.assertNotIn("load_state", self.events)

    def test_group_membership_and_order_mismatch_rejected_before_prepare(self):
        source = self.trainer("manifest-source")
        path = Path(source.save_checkpoint()["state_path"])
        state_file = path / "trainer_state.json"
        metadata = json.loads(state_file.read_text())
        for change in ("group_order", "membership"):
            with self.subTest(change=change):
                altered = deepcopy(metadata)
                if change == "group_order":
                    altered["optimizer_groups"].reverse()
                else:
                    altered["optimizer_groups"][0]["parameters"].pop()
                state_file.write_text(json.dumps(altered))
                self.events.clear()
                with self.assertRaisesRegex(ValueError, "group|parameter|optimizer"):
                    self.trainer("invalid-groups", resume=path)
                self.assertNotIn("prepare", self.events)
                self.assertNotIn("load_state", self.events)


if __name__ == "__main__":
    unittest.main()
