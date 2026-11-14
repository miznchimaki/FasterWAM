"""CPU regressions for component learning rates and their visible metrics.

Use real PEFT, autograd and AdamW with the recording Accelerator fixture.
These tests do not run DeepSpeed collectives or GPU kernels.
"""

from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

import fasterwam.trainer as trainer_module
from test_lora_training import TinyLoraPolicy, adapter_config
from test_trainer_deepspeed import RecordingAccelerator


REPO_ROOT = Path(__file__).resolve().parents[1]


class LearningRateTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="fasterwam-lr-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.rates = {"video": 0.001, "action": 0.002, "fusion": 0.003}

    def config(self, name, **overrides):
        config = {
            "output_dir": str(self.root / name), "resume": None,
            "learning_rate": 0.01, "learning_rates": self.rates,
            "weight_decay": 0.0, "batch_size": 1, "num_workers": 0,
            "num_epochs": 1, "max_steps": 1, "log_every": 0,
            "save_every": 0, "eval_every": 0, "eval_num_inference_steps": 1,
            "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
            "seed": 42, "mixed_precision": "no", "lr_scheduler_type": "constant",
            "wandb": {"enabled": False},
        }
        config.update(overrides)
        return OmegaConf.create(config)

    def model(self, *, video_lora=False, action_lora=False,
              video_freeze=False, action_freeze=False):
        return TinyLoraPolicy(
            adapter_config(video_lora), adapter_config(action_lora),
            video_freeze=video_freeze, action_freeze=action_freeze,
        )

    def trainer(self, model, cfg, *, stage=2):
        events = []
        with patch.object(
            trainer_module, "Accelerator",
            side_effect=lambda **kwargs: RecordingAccelerator(events, stage, True, **kwargs),
        ):
            return trainer_module.Wan22Trainer(model, [{"x": torch.ones(2)}], cfg=cfg)

    @staticmethod
    def trainable_ids(module):
        return {id(parameter) for parameter in module.parameters() if parameter.requires_grad}

    def assert_group_coverage(self, trainer, model):
        groups = {group["name"]: group for group in trainer.optimizer.param_groups}
        expected = {
            "video": self.trainable_ids(model.video_expert),
            "action": self.trainable_ids(model.action_expert) | self.trainable_ids(model.proprio_encoder),
            "fusion": self.trainable_ids(model.mot.fusion),
        }
        expected = {name: ids for name, ids in expected.items() if ids}
        self.assertEqual(set(groups), set(expected))
        for name, ids in expected.items():
            self.assertEqual({id(parameter) for parameter in groups[name]["params"]}, ids)
            self.assertEqual(groups[name]["lr"], self.rates[name])
        parameters = [parameter for group in groups.values() for parameter in group["params"]]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertEqual({id(parameter) for parameter in parameters}, self.trainable_ids(model))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.vae.parameters()))

    def test_different_rates_control_real_adamw_updates(self):
        for stage in (2, 3):
            with self.subTest(stage=stage):
                model = self.model()
                trainer = self.trainer(model, self.config(f"updates-{stage}"), stage=stage)
                self.assert_group_coverage(trainer, model)
                before = {id(parameter): parameter.detach().clone() for parameter in model.parameters()}
                loss, _ = trainer.model({"x": torch.ones(1, 2)})
                loss.backward()
                trainer.optimizer.step()
                # The fixture has positive inputs/weights, so the first AdamW
                # update has positive gradients and magnitude equal to its LR.
                for group in trainer.optimizer.param_groups:
                    for parameter in group["params"]:
                        self.assertTrue(torch.all(parameter.grad > 0))
                        torch.testing.assert_close(
                            before[id(parameter)] - parameter,
                            torch.full_like(parameter, self.rates[group["name"]]),
                            rtol=1e-4, atol=1e-7,
                        )

    def test_freeze_and_lora_preserve_complete_disjoint_group_membership(self):
        for stage in (2, 3):
            for video_lora, action_lora, video_freeze, action_freeze in product((False, True), repeat=4):
                with self.subTest(stage=stage, video_lora=video_lora, action_lora=action_lora,
                                  video_freeze=video_freeze, action_freeze=action_freeze):
                    model = self.model(
                        video_lora=video_lora, action_lora=action_lora,
                        video_freeze=video_freeze, action_freeze=action_freeze,
                    )
                    trainer = self.trainer(model, self.config("switches"), stage=stage)
                    self.assert_group_coverage(trainer, model)
                    active = {"action", "fusion"}
                    if video_lora or not video_freeze:
                        active.add("video")
                    self.assertEqual(trainer._current_learning_rates(), {
                        name: rate for name, rate in self.rates.items() if name in active
                    })
                    # Mode restoration must not add frozen tensors to any group.
                    trainer._set_dit_only_train_mode()
                    self.assert_group_coverage(trainer, model)

    def test_equal_rates_preserve_flat_optimizer_numerics(self):
        model = self.model()
        reference = deepcopy(model)
        rates = {name: 0.01 for name in self.rates}
        cfg = self.config("equal-rates", learning_rates=rates, weight_decay=0.01)
        trainer = self.trainer(model, cfg)
        trainer_module.Wan22Trainer._apply_dit_only_train_mode(reference)
        reference_parameters = [p for p in reference.dit.parameters() if p.requires_grad]
        reference_parameters.extend(p for p in reference.proprio_encoder.parameters() if p.requires_grad)
        reference_optimizer = torch.optim.AdamW(
            reference_parameters, lr=0.01, weight_decay=0.01, betas=(0.9, 0.95),
        )
        for value in (1.0, 2.0, 0.5):
            sample = {"x": torch.full((1, 2), value)}
            trainer.model(sample)[0].backward()
            reference(sample)[0].backward()
            trainer.optimizer.step()
            reference_optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_console_and_wandb_report_only_active_learning_rates(self):
        for frozen_video in (False, True):
            with self.subTest(frozen_video=frozen_video):
                model = self.model(video_freeze=frozen_video, action_freeze=True)
                trainer = self.trainer(model, self.config("logs", log_every=1))
                expected = {"action": self.rates["action"], "fusion": self.rates["fusion"]}
                if not frozen_video:
                    expected["video"] = self.rates["video"]
                with patch.object(trainer, "_wandb_log") as wandb_log, \
                        self.assertLogs(trainer_module.logger, level="INFO") as captured:
                    trainer.train()
                payload = wandb_log.call_args.args[0]
                self.assertEqual({key: value for key, value in payload.items() if key.startswith("train/lr")}, {
                    f"train/lr_{name}": value for name, value in expected.items()
                })
                train_lines = [line for line in captured.output if "[train]" in line]
                self.assertEqual(len(train_lines), 1)
                for name, value in expected.items():
                    self.assertIn(f"lr_{name}={value:.2e}", train_lines[0])
                self.assertNotIn(" lr=", train_lines[0])
                if frozen_video:
                    self.assertNotIn("lr_video=", train_lines[0])

    def test_metrics_survive_deepspeed_empty_parameter_lists(self):
        trainer = self.trainer(self.model(video_freeze=True), self.config("partitioned"), stage=3)
        expected = trainer._current_learning_rates()
        for group in trainer.optimizer.param_groups:
            # Some ZeRO optimizer wrappers expose LR metadata while the local
            # public param lists are empty; this is not a frozen component.
            group["params"] = []
            group["lr"] *= 0.5
        self.assertEqual(trainer._current_learning_rates(), {
            name: rate * 0.5 for name, rate in expected.items()
        })

    def test_legacy_scalar_and_partial_mapping_fall_back_per_component(self):
        cfg = self.config("scalar")
        del cfg.learning_rates
        trainer = self.trainer(self.model(), cfg)
        self.assertEqual(trainer._current_learning_rates(), dict.fromkeys(self.rates, cfg.learning_rate))
        cfg = self.config("partial", learning_rates={"action": 0.003})
        trainer = self.trainer(self.model(), cfg)
        self.assertEqual(trainer._current_learning_rates(), {
            "video": cfg.learning_rate, "action": 0.003, "fusion": cfg.learning_rate,
        })

    def test_hydra_defaults_scalar_compatibility_and_independent_overrides(self):
        with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
            for task in ("libero_fasterwam_2cam224_1e-4", "robotwin_fasterwam_3cam_384_1e-4"):
                with self.subTest(task=task):
                    cfg = compose(config_name="train", overrides=[f"task={task}"])
                    self.assertEqual(dict(cfg.learning_rates), dict.fromkeys(self.rates, 1e-4))
                    cfg = compose(config_name="train", overrides=[f"task={task}", "learning_rate=2e-4"])
                    self.assertEqual(dict(cfg.learning_rates), dict.fromkeys(self.rates, 2e-4))
                    cfg = compose(config_name="train", overrides=[
                        f"task={task}", "learning_rate=2e-4", "learning_rates.action=3e-4",
                        "learning_rates.fusion=5e-5",
                    ])
                    self.assertEqual(dict(cfg.learning_rates), {
                        "video": 2e-4, "action": 3e-4, "fusion": 5e-5,
                    })


if __name__ == "__main__":
    unittest.main()
