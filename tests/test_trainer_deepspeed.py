"""CPU trainer regressions with recording Accelerator/engine substitutes.

These exercise the real trainer, Torch autograd, optimizer, and checkpoint files.
They do not import DeepSpeed or validate its distributed collectives/GPU kernels.
Run with ``PYTHONPATH=src python -m unittest discover -s tests -v``.
"""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from omegaconf import OmegaConf

import fasterwam.trainer as trainer_module


class TinyPolicy(nn.Module):
    """Retain FasterWAM's public aliases and portable checkpoint structure."""

    def __init__(self, events):
        super().__init__()
        self.events = events
        self.video_expert = nn.Linear(1, 1, bias=False)
        self.action_expert = nn.Linear(1, 1, bias=False)
        self.mot = nn.Module()
        self.mot.mixtures = nn.ModuleDict({
            "video": self.video_expert, "action": self.action_expert,
        })
        self.dit = self.mot
        self.vae = nn.Linear(1, 1, bias=False)
        self.proprio_encoder = None
        self.torch_dtype = torch.float32
        with torch.no_grad():
            self.video_expert.weight.fill_(0.5)
            self.action_expert.weight.fill_(0.5)

    def forward(self, sample):
        return self.training_loss(sample)

    def training_loss(self, sample):
        self.events.append("policy_loss")
        loss = self.action_expert(self.video_expert(sample["x"])).square().mean()
        return loss, {"loss_action": float(loss.detach())}

    def load_checkpoint(self, path, optimizer=None):
        self.events.append("load_weights")
        self.mot.load_state_dict(torch.load(path, weights_only=True)["mot"])

    def save_checkpoint(self, path, optimizer=None, step=None):
        self.events.append("save_weights")
        torch.save({"mot": self.mot.state_dict(), "step": step}, path)


class RecordingEngine(nn.Module):
    def __init__(self, module, events):
        super().__init__()
        self.module = module
        self.events = events

    def forward(self, sample):
        self.events.append("prepared_forward")
        return self.module(sample)

    def training_loss(self, sample):
        raise AssertionError("The trainer bypassed the prepared model's forward.")


class RecordingAccelerator:
    def __init__(self, events, stage, main, **kwargs):
        self.events = events
        self.device = torch.device("cpu")
        self.distributed_type = "DEEPSPEED"
        self.mixed_precision = kwargs["mixed_precision"]
        self.num_processes = 1
        self.process_index = 0 if main else 1
        self.is_main_process = main
        self.sync_gradients = True
        self.optimizer_step_was_skipped = False
        self.state = SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(deepspeed_config={
                "zero_optimization": {
                    "stage": stage,
                    "stage3_gather_16bit_weights_on_model_save": True,
                },
            })
        )

    def prepare(self, model, optimizer, loader, scheduler):
        self.events.append("prepare")
        self.config_at_prepare = deepcopy(self.state.deepspeed_plugin.deepspeed_config)
        self.weights_at_prepare = [p.detach().clone() for p in model.dit.parameters()]
        self.engine = RecordingEngine(model, self.events)
        return self.engine, optimizer, loader, scheduler

    @staticmethod
    def unwrap_model(model):
        return model.module if isinstance(model, RecordingEngine) else model

    @staticmethod
    def gather(tensor):
        return tensor

    def accumulate(self, model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        self.events.append("backward")
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters, max_norm):
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))

    def wait_for_everyone(self):
        self.events.append("barrier")

    def get_state_dict(self, model):
        self.events.append("get_state_dict")
        if self.is_main_process:
            return self.unwrap_model(model).state_dict()
        # DeepSpeed's non-main ranks participate but do not receive full weights.
        return None

    def save_state(self, output_dir):
        self.events.append("save_state")
        Path(output_dir, "recorded-state").write_text("collective participated")

    def load_state(self, input_dir):
        self.events.append("load_state")


class TrainerDeepSpeedTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="fasterwam-trainer-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def config(self, output_dir, resume=None):
        return OmegaConf.create({
            "output_dir": str(output_dir), "resume": str(resume) if resume else None,
            "learning_rate": 0.01, "weight_decay": 0.0,
            "batch_size": 1, "num_workers": 0, "num_epochs": 1, "max_steps": 1,
            "log_every": 0, "save_every": 0, "eval_every": 0,
            "eval_num_inference_steps": 1, "gradient_accumulation_steps": 1,
            "max_grad_norm": 0.37, "seed": 42, "mixed_precision": "no",
            "lr_scheduler_type": "cosine", "wandb": {"enabled": False},
        })

    @contextmanager
    def harness(self, cfg, stage=2, main=True):
        events = []
        model = TinyPolicy(events)
        original_adamw = torch.optim.AdamW
        optimizer_weights = []

        def create_optimizer(parameters, **kwargs):
            events.append("optimizer_created")
            parameters = list(parameters)
            optimizer_weights.extend(p.detach().clone() for p in parameters)
            return original_adamw(parameters, **kwargs)

        def create_accelerator(**kwargs):
            return RecordingAccelerator(events, stage, main, **kwargs)

        data = [{"x": torch.tensor([1.0])}, {"x": torch.tensor([2.0])}]
        with patch.object(trainer_module, "Accelerator", side_effect=create_accelerator), \
                patch.object(torch.optim, "AdamW", side_effect=create_optimizer):
            trainer = trainer_module.Wan22Trainer(model, data, cfg=cfg)
            yield trainer, model, events, optimizer_weights

    def test_weight_warm_start_precedes_optimizer_and_prepare(self):
        checkpoint = self.root / "warm.pt"
        torch.save({"mot": {
            "mixtures.video.weight": torch.tensor([[3.0]]),
            "mixtures.action.weight": torch.tensor([[4.0]]),
        }}, checkpoint)
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                cfg = self.config(self.root / f"warm-stage-{stage}", checkpoint)
                with self.harness(cfg, stage=stage) as (trainer, _, events, weights):
                    self.assertLess(events.index("load_weights"), events.index("optimizer_created"))
                    self.assertLess(events.index("optimizer_created"), events.index("prepare"))
                    self.assertEqual(events.count("load_weights"), 1)
                    self.assertNotIn("load_state", events)
                    self.assertEqual([p.item() for p in weights], [3.0, 4.0])
                    self.assertEqual([p.item() for p in trainer.accelerator.weights_at_prepare], [3.0, 4.0])

    def test_full_state_resume_follows_prepare_and_restores_progress(self):
        state_dir = self.root / "resume-state"
        state_dir.mkdir()
        (state_dir / "trainer_state.json").write_text(json.dumps({
            "global_step": 7, "epoch": 2, "batch_in_epoch": 3,
        }))
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                cfg = self.config(self.root / f"resume-stage-{stage}", state_dir)
                with self.harness(cfg, stage=stage) as (trainer, _, events, _):
                    self.assertLess(events.index("optimizer_created"), events.index("prepare"))
                    self.assertLess(events.index("prepare"), events.index("load_state"))
                    self.assertNotIn("load_weights", events)
                    self.assertEqual((trainer.global_step, trainer.epoch, trainer.batch_in_epoch), (7, 2, 3))
                    self.assertEqual(trainer.train_sampler.epoch_offset, 2)
                    self.assertEqual(trainer.train_sampler.resume_batch_offset, 3)

    def test_zero3_save_is_collective_but_only_main_writes_portable_weights(self):
        for main in (False, True):
            with self.subTest(main=main):
                cfg = self.config(self.root / f"save-main-{main}")
                with self.harness(cfg, stage=3, main=main) as (trainer, model, events, _):
                    trainer.global_step = 11
                    events.clear()
                    result = trainer.save_checkpoint()
                    self.assertEqual(events.count("get_state_dict"), 1)
                    self.assertEqual(events.count("save_state"), 1)
                    self.assertLess(events.index("get_state_dict"), events.index("save_state"))
                    self.assertTrue(Path(result["state_path"], "recorded-state").is_file())
                    weight_files = list(Path(trainer.weights_dir).glob("*.pt"))
                    self.assertEqual(len(weight_files), int(main))
                    self.assertEqual(Path(result["state_path"], "trainer_state.json").exists(), main)
                    if main:
                        payload = torch.load(result["weights_path"], weights_only=True)
                        self.assertEqual(set(payload), {"mot", "step", "torch_dtype"})
                        self.assertEqual(payload["step"], 11)
                        for key, value in model.mot.state_dict().items():
                            torch.testing.assert_close(payload["mot"][key], value)
                    else:
                        self.assertIsNone(result["weights_path"])

    def test_training_uses_prepared_forward_and_supplies_clipping_before_prepare(self):
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                cfg = self.config(self.root / f"train-stage-{stage}")
                with self.harness(cfg, stage=stage) as (trainer, model, events, _):
                    self.assertEqual(trainer.accelerator.config_at_prepare["gradient_clipping"], 0.37)
                    before = model.action_expert.weight.detach().clone()
                    trainer.train()
                    self.assertEqual(events.count("prepared_forward"), 1)
                    self.assertLess(events.index("prepared_forward"), events.index("policy_loss"))
                    self.assertLess(events.index("policy_loss"), events.index("backward"))
                    self.assertFalse(torch.equal(before, model.action_expert.weight))
                    self.assertEqual(trainer.global_step, 1)


if __name__ == "__main__":
    unittest.main()
