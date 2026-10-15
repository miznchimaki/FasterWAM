"""CPU integration tests using real PEFT, autograd, AdamW, and the trainer.

The recording Accelerator substitutes test stage-specific trainer behavior;
these are not DeepSpeed distributed/GPU integration tests.
Run with ``PYTHONPATH=src python -m unittest discover -s tests -v``.
"""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from torch import nn

import fasterwam.trainer as trainer_module
from fasterwam.utils.deepspeed_utils import build_weights_checkpoint_payload
from fasterwam.utils.lora import configure_expert_lora
from test_trainer_deepspeed import RecordingAccelerator


class TinyExpert(nn.Module):
    def __init__(self, action=False):
        super().__init__()
        self.q = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)
        self.offset = nn.Parameter(torch.full((2,), 0.2))
        if action:
            self.action_encoder = nn.Linear(2, 2)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.2)

    def forward(self, x):
        if hasattr(self, "action_encoder"):
            x = self.action_encoder(x)
        return self.head(self.q(x)) + self.offset


def adapter_config(enabled=True, **overrides):
    return {
        "enabled": enabled, "r": 1, "lora_alpha": 2,
        "lora_dropout": 0.0, "target_modules": ["q"], **overrides,
    }


class TinyLoraPolicy(nn.Module):
    def __init__(self, video, action):
        super().__init__()
        self.video_expert = TinyExpert()
        self.action_expert = TinyExpert(action=True)
        configure_expert_lora(self.video_expert, video, branch="video")
        configure_expert_lora(self.action_expert, action, branch="action")
        self.mot = nn.Module()
        self.mot.mixtures = nn.ModuleDict({
            "video": self.video_expert, "action": self.action_expert,
        })
        self.mot.fusion = nn.Linear(2, 2)
        self.dit = self.mot
        self.proprio_encoder = nn.Linear(2, 2)
        self.vae = nn.Linear(2, 2)
        self.torch_dtype = torch.float32
        with torch.no_grad():
            for module in (self.mot.fusion, self.proprio_encoder, self.vae):
                for parameter in module.parameters():
                    parameter.fill_(0.2)

    def forward(self, sample):
        x = sample["x"]
        output = (self.video_expert(x) + self.action_expert(x)
                  + self.mot.fusion(x) + self.proprio_encoder(x) + self.vae(x))
        loss = output.square().mean()
        return loss, {"loss_action": float(loss.detach())}

    def save_checkpoint(self, path, optimizer=None, step=None):
        torch.save(build_weights_checkpoint_payload(self, self.state_dict(), step=step), path)


class LoraTrainingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="fasterwam-lora-training-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def config(self, name, resume=None):
        return OmegaConf.create({
            "output_dir": str(self.root / name), "resume": str(resume) if resume else None,
            "learning_rate": 0.01, "weight_decay": 0.0,
            "batch_size": 1, "num_workers": 0, "num_epochs": 1, "max_steps": 2,
            "log_every": 0, "save_every": 0, "eval_every": 0,
            "eval_num_inference_steps": 1, "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0, "seed": 42, "mixed_precision": "no",
            "lr_scheduler_type": "cosine", "wandb": {"enabled": False},
        })

    @contextmanager
    def accelerator(self, stage, events):
        with patch.object(trainer_module, "Accelerator", side_effect=lambda **kwargs:
                          RecordingAccelerator(events, stage, True, **kwargs)):
            yield

    def make_trainer(self, model, cfg):
        data = [{"x": torch.ones(2)}, {"x": torch.full((2,), 2.0)}]
        return trainer_module.Wan22Trainer(model, data, cfg=cfg)

    @staticmethod
    def snapshot(module):
        return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}

    def test_real_updates_respect_each_expert_switch_and_preserve_freezing(self):
        for stage in (2, 3):
            for video_enabled, action_enabled in ((False, True), (True, False), (True, True), (False, False)):
                with self.subTest(stage=stage, video=video_enabled, action=action_enabled):
                    torch.manual_seed(7)
                    model = TinyLoraPolicy(adapter_config(video_enabled), adapter_config(action_enabled))
                    events = []
                    with self.accelerator(stage, events):
                        trainer = self.make_trainer(model, self.config(f"train-{stage}-{video_enabled}-{action_enabled}"))
                        optimizer_parameters = [p for group in trainer.optimizer.param_groups for p in group["params"]]
                        expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
                        self.assertEqual({id(p) for p in optimizer_parameters}, expected_ids)
                        self.assertEqual(len(optimizer_parameters), len(expected_ids))
                        self.assertTrue(all(not p.requires_grad for p in model.vae.parameters()))

                        before = self.snapshot(model)
                        mask = {name: p.requires_grad for name, p in model.named_parameters()}
                        model.eval()
                        trainer._set_dit_only_train_mode()
                        self.assertEqual({name: p.requires_grad for name, p in model.named_parameters()}, mask)
                        self.assertFalse(model.vae.training)
                        trainer.train()
                        self.assertEqual(events.count("prepared_forward"), 2)
                        self.assertEqual({name: p.requires_grad for name, p in model.named_parameters()}, mask)
                        after = dict(model.named_parameters())
                        for name, trainable in mask.items():
                            if not trainable:
                                self.assertTrue(torch.equal(before[name], after[name]), name)

                        for expert, enabled in ((model.video_expert, video_enabled), (model.action_expert, action_enabled)):
                            names = {id(p): name for name, p in model.named_parameters()}
                            changed_adapters = []
                            for local_name, parameter in expert.named_parameters():
                                is_adapter = ".lora_A." in local_name or ".lora_B." in local_name
                                is_action_io = expert is model.action_expert and local_name.startswith(("head.", "action_encoder."))
                                self.assertEqual(parameter.requires_grad, is_adapter or is_action_io if enabled else True, local_name)
                                changed = not torch.equal(before[names[id(parameter)]], parameter)
                                if enabled and is_adapter:
                                    changed_adapters.append(changed)
                                elif not enabled or is_action_io:
                                    self.assertTrue(changed, local_name)
                            if enabled:
                                self.assertTrue(any(changed_adapters), "LoRA received no optimizer update")
                        self.assertIsInstance(model.action_expert.action_encoder, nn.Linear)
                        self.assertIsInstance(model.action_expert.head, nn.Linear)
                        for module in (model.mot.fusion, model.proprio_encoder):
                            for parameter in module.parameters():
                                self.assertTrue(parameter.requires_grad)
                                name = next(name for name, p in after.items() if p is parameter)
                                self.assertFalse(torch.equal(before[name], parameter), name)

    def test_full_state_records_metadata_and_matching_config_resumes(self):
        for stage in (2, 3):
            with self.subTest(stage=stage):
                configs = (adapter_config(False), adapter_config())
                events = []
                with self.accelerator(stage, events):
                    model = TinyLoraPolicy(*configs)
                    trainer = self.make_trainer(model, self.config(f"save-{stage}"))
                    trainer.global_step = 7
                    state_path = Path(trainer.save_checkpoint()["state_path"])
                    metadata = json.loads((state_path / "trainer_state.json").read_text())
                    self.assertEqual(metadata["lora"]["video"], {"enabled": False})
                    self.assertEqual(metadata["lora"]["action"], adapter_config())
                    expected_spec = [
                        ("dit.mixtures.video.offset", [2]),
                        ("dit.mixtures.video.q.weight", [2, 2]),
                        ("dit.mixtures.video.q.bias", [2]),
                        ("dit.mixtures.video.head.weight", [2, 2]),
                        ("dit.mixtures.video.head.bias", [2]),
                        ("dit.mixtures.action.q.lora_A.default.weight", [1, 2]),
                        ("dit.mixtures.action.q.lora_B.default.weight", [2, 1]),
                        ("dit.mixtures.action.head.weight", [2, 2]),
                        ("dit.mixtures.action.head.bias", [2]),
                        ("dit.mixtures.action.action_encoder.weight", [2, 2]),
                        ("dit.mixtures.action.action_encoder.bias", [2]),
                        ("dit.fusion.weight", [2, 2]), ("dit.fusion.bias", [2]),
                        ("proprio_encoder.weight", [2, 2]), ("proprio_encoder.bias", [2]),
                    ]
                    self.assertEqual(metadata["trainable_parameters"], [
                        {"name": name, "shape": shape} for name, shape in expected_spec
                    ])
                    self.assertIn("save_state", events)
                    events.clear()
                    resumed = self.make_trainer(TinyLoraPolicy(*configs), self.config(f"resume-{stage}", state_path))
                    self.assertLess(events.index("prepare"), events.index("load_state"))
                    self.assertEqual(resumed.global_step, 7)

    def test_full_state_rejects_rank_or_alpha_change_before_prepare(self):
        for stage in (2, 3):
            events = []
            with self.accelerator(stage, events):
                trainer = self.make_trainer(
                    TinyLoraPolicy(adapter_config(False), adapter_config()), self.config(f"source-{stage}"))
                state_path = trainer.save_checkpoint()["state_path"]
                for override in ({"r": 2}, {"lora_alpha": 4}):
                    with self.subTest(stage=stage, override=override):
                        events.clear()
                        model = TinyLoraPolicy(adapter_config(False), adapter_config(**override))
                        with self.assertRaisesRegex(ValueError, "[Ll]o[Rr][Aa]|[Cc]onfig|[Mm]ismatch"):
                            self.make_trainer(model, self.config(f"bad-{stage}", state_path))
                        self.assertNotIn("prepare", events)
                        self.assertNotIn("load_state", events)

    def test_full_state_rejects_changed_optimizer_parameter_manifest(self):
        events = []
        with self.accelerator(3, events):
            configs = (adapter_config(False), adapter_config())
            trainer = self.make_trainer(TinyLoraPolicy(*configs), self.config("manifest-source"))
            state_path = Path(trainer.save_checkpoint()["state_path"])
            state_file = state_path / "trainer_state.json"
            saved = json.loads(state_file.read_text())
            for mismatch in ("shape", "order", "missing", "old_frozen_io"):
                with self.subTest(mismatch=mismatch):
                    changed = deepcopy(saved)
                    if mismatch == "shape":
                        changed["trainable_parameters"][0]["shape"] = [999]
                    elif mismatch == "order":
                        changed["trainable_parameters"].reverse()
                    elif mismatch == "missing":
                        changed.pop("trainable_parameters")
                    else:
                        changed["trainable_parameters"] = [
                            item for item in changed["trainable_parameters"]
                            if not item["name"].startswith(("dit.mixtures.action.head.", "dit.mixtures.action.action_encoder."))
                        ]
                    state_file.write_text(json.dumps(changed))
                    events.clear()
                    with self.assertRaisesRegex(ValueError, "parameter"):
                        self.make_trainer(TinyLoraPolicy(*configs), self.config("bad-manifest", state_path))
                    self.assertNotIn("prepare", events)
                    self.assertNotIn("load_state", events)

    def test_preprocessing_loaders_strip_lora_constructor_options(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/model/fasterwam.yaml"
        for filename in ("preprocess_action_dit_backbone.py", "preprocess_sparse_action_dit_backbone.py"):
            with self.subTest(script=filename):
                spec = importlib.util.spec_from_file_location("preprocess_under_test", root / "scripts" / filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                video, action, original = module._load_model_config(config_path)
                self.assertNotIn("lora", video)
                self.assertNotIn("lora", action)
                self.assertIn("lora", original.video_dit_config)
                self.assertIn("lora", original.action_dit_config)
                self.assertEqual(video["num_layers"], 30)
                self.assertEqual(action["num_layers"], 30)

    def test_factory_injects_real_adapters_after_dense_pretrained_loading(self):
        import fasterwam.models.wan22.fasterwam as factory_module

        events = []
        experts = {branch: TinyExpert(action=branch == "action") for branch in ("video", "action")}
        for expert in experts.values():
            expert.blocks = nn.ModuleList([nn.Identity()])
            expert.num_heads = expert.attn_head_dim = 2
            expert.hidden_dim = 4

        def load_video(**kwargs):
            self.assertNotIn("lora", kwargs["dit_config"])
            events.append("load_video")
            with torch.no_grad():
                experts["video"].q.weight.fill_(3)
            return SimpleNamespace(dit=experts["video"], vae=nn.Identity(), text_encoder=None,
                                   tokenizer=None, dit_path="video.pt", vae_path="vae.pt",
                                   text_encoder_path=None, tokenizer_path=None)

        def load_action(**kwargs):
            self.assertNotIn("lora", kwargs["action_dit_config"])
            events.append("load_action")
            with torch.no_grad():
                experts["action"].q.weight.fill_(4)
            return experts["action"]

        class FactoryPolicy(factory_module.FasterWAM):
            def __init__(self, **kwargs):
                nn.Module.__init__(self)
                self.video_expert = kwargs["video_expert"]
                self.action_expert = kwargs["action_expert"]
                self.mot = kwargs["mot"]
                events.append("model_init")

        def inject(expert, config):
            branch = "video" if expert is experts["video"] else "action"
            self.assertFalse(hasattr(expert.q, "lora_A"))
            expected = 3 if branch == "video" else 4
            torch.testing.assert_close(expert.q.weight, torch.full_like(expert.q.weight, expected))
            self.assertIn("model_init", events)
            events.append(f"inject_{branch}")
            return configure_expert_lora(expert, config, branch=branch)

        video_config = {"text_dim": 2, "lora": adapter_config()}
        action_config = {"num_layers": 1, "lora": adapter_config()}
        original = deepcopy((video_config, action_config))
        with patch.object(factory_module, "load_wan22_ti2v_5b_components", side_effect=load_video), \
                patch.object(factory_module.SparseActionDiT, "from_pretrained", side_effect=load_action), \
                patch.object(factory_module, "SparseMoT", side_effect=lambda **kwargs: nn.Module()), \
                patch.object(factory_module, "configure_expert_lora", side_effect=inject):
            model = FactoryPolicy.from_wan22_pretrained(
                device="cpu", torch_dtype=torch.float32, video_dit_config=video_config,
                action_dit_config=action_config, condition_layers=[0])
        self.assertEqual(events, ["load_video", "load_action", "model_init", "inject_video", "inject_action"])
        self.assertEqual((video_config, action_config), original)
        self.assertTrue(hasattr(model.video_expert.q, "lora_A"))
        self.assertTrue(hasattr(model.action_expert.q, "lora_A"))


if __name__ == "__main__":
    unittest.main()
