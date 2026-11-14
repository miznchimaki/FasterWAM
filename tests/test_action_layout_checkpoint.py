"""Compact layer metadata guards, including same-shape semantic mismatches."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import fasterwam.trainer as trainer_module
from fasterwam.models.wan22.action_layout import resolve_action_layout
from fasterwam.utils.action_layout_checkpoint import get_action_layout_metadata
from fasterwam.utils.deepspeed_utils import build_weights_checkpoint_payload
from fasterwam.utils.lora_merge import merge_lora_checkpoint_payload
from test_lora_checkpoint import TinyPolicy
from test_lora_merge import make_payload
from test_lora_training import TinyLoraPolicy, adapter_config
from test_trainer_deepspeed import RecordingAccelerator


def set_layout(model, *, indices=(0, 1, 4), conditions=(0, 4), video_depth=6):
    # Deliberately use identical tensor schemas for different layer mappings:
    # names/shapes alone must not make an incompatible checkpoint loadable.
    model.mot.num_layers = video_depth
    model.mot.condition_layers = conditions
    model.mot.action_layer_indices = indices
    return model


class ActionLayoutCheckpointTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="fasterwam-action-layout-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.path = self.root / "weights.pt"

    def assert_rejected_without_mutation(self, model):
        before = {name: value.clone() for name, value in model.state_dict().items()}
        modules = {name: id(module) for name, module in model.named_modules()}
        with self.assertRaisesRegex(ValueError, "action_layout"):
            model.load_checkpoint(self.path)
        self.assertEqual(modules, {name: id(module) for name, module in model.named_modules()})
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_dense_and_lora_compact_roundtrip_and_zero3_export(self):
        for rank in (None, 2):
            with self.subTest(rank=rank):
                source = set_layout(TinyPolicy(action_rank=rank))
                source.save_checkpoint(self.path, step=7)
                payload = torch.load(self.path, weights_only=True)
                self.assertEqual(payload["action_layout"], get_action_layout_metadata(source)["action_layout"])
                gathered = build_weights_checkpoint_payload(source, source.state_dict(), step=7)
                self.assertEqual(gathered["action_layout"], payload["action_layout"])
                target = set_layout(TinyPolicy())
                target.load_checkpoint(self.path)
                for name, value in source.mot.state_dict().items():
                    torch.testing.assert_close(value, target.mot.state_dict()[name], rtol=0, atol=0)

    def test_same_shape_wrong_mapping_rejected_before_lora_injection(self):
        source = set_layout(TinyPolicy(action_rank=2))
        source.save_checkpoint(self.path)
        for changes in (
            {"indices": (0, 2, 4)},
            {"indices": (0, 1, 5), "conditions": (0, 5)},
            {"video_depth": 7},
        ):
            with self.subTest(changes=changes):
                self.assert_rejected_without_mutation(set_layout(TinyPolicy(), **changes))

    def test_removed_prefix_layout_roundtrip_and_same_depth_mismatch(self):
        without_prefix = resolve_action_layout(8, [2, 5], 1, 1, leading_noncondition_layers=0)
        with_prefix = resolve_action_layout(8, [2, 5], 1, 0, leading_noncondition_layers=1)
        self.assertEqual(without_prefix.num_action_layers, with_prefix.num_action_layers)
        source = set_layout(
            TinyPolicy(action_rank=2), indices=without_prefix.action_layer_indices,
            conditions=(2, 5), video_depth=8,
        )
        source.save_checkpoint(self.path)
        matching = set_layout(
            TinyPolicy(), indices=without_prefix.action_layer_indices, conditions=(2, 5), video_depth=8,
        )
        matching.load_checkpoint(self.path)
        for name, value in source.mot.state_dict().items():
            torch.testing.assert_close(value, matching.mot.state_dict()[name], rtol=0, atol=0)
        wrong_prefix = set_layout(
            TinyPolicy(), indices=with_prefix.action_layer_indices, conditions=(2, 5), video_depth=8,
        )
        self.assert_rejected_without_mutation(wrong_prefix)

    def test_compact_metadata_required_but_identity_legacy_payload_unchanged(self):
        source = set_layout(TinyPolicy())
        source.save_checkpoint(self.path)
        payload = torch.load(self.path, weights_only=True)
        payload.pop("action_layout")
        torch.save(payload, self.path)
        self.assert_rejected_without_mutation(set_layout(TinyPolicy()))

        identity_source = set_layout(TinyPolicy(), indices=tuple(range(6)))
        identity_source.save_checkpoint(self.path)
        legacy = torch.load(self.path, weights_only=True)
        self.assertNotIn("action_layout", legacy)
        identity_target = set_layout(TinyPolicy(), indices=tuple(range(6)))
        identity_target.load_checkpoint(self.path)
        for name, value in identity_source.mot.state_dict().items():
            torch.testing.assert_close(value, identity_target.mot.state_dict()[name], rtol=0, atol=0)

    def test_video_only_warm_start_keeps_existing_behavior(self):
        source = TinyPolicy()
        torch.save({"dit": source.video_expert.state_dict()}, self.path)
        target = set_layout(TinyPolicy())
        action_before = deepcopy(target.action_expert.state_dict())
        target.load_checkpoint(self.path)
        for name, value in source.video_expert.state_dict().items():
            torch.testing.assert_close(value, target.video_expert.state_dict()[name], rtol=0, atol=0)
        for name, value in action_before.items():
            torch.testing.assert_close(value, target.action_expert.state_dict()[name], rtol=0, atol=0)

    def test_lora_merge_preserves_layout_and_rejects_invalid_metadata(self):
        _, payload = make_payload()
        metadata = get_action_layout_metadata(set_layout(TinyPolicy()))["action_layout"]
        payload["action_layout"] = metadata
        merged = merge_lora_checkpoint_payload(payload)
        self.assertEqual(merged["action_layout"], metadata)
        self.assertIsNot(merged["action_layout"], metadata)
        self.assertNotIn("lora", merged)
        for field, value in (("version", True), ("video_num_layers", 0),
                             ("condition_layers", [0, 5]), ("action_layer_indices", [0, 1, 1, 4])):
            with self.subTest(field=field):
                invalid = dict(payload, action_layout={**metadata, field: value})
                with self.assertRaisesRegex(ValueError, "action_layout"):
                    merge_lora_checkpoint_payload(invalid)

    def config(self, name, resume=None):
        return OmegaConf.create({
            "output_dir": str(self.root / name), "resume": str(resume) if resume else None,
            "learning_rate": 1e-4, "weight_decay": 0.0,
            "batch_size": 1, "num_workers": 0, "num_epochs": 1, "max_steps": 1,
            "log_every": 0, "save_every": 0, "eval_every": 0,
            "eval_num_inference_steps": 1, "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0, "seed": 42, "mixed_precision": "no",
            "lr_scheduler_type": "cosine", "wandb": {"enabled": False},
        })

    def make_trainer(self, config):
        model = set_layout(TinyLoraPolicy(adapter_config(False), adapter_config(False)))
        return trainer_module.Wan22Trainer(model, [{"x": torch.ones(2)}], cfg=config)

    def test_full_state_layout_roundtrip_and_mismatch_rejected_before_prepare(self):
        events = []
        with patch.object(trainer_module, "Accelerator", side_effect=lambda **kwargs:
                          RecordingAccelerator(events, 2, True, **kwargs)):
            source = self.make_trainer(self.config("source"))
            state_path = Path(source.save_checkpoint()["state_path"])
            metadata_file = state_path / "trainer_state.json"
            saved = json.loads(metadata_file.read_text())
            self.assertEqual(saved["action_layout"], source.training_action_layout["action_layout"])
            self.make_trainer(self.config("matching", state_path))
            self.assertIn("load_state", events)
            for mode in ("wrong_mapping", "missing_layout", "missing_manifest"):
                with self.subTest(mode=mode):
                    changed = deepcopy(saved)
                    if mode == "wrong_mapping":
                        changed["action_layout"]["action_layer_indices"] = [0, 2, 4]
                    elif mode == "missing_layout":
                        changed.pop("action_layout")
                    metadata_file.write_text(json.dumps(changed))
                    if mode == "missing_manifest":
                        metadata_file.unlink()
                    events.clear()
                    with self.assertRaisesRegex(ValueError, "action_layout"):
                        self.make_trainer(self.config("bad-resume", state_path))
                    self.assertNotIn("prepare", events)
                    self.assertNotIn("load_state", events)


if __name__ == "__main__":
    unittest.main()
