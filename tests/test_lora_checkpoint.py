"""Real CPU PEFT checkpoint round trips and fail-before-mutation checks."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fasterwam.models.wan22.fastwam import FastWAM  # noqa: E402
from fasterwam.utils.deepspeed_utils import build_weights_checkpoint_payload  # noqa: E402
from fasterwam.utils.lora import apply_expert_trainability, configure_expert_lora, get_expert_lora_config  # noqa: E402
from fasterwam.utils.lora_checkpoint import get_lora_checkpoint_metadata  # noqa: E402


class TinyExpert(nn.Module):
    def __init__(self, action=False):
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)
        if action:
            self.action_encoder = nn.Linear(4, 4)

    def forward(self, value):
        if hasattr(self, "action_encoder"):
            value = self.action_encoder(value)
        return self.head(torch.tanh(self.q(value)))


class TinyPolicy(FastWAM):
    """Use real FastWAM save/load without constructing the multi-billion DiTs."""

    def __init__(self, *, action_rank=None, video_rank=None, targets=("q",), proprio=True,
                 allow_legacy_action_io=False):
        nn.Module.__init__(self)
        self.video_expert = TinyExpert()
        self.action_expert = TinyExpert(action=True)
        self.mot = nn.Module()
        self.mot.mixtures = nn.ModuleDict({"video": self.video_expert, "action": self.action_expert})
        self.mot.video_kv_fusion_logits = nn.ParameterList([nn.Parameter(torch.randn(2))])
        self.dit = self.mot
        self.proprio_encoder = nn.Linear(3, 4) if proprio else None
        self.torch_dtype = torch.float32
        self.vae = nn.Linear(2, 2)
        for branch, rank in (("video", video_rank), ("action", action_rank)):
            if rank is not None:
                configure_expert_lora(
                    getattr(self, branch + "_expert"),
                    {"enabled": True, "r": rank, "lora_alpha": 2 * rank,
                     "lora_dropout": 0.2, "target_modules": list(targets)},
                    branch=branch,
                    allow_legacy_action_io=allow_legacy_action_io,
                )
        self.eval()

    def predict(self, value):
        return self.action_expert(self.video_expert(value))


def change_adapter_weights(model):
    with torch.no_grad():
        for name, parameter in model.mot.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                parameter.fill_(0.15)


class LoRACheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name) / "weights.pt"
        self.input = torch.randn(3, 4)

    def assert_weights_equal(self, first, second):
        first_state, second_state = first.mot.state_dict(), second.mot.state_dict()
        self.assertEqual(set(first_state), set(second_state))
        for key in first_state:
            torch.testing.assert_close(first_state[key], second_state[key], rtol=0, atol=0)
        torch.testing.assert_close(first.predict(self.input), second.predict(self.input), rtol=0, atol=0)

    def save_modified_payload(self, source, change):
        source.save_checkpoint(self.path, step=17)
        payload = torch.load(self.path, weights_only=True)
        change(payload)
        torch.save(payload, self.path)

    def assert_rejected_without_mutation(self, target, pattern, **kwargs):
        before = {key: value.clone() for key, value in target.state_dict().items()}
        module_ids = {name: id(module) for name, module in target.named_modules()}
        with self.assertRaisesRegex((ValueError, TypeError), pattern):
            target.load_checkpoint(self.path, **kwargs)
        self.assertEqual({name: id(module) for name, module in target.named_modules()}, module_ids)
        self.assertEqual(set(target.state_dict()), set(before))
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_full_lora_checkpoint_restores_default_dense_evaluation_model(self):
        source = TinyPolicy(action_rank=2)
        change_adapter_weights(source)
        source.save_checkpoint(self.path, step=17)
        target = TinyPolicy()
        action_object = target.action_expert
        payload = target.load_checkpoint(self.path)
        target.eval()
        self.assert_weights_equal(source, target)
        self.assertIs(target.action_expert, action_object)
        self.assertIs(target.mot.mixtures["action"], action_object)
        self.assertEqual(payload["checkpoint_format_version"], 2)
        self.assertEqual(payload["step"], 17)
        self.assertTrue(payload["lora"]["action"]["enabled"])
        self.assertIn("mixtures.action.q.base_layer.weight", payload["mot"])
        torch.testing.assert_close(source.proprio_encoder.weight, target.proprio_encoder.weight)

    def test_metadata_reconfigures_different_rank_targets_and_branch_enablement(self):
        source = TinyPolicy(action_rank=3, targets=("head",), allow_legacy_action_io=True)
        change_adapter_weights(source)
        source.save_checkpoint(self.path)
        target = TinyPolicy(action_rank=1, video_rank=1)
        target.load_checkpoint(self.path)
        target.eval()
        self.assert_weights_equal(source, target)
        self.assertFalse(get_expert_lora_config(target.video_expert)["enabled"])
        self.assertEqual(get_expert_lora_config(target.action_expert)["r"], 3)

    def test_legacy_action_io_adapters_preserve_predictions_but_cannot_start_training(self):
        source = TinyPolicy(action_rank=2, targets=("q", "action_encoder", "head"),
                            allow_legacy_action_io=True)
        change_adapter_weights(source)
        source.save_checkpoint(self.path)
        target = TinyPolicy(action_rank=2)
        target.load_checkpoint(self.path)
        target.eval()
        self.assert_weights_equal(source, target)
        self.assertTrue(hasattr(target.action_expert.action_encoder, "lora_A"))
        self.assertTrue(hasattr(target.action_expert.head, "lora_A"))
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            apply_expert_trainability(target.action_expert)
        for training_target in (TinyPolicy(action_rank=2), target):
            self.assert_rejected_without_mutation(
                training_target, "action_encoder/head", lora_config_policy="match")
        self.assert_rejected_without_mutation(
            target, "action_encoder/head", optimizer=torch.optim.AdamW(target.parameters()))

    def test_dense_warmstart_remaps_base_and_resets_previous_adapter_delta(self):
        source = TinyPolicy()
        source.save_checkpoint(self.path)
        target = TinyPolicy(action_rank=2, video_rank=2)
        change_adapter_weights(target)
        target.load_checkpoint(self.path, lora_config_policy="match")
        target.eval()
        torch.testing.assert_close(source.predict(self.input), target.predict(self.input), rtol=0, atol=0)
        for branch in ("video", "action"):
            original = getattr(source, branch + "_expert")
            loaded = getattr(target, branch + "_expert")
            torch.testing.assert_close(original.q.weight, loaded.q.base_layer.weight, rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(loaded.q.lora_B["default"].weight).item(), 0)

    def test_stage3_full_state_export_keeps_metadata_and_all_base_parameters(self):
        source = TinyPolicy(action_rank=2, video_rank=2)
        change_adapter_weights(source)
        payload = build_weights_checkpoint_payload(source, source.state_dict(), step=23)
        self.assertEqual(payload["lora"], get_lora_checkpoint_metadata(source)["lora"])
        self.assertFalse(any(key.startswith("vae") for key in payload["mot"]))
        torch.save(payload, self.path)
        target = TinyPolicy()
        target.load_checkpoint(self.path)
        target.eval()
        self.assert_weights_equal(source, target)

    def test_match_policy_rejects_changed_lora_configuration(self):
        source = TinyPolicy(action_rank=2)
        source.save_checkpoint(self.path)
        self.assert_rejected_without_mutation(TinyPolicy(action_rank=1), "differs", lora_config_policy="match")

    def test_existing_optimizer_rejects_adapter_topology_change(self):
        source = TinyPolicy(action_rank=2)
        source.save_checkpoint(self.path)
        target = TinyPolicy()
        optimizer = torch.optim.AdamW(target.parameters())
        self.assert_rejected_without_mutation(target, "existing optimizer", optimizer=optimizer)

    def test_partitioned_model_cannot_be_reconfigured_or_loaded_as_dense_weights(self):
        source = TinyPolicy(action_rank=2)
        source.save_checkpoint(self.path)
        target = TinyPolicy()
        next(target.parameters()).ds_id = 0
        self.assert_rejected_without_mutation(target, "before DeepSpeed")

    def test_missing_metadata_and_incomplete_effective_settings_are_rejected(self):
        source = TinyPolicy(action_rank=2)
        def remove_metadata(payload):
            del payload["lora"]
            del payload["checkpoint_format_version"]
        self.save_modified_payload(source, remove_metadata)
        self.assert_rejected_without_mutation(TinyPolicy(), "no configuration metadata")
        self.save_modified_payload(source, lambda payload: payload["lora"]["action"].pop("lora_alpha"))
        self.assert_rejected_without_mutation(TinyPolicy(), "all effective settings")

    def test_corrupt_keys_and_shapes_are_rejected_before_adapter_injection(self):
        source = TinyPolicy(action_rank=2)
        corruptions = (
            (lambda payload: payload["mot"].pop("mixtures.action.q.base_layer.weight"), "missing="),
            (lambda payload: payload["mot"].update({"unknown.weight": torch.ones(1)}), "unexpected="),
            (lambda payload: payload["mot"].update({"mixtures.action.q.lora_A.default.weight": torch.ones(9, 4)}), "shape mismatch"),
            (lambda payload: payload["proprio_encoder"].update({"weight": torch.ones(8, 8)}), "shape mismatch"),
        )
        for mutate, pattern in corruptions:
            with self.subTest(pattern=pattern):
                self.save_modified_payload(source, mutate)
                self.assert_rejected_without_mutation(TinyPolicy(), pattern)

    def test_corrupt_dense_checkpoint_cannot_silently_skip_base_weights(self):
        source = TinyPolicy()
        self.save_modified_payload(source, lambda payload: payload["mot"].pop("mixtures.action.q.weight"))
        self.assert_rejected_without_mutation(TinyPolicy(action_rank=2), "missing=")

    def test_full_lora_checkpoint_requires_matching_proprio_presence(self):
        source = TinyPolicy(action_rank=2)
        self.save_modified_payload(source, lambda payload: payload.pop("proprio_encoder"))
        self.assert_rejected_without_mutation(TinyPolicy(), "proprio_encoder configuration")
        source.save_checkpoint(self.path)
        self.assert_rejected_without_mutation(TinyPolicy(proprio=False), "proprio_encoder configuration")

    def test_legacy_dit_only_loads_video_and_keeps_action_and_proprio(self):
        source = TinyPolicy()
        torch.save({"dit": source.video_expert.state_dict()}, self.path)
        target = TinyPolicy(action_rank=2, video_rank=2)
        change_adapter_weights(target)
        action_before = deepcopy(target.action_expert.state_dict())
        proprio_before = deepcopy(target.proprio_encoder.state_dict())
        target.load_checkpoint(self.path)
        target.eval()
        torch.testing.assert_close(source.video_expert(self.input), target.video_expert(self.input), rtol=0, atol=0)
        for key, value in target.action_expert.state_dict().items():
            torch.testing.assert_close(value, action_before[key], rtol=0, atol=0)
        for key, value in target.proprio_encoder.state_dict().items():
            torch.testing.assert_close(value, proprio_before[key], rtol=0, atol=0)

    def test_dense_format_and_optional_proprio_remain_backward_compatible(self):
        source = TinyPolicy(proprio=False)
        source.save_checkpoint(self.path, step=9)
        payload = torch.load(self.path, weights_only=True)
        self.assertEqual(set(payload), {"mot", "step", "torch_dtype"})
        target = TinyPolicy(proprio=False)
        target.load_checkpoint(self.path)
        self.assert_weights_equal(source, target)


if __name__ == "__main__":
    unittest.main()
