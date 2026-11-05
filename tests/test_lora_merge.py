"""Offline export against real PEFT 0.14 layers, without constructing a DiT."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora.layer import Linear as LoRALinear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fasterwam.utils.lora_checkpoint import load_fastwam_checkpoint  # noqa: E402
from fasterwam.utils.lora_merge import merge_lora_checkpoint, merge_lora_checkpoint_payload  # noqa: E402


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.ModuleDict({"q": nn.Linear(4, 4)}) for _ in range(2)])
        self.head = nn.Linear(4, 4)

    def forward(self, value):
        value = self.action_encoder(value)
        for block in self.blocks:
            value = torch.tanh(block["q"](value))
        return self.head(value)


class TinyPolicy(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.mot = nn.Module()
        self.mot.mixtures = nn.ModuleDict({"video": TinyExpert(), "action": TinyExpert()})
        self.mot.video_kv_fusion_logits = nn.ParameterList([nn.Parameter(torch.randn(2))])
        self.video_expert = self.mot.mixtures["video"]
        self.action_expert = self.mot.mixtures["action"]
        self.proprio_encoder = nn.Linear(3, 4)
        self.to(dtype)

    def forward(self, value):
        return self.action_expert(self.video_expert(value))


def make_payload(*, dtype=torch.float32, legacy_io=False, video=True):
    model = TinyPolicy(dtype)
    metadata = {"version": 1, "implementation": "peft", "peft_version": "0.14.0", "adapter_name": "default"}
    for branch in ("video", "action"):
        if branch == "video" and not video:
            metadata[branch] = {"enabled": False}
            continue
        targets = ["q"] + (["action_encoder", "head"] if branch == "action" and legacy_io else [])
        config = {"enabled": True, "r": 2, "lora_alpha": 3, "lora_dropout": 0.2, "target_modules": targets}
        inject_adapter_in_model(LoraConfig(r=2, lora_alpha=3, lora_dropout=0.2, target_modules=targets),
                                model.mot.mixtures[branch])
        metadata[branch] = config
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, LoRALinear):
                layer.lora_A["default"].weight.uniform_(-0.4, 0.4)
                layer.lora_B["default"].weight.uniform_(-0.4, 0.4)
    model.eval()
    payload = {"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict(),
               "step": 17, "torch_dtype": str(dtype), "checkpoint_format_version": 2, "lora": metadata,
               "optimizer": {"state": {0: {"exp_avg": torch.ones(2)}}},
               "scheduler": {"last_epoch": 17}, "trainer_state": {"global_step": 17}}
    return model, payload


class LoRAMergeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.input = Path(self.scratch.name) / "lora.pt"
        self.output = Path(self.scratch.name) / "dense.pt"

    def test_fp32_real_peft_merge_legacy_io_and_both_dense_loaders(self):
        source, payload = make_payload(legacy_io=True)
        before = deepcopy(payload)
        dense = merge_lora_checkpoint_payload(payload)
        self.assertEqual(set(dense), {"mot", "proprio_encoder", "step", "torch_dtype"})
        self.assertFalse(any("lora_" in key or "base_layer" in key for key in dense["mot"]))
        for path, layer in source.mot.named_modules():
            if isinstance(layer, LoRALinear):
                reference = deepcopy(layer)
                reference.merge(safe_merge=True)
                torch.testing.assert_close(dense["mot"][path + ".weight"], reference.base_layer.weight, rtol=0, atol=0)
        legacy = TinyPolicy().eval()
        incompatible = legacy.mot.load_state_dict(dense["mot"], strict=False)
        self.assertFalse(incompatible.missing_keys or incompatible.unexpected_keys)
        torch.save(dense, self.output)
        current = TinyPolicy().eval()
        load_fastwam_checkpoint(current, self.output)
        value = torch.randn(3, 4)
        torch.testing.assert_close(source(value), current(value), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(legacy(value), current(value), rtol=0, atol=0)
        torch.testing.assert_close(current.proprio_encoder.weight, source.proprio_encoder.weight, rtol=0, atol=0)
        for component in ("mot", "proprio_encoder"):
            for key, tensor in payload[component].items():
                torch.testing.assert_close(tensor, before[component][key], rtol=0, atol=0)

    def test_modern_dense_action_io_and_untouched_fusion_are_preserved(self):
        _, payload = make_payload(video=False)
        dense = merge_lora_checkpoint_payload(payload)
        for key in ("mixtures.action.action_encoder.weight", "mixtures.action.head.weight",
                    "mixtures.video.blocks.0.q.weight", "video_kv_fusion_logits.0"):
            self.assertIs(dense["mot"][key], payload["mot"][key])
        self.assertIs(dense["proprio_encoder"], payload["proprio_encoder"])
        self.assertEqual(dense["step"], 17)

    def test_native_sparse_mot_video_action_and_cache_outputs_with_modern_and_legacy_io(self):
        from test_lora import tiny_mot, inputs_for, forward_mot
        from fasterwam.utils.lora import configure_expert_lora, DEFAULT_LORA_TARGET_MODULES
        from fasterwam.utils.lora_checkpoint import get_lora_checkpoint_metadata

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        for legacy_io in (False, True):
            with self.subTest(legacy_io=legacy_io):
                source = tiny_mot(checkpoint=False)
                for branch in ("video", "action"):
                    targets = list(DEFAULT_LORA_TARGET_MODULES)
                    if branch == "action" and legacy_io:
                        targets += ["action_encoder", "head"]
                    configure_expert_lora(source.mixtures[branch],
                                         {"enabled": True, "r": 2, "lora_alpha": 3, "target_modules": targets},
                                         branch=branch, allow_legacy_action_io=legacy_io)
                with torch.no_grad():
                    for layer in source.modules():
                        if isinstance(layer, LoRALinear):
                            layer.lora_A["default"].weight.uniform_(-0.06, 0.06)
                            layer.lora_B["default"].weight.uniform_(-0.06, 0.06)
                    for name in ("action_encoder", "head"):
                        layer = getattr(source.mixtures["action"], name)
                        base = layer.get_base_layer() if legacy_io else layer
                        base.weight.add_(0.05)  # Include learned dense I/O weights, independent of adapter deltas.
                source.eval()
                payload = {"mot": source.state_dict(), "step": 9, "torch_dtype": "torch.float32"}
                payload.update(get_lora_checkpoint_metadata(SimpleNamespace(
                    video_expert=source.mixtures["video"], action_expert=source.mixtures["action"])))
                torch.save(payload, self.input)
                output = self.output.with_name(f"dense_{legacy_io}.pt")
                merge_lora_checkpoint(self.input, output)
                dense = torch.load(output, weights_only=True)
                target = tiny_mot(checkpoint=False).eval()
                target.load_state_dict(dense["mot"], strict=True)
                inputs = inputs_for(torch.float32)
                with torch.no_grad():
                    reference, actual = forward_mot(source, inputs), forward_mot(target, inputs)
                    for expected, result in zip(reference, actual):
                        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(forward_mot(target, inputs, cached=True),
                                               forward_mot(source, inputs, cached=True), rtol=2e-5, atol=2e-6)
                for name in ("action_encoder", "head"):
                    source_layer = getattr(source.mixtures["action"], name)
                    merged_weight = getattr(target.mixtures["action"], name).weight
                    if legacy_io:
                        self.assertFalse(torch.equal(merged_weight, source_layer.base_layer.weight))
                    else:
                        torch.testing.assert_close(merged_weight, source_layer.weight, rtol=0, atol=0)

    def test_fp16_bf16_fp32_accumulation_and_peft_finite_precision_tolerance(self):
        for dtype, tolerance in ((torch.float16, 0.002), (torch.bfloat16, 0.016)):
            with self.subTest(dtype=dtype):
                source, payload = make_payload(dtype=dtype, legacy_io=True)
                dense = merge_lora_checkpoint_payload(payload)
                for path, layer in source.mot.named_modules():
                    if isinstance(layer, LoRALinear):
                        fp32_reference = deepcopy(layer).float()
                        fp32_reference.merge(safe_merge=True)
                        actual = dense["mot"][path + ".weight"]
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, fp32_reference.base_layer.weight.to(dtype), rtol=0, atol=0)
                        native_reference = deepcopy(layer)
                        native_reference.merge(safe_merge=True)
                        torch.testing.assert_close(actual, native_reference.base_layer.weight, rtol=tolerance, atol=tolerance)
                target = TinyPolicy(dtype).eval()
                target.mot.load_state_dict(dense["mot"])
                value = torch.randn(3, 4).to(dtype)
                torch.testing.assert_close(source(value), target(value), rtol=2 * tolerance, atol=2 * tolerance)

    def test_bf16_single_rounding_is_intentional_and_differs_from_native_peft(self):
        source, payload = make_payload(dtype=torch.bfloat16, video=False)
        layer = source.action_expert.blocks[0]["q"]
        payload["lora"]["action"]["lora_alpha"] = 2
        for candidate in source.action_expert.modules():
            if isinstance(candidate, LoRALinear):
                candidate.scaling["default"] = 1
        with torch.no_grad():
            layer.base_layer.weight.fill_(1)
            layer.lora_A["default"].weight[0].fill_(1)
            layer.lora_A["default"].weight[1].fill_(0.0625)
            layer.lora_B["default"].weight[:, 0].fill_(0.00390625)
            layer.lora_B["default"].weight[:, 1].fill_(0.000030517578125)
        dense = merge_lora_checkpoint_payload(payload)
        layer.merge(safe_merge=True)
        self.assertEqual(layer.base_layer.weight[0, 0].item(), 1)
        self.assertEqual(dense["mot"]["mixtures.action.blocks.0.q.weight"][0, 0].item(), 1.0078125)

    def test_invalid_metadata_and_adapter_layout_are_rejected(self):
        _, original = make_payload()
        key = "mixtures.action.blocks.0.q."
        corruptions = (
            lambda p: p.pop("lora"),
            lambda p: p["lora"]["action"].pop("lora_alpha"),
            lambda p: p["lora"]["action"].update({"target_modules": ["does_not_exist"]}),
            lambda p: p["lora"]["action"].update({"enabled": False}),
            lambda p: p["mot"].pop(key + "base_layer.weight"),
            lambda p: p["mot"].pop(key + "lora_B.default.weight"),
            lambda p: p["mot"].update({key + "lora_A.other.weight": torch.ones(2, 4)}),
            lambda p: p["mot"].update({key + "lora_B.default.bias": torch.ones(4)}),
            lambda p: p["mot"].update({key + "lora_magnitude_vector.default.weight": torch.ones(4)}),
            lambda p: p["mot"].update({key + "modules_to_save.default.weight": torch.ones(4, 4)}),
            lambda p: p["mot"].update({key + "weight": torch.ones(4, 4)}),
            lambda p: p["mot"].update({key + "lora_A.default.weight": torch.ones(3, 4)}),
            lambda p: p["mot"].update({key + "base_layer.bias": torch.ones(3)}),
            lambda p: p["mot"].update({key + "base_layer.weight": torch.ones(4, 4, 1)}),
            lambda p: p["mot"].update({"mixtures.action.extra.q.weight": torch.ones(4, 4)}),
            lambda p: p["mot"].update({"module.zero_shard": torch.ones(4)}),
            lambda p: p["proprio_encoder"].pop("bias"),
            lambda p: p.update({"mot": {k: v for k, v in p["mot"].items() if not k.startswith("mixtures.video.")}}),
        )
        for index, change in enumerate(corruptions):
            with self.subTest(index=index):
                payload = deepcopy(original)
                change(payload)
                with self.assertRaises((ValueError, TypeError)):
                    merge_lora_checkpoint_payload(payload)

    def test_nonfinite_overflow_and_partitioned_tensors_are_rejected(self):
        _, original = make_payload(dtype=torch.float16)
        key = "mixtures.action.blocks.0.q."
        for kind in ("nan", "input_inf", "cast_overflow", "fp32_overflow", "empty", "sparse", "wrong_dtype"):
            with self.subTest(kind=kind):
                payload = deepcopy(original)
                if kind in {"nan", "input_inf"}:
                    payload["mot"][key + "lora_A.default.weight"].fill_(float("nan" if kind == "nan" else "inf"))
                elif kind == "cast_overflow":
                    payload["mot"][key + "lora_A.default.weight"].fill_(60000)
                    payload["mot"][key + "lora_B.default.weight"].fill_(60000)
                elif kind == "fp32_overflow":
                    payload["mot"][key + "lora_A.default.weight"] = torch.full((2, 4), 3e38)
                    payload["mot"][key + "lora_B.default.weight"] = torch.full((4, 2), 3e38)
                elif kind == "empty":
                    payload["mot"][key + "base_layer.weight"] = torch.empty(0)
                elif kind == "sparse":
                    payload["mot"][key + "base_layer.weight"] = payload["mot"][key + "base_layer.weight"].to_sparse()
                else:
                    payload["mot"][key + "lora_A.default.weight"] = torch.ones(2, 4, dtype=torch.int64)
                with self.assertRaisesRegex(ValueError, "non-finite|empty or partitioned|strided CPU|Only FP32"):
                    merge_lora_checkpoint_payload(payload)

    def test_file_export_preserves_input_and_refuses_existing_or_same_output(self):
        _, payload = make_payload()
        torch.save(payload, self.input)
        digest = hashlib.sha256(self.input.read_bytes()).digest()
        self.assertEqual(merge_lora_checkpoint(self.input, self.output), self.output)
        self.assertEqual(hashlib.sha256(self.input.read_bytes()).digest(), digest)
        self.assertNotIn("lora", torch.load(self.output, weights_only=True))
        with self.assertRaisesRegex(ValueError, "different files"):
            merge_lora_checkpoint(self.input, self.input)
        with self.assertRaises(FileExistsError):
            merge_lora_checkpoint(self.input, self.output)
        with self.assertRaisesRegex(ValueError, "directory"):
            merge_lora_checkpoint(self.input.parent, self.input.parent / "other.pt")
        alias = self.input.parent / "source_alias.pt"
        os.link(self.input, alias)
        with self.assertRaises(FileExistsError):
            merge_lora_checkpoint(self.input, alias)

    def test_failed_save_and_concurrent_output_do_not_publish_partial_file(self):
        _, payload = make_payload()
        torch.save(payload, self.input)
        digest = self.input.read_bytes()
        def failed_save(_payload, stream):
            stream.write(b"partial checkpoint")
            raise OSError("simulated disk failure")
        with patch("fasterwam.utils.lora_merge.torch.save", side_effect=failed_save):
            with self.assertRaisesRegex(OSError, "disk failure"):
                merge_lora_checkpoint(self.input, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(set(self.input.parent.iterdir()), {self.input})
        original_link = os.link
        def concurrent_link(source, target):
            Path(target).write_bytes(b"other writer")
            original_link(source, target)
        with patch("fasterwam.utils.lora_merge.os.link", side_effect=concurrent_link):
            with self.assertRaises(FileExistsError):
                merge_lora_checkpoint(self.input, self.output)
        self.assertEqual(self.output.read_bytes(), b"other writer")
        self.assertEqual(self.input.read_bytes(), digest)
        self.assertEqual(set(self.input.parent.iterdir()), {self.input, self.output})

    def test_cli_requires_no_peft_or_model_import_and_supports_old_torch_serialization(self):
        _, payload = make_payload(legacy_io=True)
        torch.save(payload, self.input, _use_new_zipfile_serialization=False)
        # The subprocess import hook makes any accidental heavyweight dependency
        # an error, even though the test environment itself has PEFT installed.
        program = """
import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'peft', 'transformers', 'omegaconf', 'datasets'} or name.startswith('fasterwam.models'):
        raise AssertionError('Exporter imported a forbidden dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--input', sys.argv[2], '--output', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", program, str(ROOT / "scripts/merge_lora_checkpoint.py"),
                                 str(self.input), str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved dense inference checkpoint", result.stdout)
        self.assertNotIn("checkpoint_format_version", torch.load(self.output, weights_only=True))


if __name__ == "__main__":
    unittest.main()
