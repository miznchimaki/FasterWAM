"""Exercise native DiT LoRA training and SparseMoT's direct layer access on CPU."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fasterwam.models.wan22.sparse_action_dit import SparseActionDiT  # noqa: E402
from fasterwam.models.wan22.sparse_mot import SparseMoT  # noqa: E402
from fasterwam.models.wan22.wan_video_dit import WanVideoDiT  # noqa: E402
from fasterwam.utils.lora import (  # noqa: E402
    apply_expert_trainability,
    configure_expert_lora,
    get_expert_lora_config,
    normalize_lora_config,
    remove_expert_lora,
    reset_expert_lora_parameters,
    resolve_lora_target_modules,
    split_lora_config,
)


def tiny_mot(dtype=torch.float32, checkpoint=True):
    video = WanVideoDiT(
        hidden_dim=16, in_dim=4, ffn_dim=32, out_dim=4, text_dim=12,
        freq_dim=8, eps=1e-6, patch_size=(1, 1, 1), num_heads=2,
        attn_head_dim=8, num_layers=3, has_image_input=False,
        seperated_timestep=True, video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=checkpoint,
    )
    action = SparseActionDiT(
        hidden_dim=8, action_dim=3, ffn_dim=16, text_dim=12, freq_dim=8,
        eps=1e-6, num_heads=2, attn_head_dim=8, num_layers=3,
        noncondition_num_heads=1, condition_layers=[0, 2],
        use_gradient_checkpointing=checkpoint,
    )
    return SparseMoT(
        {"video": video, "action": action}, condition_layers=[0, 2],
        mot_checkpoint_mixed_attn=checkpoint,
        video_kv_fusion="interval_weighted_sum",
    ).to(dtype=dtype)


def inputs_for(dtype):
    return (
        torch.randn(2, 4, 2, 2, 2, dtype=dtype),
        torch.randn(2, 4, 3, dtype=dtype),
        torch.tensor([500, 750], dtype=dtype),
        torch.randn(2, 2, 12, dtype=dtype),
    )


def forward_mot(mot, inputs, *, cached=False):
    video, action = mot.mixtures["video"], mot.mixtures["action"]
    latents, actions, timesteps, text = inputs
    states = {
        "video": video.pre_dit(latents, timesteps, text, fuse_vae_embedding_in_latents=True),
        "action": action.pre_dit(actions, timesteps, text),
    }
    nv, na = (states[name]["tokens"].shape[1] for name in ("video", "action"))
    mask = torch.ones(nv + na, nv + na, dtype=torch.bool)
    mask[:nv, nv:] = False
    mask[:nv, :nv] = video.build_video_to_video_mask(
        nv, states["video"]["meta"]["tokens_per_frame"], latents.device,
    )
    contexts = {
        name: {"context": state["context"], "mask": state["context_mask"]}
        for name, state in states.items()
    }
    if cached:
        cache = mot.prefill_video_cache(
            states["video"]["tokens"], states["video"]["freqs"],
            states["video"]["t_mod"], contexts["video"], mask[:nv, :nv],
        )
        tokens = mot.forward_action_with_video_cache(
            states["action"]["tokens"], states["action"]["freqs"],
            states["action"]["t_mod"], contexts["action"], cache, mask, nv,
        )
        return action.post_dit(tokens, states["action"])
    tokens = mot(
        embeds_all={name: state["tokens"] for name, state in states.items()},
        attention_mask=mask,
        freqs_all={name: state["freqs"] for name, state in states.items()},
        context_all=contexts,
        t_mod_all={name: state["t_mod"] for name, state in states.items()},
    )
    return video.post_dit(tokens["video"], states["video"]), action.post_dit(tokens["action"], states["action"])


class LoRAConfigurationTests(unittest.TestCase):
    def test_split_is_nonmutating_and_absent_block_preserves_legacy_behavior(self):
        original = {"hidden_dim": 8, "lora": {"enabled": True, "target_modules": ["head"]}}
        before = deepcopy(original)
        clean, config = split_lora_config(original)
        self.assertEqual(original, before)
        self.assertEqual(clean, {"hidden_dim": 8})
        config["target_modules"].append("action_encoder")
        self.assertEqual(original, before)
        self.assertFalse(split_lora_config({"hidden_dim": 8})[1]["enabled"])
        self.assertEqual(get_expert_lora_config(nn.Linear(2, 2)), {"enabled": False})

    def test_bad_configuration_fails_early(self):
        bad_configs = (
            True, {"enabled": "true"}, {"r": 0}, {"r": True}, {"r": 2.5},
            {"lora_alpha": float("nan")}, {"lora_alpha": 0},
            {"lora_dropout": 1.0}, {"lora_dropout": -0.1},
            {"target_modules": []}, {"target_modules": "all-linear"},
            {"target_modules": [""]}, {"target_modules": [" head"]},
            {"target_modules": ["head", "head"]}, {"modules_to_save": ["head"]},
        )
        for config in bad_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                normalize_lora_config(config)

    def test_invalid_target_preserves_existing_adapter_and_weights(self):
        expert = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        configure_expert_lora(expert, {"enabled": True, "target_modules": ["0"]})
        adapter = expert[0]
        for targets in (["missing"], ["0", "typo"], ["1"]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                configure_expert_lora(expert, {"enabled": True, "target_modules": targets})
            self.assertIs(expert[0], adapter)

    def test_action_io_targets_are_rejected_after_suffix_resolution(self):
        expert = nn.Module()
        expert.blocks = nn.Sequential(nn.Linear(3, 3))
        expert.action_encoder = nn.Sequential(nn.Linear(3, 3))
        expert.head = nn.Sequential(nn.Linear(3, 3))
        configure_expert_lora(expert, {"enabled": True, "target_modules": ["blocks.0"]})
        adapter = expert.blocks[0]
        original = deepcopy(expert.state_dict())
        for target in ("action_encoder", "head", "0", "action_encoder.0", "head.0"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "action_encoder/head"):
                configure_expert_lora(expert, {"enabled": True, "target_modules": ["blocks.0", target]})
            self.assertIs(expert.blocks[0], adapter)
            for name, value in expert.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)

        # A custom alias cannot bypass protection of the same physical layer.
        aliased = nn.Module()
        aliased.shared = nn.Linear(3, 3)
        aliased.head = aliased.shared
        with self.assertRaisesRegex(ValueError, "action_encoder/head"):
            configure_expert_lora(aliased, {"enabled": True, "target_modules": ["shared"]}, branch="action")
        # The restriction is action-specific; a video head remains selectable.
        configure_expert_lora(aliased, {"enabled": True, "target_modules": ["shared"]}, branch="video")
        self.assertTrue(hasattr(aliased.shared, "lora_A"))

    def test_reconfigure_remove_reset_and_same_config_are_safe_before_prepare(self):
        expert = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 3))
        base0, base1 = expert[0], expert[1]
        original = deepcopy(expert.state_dict())
        config = {"enabled": True, "r": 4, "target_modules": ["0"]}
        self.assertIs(configure_expert_lora(expert, config), expert)
        adapter = expert[0]
        with torch.no_grad():
            adapter.lora_B["default"].weight.fill_(0.5)
        self.assertIs(configure_expert_lora(expert, config)[0], adapter)
        self.assertTrue(torch.all(adapter.lora_B["default"].weight == 0.5))
        resolved = resolve_lora_target_modules(expert, config)
        self.assertEqual(resolved, ["0"])
        metadata = get_expert_lora_config(expert)
        metadata["target_modules"].append("1")
        self.assertEqual(get_expert_lora_config(expert)["target_modules"], ["0"])
        reset_expert_lora_parameters(expert)
        self.assertEqual(torch.count_nonzero(adapter.lora_B["default"].weight), 0)
        configure_expert_lora(expert, {"enabled": True, "r": 2, "target_modules": ["1"]})
        self.assertIs(expert[0], base0)
        self.assertIs(expert[1].get_base_layer(), base1)
        self.assertEqual(expert[1].lora_A["default"].weight.shape, (2, 4))
        remove_expert_lora(expert)
        self.assertIs(expert[1], base1)
        self.assertFalse(hasattr(expert, "peft_config"))
        self.assertTrue(all(parameter.requires_grad for parameter in expert.parameters()))
        for name, value in expert.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_zero_partitioned_expert_rejects_adapter_structure_changes(self):
        expert = nn.Sequential(nn.Linear(3, 4))
        config = {"enabled": True, "target_modules": ["0"]}
        configure_expert_lora(expert, config)
        adapter = expert[0]
        next(expert.parameters()).ds_id = 1
        for operation in (
            lambda: configure_expert_lora(expert, {"enabled": False}),
            lambda: remove_expert_lora(expert),
            lambda: reset_expert_lora_parameters(expert),
        ):
            with self.assertRaisesRegex(RuntimeError, "before Accelerator.prepare"):
                operation()
            self.assertIs(expert[0], adapter)

    def test_injection_preserves_eval_mode_and_disabling_restores_dense_expert(self):
        expert = nn.Sequential(nn.Linear(3, 4)).eval()
        base = expert[0]
        configure_expert_lora(expert, {"enabled": True, "target_modules": ["0"], "lora_dropout": 0.5})
        self.assertFalse(expert[0].training)
        self.assertFalse(expert[0].lora_dropout["default"].training)
        configure_expert_lora(expert, {"enabled": False})
        self.assertIs(expert[0], base)
        self.assertFalse(expert[0].training)
        self.assertTrue(all(parameter.requires_grad for parameter in expert.parameters()))


class NativeSparseMoTLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_identity_initialization_backprop_and_cache_with_native_experts(self):
        for dtype in (torch.float32, torch.bfloat16):
            for video_enabled in (False, True):
                with self.subTest(dtype=dtype, video_enabled=video_enabled):
                    torch.manual_seed(53)
                    mot = tiny_mot(dtype)
                    video, action = mot.mixtures["video"], mot.mixtures["action"]
                    inputs = inputs_for(dtype)
                    mot.eval()
                    with torch.no_grad():
                        expected = forward_mot(mot, inputs)
                    original_action = {id(p): p.detach().clone() for p in action.parameters()}
                    blocks = action.blocks
                    self.assertIs(configure_expert_lora(video, {"enabled": video_enabled}), video)
                    self.assertIs(configure_expert_lora(action, {"enabled": True}), action)
                    self.assertIs(action.blocks, blocks)
                    self.assertIsInstance(video, WanVideoDiT)
                    self.assertIsInstance(action, SparseActionDiT)
                    self.assertIsInstance(action.action_encoder, nn.Linear)
                    self.assertIsInstance(action.head, nn.Linear)
                    self.assertFalse(hasattr(action.action_encoder, "lora_A"))
                    self.assertFalse(hasattr(action.head, "lora_A"))
                    self.assertTrue(hasattr(action.blocks[1].self_attn.q, "lora_A"))
                    self.assertFalse(hasattr(action.blocks[1], "cross_attn"))
                    self.assertTrue(hasattr(action.blocks[0].cross_attn.q, "lora_A"))

                    # Simulate Trainer's global trainability reset.
                    mot.requires_grad_(False)
                    mot.requires_grad_(True)
                    apply_expert_trainability(video)
                    apply_expert_trainability(action)
                    for name, parameter in action.named_parameters():
                        is_adapter = ".lora_A.default." in name or ".lora_B.default." in name
                        is_io = name.startswith(("action_encoder.", "head."))
                        self.assertEqual(parameter.requires_grad, is_adapter or is_io, name)
                        self.assertEqual(parameter.dtype, dtype, name)
                    with torch.no_grad():
                        actual = forward_mot(mot, inputs)
                        for out, reference in zip(actual, expected):
                            self.assertEqual(out.dtype, dtype)
                            torch.testing.assert_close(out, reference, rtol=0, atol=0)

                    mot.train()
                    optimizer = torch.optim.SGD((p for p in mot.parameters() if p.requires_grad), lr=0.02)
                    video_output, action_output = forward_mot(mot, inputs)
                    (video_output.float().square().mean() + action_output.float().square().mean()).backward()
                    self.assertGreater(action.head.weight.grad.float().abs().sum().item(), 0)
                    self.assertGreater(action.action_encoder.weight.grad.float().abs().sum().item(), 0)
                    for name, parameter in action.named_parameters():
                        if parameter.requires_grad:
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        else:
                            self.assertIsNone(parameter.grad, name)
                    if video_enabled:
                        for name, parameter in video.named_parameters():
                            if not parameter.requires_grad:
                                self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(video.patch_embedding.weight.grad)
                    self.assertTrue(all(p.requires_grad for p in mot.video_kv_fusion_logits))
                    optimizer.step()
                    for parameter in action.parameters():
                        if id(parameter) in original_action and not parameter.requires_grad:
                            torch.testing.assert_close(parameter, original_action[id(parameter)], rtol=0, atol=0)
                    for module in (action.action_encoder, action.head):
                        self.assertFalse(torch.equal(module.weight, original_action[id(module.weight)]))
                    mot.eval()
                    with torch.no_grad():
                        _, uncached = forward_mot(mot, inputs)
                        cached = forward_mot(mot, inputs, cached=True)
                    torch.testing.assert_close(cached, uncached)
                    self.assertFalse(torch.equal(uncached, expected[1]))


if __name__ == "__main__":
    unittest.main()
