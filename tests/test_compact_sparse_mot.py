"""Native compact SparseMoT attention, gradients and video-cache regressions."""

from copy import deepcopy
import re
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fasterwam.models.wan22.action_layout import resolve_action_layout
from fasterwam.models.wan22.sparse_action_dit import SparseActionDiT
from fasterwam.models.wan22.sparse_mot import SparseMoT
from fasterwam.models.wan22.wan_video_dit import WanVideoDiT
from test_lora import forward_mot, inputs_for


def dense_mot(*, fusion, checkpoint=False, mapping=None, conditions=(0, 3)):
    video = WanVideoDiT(
        hidden_dim=16, in_dim=4, ffn_dim=32, out_dim=4, text_dim=12,
        freq_dim=8, eps=1e-6, patch_size=(1, 1, 1), num_heads=2,
        attn_head_dim=8, num_layers=6, has_image_input=False,
        seperated_timestep=True, video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=checkpoint,
    )
    action = SparseActionDiT(
        hidden_dim=8, action_dim=3, ffn_dim=16, text_dim=12, freq_dim=8,
        eps=1e-6, num_heads=2, attn_head_dim=8, num_layers=6,
        noncondition_num_heads=1, condition_layers=conditions,
        use_gradient_checkpointing=checkpoint,
    )
    return SparseMoT(
        {"video": video, "action": action}, condition_layers=conditions,
        video_kv_fusion=fusion, mot_checkpoint_mixed_attn=checkpoint,
        action_layer_indices=mapping,
    )


def compact_from(dense, indices=(0, 1, 3, 4)):
    video = deepcopy(dense.mixtures["video"])
    action = deepcopy(dense.mixtures["action"])
    action.blocks = nn.ModuleList([action.blocks[index] for index in indices])
    action.condition_layers = tuple(indices.index(index) for index in dense.condition_layers)
    action.condition_layer_set = frozenset(action.condition_layers)
    compact = SparseMoT(
        {"video": video, "action": action}, condition_layers=dense.condition_layers,
        mot_checkpoint_mixed_attn=dense.mot_checkpoint_mixed_attn,
        video_kv_fusion=dense.video_kv_fusion, action_layer_indices=indices,
    )
    compact.video_kv_fusion_logits.load_state_dict(dense.video_kv_fusion_logits.state_dict())
    return compact


def zero_refinement_updates(block):
    """Make a dense non-condition block identity without editing MoT's loop."""
    with torch.no_grad():
        for layer in (block.self_attn.o, block.ffn[-1]):
            layer.weight.zero_()
            layer.bias.zero_()


def dense_parameter_name(compact_name, indices):
    return re.sub(
        r"^(mixtures\.action\.blocks\.)(\d+)(\.)",
        lambda match: match[1] + str(indices[int(match[2])]) + match[3],
        compact_name,
    )


class CompactSparseMoTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_default_identity_mapping_preserves_initialization_and_execution(self):
        for fusion in (None, "interval_weighted_sum"):
            with self.subTest(fusion=fusion):
                torch.manual_seed(11)
                implicit = dense_mot(fusion=fusion)
                rng = torch.get_rng_state()
                torch.manual_seed(11)
                explicit = dense_mot(fusion=fusion, mapping=list(range(6)))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(set(implicit.state_dict()), set(explicit.state_dict()))
                for name, value in implicit.state_dict().items():
                    torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
                inputs = inputs_for(torch.float32)
                for left, right in zip(forward_mot(implicit, inputs), forward_mot(explicit, inputs)):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_compact_forward_and_gradients_match_dense_identity_refinement_reference(self):
        for fusion in (None, "interval_weighted_sum"):
            for checkpoint in (False, True):
                with self.subTest(fusion=fusion, checkpoint=checkpoint):
                    torch.manual_seed(19)
                    dense = dense_mot(fusion=fusion, checkpoint=checkpoint)
                    # Give every video K/V in a fusion interval appreciable weight.
                    for logits in dense.video_kv_fusion_logits:
                        with torch.no_grad():
                            logits.zero_()
                    compact = compact_from(dense)
                    for index in (2, 5):
                        zero_refinement_updates(dense.mixtures["action"].blocks[index])
                    inputs = inputs_for(torch.float32)
                    reference_inputs = tuple(item.detach().clone().requires_grad_() for item in inputs)
                    compact_inputs = tuple(item.detach().clone().requires_grad_() for item in inputs)
                    reference = forward_mot(dense, reference_inputs)
                    actual = forward_mot(compact, compact_inputs)
                    for wanted, got in zip(reference, actual):
                        torch.testing.assert_close(got, wanted, rtol=2e-5, atol=2e-6)
                    sum(output.square().mean() for output in reference).backward()
                    sum(output.square().mean() for output in actual).backward()
                    dense_parameters = dict(dense.named_parameters())
                    for name, parameter in compact.named_parameters():
                        wanted = dense_parameters[dense_parameter_name(name, compact.action_layer_indices)].grad
                        if wanted is None:
                            self.assertIsNone(parameter.grad, name)
                        else:
                            self.assertIsNotNone(parameter.grad, name)
                            torch.testing.assert_close(parameter.grad, wanted, rtol=3e-4, atol=3e-6, msg=name)
                    for wanted, got in zip(reference_inputs, compact_inputs):
                        torch.testing.assert_close(got.grad, wanted.grad, rtol=3e-4, atol=3e-6)
                    self.assertEqual(compact.action_num_layers, 4)
                    self.assertEqual(compact.mixtures["action"].condition_layers, (0, 2))
                    self.assertLess(sum(p.numel() for p in compact.parameters()),
                                    sum(p.numel() for p in dense.parameters()))
                    self.assertFalse(any("mixtures.action.blocks.4." in key or "mixtures.action.blocks.5." in key
                                         for key in compact.state_dict()))

    def test_cache_keeps_video_coordinates_and_matches_joint_action(self):
        for fusion in (None, "interval_weighted_sum"):
            for checkpoint in (False, True):
                with self.subTest(fusion=fusion, checkpoint=checkpoint):
                    torch.manual_seed(37)
                    mot = compact_from(dense_mot(fusion=fusion, checkpoint=checkpoint))
                    inputs = inputs_for(torch.float32)
                    with patch.object(mot, "_build_expert_attention_io", wraps=mot._build_expert_attention_io) as build:
                        joint_action = forward_mot(mot, inputs)[1]
                    video_blocks = mot.mixtures["video"].blocks
                    action_blocks = mot.mixtures["action"].blocks
                    self.assertEqual([call.kwargs["block"] for call in build.call_args_list
                                      if call.kwargs["expert"] is mot.mixtures["video"]], list(video_blocks))
                    self.assertEqual([call.kwargs["block"] for call in build.call_args_list
                                      if call.kwargs["expert"] is mot.mixtures["action"]], list(action_blocks))
                    caches = []
                    original_prefill = mot.prefill_video_cache

                    def capture_cache(*args, **kwargs):
                        result = original_prefill(*args, **kwargs)
                        caches.append(result)
                        return result

                    with patch.object(mot, "_build_expert_attention_io", wraps=mot._build_expert_attention_io) as build, \
                            patch.object(mot, "prefill_video_cache", side_effect=capture_cache) as prefill:
                        cached_action = forward_mot(mot, inputs, cached=True)
                    self.assertEqual([call.kwargs["block"] for call in build.call_args_list
                                      if call.kwargs["expert"] is mot.mixtures["video"]], list(video_blocks))
                    self.assertEqual([call.kwargs["block"] for call in build.call_args_list
                                      if call.kwargs["expert"] is mot.mixtures["action"]], list(action_blocks))
                    self.assertEqual(prefill.call_count, 1)
                    self.assertEqual(len(caches[0]), mot.num_layers)
                    if fusion is not None:
                        self.assertEqual([i for i, entry in enumerate(caches[0]) if entry is not None], [0, 3])
                    else:
                        self.assertTrue(all(entry is not None for entry in caches[0]))
                    torch.testing.assert_close(cached_action, joint_action, rtol=2e-5, atol=2e-6)
                    self.assertEqual(mot.condition_intervals, ((0,), (1, 2, 3)))
                    # No action block exists at video slot 2, but its K/V still
                    # contributes to the fused condition-layer-3 cache.
                    if fusion is not None:
                        self.assertEqual([logits.numel() for logits in mot.video_kv_fusion_logits], [1, 3])

    def test_invalid_mapping_and_local_condition_types_fail_at_construction(self):
        source = dense_mot(fusion=None)
        compact = compact_from(source)
        experts = compact.mixtures
        bad_mappings = (None, "0,1,3,4", 3, [0, 1, 3], [0, 1, 3, True],
                        [0, 1, 3, 6], [0, -1, 3, 4], [0, 1, 3, 3],
                        [0, 3, 1, 4], [0, 1, 2, 4])
        for mapping in bad_mappings:
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "action_layer_indices"):
                SparseMoT(experts, condition_layers=[0, 3], action_layer_indices=mapping)
        wrong_local_conditions = deepcopy(experts)
        wrong_local_conditions["action"].condition_layers = (0, 3)
        with self.assertRaisesRegex(ValueError, "local condition"):
            SparseMoT(wrong_local_conditions, condition_layers=[0, 3], action_layer_indices=[0, 1, 3, 4])

    def test_removed_leading_layers_preserve_forward_gradients_and_video_cache(self):
        for fusion in (None, "interval_weighted_sum"):
            for leading in (0, 1):
                for checkpoint in (False, True):
                    with self.subTest(fusion=fusion, leading=leading, checkpoint=checkpoint):
                        torch.manual_seed(53)
                        dense = dense_mot(fusion=fusion, checkpoint=checkpoint, conditions=(2, 5))
                        for logits in dense.video_kv_fusion_logits:
                            with torch.no_grad():
                                logits.zero_()
                        layout = resolve_action_layout(
                            6, [2, 5], 1, 0, leading_noncondition_layers=leading,
                        )
                        compact = compact_from(dense, layout.action_layer_indices)
                        for index in set(range(6)) - set(layout.action_layer_indices):
                            zero_refinement_updates(dense.mixtures["action"].blocks[index])
                        inputs = inputs_for(torch.float32)
                        reference_inputs = tuple(item.detach().clone().requires_grad_() for item in inputs)
                        compact_inputs = tuple(item.detach().clone().requires_grad_() for item in inputs)
                        expected = forward_mot(dense, reference_inputs)
                        actual = forward_mot(compact, compact_inputs)
                        for wanted, got in zip(expected, actual):
                            torch.testing.assert_close(got, wanted, rtol=2e-5, atol=2e-6)
                        sum(output.square().mean() for output in expected).backward()
                        sum(output.square().mean() for output in actual).backward()
                        dense_parameters = dict(dense.named_parameters())
                        for name, parameter in compact.named_parameters():
                            wanted = dense_parameters[dense_parameter_name(name, layout.action_layer_indices)].grad
                            if wanted is None:
                                self.assertIsNone(parameter.grad, name)
                            else:
                                self.assertIsNotNone(parameter.grad, name)
                                torch.testing.assert_close(parameter.grad, wanted, rtol=3e-4, atol=3e-6, msg=name)
                        for wanted, got in zip(reference_inputs, compact_inputs):
                            torch.testing.assert_close(got.grad, wanted.grad, rtol=3e-4, atol=3e-6)
                        cached = forward_mot(compact, inputs, cached=True)
                        torch.testing.assert_close(cached, actual[1], rtol=2e-5, atol=2e-6)
                        self.assertEqual(compact.mixtures["action"].condition_layers, (leading, leading + 2))
                        # Removing action prefix blocks must not remove video K/V
                        # from the first condition's fusion interval.
                        self.assertEqual(compact.condition_intervals, ((0, 1, 2), (3, 4, 5)))
                        if fusion is not None:
                            self.assertEqual([logits.numel() for logits in compact.video_kv_fusion_logits], [3, 3])

    def test_layer_specific_masks_use_video_indices_in_joint_and_cache_paths(self):
        def run_with_distinct_masks(mot, inputs, *, cached=False):
            resolve = mot._resolve_attention_masks

            def layer_masks(mask, *, expected_seq_len):
                masks = [layer.clone() for layer in resolve(mask, expected_seq_len=expected_seq_len)]
                video_len = expected_seq_len - inputs[1].shape[1]
                for video_idx, layer in enumerate(masks):
                    layer[video_len:, :video_len] = False
                    layer[video_len:, :min(video_idx + 1, video_len)] = True
                return masks

            with patch.object(mot, "_resolve_attention_masks", side_effect=layer_masks):
                return forward_mot(mot, inputs, cached=cached)

        for fusion in (None, "interval_weighted_sum"):
            with self.subTest(fusion=fusion):
                torch.manual_seed(43)
                reference = dense_mot(fusion=fusion)
                compact = compact_from(reference)
                for index in (2, 5):
                    zero_refinement_updates(reference.mixtures["action"].blocks[index])
                inputs = inputs_for(torch.float32)
                expected = run_with_distinct_masks(reference, inputs)[1]
                actual = run_with_distinct_masks(compact, inputs)[1]
                cached = run_with_distinct_masks(compact, inputs, cached=True)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(cached, expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
