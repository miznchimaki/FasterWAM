"""CPU freeze regressions using native SparseMoT attention and real autograd.

These tests exercise checkpointed Torch execution, not DeepSpeed collectives.
"""

from copy import deepcopy
import unittest

import torch
from torch import nn

from fasterwam.trainer import Wan22Trainer
from fasterwam.utils.lora import (
    configure_expert_freeze,
    get_expert_freeze,
    split_dit_training_config,
)
from fasterwam.utils.zero3_model import prepare_model_for_zero3
from test_lora import inputs_for, tiny_mot


def action_loss_through_video_context(model, inputs):
    """The proprio gradient must cross frozen video and action blocks.

    Detaching only action's context prevents its direct cross-attention path
    from concealing an accidentally detached/frozen video computation graph.
    """
    latents, actions, timesteps, _ = inputs
    proprio = torch.arange(12, dtype=latents.dtype).reshape(2, 2, 3) / 12
    context = model.proprio_encoder(proprio)
    video, action = model.video_expert, model.action_expert
    states = {
        "video": video.pre_dit(latents, timesteps, context, fuse_vae_embedding_in_latents=True),
        "action": action.pre_dit(actions, timesteps, context.detach()),
    }
    nv, na = (states[branch]["tokens"].shape[1] for branch in ("video", "action"))
    mask = torch.ones(nv + na, nv + na, dtype=torch.bool)
    mask[:nv, nv:] = False
    mask[:nv, :nv] = video.build_video_to_video_mask(
        nv, states["video"]["meta"]["tokens_per_frame"], latents.device,
    )
    tokens = model.mot(
        embeds_all={branch: state["tokens"] for branch, state in states.items()},
        attention_mask=mask,
        freqs_all={branch: state["freqs"] for branch, state in states.items()},
        context_all={branch: {"context": state["context"], "mask": state["context_mask"]}
                     for branch, state in states.items()},
        t_mod_all={branch: state["t_mod"] for branch, state in states.items()},
    )
    output = action.post_dit(tokens["action"], states["action"])
    return output, output.square().mean()


class DiTFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_freeze_config_is_boolean_optional_and_stripped_without_mutation(self):
        original = {"hidden_dim": 8, "freeze": True, "lora": {"enabled": True, "target_modules": ["q"]}}
        before = deepcopy(original)
        clean, lora, frozen = split_dit_training_config(original)
        self.assertEqual(clean, {"hidden_dim": 8})
        self.assertTrue(frozen)
        self.assertTrue(lora["enabled"])
        lora["target_modules"].append("k")
        self.assertEqual(original, before)
        self.assertIs(split_dit_training_config({"hidden_dim": 8})[2], False)
        expert = nn.Linear(2, 2)
        self.assertIs(get_expert_freeze(expert), False)
        for invalid in ("false", "true", 0, 1, None, []):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    split_dit_training_config({"freeze": invalid})
                with self.assertRaises(ValueError):
                    configure_expert_freeze(expert, invalid, branch="video")
                self.assertIs(get_expert_freeze(expert), False)
                self.assertTrue(all(p.requires_grad for p in expert.parameters()))

    def test_frozen_native_experts_backpropagate_to_action_io_proprio_and_fusion(self):
        reference_output = None
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                torch.manual_seed(53)
                model = nn.Module()
                model.mot = tiny_mot(checkpoint=checkpoint)
                model.dit = model.mot
                model.video_expert = model.mot.mixtures["video"]
                model.action_expert = model.mot.mixtures["action"]
                model.proprio_encoder = nn.Linear(3, 12)
                model.vae = nn.Linear(2, 2)
                for branch in ("video", "action"):
                    configure_expert_freeze(getattr(model, branch + "_expert"), True, branch=branch)
                Wan22Trainer._apply_dit_only_train_mode(model)
                self.assertTrue(all(not p.requires_grad for p in model.vae.parameters()))
                self.assertTrue(all(not p.requires_grad for p in model.video_expert.parameters()))
                for name, parameter in model.action_expert.named_parameters():
                    self.assertEqual(parameter.requires_grad, name.startswith(("action_encoder.", "head.")), name)
                mask = {id(p): p.requires_grad for p in model.parameters()}
                prepare_model_for_zero3(model)
                self.assertEqual({id(p): p.requires_grad for p in model.parameters()}, mask)
                model.eval()
                Wan22Trainer._apply_dit_only_train_mode(model)
                self.assertEqual({id(p): p.requires_grad for p in model.parameters()}, mask)

                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
                output, loss = action_loss_through_video_context(model, inputs_for(torch.float32))
                if reference_output is None:
                    reference_output = output.detach().clone()
                else:
                    torch.testing.assert_close(output, reference_output)
                loss.backward()
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad, name)
                updated_modules = (model.action_expert.action_encoder, model.action_expert.head,
                                   model.proprio_encoder)
                for module in updated_modules:
                    for parameter in module.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                fusion = model.mot.video_kv_fusion_logits[-1]
                self.assertGreater(fusion.numel(), 1)
                self.assertTrue(fusion.requires_grad)
                self.assertIsNotNone(fusion.grad)
                self.assertTrue(torch.isfinite(fusion.grad).all())
                self.assertGreater(fusion.grad.abs().sum().item(), 0)
                optimizer.step()
                names = {id(p): name for name, p in model.named_parameters()}
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                for module in updated_modules:
                    self.assertFalse(torch.equal(module.weight, before[names[id(module.weight)]]))
                self.assertFalse(torch.equal(fusion, before[names[id(fusion)]]))


if __name__ == "__main__":
    unittest.main()
