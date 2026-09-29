"""Check ZeRO-3 module ownership with small CPU-only PyTorch models.

Run with ``python -m unittest discover -s tests -p test_zero3_model.py -v``.
DeepSpeed and model/media dependencies are not required.
"""

from collections import Counter
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fasterwam.utils.zero3_model import prepare_model_for_zero3  # noqa: E402


class TinyMoT(nn.Module):
    def __init__(self, video, action):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})
        self.video_kv_fusion_logits = nn.ParameterList([nn.Parameter(torch.zeros(2))])

    def forward(self, value):
        weights = self.video_kv_fusion_logits[0].softmax(dim=0)
        return weights[0] * self.mixtures["video"](value) + weights[1] * self.mixtures["action"](value)


class TinyWAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_expert = nn.Sequential(nn.Linear(3, 3), nn.Tanh())
        self.action_expert = nn.Sequential(nn.Linear(3, 3), nn.Tanh())
        self.mot = TinyMoT(self.video_expert, self.action_expert)
        self.dit = self.mot
        self.proprio_encoder = nn.Linear(3, 3)

    def forward(self, value):
        # Exercise both ordinary public aliases and the canonical module path.
        return self.mot(value) + self.video_expert(value) + self.proprio_encoder(value)


def recursive_children(module):
    """Match DeepSpeed 0.18.5's hook traversal, without a global visited set."""
    yield module
    for child in module.children():
        yield from recursive_children(child)


class Zero3ModelTests(unittest.TestCase):
    def test_each_module_receives_one_recursive_visit_and_aliases_remain(self):
        model = TinyWAM()
        parameters_before = {id(parameter) for parameter in model.parameters()}
        counts_before = Counter(id(module) for module in recursive_children(model))
        self.assertEqual(counts_before[id(model.video_expert[0])], 2)
        self.assertEqual(counts_before[id(model.mot)], 1)

        prepare_model_for_zero3(model)

        self.assertTrue(all(count == 1 for count in Counter(id(module) for module in recursive_children(model)).values()))
        self.assertEqual(set(model._modules), {"mot", "proprio_encoder"})
        self.assertIs(model.video_expert, model.mot.mixtures["video"])
        self.assertIs(model.action_expert, model.mot.mixtures["action"])
        self.assertIs(model.dit, model.mot)
        self.assertEqual({id(parameter) for parameter in model.parameters()}, parameters_before)
        model.eval()
        self.assertFalse(model.video_expert.training)
        model.train()
        self.assertTrue(model.action_expert.training)

    def test_values_gradients_optimizer_and_canonical_checkpoint_are_preserved(self):
        torch.manual_seed(17)
        model = TinyWAM()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer_parameters = tuple(optimizer.param_groups[0]["params"])
        expected_mot_state = {name: value.clone() for name, value in model.mot.state_dict().items()}
        value = torch.randn(2, 3)
        before = model(value)
        before.square().sum().backward()
        expected_gradients = {id(parameter): parameter.grad.clone() for parameter in model.parameters()}
        model.zero_grad(set_to_none=True)

        prepare_model_for_zero3(model)

        after = model(value)
        torch.testing.assert_close(after, before)
        after.square().sum().backward()
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, expected_gradients[id(parameter)])
        self.assertEqual(tuple(id(parameter) for parameter in optimizer.param_groups[0]["params"]),
                         tuple(id(parameter) for parameter in optimizer_parameters))
        self.assertEqual(set(model.mot.state_dict()), set(expected_mot_state))
        for name, tensor in model.mot.state_dict().items():
            torch.testing.assert_close(tensor, expected_mot_state[name])
        self.assertTrue(all(name.startswith(("mot.", "proprio_encoder.")) for name in model.state_dict()))

    def test_preparation_is_idempotent(self):
        model = TinyWAM()
        prepare_model_for_zero3(model)
        registrations = dict(model._modules)
        names = tuple(model.state_dict())
        prepare_model_for_zero3(model)
        self.assertEqual(model._modules, registrations)
        self.assertEqual(tuple(model.state_dict()), names)

    def test_mismatched_alias_is_rejected_before_any_mutation(self):
        model = TinyWAM()
        model.action_expert = nn.Linear(3, 3)
        registrations = dict(model._modules)
        with self.assertRaisesRegex(ValueError, "model.action_expert"):
            prepare_model_for_zero3(model)
        self.assertEqual(model._modules, registrations)


if __name__ == "__main__":
    unittest.main()
