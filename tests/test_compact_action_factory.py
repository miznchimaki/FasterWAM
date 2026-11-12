"""CPU tests for compact FasterWAM construction and Hydra routing."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch import nn

import fasterwam.models.wan22.fasterwam as factory_module
from fasterwam.models.wan22.fasterwam import FasterWAM
from fasterwam.models.wan22.sparse_action_dit import SparseActionDiT
from fasterwam.models.wan22.sparse_mot import SparseMoT
from fasterwam.models.wan22.wan_video_dit import WanVideoDiT
from fasterwam.runtime import create_fasterwam
from fasterwam.utils.lora import configure_expert_freeze, configure_expert_lora


def tiny_configs():
    video = dict(
        hidden_dim=16, in_dim=4, ffn_dim=32, out_dim=4, text_dim=12,
        freq_dim=8, eps=1e-6, patch_size=(1, 1, 1), num_heads=2,
        attn_head_dim=8, num_layers=10, has_image_input=False,
        seperated_timestep=True, video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=False,
    )
    action = dict(
        hidden_dim=8, action_dim=3, ffn_dim=16, text_dim=12, freq_dim=8,
        eps=1e-6, num_heads=2, attn_head_dim=8, num_layers=10,
        noncondition_num_heads=1, condition_layers=[0, 4, 8],
        use_gradient_checkpointing=False,
    )
    return video, action


def tiny_components(**kwargs):
    return SimpleNamespace(
        dit=WanVideoDiT(**kwargs["dit_config"]), vae=nn.Linear(2, 2),
        text_encoder=None, tokenizer=None, dit_path="tiny-video", vae_path="tiny-vae",
        text_encoder_path=None, tokenizer_path=None,
    )


class CompactActionFactoryTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch.object(
            factory_module, "load_wan22_ti2v_5b_components", side_effect=tiny_components,
        )
        self.loader_mock = self.loader.start()
        self.addCleanup(self.loader.stop)

    @staticmethod
    def build(*, video=None, action=None, **kwargs):
        default_video, default_action = tiny_configs()
        return FasterWAM.from_wan22_pretrained(
            video_dit_config=default_video if video is None else video,
            action_dit_config=default_action if action is None else action,
            device="cpu", torch_dtype=torch.float32, load_text_encoder=False,
            proprio_dim=4, mot_checkpoint_mixed_attn=False,
            video_kv_fusion="interval_weighted_sum", **kwargs,
        )

    def test_random_initialization_allocates_only_compact_blocks(self):
        for skip_pretrain in (False, True):
            with self.subTest(skip_pretrain=skip_pretrain), patch.object(
                SparseActionDiT, "from_pretrained", wraps=SparseActionDiT.from_pretrained,
            ) as loader:
                model = self.build(
                    action_noncondition_layers_per_interval=1,
                    action_dit_pretrained_path="ignored.pt" if skip_pretrain else None,
                    skip_dit_load_from_pretrain=skip_pretrain,
                )
                loaded_config = loader.call_args.kwargs["action_dit_config"]
                self.assertEqual(loaded_config["num_layers"], 6)
                self.assertEqual(loaded_config["condition_layers"], [0, 2, 4])
                self.assertEqual(len(model.video_expert.blocks), 10)
                self.assertEqual(len(model.action_expert.blocks), 6)
                self.assertEqual(model.action_layer_indices, (0, 1, 4, 5, 8, 9))
                self.assertEqual(model.condition_layers, (0, 4, 8))
                self.assertEqual(model.action_expert.condition_layers, (0, 2, 4))

    def test_pretrained_dense_backbone_selects_exact_retained_block_weights(self):
        video_config, action_config = tiny_configs()
        source = SparseActionDiT(**action_config)
        with torch.no_grad():
            for index, block in enumerate(source.blocks):
                for parameter in block.parameters():
                    parameter.fill_(0.01 * (index + 1))
        payload = {
            "meta": {key: action_config[key] for key in SparseActionDiT.ACTION_BACKBONE_META_KEYS},
            "backbone_state_dict": {
                key: value for key, value in source.state_dict().items()
                if key in SparseActionDiT.backbone_key_set(source.state_dict())
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "dense-backbone.pt")
            torch.save(payload, path)
            with patch.object(SparseActionDiT, "from_pretrained", wraps=SparseActionDiT.from_pretrained) as loader:
                model = self.build(
                    video=video_config, action=action_config,
                    action_dit_pretrained_path=str(path),
                    action_noncondition_layers_per_interval=[1, 2],
                    action_trailing_noncondition_layers=0,
                )
                self.assertEqual(loader.call_args.kwargs["action_dit_config"]["num_layers"], 10)
                self.assertEqual(loader.call_args.kwargs["action_dit_config"]["condition_layers"], [0, 4, 8])
        self.assertEqual(model.action_layer_indices, (0, 1, 4, 5, 6, 8))
        self.assertEqual(model.action_expert.condition_layers, (0, 2, 5))
        for local_index, original_index in enumerate(model.action_layer_indices):
            for name, value in model.action_expert.blocks[local_index].state_dict().items():
                torch.testing.assert_close(value, source.blocks[original_index].state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(isinstance(block, nn.Module) for block in model.action_expert.blocks))

    @staticmethod
    def legacy_default_model():
        # Reference the pre-compaction construction order explicitly. In
        # particular, video, VAE, action, and proprio initialization consume RNG.
        video_config, action_config = tiny_configs()
        components = tiny_components(dit_config=video_config)
        action = SparseActionDiT.from_pretrained(
            action_config, device="cpu", torch_dtype=torch.float32,
        )
        mot = SparseMoT(
            {"video": components.dit, "action": action}, condition_layers=[0, 4, 8],
            mot_checkpoint_mixed_attn=False, video_kv_fusion="interval_weighted_sum",
        )
        model = FasterWAM(
            video_expert=components.dit, action_expert=action, mot=mot,
            vae=components.vae, text_dim=12, proprio_dim=4,
            device="cpu", torch_dtype=torch.float32,
            mot_action_video_attention_layers=[0, 4, 8],
        )
        for branch in ("video", "action"):
            expert = getattr(model, branch + "_expert")
            configure_expert_lora(expert, {"enabled": False})
            configure_expert_freeze(expert, False, branch=branch)
        return model

    def test_default_preserves_legacy_rng_and_state_dict_keys_and_values(self):
        torch.manual_seed(17)
        reference = self.legacy_default_model()
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(17)
        actual = self.build()
        self.assertEqual(list(reference.state_dict()), list(actual.state_dict()))
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(reference_rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(actual.action_layer_indices, tuple(range(10)))

    def test_explicit_none_condition_layers_preserves_original_default(self):
        video, action = tiny_configs()
        video["num_layers"] = action["num_layers"] = 30
        action["condition_layers"] = None
        model = self.build(video=video, action=action, condition_layers=None)
        expected = (0, 4, 8, 12, 16, 20, 24, 28)
        self.assertEqual(model.condition_layers, expected)
        self.assertEqual(model.action_expert.condition_layers, expected)
        self.assertEqual(model.action_layer_indices, tuple(range(30)))
        self.assertIsNone(action["condition_layers"])

    def test_compact_structure_precedes_lora_and_freeze_policy(self):
        for lora_enabled in (False, True):
            with self.subTest(lora_enabled=lora_enabled):
                video_config, action_config = tiny_configs()
                action_config.update({
                    "freeze": True,
                    "lora": {
                        "enabled": lora_enabled, "r": 2, "lora_alpha": 2,
                        "target_modules": ["self_attn.q"],
                    },
                })
                model = self.build(video=video_config, action=action_config,
                                   action_noncondition_layers_per_interval=1)
                self.assertEqual(len(model.action_expert.blocks), 6)
                for name, parameter in model.action_expert.named_parameters():
                    expected = name.startswith(("action_encoder.", "head.")) or (
                        lora_enabled and (".lora_A." in name or ".lora_B." in name)
                    )
                    self.assertEqual(parameter.requires_grad, expected, name)
                self.assertIsInstance(model.action_expert.action_encoder, nn.Linear)
                self.assertIsInstance(model.action_expert.head, nn.Linear)
                self.assertTrue(all(parameter.requires_grad for parameter in model.proprio_encoder.parameters()))
                self.assertTrue(all(parameter.requires_grad for parameter in model.mot.video_kv_fusion_logits))

    def test_invalid_dense_reference_or_layout_fails_before_model_loading(self):
        video, action = tiny_configs()
        action["num_layers"] = 6
        with self.assertRaisesRegex(ValueError, "dense reference depth"):
            self.build(video=video, action=action)
        self.loader_mock.assert_not_called()
        with self.assertRaises(ValueError):
            self.build(action_noncondition_layers_per_interval=[4, 1])
        self.loader_mock.assert_not_called()

    def test_runtime_preserves_listconfig_controls_and_validation(self):
        video, action = tiny_configs()
        kwargs = dict(
            model_id="test", tokenizer_model_id="test", video_dit_config=OmegaConf.create(video),
            action_dit_config=OmegaConf.create(action),
            action_scheduler={"train_shift": 5, "infer_shift": 5, "num_train_timesteps": 1000},
            condition_layers=OmegaConf.create([0, 4, 8]),
            action_noncondition_layers_per_interval=OmegaConf.create([1, 2]),
            action_trailing_noncondition_layers=0,
            device="cpu", model_dtype=torch.float32,
        )
        model = create_fasterwam(**kwargs)
        self.assertEqual(model.action_layer_indices, (0, 1, 4, 5, 6, 8))
        kwargs["action_noncondition_layers_per_interval"] = OmegaConf.create([1.5, 2])
        with self.assertRaises(ValueError):
            create_fasterwam(**kwargs)

    def test_hydra_overrides_route_new_controls_without_mutating_configs(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="train", overrides=[
                "task=libero_fasterwam_2cam224_1e-4",
                "model.condition_layers=[0,12,24,28]",
                "model.action_noncondition_layers_per_interval=[2,3,1]",
                "model.action_trailing_noncondition_layers=0",
                "model.action_dit_pretrained_path=null",
            ])
        original = deepcopy(OmegaConf.to_container(cfg.model, resolve=True))
        with patch.object(FasterWAM, "from_wan22_pretrained", return_value="constructed") as constructor:
            result = instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
        self.assertEqual(result, "constructed")
        arguments = constructor.call_args.kwargs
        self.assertEqual(arguments["condition_layers"], [0, 12, 24, 28])
        self.assertEqual(arguments["action_noncondition_layers_per_interval"], [2, 3, 1])
        self.assertEqual(arguments["action_trailing_noncondition_layers"], 0)
        self.assertIsNone(arguments["action_dit_pretrained_path"])
        self.assertEqual(OmegaConf.to_container(cfg.model, resolve=True), original)


if __name__ == "__main__":
    unittest.main()
