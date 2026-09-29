"""Validate checkpoint format selection without GPU or DeepSpeed dependencies."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fasterwam.utils.deepspeed_utils import (  # noqa: E402
    build_weights_checkpoint_payload,
    get_deepspeed_config,
    get_deepspeed_stage,
)


class DeepSpeedConfigurationTests(unittest.TestCase):
    def test_no_plugin_is_distinct_from_zero_stage_zero(self):
        accelerator = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None))
        self.assertEqual(get_deepspeed_config(accelerator), {})
        self.assertIsNone(get_deepspeed_stage(accelerator))
        for configured, expected in ((0, 0), (1, 1), (2, 2), ("3", 3)):
            config = {"zero_optimization": {"stage": configured}}
            accelerator.state.deepspeed_plugin = SimpleNamespace(deepspeed_config=config)
            self.assertIs(get_deepspeed_config(accelerator), config)
            self.assertEqual(get_deepspeed_stage(accelerator), expected)


class PortableCheckpointTests(unittest.TestCase):
    @staticmethod
    def model(*, mot=True, proprio=True):
        def reject_partitioned_state_dict():
            raise AssertionError("Export must not reread the partitioned model state.")

        model = SimpleNamespace(
            dit=object(),
            torch_dtype="torch.bfloat16",
            state_dict=reject_partitioned_state_dict,
            proprio_encoder=object() if proprio else None,
        )
        if mot:
            model.mot = model.dit
        return model

    def test_fasterwam_exports_canonical_mot_without_aliases_or_frozen_weights(self):
        video_weight = object()
        action_weight = object()
        fusion_weight = object()
        proprio_weight = object()
        full_state = {
            "video_expert.blocks.0.weight": video_weight,
            "action_expert.blocks.0.weight": action_weight,
            "mot.mixtures.video.blocks.0.weight": video_weight,
            "mot.mixtures.action.blocks.0.weight": action_weight,
            "mot.video_kv_fusion_logits.0": fusion_weight,
            # DeepSpeed's consolidated dict may omit the duplicate .dit alias.
            "proprio_encoder.weight": proprio_weight,
            "vae.model.weight": object(),
            "text_encoder.weight": object(),
        }
        payload = build_weights_checkpoint_payload(self.model(), full_state, step=21700)
        self.assertEqual(set(payload), {"mot", "proprio_encoder", "step", "torch_dtype"})
        self.assertEqual(
            set(payload["mot"]),
            {"mixtures.video.blocks.0.weight", "mixtures.action.blocks.0.weight", "video_kv_fusion_logits.0"},
        )
        self.assertIs(payload["mot"]["mixtures.video.blocks.0.weight"], video_weight)
        self.assertIs(payload["mot"]["video_kv_fusion_logits.0"], fusion_weight)
        self.assertIs(payload["proprio_encoder"]["weight"], proprio_weight)
        self.assertEqual(payload["step"], 21700)
        self.assertEqual(payload["torch_dtype"], "torch.bfloat16")
        self.assertIn("vae.model.weight", full_state)

    def test_wan22_keeps_legacy_dit_format(self):
        weight = object()
        payload = build_weights_checkpoint_payload(
            self.model(mot=False, proprio=False), {"dit.blocks.0.weight": weight}
        )
        self.assertEqual(set(payload), {"dit", "step", "torch_dtype"})
        self.assertIs(payload["dit"]["blocks.0.weight"], weight)
        self.assertIsNone(payload["step"])

    def test_mot_without_proprio_does_not_export_unexpected_proprio_keys(self):
        payload = build_weights_checkpoint_payload(
            self.model(proprio=False),
            {"mot.mixtures.video.weight": object(), "proprio_encoder.weight": object()},
        )
        self.assertNotIn("proprio_encoder", payload)

    def test_missing_mot_cannot_silently_fall_back_to_dit_alias(self):
        with self.assertRaisesRegex(ValueError, "mot\\."):
            build_weights_checkpoint_payload(self.model(), {"dit.weight": object()})

    def test_missing_proprio_or_dit_fails_before_saving_incomplete_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "proprio_encoder\\."):
            build_weights_checkpoint_payload(self.model(), {"mot.weight": object()})
        with self.assertRaisesRegex(ValueError, "dit\\."):
            build_weights_checkpoint_payload(self.model(mot=False, proprio=False), {"vae.weight": object()})

    def test_nonmain_rank_result_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "rank zero"):
            build_weights_checkpoint_payload(self.model(), None)


if __name__ == "__main__":
    unittest.main()
