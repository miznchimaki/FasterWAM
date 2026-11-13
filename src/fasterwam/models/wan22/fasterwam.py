from typing import Any, Optional, Sequence

import torch

from .sparse_action_dit import SparseActionDiT, DEFAULT_SPARSE_MOT_CONDITION_LAYERS
from .action_layout import resolve_action_layout
from .jointwam import JointWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .sparse_mot import SparseMoT
from fasterwam.utils.lora import (
    configure_expert_freeze,
    configure_expert_lora,
    split_dit_training_config,
)
from fasterwam.utils.logging_config import get_logger


logger = get_logger(__name__)


class FasterWAM(JointWAM):
    """Final FasterWAM system built from SparseMoT and SparseActionDiT."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        condition_layers: Optional[Sequence[int]] = None,
        mot_action_video_attention_layers: Optional[Sequence[int]] = None,
        video_kv_fusion: Optional[str] = None,
        video_kv_fusion_init: str = "current_layer",
        action_noncondition_layers_per_interval: int | Sequence[int] | None = None,
        action_trailing_noncondition_layers: int | None = None,
        action_leading_noncondition_layers: int | None = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FasterWAM.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FasterWAM.")
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FasterWAM requires `video_dit_config['action_conditioned']=false`."
            )
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for FasterWAM.")

        # LoRA/freeze options are not constructor arguments of either dense DiT.
        # Load the original pretrained weights before changing Linear modules.
        video_dit_config, video_lora_config, video_freeze = split_dit_training_config(video_dit_config)
        action_dit_config, action_lora_config, action_freeze = split_dit_training_config(action_dit_config)

        if condition_layers is None:
            condition_layers = action_dit_config.get("condition_layers", DEFAULT_SPARSE_MOT_CONDITION_LAYERS)
        if condition_layers is None:
            condition_layers = DEFAULT_SPARSE_MOT_CONDITION_LAYERS
        video_num_layers = video_dit_config["num_layers"]
        if action_dit_config["num_layers"] != video_num_layers:
            raise ValueError(
                "`action_dit_config.num_layers` is the dense reference depth and must match "
                "`video_dit_config.num_layers`; use the action non-condition layer counts "
                "to select the compact action depth."
            )
        layout = resolve_action_layout(
            video_num_layers,
            condition_layers,
            noncondition_layers_per_interval=action_noncondition_layers_per_interval,
            trailing_noncondition_layers=action_trailing_noncondition_layers,
            leading_noncondition_layers=action_leading_noncondition_layers,
        )
        normalized_condition_layers = layout.condition_layers
        action_dit_config = dict(action_dit_config)
        action_dit_config["condition_layers"] = list(normalized_condition_layers)

        if (
            mot_action_video_attention_layers is not None
            and tuple(mot_action_video_attention_layers) != normalized_condition_layers
        ):
            raise ValueError(
                "`mot_action_video_attention_layers` must be omitted or equal to `condition_layers` "
                "for FasterWAM."
            )

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        if len(video_expert.blocks) != layout.video_num_layers:
            raise ValueError("Loaded video DiT depth does not match the requested action layout.")
        random_action_init = skip_dit_load_from_pretrain or not action_dit_pretrained_path
        if random_action_init:
            # Allocate only the retained blocks. Their local indices are dense;
            # SparseMoT maps them to the original video-layer coordinates.
            action_dit_config["num_layers"] = layout.num_action_layers
            action_dit_config["condition_layers"] = list(layout.action_condition_layers)
        action_expert = SparseActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if not random_action_init and layout.num_action_layers != layout.video_num_layers:
            # Existing offline backbone exports use dense video-layer indices.
            # Load them with the original strict validation before selecting
            # blocks, then inject LoRA only into the retained structure below.
            action_expert.blocks = torch.nn.ModuleList(
                [action_expert.blocks[index] for index in layout.action_layer_indices]
            )
            action_expert.condition_layers = layout.action_condition_layers
            action_expert.condition_layer_set = frozenset(layout.action_condition_layers)

        if len(action_expert.blocks) != layout.num_action_layers:
            raise ValueError("Loaded action DiT depth does not match the requested action layout.")
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("SparseActionDiT condition-layer `num_heads` must match video expert.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("SparseActionDiT `attn_head_dim` must match video expert.")
        if int(action_expert.num_heads) * int(action_expert.attn_head_dim) != int(video_expert.hidden_dim):
            raise ValueError(
                "SparseActionDiT condition-layer attention dim must equal video attention dim: "
                f"{action_expert.num_heads} * {action_expert.attn_head_dim} != {video_expert.hidden_dim}."
            )

        mot = SparseMoT(
            mixtures={"video": video_expert, "action": action_expert},
            condition_layers=normalized_condition_layers,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            video_kv_fusion=video_kv_fusion,
            video_kv_fusion_init=video_kv_fusion_init,
            action_layer_indices=layout.action_layer_indices,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            mot_action_video_attention_layers=normalized_condition_layers,
        )
        model.condition_layers = normalized_condition_layers
        model.action_layer_indices = layout.action_layer_indices
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "condition_layers": list(normalized_condition_layers),
            "action_layer_indices": list(layout.action_layer_indices),
            "action_condition_layers": list(layout.action_condition_layers),
            "action_num_layers": layout.num_action_layers,
            "video_kv_fusion": video_kv_fusion,
            "video_kv_fusion_init": video_kv_fusion_init,
        }
        # Inject only after all original modules (including proprio) have been
        # initialized, preserving the base initialization's RNG sequence.
        configure_expert_lora(model.video_expert, video_lora_config)
        configure_expert_lora(model.action_expert, action_lora_config)
        configure_expert_freeze(model.video_expert, video_freeze, branch="video")
        configure_expert_freeze(model.action_expert, action_freeze, branch="action")
        logger.info(
            "FasterWAM layer layout: video_depth=%d action_depth=%d "
            "condition_video_layers=%s condition_action_layers=%s action_to_video_layers=%s",
            layout.video_num_layers, layout.num_action_layers,
            list(layout.condition_layers), list(layout.action_condition_layers),
            list(layout.action_layer_indices),
        )
        return model
