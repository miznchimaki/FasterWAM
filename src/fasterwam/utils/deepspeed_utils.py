"""DeepSpeed configuration and portable weight-checkpoint helpers.

Keep the collective operations in the trainer: every rank must call
``accelerator.get_state_dict(prepared_model)`` for ZeRO-3. Only rank zero
receives the consolidated state dictionary and builds the payload below.
"""

from collections.abc import Mapping
from typing import Any


def get_deepspeed_config(accelerator) -> Mapping[str, Any]:
    """Return the active plugin configuration, or an empty mapping without it."""
    state = getattr(accelerator, "state", None)
    plugin = getattr(state, "deepspeed_plugin", None)
    if plugin is None:
        return {}
    config = getattr(plugin, "deepspeed_config", None)
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("DeepSpeed plugin configuration must be a mapping.")
    return config


def get_deepspeed_stage(accelerator) -> int | None:
    """Return the configured ZeRO stage; ``None`` means no stage is configured."""
    zero_config = get_deepspeed_config(accelerator).get("zero_optimization", {})
    if not isinstance(zero_config, Mapping):
        raise TypeError("DeepSpeed zero_optimization configuration must be a mapping.")
    stage = zero_config.get("stage")
    if stage is None:
        return None
    stage = int(stage)
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported DeepSpeed ZeRO stage: {stage}")
    return stage


def _extract_component(state_dict: Mapping[str, Any], name: str) -> dict[str, Any]:
    prefix = name + "."
    weights = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix) and len(key) > len(prefix)
    }
    if not weights:
        raise ValueError(
            f"Consolidated model state dictionary has no weights under {prefix!r}; "
            "refusing to write an empty checkpoint component."
        )
    return weights


def build_weights_checkpoint_payload(
    model,
    state_dict: Mapping[str, Any],
    *,
    step: int | None = None,
) -> dict[str, Any]:
    """Select the existing FasterWAM/Wan22 checkpoint format from full weights.

    ``model`` must be unwrapped. ``state_dict`` must be the full consolidated
    state returned on rank zero, not the partitioned result of
    ``model.state_dict()``. Tensor references are reused without cloning the
    large CPU checkpoint. Frozen VAE/text components and duplicate top-level
    expert aliases are deliberately excluded.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "Expected a consolidated model state dictionary on rank zero, "
            "not the None returned on other ZeRO-3 ranks."
        )
    if getattr(model, "mot", None) is not None:
        # FastWAM registers .dit as an alias of .mot. DeepSpeed's traversal
        # can omit that duplicate alias; .mot is the canonical checkpoint key.
        component = "mot"
    elif getattr(model, "dit", None) is not None:
        component = "dit"
    else:
        raise ValueError("Expected a FasterWAM/MoT or Wan22/DiT model for checkpoint export.")

    payload = {
        component: _extract_component(state_dict, component),
        "step": step,
        "torch_dtype": str(model.torch_dtype),
    }
    if component == "mot" and getattr(model, "proprio_encoder", None) is not None:
        payload["proprio_encoder"] = _extract_component(state_dict, "proprio_encoder")
    if component == "mot":
        from .lora_checkpoint import get_lora_checkpoint_metadata

        payload.update(get_lora_checkpoint_metadata(model))
    return payload
