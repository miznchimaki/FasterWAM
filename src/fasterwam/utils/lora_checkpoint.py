"""Self-contained FasterWAM checkpoints with explicit LoRA architecture metadata."""

from collections.abc import Mapping
from importlib.metadata import version as package_version
import logging
from typing import Any


logger = logging.getLogger(__name__)
CHECKPOINT_FORMAT_VERSION = 2
LORA_METADATA_VERSION = 1


def _experts(model):
    return {
        "video": getattr(model, "video_expert", None),
        "action": getattr(model, "action_expert", None),
    }


def get_lora_checkpoint_metadata(model) -> dict[str, Any]:
    """Return fields to add to a full checkpoint, or {} for a dense model.

    This reads configuration only, so it is also safe after ZeRO partitioning.
    Dense checkpoint consumers retain the original payload schema.
    """
    experts = _experts(model)
    if any(expert is None for expert in experts.values()):
        return {}
    from .lora import get_expert_lora_config

    configs = {name: get_expert_lora_config(expert) for name, expert in experts.items()}
    for name, config in configs.items():
        if not config["enabled"]:
            configs[name] = {"enabled": False}
        else:
            config["target_modules"] = sorted(config["target_modules"])
    if not any(config["enabled"] for config in configs.values()):
        return {}
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "lora": {
            "version": LORA_METADATA_VERSION,
            "implementation": "peft",
            "peft_version": package_version("peft"),
            "adapter_name": "default",
            **configs,
        },
    }


def _adapter_modules(module):
    # Only the standard Linear LoRA wrappers installed by utils.lora are
    # supported. Attribute checks keep dense checkpoint loading PEFT-optional.
    return {
        name: child
        for name, child in module.named_modules()
        if hasattr(child, "base_layer") and hasattr(child, "lora_A") and hasattr(child, "lora_B")
    }


def _has_adapter_keys(state_dict) -> bool:
    return any(".lora_" in key or key.startswith("lora_") for key in state_dict)


def _shape_schema(state_dict) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in state_dict.items()}


def _dense_schema(module) -> dict[str, tuple[int, ...]]:
    """Inspect base shapes without copying weights or changing adapter modules."""
    adapters = _adapter_modules(module)
    result = {}
    for key, value in module.state_dict().items():
        for path in adapters:
            prefix = path + "." if path else ""
            if key.startswith(prefix + "base_layer."):
                key = prefix + key[len(prefix + "base_layer."):]
                break
            if key.startswith(prefix + "lora_"):
                key = None
                break
        if key is not None:
            if key in result:
                raise ValueError(f"Ambiguous dense checkpoint key after removing LoRA wrappers: {key}")
            result[key] = tuple(value.shape)
    return result


def _validate_state(state_dict, expected_shapes, *, component):
    import torch

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint {component!r} must be a state-dict mapping.")
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError(f"Checkpoint {component!r} keys must be strings.")
    missing = sorted(set(expected_shapes) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected_shapes))
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint {component!r} keys do not match the model: "
            f"missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint tensor {component}.{key} is not a torch.Tensor.")
        if tuple(value.shape) != expected_shapes[key]:
            raise ValueError(
                f"Checkpoint shape mismatch for {component}.{key}: "
                f"got {tuple(value.shape)}, expected {expected_shapes[key]}"
            )


def _read_lora_metadata(payload):
    metadata = payload.get("lora")
    format_version = payload.get("checkpoint_format_version")
    if metadata is None:
        if format_version is not None:
            raise ValueError("Versioned checkpoint is missing its LoRA metadata.")
        return None
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint_format_version: {format_version!r}")
    if not isinstance(metadata, Mapping):
        raise TypeError("Checkpoint LoRA metadata must be a mapping.")
    required = {"version", "implementation", "peft_version", "adapter_name", "video", "action"}
    if set(metadata) != required:
        raise ValueError("Checkpoint LoRA metadata fields are incomplete or unsupported.")
    if metadata["version"] != LORA_METADATA_VERSION or metadata["implementation"] != "peft":
        raise ValueError("Unsupported checkpoint LoRA implementation or metadata version.")
    if metadata["adapter_name"] != "default":
        raise ValueError("FasterWAM checkpoints support only the 'default' LoRA adapter.")
    if not isinstance(metadata["peft_version"], str):
        raise TypeError("Checkpoint peft_version must be a string.")
    from .lora import normalize_lora_config

    configs = {}
    for branch in ("video", "action"):
        config = metadata[branch]
        if not isinstance(config, Mapping) or "enabled" not in config:
            raise ValueError(f"Checkpoint {branch} LoRA configuration is incomplete.")
        if config["enabled"] and not all(
            key in config and config[key] is not None
            for key in ("r", "lora_alpha", "lora_dropout", "target_modules")
        ):
            raise ValueError(f"Checkpoint {branch} LoRA configuration must specify all effective settings.")
        configs[branch] = normalize_lora_config(config, branch=branch)
    if not any(config["enabled"] for config in configs.values()):
        raise ValueError("LoRA checkpoint metadata does not enable any adapter.")
    return configs


def _config_key(config):
    if not config["enabled"]:
        return (False,)
    return (
        True, config["r"], config["lora_alpha"], config["lora_dropout"],
        tuple(sorted(config["target_modules"])),
    )


def _lora_source_schema(model, configs):
    """Predict checkpoint tensor shapes for metadata before injecting anything."""
    from .lora import resolve_lora_target_modules

    shapes = _dense_schema(model.mot)
    for branch, expert in _experts(model).items():
        config = configs[branch]
        if not config["enabled"]:
            continue
        targets = resolve_lora_target_modules(expert, config, branch=branch)
        for target in targets:
            layer = expert.get_submodule(target)
            base = layer.get_base_layer() if hasattr(layer, "get_base_layer") else layer
            prefix = f"mixtures.{branch}.{target}."
            for name in ("weight", "bias"):
                if prefix + name in shapes:
                    shapes[prefix + "base_layer." + name] = shapes.pop(prefix + name)
            shapes[prefix + "lora_A.default.weight"] = (config["r"], base.in_features)
            shapes[prefix + "lora_B.default.weight"] = (base.out_features, config["r"])
    return shapes


def _load_dense_component(module, state_dict, *, component):
    """Load all base weights; only newly initialized LoRA tensors may be absent."""
    adapters = _adapter_modules(module)
    remapped = dict(state_dict)
    for path in adapters:
        prefix = path + "." if path else ""
        for name in ("weight", "bias"):
            key = prefix + name
            if key in remapped:
                remapped[prefix + "base_layer." + name] = remapped.pop(key)
    current_keys = set(module.state_dict())
    allowed_missing = {
        key for key in current_keys
        if any(
            key.startswith((path + "." if path else "") + adapter + ".default.")
            for path in adapters for adapter in ("lora_A", "lora_B")
        )
    }
    # Validate again against the actual wrappers before load_state_dict can
    # partially modify any weights, rather than trusting strict=False.
    missing = current_keys - set(remapped)
    unexpected = set(remapped) - current_keys
    if missing != allowed_missing or unexpected:
        raise ValueError(
            f"Unsafe dense-to-LoRA mapping in {component}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    incompatible = module.load_state_dict(remapped, strict=not bool(adapters))
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading validated {component} weights: {incompatible}")


def load_fastwam_checkpoint(model, path, optimizer=None, *, lora_config_policy="checkpoint"):
    """Load a full/dense checkpoint before optimizer creation or ZeRO prepare.

    Evaluation uses checkpoint metadata to configure the adapters. A training
    warm start may select ``match`` to require its explicit LoRA configuration.
    Old dense checkpoints initialize the current adapters with zero delta.
    """
    import torch

    if lora_config_policy not in {"checkpoint", "match"}:
        raise ValueError("lora_config_policy must be 'checkpoint' or 'match'.")
    if any(hasattr(parameter, "ds_id") for parameter in model.parameters()):
        raise ValueError("Load weight checkpoints before DeepSpeed/ZeRO parameter partitioning.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping.")
    if "mot" in payload:
        component, module = "mot", model.mot
    elif "dit" in payload:
        component, module = "dit", model.video_expert
    else:
        raise ValueError(f"Checkpoint missing both 'mot' and 'dit' keys: {path}")
    state_dict = payload[component]
    if not isinstance(state_dict, Mapping) or not all(isinstance(key, str) for key in state_dict):
        raise TypeError(f"Checkpoint {component!r} must be a state dict with string keys.")
    configs = _read_lora_metadata(payload)
    if configs is None and _has_adapter_keys(state_dict):
        raise ValueError("Checkpoint has LoRA tensors but no configuration metadata; cannot infer adapter scaling.")
    if configs is not None and component != "mot":
        raise ValueError("LoRA metadata requires a complete 'mot' checkpoint, not a legacy video-only 'dit'.")

    # Validate the entire incoming payload against a virtual architecture
    # before replacing adapter modules or modifying base weights.
    expected_shapes = _lora_source_schema(model, configs) if configs is not None else _dense_schema(module)
    _validate_state(state_dict, expected_shapes, component=component)
    proprio = getattr(model, "proprio_encoder", None)
    if configs is not None and (proprio is not None) != ("proprio_encoder" in payload):
        raise ValueError(
            "Complete LoRA checkpoint proprio_encoder configuration does not match the model: "
            f"model_has_proprio={proprio is not None}, checkpoint_has_proprio={'proprio_encoder' in payload}"
        )
    if proprio is not None and "proprio_encoder" in payload:
        _validate_state(payload["proprio_encoder"], _shape_schema(proprio.state_dict()), component="proprio_encoder")

    if configs is not None:
        from .lora import configure_expert_lora, get_expert_lora_config

        changes = [
            branch for branch, expert in _experts(model).items()
            if _config_key(get_expert_lora_config(expert)) != _config_key(configs[branch])
        ]
        if changes and lora_config_policy == "match":
            raise ValueError(
                f"Checkpoint LoRA configuration differs from the training model for {changes}; "
                "use the checkpoint's LoRA settings for a training warm start."
            )
        if changes and optimizer is not None:
            raise ValueError("Cannot reconfigure LoRA modules with an existing optimizer; load before optimizer creation.")
        if changes:
            logger.warning("Using checkpoint LoRA configuration instead of model configuration for %s", changes)
            for branch in changes:
                configure_expert_lora(_experts(model)[branch], configs[branch], branch=branch)
        # Check the implementation's actual schema as well as the virtual one.
        _validate_state(state_dict, _shape_schema(model.mot.state_dict()), component="mot")
        model.mot.load_state_dict(state_dict, strict=True)
    else:
        if optimizer is not None and "optimizer" in payload and _adapter_modules(module):
            raise ValueError("Cannot restore a dense optimizer state into an adapted model; use a weights-only warm start.")
        _load_dense_component(module, state_dict, component=component)
        if _adapter_modules(module):
            from .lora import reset_expert_lora_parameters

            branches = ("video", "action") if component == "mot" else ("video",)
            for branch in branches:
                reset_expert_lora_parameters(_experts(model)[branch])
        if component == "dit":
            logger.warning("Loaded legacy 'dit' checkpoint into the video expert only.")

    if proprio is not None:
        if "proprio_encoder" in payload:
            proprio.load_state_dict(payload["proprio_encoder"], strict=True)
        else:
            logger.warning("Checkpoint has no 'proprio_encoder' weights; keeping current parameters.")
    elif "proprio_encoder" in payload:
        logger.warning("Checkpoint contains proprio_encoder weights but model has proprio_dim=None; ignoring.")
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
