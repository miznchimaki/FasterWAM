"""In-place, adapter-only LoRA for FasterWAM's native DiT experts.

Adapters must be configured before optimizer/Accelerator preparation. Keeping the
expert object itself preserves SparseMoT's direct access to its blocks and methods.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Real

from torch import nn


DEFAULT_LORA_TARGET_MODULES = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
)
DEFAULT_ACTION_LORA_TARGET_MODULES = DEFAULT_LORA_TARGET_MODULES + ("action_encoder", "head")
_CONFIG_ATTRIBUTE = "_fasterwam_lora_config"
_CONFIG_KEYS = {"enabled", "r", "lora_alpha", "lora_dropout", "target_modules"}


def normalize_lora_config(config: Mapping | None, *, branch: str | None = None) -> dict:
    """Validate and copy a plain LoRA config without touching a model.

    A missing block keeps the legacy full-training behavior. An omitted target
    list is resolved when the expert branch is known; it is never ``all-linear``
    (PEFT 0.14 only supports that shorthand for Transformers PreTrainedModel).
    """
    if branch not in (None, "video", "action"):
        raise ValueError(f"Unknown LoRA expert branch: {branch!r}")
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("The `lora` block must be a mapping.")
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown LoRA configuration keys: {sorted(map(str, unknown))}")

    enabled = config.get("enabled", False)
    rank = config.get("r", 16)
    alpha = config.get("lora_alpha", 16)
    dropout = config.get("lora_dropout", 0.0)
    if not isinstance(enabled, bool):
        raise ValueError("LoRA `enabled` must be a boolean.")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA `r` must be a positive integer.")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA `lora_alpha` must be a finite positive number.")
    if isinstance(dropout, bool) or not isinstance(dropout, Real) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LoRA `lora_dropout` must be a finite number in [0, 1).")

    targets = config.get("target_modules")
    if targets is None and branch is not None:
        targets = DEFAULT_ACTION_LORA_TARGET_MODULES if branch == "action" else DEFAULT_LORA_TARGET_MODULES
    if targets is not None:
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence) or not targets:
            raise ValueError("LoRA `target_modules` must be a nonempty list of module-name suffixes.")
        if any(not isinstance(name, str) or not name or name.strip() != name for name in targets):
            raise ValueError("LoRA targets must be nonempty module-name suffixes without surrounding whitespace.")
        if len(set(targets)) != len(targets):
            raise ValueError("LoRA `target_modules` must not contain duplicates.")
        targets = list(targets)
    return {
        "enabled": enabled,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": float(dropout),
        "target_modules": targets,
    }


def split_lora_config(dit_config: Mapping) -> tuple[dict, dict]:
    """Remove the nested LoRA block from a copied DiT constructor config."""
    if not isinstance(dit_config, Mapping):
        raise ValueError("The DiT configuration must be a mapping.")
    clean = deepcopy(dict(dit_config))
    return clean, normalize_lora_config(clean.pop("lora", None))


def get_expert_lora_config(expert: nn.Module) -> dict:
    """Return serializable metadata without exposing mutable internal state."""
    return deepcopy(getattr(expert, _CONFIG_ATTRIBUTE, {"enabled": False}))


def _branch_of(expert: nn.Module, branch: str | None) -> str:
    return branch if branch is not None else ("action" if hasattr(expert, "action_encoder") else "video")


def _assert_before_zero_prepare(expert: nn.Module) -> None:
    if any(hasattr(parameter, "ds_id") for parameter in expert.parameters()):
        raise RuntimeError("Configure, remove, or reset LoRA before Accelerator.prepare; ZeRO-partitioned parameters found.")


def _adapter_layers(expert: nn.Module) -> list[tuple[str, nn.Module]]:
    # Avoid importing PEFT for a legacy expert with no adapters.
    if not hasattr(expert, "peft_config") and not any(hasattr(module, "lora_A") for module in expert.modules()):
        return []
    from peft.tuners.lora.layer import Linear
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = []
    for name, module in expert.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        if not isinstance(module, Linear) or not isinstance(module.get_base_layer(), nn.Linear):
            raise ValueError("FasterWAM supports only standard PEFT LoRA Linear adapters.")
        if set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"}:
            raise ValueError("FasterWAM requires exactly one LoRA adapter named 'default'.")
        if module.merged or module.use_dora.get("default", False) or module.lora_bias.get("default", False):
            raise ValueError("Merged, DoRA, and bias-bearing LoRA adapters cannot be reconfigured here.")
        layers.append((name, module))
    if hasattr(expert, "peft_config") and not layers:
        raise ValueError("The expert has PEFT configuration but no supported LoRA Linear adapters.")
    return layers


def resolve_lora_target_modules(expert: nn.Module, config: Mapping, *, branch: str | None = None) -> list[str]:
    """Resolve suffixes to canonical Linear paths, ignoring adapter internals."""
    normalized = normalize_lora_config(config, branch=_branch_of(expert, branch))
    if not normalized["enabled"]:
        return []
    layers = dict(_adapter_layers(expert))
    matched = {suffix: [] for suffix in normalized["target_modules"]}
    resolved = []
    for name, module in expert.named_modules():
        if not name or any(name.startswith(path + ".") for path in layers):
            continue
        base = layers[name].get_base_layer() if name in layers else module
        suffixes = [suffix for suffix in matched if name == suffix or name.endswith("." + suffix)]
        if not suffixes:
            continue
        if not isinstance(base, nn.Linear):
            raise ValueError(f"LoRA target {name!r} is {type(base).__name__}, not nn.Linear.")
        resolved.append(name)
        for suffix in suffixes:
            matched[suffix].append(name)
    missing = [suffix for suffix, names in matched.items() if not names]
    if missing:
        raise ValueError(f"LoRA targets do not match any Linear modules: {missing}")
    return resolved


def remove_expert_lora(expert: nn.Module) -> nn.Module:
    """Remove unmerged adapters, preserving the exact original base parameters."""
    _assert_before_zero_prepare(expert)
    layers = _adapter_layers(expert)
    for name, module in layers:
        parent_name, _, child_name = name.rpartition(".")
        parent = expert.get_submodule(parent_name) if parent_name else expert
        setattr(parent, child_name, module.get_base_layer())
    if hasattr(expert, "peft_config"):
        delattr(expert, "peft_config")
    setattr(expert, _CONFIG_ATTRIBUTE, {"enabled": False})
    expert.requires_grad_(True)
    return expert


def configure_expert_lora(expert: nn.Module, config: Mapping | None, *, branch: str | None = None) -> nn.Module:
    """Inject or replace adapters in place before optimizer creation/prepare.

    Reapplying an identical config preserves learned adapters. A changed config
    discards the old delta and preserves the underlying base weights; checkpoint
    loaders can then restore the matching full state. Adapter dtype follows the
    base Linear dtype, including BF16 DeepSpeed initialization.
    """
    normalized = normalize_lora_config(config, branch=_branch_of(expert, branch))
    _assert_before_zero_prepare(expert)
    # Validate targets and all existing adapters before changing the module tree.
    existing = _adapter_layers(expert)
    targets = resolve_lora_target_modules(expert, normalized, branch=branch) if normalized["enabled"] else []
    if existing and get_expert_lora_config(expert) == normalized:
        apply_expert_trainability(expert)
        return expert
    if normalized["enabled"]:
        from peft import LoraConfig, inject_adapter_in_model

        peft_config = LoraConfig(
            r=normalized["r"],
            lora_alpha=normalized["lora_alpha"],
            lora_dropout=normalized["lora_dropout"],
            target_modules=targets,
            bias="none",
            init_lora_weights=True,
            use_rslora=False,
            use_dora=False,
        )
    if existing:
        remove_expert_lora(expert)
    if normalized["enabled"]:
        result = inject_adapter_in_model(peft_config, expert, adapter_name="default", low_cpu_mem_usage=False)
        if result is not expert:
            raise RuntimeError("PEFT injection unexpectedly replaced the native expert object.")
        # Newly created LoRA Dropout modules otherwise default to training mode,
        # even when adapters are restored into an expert already in eval mode.
        for _, layer in _adapter_layers(expert):
            layer.train(layer.get_base_layer().training)
    setattr(expert, _CONFIG_ATTRIBUTE, normalized)
    apply_expert_trainability(expert)
    return expert


def apply_expert_trainability(expert: nn.Module) -> None:
    """Restore full training when disabled, or train only LoRA A/B when enabled."""
    if not get_expert_lora_config(expert)["enabled"]:
        expert.requires_grad_(True)
        return
    layers = _adapter_layers(expert)
    if not layers:
        raise ValueError("LoRA is enabled but the expert has no adapters.")
    expert.requires_grad_(False)
    for _, layer in layers:
        layer.lora_A["default"].requires_grad_(True)
        layer.lora_B["default"].requires_grad_(True)


def reset_expert_lora_parameters(expert: nn.Module) -> nn.Module:
    """Reset a dense-checkpoint warm start to identity adapters before prepare."""
    _assert_before_zero_prepare(expert)
    for _, layer in _adapter_layers(expert):
        layer.reset_lora_parameters("default", init_lora_weights=True)
    return expert
