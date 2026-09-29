"""Prepare the model's module registration tree for ZeRO-3 hooks."""

from __future__ import annotations

from torch import nn


def prepare_model_for_zero3(model: nn.Module) -> None:
    """Keep MoT as the only registered owner of the two experts.

    FastWAM/FasterWAM expose the experts and ``dit`` as convenience aliases.
    DeepSpeed 0.18.5 recursively installs ZeRO-3 hooks with ``children()``;
    that traversal does not deduplicate modules reached through different
    parents. Remove the redundant root registrations before Accelerator's
    ``prepare`` while retaining the same objects at their public attributes.

    This changes root state-dict keys to the canonical ``mot.*`` tree, but
    preserves ``mot.state_dict()``, parameter objects, and optimizer references.
    It must only be called for ZeRO-3, before hooks have been installed.
    """
    mot = getattr(model, "mot", None)
    mixtures = getattr(mot, "mixtures", None)
    if model._modules.get("mot") is not mot or not isinstance(mixtures, nn.ModuleDict):
        raise ValueError("ZeRO-3 requires a registered model.mot with a mixtures ModuleDict.")
    if "video" not in mixtures or "action" not in mixtures:
        raise ValueError("ZeRO-3 requires both video and action experts in model.mot.mixtures.")

    aliases = {
        "video_expert": mixtures["video"],
        "action_expert": mixtures["action"],
        "dit": mot,
    }
    # Validate every alias before mutating the registration tree. Also allow
    # ordinary attributes so applying this preparation twice is harmless.
    for name, target in aliases.items():
        if getattr(model, name, None) is not target:
            raise ValueError(f"ZeRO-3 requires model.{name} to refer to its canonical MoT module.")
        if name in model._modules and model._modules[name] is not target:
            raise ValueError(f"Unexpected registered module at model.{name}.")

    for name, target in aliases.items():
        model._modules.pop(name, None)
        object.__setattr__(model, name, target)
