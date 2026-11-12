"""Checkpoint compatibility for compact action blocks aligned to video layers."""

from collections.abc import Mapping


def _normalize_action_layout(layout):
    fields = {"version", "video_num_layers", "condition_layers", "action_layer_indices"}
    if not isinstance(layout, Mapping) or set(layout) != fields:
        raise ValueError("Checkpoint action_layout fields are incomplete or unsupported.")
    if type(layout["version"]) is not int or layout["version"] != 1:
        raise ValueError("Unsupported checkpoint action_layout version.")
    depth = layout["video_num_layers"]
    if type(depth) is not int or depth <= 0:
        raise ValueError("action_layout.video_num_layers must be a positive integer.")
    result = {"version": 1, "video_num_layers": depth}
    for key in ("condition_layers", "action_layer_indices"):
        indices = layout[key]
        if not isinstance(indices, (list, tuple)) or not indices:
            raise ValueError(f"action_layout.{key} must be a nonempty sequence.")
        if any(type(index) is not int or not 0 <= index < depth for index in indices):
            raise ValueError(f"action_layout.{key} contains invalid video-layer indices.")
        if list(indices) != sorted(set(indices)):
            raise ValueError(f"action_layout.{key} must be strictly increasing.")
        result[key] = list(indices)
    if not set(result["condition_layers"]).issubset(result["action_layer_indices"]):
        raise ValueError("action_layout must retain every condition layer.")
    return result


def _model_action_layout(model):
    mot = getattr(model, "mot", None)
    if mot is None or not hasattr(mot, "action_layer_indices"):
        return None
    return _normalize_action_layout({
        "version": 1,
        "video_num_layers": mot.num_layers,
        "condition_layers": list(mot.condition_layers),
        "action_layer_indices": list(mot.action_layer_indices),
    })


def get_action_layout_metadata(model):
    """Keep the legacy payload schema for unchanged one-to-one layer layouts."""
    layout = _model_action_layout(model)
    if layout is None or layout["action_layer_indices"] == list(range(layout["video_num_layers"])):
        return {}
    return {"action_layout": layout}


def validate_action_layout_metadata(model, payload):
    """Reject semantic layout mismatches even when all tensor shapes agree.

    This reads ordinary attributes only and must run before loading any model,
    adapter or optimizer weights. Evaluation must use the training layout.
    """
    expected = _model_action_layout(model)
    if "action_layout" not in payload:
        if get_action_layout_metadata(model):
            raise ValueError(
                "Compact action checkpoint is missing action_layout metadata. "
                "Use a checkpoint saved with the same compact action configuration."
            )
        return
    actual = _normalize_action_layout(payload["action_layout"])
    if expected is None or actual != expected:
        raise ValueError(
            "Checkpoint action_layout does not match the model's video depth, "
            "condition layers, or action-to-video layer mapping. "
            "Restore the training model configuration before loading this checkpoint."
        )


def copy_action_layout_metadata(payload):
    """Validate and preserve the architecture when exporting merged LoRA weights."""
    if "action_layout" not in payload:
        return {}
    return {"action_layout": _normalize_action_layout(payload["action_layout"])}
