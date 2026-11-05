"""Per-component optimizer groups and a checkpointable relative LR schedule."""

from math import cos, isfinite, pi


LR_COMPONENTS = ("video", "action", "fusion")


def resolve_learning_rates(cfg):
    """Keep the scalar learning_rate as the fallback for old configs/overrides."""
    overrides = cfg.get("learning_rates", {})
    unknown = set(overrides) - set(LR_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown learning_rates components: {sorted(unknown)}")
    rates = {}
    for name in LR_COMPONENTS:
        value = float(overrides.get(name, cfg.learning_rate))
        if not isfinite(value) or value < 0:
            raise ValueError(f"learning_rates.{name} must be finite and non-negative, got {value}.")
        rates[name] = value
    return rates


def build_component_param_groups(model, named_parameters, learning_rates):
    """Partition the existing trainable set, accounting for MoT/expert aliases.

    Proprio is shared conditioning; as an explicit optimizer policy it follows
    the action LR. The remaining MoT parameters (KV fusion logits in FasterWAM)
    use the fusion LR. This does not change any requires_grad flags.
    """
    video = getattr(model, "video_expert", None)
    action = getattr(model, "action_expert", None)
    if video is None and action is None:
        # Preserve the trainer's plain video-DiT model support as well.
        video = model.dit
    video_ids = {id(p) for p in video.parameters()} if video is not None else set()
    action_ids = {id(p) for p in action.parameters()} if action is not None else set()
    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        action_ids.update(id(p) for p in proprio.parameters())
    if video_ids & action_ids:
        raise ValueError("Video and action LR groups must not share parameter objects.")

    groups = {name: {"name": name, "params": [], "lr": learning_rates[name]}
              for name in LR_COMPONENTS}
    names = {name: [] for name in LR_COMPONENTS}
    seen = set()
    for name, parameter in named_parameters:
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"Duplicate trainable parameter in optimizer input: {name}")
        seen.add(identity)
        component = "video" if identity in video_ids else "action" if identity in action_ids else "fusion"
        groups[component]["params"].append(parameter)
        names[component].append(name)
    active = [name for name in LR_COMPONENTS if groups[name]["params"]]
    return ([groups[name] for name in active],
            [{"name": name, "parameters": names[name]} for name in active])


class CosineDecay:
    """Serializable LambdaLR multiplier with a per-group relative minimum.

    LambdaLR saves callable-object attributes, unlike a lambda's closed-over
    values, so full-state resume restores the original schedule horizon.
    """

    def __init__(self, total_steps, min_factor=0.01):
        self.total_steps = total_steps
        self.min_factor = min_factor

    def __call__(self, step):
        return self.min_factor + (1.0 - self.min_factor) * (1.0 + cos(pi * step / self.total_steps)) / 2.0
