"""Resolve compact action depth while keeping video-layer coordinates stable."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionLayout:
    video_num_layers: int
    condition_layers: tuple[int, ...]
    action_layer_indices: tuple[int, ...]
    action_condition_layers: tuple[int, ...]
    noncondition_layers_per_interval: tuple[int, ...]
    trailing_noncondition_layers: int
    leading_noncondition_layers: int = 0

    @property
    def num_action_layers(self):
        return len(self.action_layer_indices)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"`{name}` must be an integer >= {minimum}, got {value!r}.")
    return value


def resolve_action_layout(
    video_num_layers,
    condition_layers,
    noncondition_layers_per_interval=None,
    trailing_noncondition_layers=None,
    leading_noncondition_layers=None,
):
    """Select the first N video slots in each action-refinement interval.

    Condition indices always refer to VIDEO layers. Action-only blocks do not
    consume the video state at their slot; placement within a gap only gives a
    deterministic mapping for retained pretrained weights. Leading and trailing
    counts select the first N available slots before/after the condition layers.
    None preserves all available layers; an integer applies to every gap, and
    a sequence specifies one positive count per adjacent condition pair.
    """
    depth = _integer(video_num_layers, "video_num_layers", 1)
    if not isinstance(condition_layers, Sequence) or isinstance(condition_layers, (str, bytes)):
        raise ValueError("`condition_layers` must be a nonempty sequence of video layer indices.")
    conditions = tuple(_integer(index, "condition_layers") for index in condition_layers)
    if not conditions or any(index >= depth for index in conditions):
        raise ValueError(f"`condition_layers` must be nonempty and within [0, {depth - 1}].")
    if any(left >= right for left, right in zip(conditions, conditions[1:])):
        raise ValueError("`condition_layers` must be strictly increasing without duplicates.")
    capacities = tuple(right - left - 1 for left, right in zip(conditions, conditions[1:]))
    if any(capacity < 1 for capacity in capacities):
        raise ValueError("Adjacent condition layers must leave at least one non-condition video slot between them.")

    field = "action_noncondition_layers_per_interval"
    if noncondition_layers_per_interval is None:
        counts = capacities
    elif isinstance(noncondition_layers_per_interval, int):
        count = _integer(noncondition_layers_per_interval, field, 1)
        counts = (count,) * len(capacities)
    elif isinstance(noncondition_layers_per_interval, Sequence) and not isinstance(
        noncondition_layers_per_interval, (str, bytes)
    ):
        counts = tuple(_integer(count, field, 1) for count in noncondition_layers_per_interval)
        if len(counts) != len(capacities):
            raise ValueError(f"`{field}` needs {len(capacities)} counts for {len(conditions)} condition layers, got {len(counts)}.")
    else:
        raise ValueError(f"`{field}` must be null, a positive integer, or a sequence of positive integers.")
    for index, (count, capacity) in enumerate(zip(counts, capacities)):
        if count > capacity:
            raise ValueError(
                f"`{field}[{index}]`={count} exceeds {capacity} available layers between "
                f"video condition layers {conditions[index]} and {conditions[index + 1]}."
            )

    leading_capacity = conditions[0]
    leading = leading_capacity if leading_noncondition_layers is None else _integer(
        leading_noncondition_layers, "action_leading_noncondition_layers",
    )
    if leading > leading_capacity:
        raise ValueError(
            f"`action_leading_noncondition_layers`={leading} exceeds {leading_capacity} available layers "
            f"before video condition layer {conditions[0]}."
        )

    tail_capacity = depth - conditions[-1] - 1
    tail = tail_capacity if trailing_noncondition_layers is None else _integer(
        trailing_noncondition_layers, "action_trailing_noncondition_layers",
    )
    if tail > tail_capacity:
        raise ValueError(
            f"`action_trailing_noncondition_layers`={tail} exceeds {tail_capacity} available layers "
            f"after video condition layer {conditions[-1]}."
        )

    retained = list(range(leading))
    local_conditions = []
    for index, condition in enumerate(conditions):
        local_conditions.append(len(retained))
        retained.append(condition)
        count = counts[index] if index < len(counts) else tail
        retained.extend(range(condition + 1, condition + 1 + count))
    return ActionLayout(depth, conditions, tuple(retained), tuple(local_conditions), counts, tail, leading)
