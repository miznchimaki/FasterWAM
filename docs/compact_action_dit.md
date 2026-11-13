# Compact action DiT

FasterWAM can retain every video DiT layer while reducing the action-only
refinement blocks before, between, and after condition layers. The default model is unchanged:
30 video layers, 30 action layers, and condition layers at video indices
`[0, 4, 8, 12, 16, 20, 24, 28]`.

`configs/model/fasterwam.yaml` exposes three Hydra options:

```yaml
action_leading_noncondition_layers: null
action_noncondition_layers_per_interval: null
action_trailing_noncondition_layers: null
```

- `action_leading_noncondition_layers`: `null` retains all layers before the
  first condition layer; an integer sets their count. It must satisfy
  `0 <= count <= first_condition`. When the first condition index is 0,
  only `null` or 0 is valid, and both mean no leading action layers.
- `action_noncondition_layers_per_interval`: `null` retains all available
  non-condition layers; an integer applies the same count to every gap; a
  list supplies one count for each adjacent condition pair.
- `action_trailing_noncondition_layers`: `null` retains all layers after the
  last condition layer; an integer sets their count. Zero is allowed here.
- `condition_layers` continues to use **video layer indices**. It must be
  nonempty, strictly increasing, and within the video depth. Each pair must
  leave at least one non-condition slot, and its requested count must satisfy
  `1 <= count <= next_condition - current_condition - 1`.
- The tail must satisfy `0 <= count <= video_num_layers - last_condition - 1`.
  Counts must be integers, not booleans or floating-point values. A list must
  contain exactly `len(condition_layers) - 1` entries.

The existing `action_dit_config.num_layers` remains the **dense reference
depth** and must match the video depth. Do not manually lower it: the factory
derives the actual action depth from the leading/interval/tail settings and logs both
depths and the complete local-to-video mapping.

For 30 video layers, representative layouts are:

| Video condition indices | Layers per gap | Tail | Actual action layers |
| --- | --- | --- | --- |
| `[0,4,8,12,16,20,24,28]` | `null` (3 each) | `null` (1) | 30 |
| same | 2 | 1 | 23 |
| same | 1 | 1 | 16 |
| same | 1 | 0 | 15 |
| `[0,12,24,28]` | 1 | 1 | 8 |

Append these overrides to the existing training command for a randomly
initialized 16-layer action model:

```bash
model.action_dit_pretrained_path=null \
model.action_noncondition_layers_per_interval=1 \
model.action_trailing_noncondition_layers=1
```

For an 8-layer action model with four condition layers:

```bash
model.action_dit_pretrained_path=null \
'model.condition_layers=[0,12,24,28]' \
model.action_noncondition_layers_per_interval=1 \
model.action_trailing_noncondition_layers=1
```

Unequal gap counts are also supported, for example
`'model.action_noncondition_layers_per_interval=[2,3,1]'` with those four
condition indices gives 11 action layers including the default tail of 1.
All LoRA, freeze, learning-rate, batch-size and other experiment settings
continue to come from the existing configuration and overrides.

For a nonzero first condition index, the following keeps two leading action
layers, one layer per gap, and one trailing layer (10 action layers total):

```bash
model.action_dit_pretrained_path=null \
'model.condition_layers=[5,12,24,28]' \
model.action_leading_noncondition_layers=2 \
model.action_noncondition_layers_per_interval=1 \
model.action_trailing_noncondition_layers=1
```

Here the action-to-video mapping is `[0,1,5,6,12,13,24,25,28,29]`, so video
condition index 5 corresponds to local action index 2 (the third action layer).
With leading=`null` (or 5), it would be the sixth action layer; with leading=0,
it is the first action layer. All video layers 0 through 5 still contribute
to the first fused K/V when interval fusion is enabled.

## Execution and initialization

The action model stores only retained blocks in a contiguous `ModuleList`.
For example, eight condition layers with one refinement per gap and a tail
of one give this mapping from local action index to video index:

```text
[0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29]
```

Local action condition indices are then `[0,2,4,6,8,10,12,14]`. Within the leading
region, each gap, and the tail, retained action-only blocks use the first available video slots. They
do not read those slots' video states; the placement defines a deterministic
mapping and which weights are retained when using a dense pretrained backbone.

Video execution, attention-mask indexing, KV-cache indexing, and KV-fusion
intervals all remain in video coordinates. Fusion still includes **every video
layer in its interval**, even where an action block has been removed. Cached
action inference executes only the retained action blocks.

Random initialization constructs the compact action model directly. When an
action pretrained path is supplied, the existing dense backbone is loaded and
its retained blocks are selected before LoRA injection and optimizer creation.
That path temporarily constructs a dense action model; it also requires the
backbone's condition-layer metadata to match the requested global conditions.
The existing offline backbone preprocessing/export format is unchanged.

## Checkpoints and evaluation

Use the same compact model configuration when loading a compact checkpoint
for evaluation, warm start, or full-state resume. Compact weight checkpoints
and trainer-state manifests include `action_layout` metadata; mismatched
video/condition/action mappings are rejected before loading weights or optimizer
state, even if tensor shapes happen to match. The default identity layout keeps
the existing checkpoint schema and can still load legacy dense checkpoints.
A dense video-only checkpoint remains a valid video-only warm start.

LoRA adapters use the compact local block names, and the LoRA merge script
preserves the layout metadata. Freeze policies still train action encoder/head,
and parameter grouping still assigns action/proprio and fusion their existing
learning rates. Changing the layout is a new architecture experiment, not a
compatible continuation of an existing full-state training checkpoint.
