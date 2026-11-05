# Component learning rates

`configs/train.yaml` exposes three Hydra overrides. Their default values
interpolate the existing scalar `learning_rate`, so the default for each is
`1e-4`. Existing `learning_rate=...` overrides still set every component that
has not been explicitly overridden.

| Override | Trainable parameters | W&B chart |
| --- | --- | --- |
| `learning_rates.video` | Video DiT parameters, or its LoRA parameters when enabled | `train/lr_video` |
| `learning_rates.action` | Action DiT parameters, including its dense `action_encoder` and `head`, plus `proprio_encoder` | `train/lr_action` |
| `learning_rates.fusion` | SparseMoT `video_kv_fusion_logits` | `train/lr_fusion` |

The proprio encoder feeds shared video/action context. Assigning it to the
action learning-rate group is an explicit training policy; it remains a
separate model module and remains trainable. LoRA parameters use their own
branch's learning rate. Frozen parameters are excluded from the optimizer.

Only groups with trainable parameters appear in the optimizer and LR logs.
A fully frozen video DiT therefore has no `train/lr_video` chart or
`lr_video` training-log field. Freezing the action backbone still leaves its
encoder/head and the proprio encoder trainable, so `train/lr_action` remains.
Fusion parameters keep training. If fusion is disabled or absent, there is no
fusion LR metric. These rules use actual trainable parameters rather than the
raw freeze flag, preserving LoRA's precedence over freeze.

## Running an experiment

Keep the existing launcher, task, GPU count, and other experiment overrides;
append the component LR overrides. For example, with the LIBERO wrapper:

```bash
NPROC_PER_NODE=8 bash scripts/train_fasterwam_libero.sh \
  learning_rates.video=1e-4 \
  learning_rates.action=2e-4 \
  learning_rates.fusion=1e-4 \
  wandb.enabled=true
```

The same overrides work with `train_zero1.sh`, `train_zero2.sh`, and
`train_zero3.sh`. To isolate a random-action-initialization experiment, retain
your existing `model.action_dit_pretrained_path=null` and other initialization
overrides; changing a learning rate does not change model initialization.

At the existing `log_every` interval, the terminal and `train.log` report
`lr_video`, `lr_action`, and `lr_fusion` for active groups. W&B receives the
same current scheduled values under the chart names above; its console logs
also show the existing Python training messages. Logging frequency and other
metrics are unchanged.

## Schedule and checkpoints

Each value is the group's base learning rate. The existing 5% linear warmup
and cosine schedule apply the same relative schedule to every group, with a
minimum of 1% of that group's base LR. The constant schedule retains its
existing behavior. Setting all three LRs equal reproduces the previous LR
schedule; this feature does not alter epoch count, batch size, optimizer
betas, weight decay, gradient clipping, or the model's trainability policy.

- Fresh training and a weights-only `.pt` warm start use the configured
  component LRs. A `.pt` warm start creates a new optimizer and schedule.
- Full-state directory resume restores the saved optimizer LRs and scheduler
  state, rather than restarting the schedule from the new config values.
  The saved parameter-group layout must match the current trainable selection.
- Older full-state checkpoints contain one optimizer group. When the requested
  active component LRs are equal, resume retains that original single group
  and its scheduler state for compatibility. Unequal active LR requests are
  rejected for these checkpoints: use the corresponding weights-only `.pt`
  checkpoint to start a new optimizer with separate LRs.
