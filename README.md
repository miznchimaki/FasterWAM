<div align="center">

<h2><nobr>Faster-WAM: Efficient Inference-Time Future Conditioning</nobr><br><nobr>for Robust World Action Models</nobr></h2>

<b>Weiheng Zhao</b><sup>1</sup> &middot; <b>Haoyi Jiang</b><sup>1</sup> &middot; <b>Xin Shi</b><sup>2</sup> &middot; <b>Liu Liu</b><sup>3</sup> &middot; <b>Zhizhong Su</b><sup>3</sup> &middot; <b>Wei Sui</b><sup>2</sup> &middot; <b>Fan Huang</b><sup>4</sup> &middot; <b>Xinggang Wang</b><sup>1</sup>

Huazhong University of Science and Technology<sup>1</sup> &middot; D-Robotics<sup>2</sup> &middot; Horizon Robotics<sup>3</sup> &middot; Xiamen University<sup>4</sup>

<a href="https://arxiv.org/pdf/2608.04404"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper arXiv"></a> <a href="https://huggingface.co/hustvl/FasterWAM"><img src="https://img.shields.io/badge/Model-HuggingFace-orange" alt="Model HuggingFace"></a>

</div>

The key insight behind **Faster-WAM** is that future representations are not merely an auxiliary training signal, but essential inference-time context for robust action prediction under distribution shifts. Guided by this principle, Faster-WAM computes future representations once and selectively reuses them during action denoising, reducing redundant video-action interaction. It achieves state-of-the-art in-distribution performance and robust OOD generalization across simulated and real-world manipulation, while substantially reducing inference latency.

<div align="center">
  <img src="assets/framework_r.png" alt="Faster-WAM framework" width="85%">
</div>

---

## Table of Contents

- [Release Progress](#release-progress)
- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset Download](#dataset-download)
- [Training](#training)
- [Released Checkpoints](#released-checkpoints)
- [Evaluation](#evaluation)
- [Latency](#latency)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Release Progress

- Training and inference code. [✔]
- LIBERO, LIBERO-Plus, and RoboTwin evaluation code. [✔]
- Model checkpoints. [✔]

## File Structure

```text
FasterWAM/
├── configs/
│   ├── data/                 # LIBERO and RoboTwin dataset configs
│   ├── model/                # FastWAM, JointWAM, and FasterWAM models
│   ├── task/                 # Benchmark-specific training configs
│   ├── sim_libero.yaml       # LIBERO evaluation defaults
│   ├── sim_libero_plus.yaml  # LIBERO-Plus evaluation defaults
│   └── sim_robotwin.yaml     # RoboTwin evaluation defaults
├── environments/             # Independent benchmark uv projects and locks
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh
│   ├── preprocess_sparse_action_dit_backbone.py
│   ├── precompute_text_embeds.py
│   └── eval_fasterwam_*.sh
├── experiments/
│   ├── libero/               # LIBERO evaluation manager and worker
│   └── robotwin/             # RoboTwin evaluation manager and policy adapter
├── src/fasterwam/            # Core model, dataset, and training code
├── third_party/RoboTwin/     # RoboTwin evaluation integration
├── checkpoints/              # Wan components and model checkpoints
├── data/                     # Preprocessed training datasets
├── runs/                     # Training outputs
└── evaluate_results/         # Evaluation outputs
```

The final system is `FasterWAM`. The repository also keeps the `FastWAM` and
`JointWAM` baselines for controlled comparisons.

## Environment Setup

Run all commands below from the repository root.

Install [uv](https://docs.astral.sh/uv/) first. The core training environment
uses Python 3.10 and the PyTorch 2.7.1 CUDA 12.8 wheels locked in `uv.lock`:

```bash
bash scripts/setup/install_core.sh
source .venv/bin/activate
```

## Model Preparation

FasterWAM uses Wan2.2-TI2V-5B. By default, missing components are downloaded
from Hugging Face and stored under `./checkpoints`. Set the directory explicitly
before model preparation, training, or evaluation:

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

To use ModelScope instead, additionally set:

```bash
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
```

### FasterWAM ActionDiT initialization

Before training FasterWAM from scratch, generate its SparseActionDiT
initialization from the Wan2.2 video DiT:

```bash
python scripts/preprocess_sparse_action_dit_backbone.py \
  --model-config configs/model/fasterwam.yaml \
  --output checkpoints/SparseActionDiT_cond_0_4_8_12_16_20_24_28_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

### Baseline ActionDiT initialization

FastWAM and JointWAM use the dense ActionDiT initialization:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## Dataset Download

### LIBERO

FasterWAM uses the same preprocessed MuJoCo 3.3.2 LIBERO dataset as FastWAM:

- [yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)

Download the four archives and extract them under `data/libero_mujoco3.3.2`:

```bash
mkdir -p data/libero_mujoco3.3.2

huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --local-dir data/libero_mujoco3.3.2

cd data/libero_mujoco3.3.2
for f in *.tar.gz; do tar -xzf "$f"; done
cd ../..
```

The resulting layout must be:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

The preprocessed RoboTwin dataset is available from:

- [yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

Download all split archives, concatenate them, and extract them as described in
the FastWAM release:

```bash
mkdir -p data/robotwin2.0

huggingface-cli download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --local-dir data/robotwin2.0

cd data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
```

The expected layout is:

```text
data/robotwin2.0/
├── dataset_stats.json
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

## Training

### Precompute instruction embeddings

Training reads cached T5 instruction embeddings. Generate them once after the
dataset has been extracted:

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_fasterwam_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_fasterwam_3cam_384_1e-4
```

For multi-GPU preprocessing:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py \
  task=libero_fasterwam_2cam224_1e-4
```

The caches are written to `data/text_embeds_cache_fasterwam/libero` and
`data/text_embeds_cache_fasterwam/robotwin`.

### Launch training

The FasterWAM model config defaults to **action backbone LoRA + full video DiT
training**. The action expert's `action_encoder` and `head` always use dense,
full-parameter training; custom LoRA targets cannot select them. For the original full-parameter baseline, pass
`model.action_dit_config.lora.enabled=false`. See
[PEFT LoRA training](docs/lora_training.md) for installation, per-expert
overrides, freezing, and checkpoint compatibility.
Training checkpoints keep base and adapter weights unmerged. To export a
separate dense checkpoint for evaluation, use the
[offline LoRA merge script](docs/merge_lora_checkpoint.md).

```bash
NPROC_PER_NODE=8 bash scripts/train_fasterwam_libero.sh

NPROC_PER_NODE=8 bash scripts/train_fasterwam_robotwin.sh
```

Both wrappers accept additional Hydra overrides. For example:

```bash
NPROC_PER_NODE=8 bash scripts/train_fasterwam_libero.sh \
  batch_size=8 \
  num_epochs=1 \
  wandb.enabled=true
```

The LIBERO/RoboTwin wrappers and `train_zero1.sh` / `train_zero2.sh` / `train_zero3.sh` save the
complete launcher and local worker stdout/stderr to `<output_dir>/train.log`,
while also displaying it in the terminal. This includes training metrics,
`print` output, progress bars, library warnings, and error tracebacks. Python
output is unbuffered, and a training failure still returns a nonzero exit code.
For example:

```text
runs/libero_fasterwam_2cam224_1e-4/2026-09-24_14-43-18/train.log
```

An `output_dir=...` override changes both the experiment directory and the log
location. Restarting with the same output directory appends to the existing log.
You do not need to add a separate `tee train.log` or `> train.log 2>&1`:

```bash
NPROC_PER_NODE=4 bash scripts/train_fasterwam_robotwin.sh \
  gradient_accumulation_steps=2 \
  output_dir=./runs/robotwin/my_experiment
```

On multiple nodes, node 0 writes `train.log`; additional nodes write
`train.node<N>.log` in the same experiment directory, each containing all local
workers. Direct `python scripts/train.py` invocations place Hydra's Python
logging file in `output_dir`; use the launch scripts above for the complete
stdout/stderr transcript. Existing logs and already-running jobs are unchanged.
Each launch script invocation runs one experiment; start separate invocations
for sweeps instead of passing Hydra's `--multirun` / `-m` option.

### ZeRO-3 on four GPUs

`scripts/train_zero3.sh` uses the same Hydra overrides and logging as the
existing ZeRO-1/2 launchers. Its configuration is in
`scripts/accelerate_configs/accelerate_zero3_ds.yaml` and
`scripts/ds_configs/ds_zero3_config.json`. Run it from the repository root in
the existing training environment; no additional package versions are needed.
The integration targets the project's pinned Accelerate **1.12.0** and
DeepSpeed **0.18.5**.

For RoboTwin on four A800s, this example preserves the paper's global batch
size of 1,024 and the task's default five epochs:

```bash
bash scripts/train_zero3.sh 4 \
  task=robotwin_fasterwam_3cam_384_1e-4 \
  model.action_dit_config.lora.enabled=false \
  batch_size=8 \
  gradient_accumulation_steps=32 \
  mixed_precision=bf16 \
  eval_every=0
```

`batch_size` is **per GPU**: global batch size is
`number_of_GPUs × batch_size × gradient_accumulation_steps`. For example,
four GPUs with the task's default `batch_size=16` and accumulation of 2 use
a global batch of 128. Adjust microbatch size to available memory and adjust
accumulation accordingly; the example above is not a measured A800 memory or
throughput result.

The new configuration shards parameters, gradients, and optimizer state and
does **not** offload parameters or the optimizer to CPU. It uses explicit
bucket sizes because this custom model does not expose the Transformers
`config.hidden_size` required by Accelerate's hidden-size-based `auto` values.
`zero3_init_flag: false` preserves the existing custom pretrained-weight
loading path: sharding begins at `Accelerator.prepare`, so each GPU must
still fit the initial model before that point. ZeRO-3 also does not remove
the need to fit activations for the selected microbatch.

ZeRO-3 enables communication overlap. The supplied eight-GPU logs show that
the previous configuration was slower than ZeRO-2 at the same microbatch;
sharding saves memory and does not guarantee greater throughput. See the
[measured comparison and controlled overlap A/B procedure](docs/zero3_performance.md).

Checkpoint behavior:

- All ranks participate in ZeRO-3 weight consolidation. Rank 0 writes the
  ordinary `checkpoints/weights/step_XXXXXX.pt` with the existing `mot` and
  optional `proprio_encoder` payload. LoRA runs additionally include adapter
  weights and architecture metadata, loaded by this branch's evaluation code.
- `resume=/path/to/checkpoints/state/step_XXXXXX` restores the distributed
  training state, including optimizer, scheduler, and recorded data progress.
  Keep the same ZeRO stage and GPU topology when resuming this state.
- `resume=/path/to/checkpoints/weights/step_XXXXXX.pt` is a weights-only warm
  start loaded **before** optimizer creation and partitioning. It starts a
  fresh optimizer, scheduler, and step count; use this form to switch stages.
- ZeRO-3 currently requires `eval_every=0` (already the RoboTwin and LIBERO
  task default). Run the existing separate evaluation scripts on the exported
  `.pt`; the trainer's unwrapped multi-step video rollout is not supported
  with partitioned parameters.

Before a full run, exercise training and both checkpoint formats with a small
hardware smoke run, then resume its state with the same batch/accumulation
settings and a larger `max_steps`:

```bash
bash scripts/train_zero3.sh 4 \
  task=robotwin_fasterwam_3cam_384_1e-4 \
  batch_size=1 gradient_accumulation_steps=2 \
  max_steps=2 save_every=1 eval_every=0 \
  output_dir=./runs/robotwin_zero3_smoke
```

The shared trainer also fixes three correctness issues affecting ZeRO-1/2:
training now enters the prepared model's `forward` so DeepSpeed's backward
and accumulation hooks run; `max_grad_norm` is passed to DeepSpeed's actual
gradient-clipping configuration; and weights-only resumes precede optimizer
master-weight initialization. Existing ZeRO-1/2 configuration files and
launch scripts retain their settings.

Implementation references: the versioned
[Accelerate preparation and checkpoint code](https://github.com/huggingface/accelerate/blob/v1.12.0/src/accelerate/accelerator.py),
[DeepSpeed forward/backward lifecycle](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.5/deepspeed/runtime/engine.py),
and [ZeRO-3 configuration schema](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.5/deepspeed/runtime/zero/config.py).

## Released Checkpoints

The released FasterWAM checkpoints and their corresponding dataset statistics
are available on [Hugging Face](https://huggingface.co/hustvl/FasterWAM).

```bash
mkdir -p checkpoints/fasterwam_release

huggingface-cli download hustvl/FasterWAM \
  --local-dir checkpoints/fasterwam_release
```

After downloading, the checkpoint directory should have the following layout:

```text
checkpoints/fasterwam_release/
├── libero/
│   ├── step_021700.pt
│   └── dataset_stats.json
└── robotwin/
    ├── step_029355.pt
    └── dataset_stats.json
```

## Evaluation

LIBERO, LIBERO-Plus, and RoboTwin are managed in three separate uv environments
to isolate their simulator dependencies. Run the corresponding setup command
once before evaluation; each evaluation launcher automatically uses the matching
environment.

### LIBERO

```bash
bash scripts/setup/install_libero.sh

TASK_NAME=libero_fasterwam_2cam224_1e-4 \
CKPT_PATH=checkpoints/fasterwam_release/libero/step_021700.pt \
DATASET_STATS_PATH=checkpoints/fasterwam_release/libero/dataset_stats.json \
NUM_GPUS=8 \
bash scripts/eval_fasterwam_libero.sh
```

### LIBERO-Plus

```bash
bash scripts/setup/install_libero_plus.sh

TASK_NAME=libero_fasterwam_2cam224_1e-4 \
CKPT_PATH=checkpoints/fasterwam_release/libero/step_021700.pt \
DATASET_STATS_PATH=checkpoints/fasterwam_release/libero/dataset_stats.json \
NUM_GPUS=8 \
bash scripts/eval_fasterwam_libero_plus.sh
```

### RoboTwin

```bash
bash scripts/setup/install_robotwin.sh

TASK_NAME=robotwin_fasterwam_3cam_384_1e-4 \
CKPT_PATH=checkpoints/fasterwam_release/robotwin/step_029355.pt \
DATASET_STATS_PATH=checkpoints/fasterwam_release/robotwin/dataset_stats.json \
NUM_GPUS=8 \
bash scripts/eval_fasterwam_robotwin.sh
```

## Latency

To measure the Inference Latency:

```bash
# All models
bash scripts/measure_latency.sh

# Selected models: jointwam, fastwam, fasterwam
bash scripts/measure_latency.sh --models jointwam fasterwam
```

## Acknowledgments

Our codebase is built upon:

- FastWAM: https://github.com/yuantianyuan01/FastWAM
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus
- RoboTwin: https://github.com/RoboTwin-Platform/RoboTwin

We thank these teams for contributing their impressive code and models to the
community.

## Citation

If you find this repository helpful for your research, please consider citing
our paper:

```bibtex
@article{zhao2026faster,
  title   = {Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models},
  author  = {Zhao, Weiheng and Jiang, Haoyi and Shi, Xin and Liu, Liu and Huang, Fan and Su, Zhizhong and Sui, Wei and Wang, Xinggang},
  journal = {arXiv preprint arXiv:2608.04404},
  year    = {2026}
}
```
