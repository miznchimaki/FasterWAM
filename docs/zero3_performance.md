# ZeRO-3 吞吐与对照测试

ZeRO-3 通过分片模型参数节省显存，使用参数前需要收集分片。ZeRO-2 保留每卡完整参数，因此两者在相同 batch size 下不保证相同吞吐。本仓库现在将 ZeRO-3 的 `overlap_comm` 从 `false` 改为 `true`，让通信有机会与计算重叠；其它桶大小、参数驻留和 offload 配置保持不变。**尚未测得此改动的提速幅度，也不保证追平 ZeRO-2。**

## 已有日志能说明什么

用户提供的 `train_zero2.log` 和 `train_zero3.log` 来自 2026-09-30 的 RoboTwin 训练。日志显示均为 8 个进程、BF16、梯度累积 1；对应任务 `robotwin_fasterwam_3cam_384_1e-4` 每卡 batch size 为 16，全局 batch size 为 128。其中 ZeRO-3 测试使用的是修改前的 `overlap_comm=false`。

为减小启动阶段的影响，取相同的 step 100 到 step 500 窗口，用日志时间戳计算经过的 400 个 optimizer steps：

| 项目 | ZeRO-2 | ZeRO-3，overlap=false |
| --- | ---: | ---: |
| 窗口耗时 | 5,518 秒 | 7,238 秒 |
| 平均每步耗时 | 13.795 秒 | 18.095 秒 |
| 按全局 batch 128 换算吞吐 | 9.279 samples/s | 7.074 samples/s |

这个窗口中，ZeRO-3 每步约多 4.300 秒，耗时增加约 31.2%，吞吐下降约 23.8%。时间戳精确到秒，因此这些是窗口平均值。日志中的 `speed` 则从训练循环启动时累计计算，包含最初较慢的步骤，不能直接当作最近十步的速度。

该任务配置关闭 activation checkpointing、训练内评测和周期性保存；在没有额外命令行覆盖的前提下，不能用这些操作解释上述差距。日志中 `After initializing ZeRO optimizer` 的显存读数分别约为 14.27 GB 和 5.44 GB；**这是初始化阶段的已分配显存，不是完整训练步骤的峰值显存。**

这些日志没有给出计算、NCCL 通信、数据等待的分项耗时，不能据此认定某一项贡献了全部差距，也不能证明两个实验的运行环境完全一致。

## 当前实现与可能的开销

- 训练通过 prepared `self.model(sample)` 进入 DeepSpeed。MoT 的 Q/K/V/O、cross attention 和 FFN 调用正常子模块，保留 ZeRO-3 的参数调度 hooks；专家的重复注册路径已在 prepare 前清理。
- 冻结的 VAE 仍在模型注册树中。DeepSpeed 0.18.5 会转换整个模型的参数，不按 `requires_grad` 排除冻结参数，所以 `no_grad` 编码仍可能触发参数收集。视频编码逐样本、逐时间块调用 encoder；实际通信次数还取决于参数缓存和复用。不能把它描述成每次都收集整个 VAE，未执行的 decoder 分支也不等于每步通信量。
- `stage3_param_persistence_threshold=100000` 使小于阈值的参数可保持完整驻留。MoT 的单个 modulation 和 fusion logits 都低于此阈值，因此不能仅凭直接访问这些参数，就推断存在逐层小 collective 的性能问题。
- 当前 reduce 和 prefetch 桶都是 50,000,000 **元素**，不是 50 MB。prefetch 50M 是该 DeepSpeed 版本的默认值。更大的桶可能减少通信次数，也可能增加峰值显存、改变重叠时机；此次不同时调整这些参数。

`overlap_comm=true` 与 DeepSpeed 0.18.5 未显式指定时的 Stage 3 默认值一致。在 GPU 上，DeepSpeed 可为参数收集和梯度通信使用独立 stream。通信与计算并行时，可能有更多参数或通信缓冲区同时存活，因此需要实测峰值显存；如果出现 OOM 或吞吐回退，可以用下面的副本恢复 `false` 做对照。把冻结 VAE 移出分片管理范围也是另一种显存换吞吐的方案，不包含在这次改动中。

官方源码依据：

- [ZeRO-3 默认 overlap 与 prefetch 配置](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.5/deepspeed/runtime/zero/config.py)：`overlap_comm_valid`、`prefetch_bucket_size`。
- [参数转换与 all-gather stream](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.5/deepspeed/runtime/zero/parameter_offload.py)：`_convert_to_zero_parameters`、`__allgather_stream`。
- [参数收集、释放与缓存复用](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.5/deepspeed/runtime/zero/partitioned_param_coordinator.py)：`fetch_sub_module`、`release_sub_module`、`__params_to_release`。
- [Stage 3 梯度通信 stream](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.5/deepspeed/runtime/zero/stage3.py)：`reduce_and_partition_stream`。

## 单机 8 卡 A/B 测试

在同一个 commit、同一组 GPU 和相同数据上比较，仅改变 `overlap_comm`。若此版本包含 LoRA 配置，**两次运行都必须设置 `model.action_dit_config.lora.enabled=false`**，保持与上述日志一致的 action 全参数微调；否则测到的同时包含训练参数范围变化。尚无 LoRA 配置的旧版本本来就是全参数微调，应省略这个覆盖项。

以下操作在仓库根目录、已激活训练环境后执行。复制配置到独立目录，不修改仓库里的训练配置：

```bash
python - <<'PY'
import copy
import json
from pathlib import Path
import yaml

root = Path("runs/zero3_overlap_ab").resolve()
root.mkdir(parents=True, exist_ok=True)
ds_base = json.loads(Path("scripts/ds_configs/ds_zero3_config.json").read_text())
accelerate_base = yaml.safe_load(Path("scripts/accelerate_configs/accelerate_zero3_ds.yaml").read_text())
for name, overlap in (("off", False), ("on", True)):
    ds_config = copy.deepcopy(ds_base)
    ds_config["zero_optimization"]["overlap_comm"] = overlap
    ds_path = root / f"deepspeed_{name}.json"
    ds_path.write_text(json.dumps(ds_config, indent=2) + "\n")
    accelerate_config = copy.deepcopy(accelerate_base)
    accelerate_config["deepspeed_config"]["deepspeed_config_file"] = str(ds_path)
    (root / f"accelerate_{name}.yaml").write_text(yaml.safe_dump(accelerate_config))
PY
```

使用已有的 Accelerate 参数启动，并将所有本地进程的 stdout/stderr 收入各自日志。这里固定 200 步，最终仍会按现有训练逻辑保存 checkpoint；比较 step 100 到 190 的时间，避开启动和最后的保存。

```bash
set -euo pipefail

# 包含 LoRA 配置的版本使用这一行：
FULL_FINETUNE_ARGS=(model.action_dit_config.lora.enabled=false)
# 尚无 LoRA 配置的旧版本，将上一行替换为：
# FULL_FINETUNE_ARGS=()

for mode in off on; do
  run_dir="runs/zero3_overlap_ab/${mode}"
  mkdir -p "${run_dir}"
  PYTHONUNBUFFERED=1 FASTERWAM_LOG_CAPTURE=1 \
    accelerate launch \
      --config_file "runs/zero3_overlap_ab/accelerate_${mode}.yaml" \
      --num_processes 8 --num_machines 1 --machine_rank 0 \
      --main_process_ip 127.0.0.1 --main_process_port 29500 \
      --deepspeed_multinode_launcher standard \
      scripts/train.py task=robotwin_fasterwam_3cam_384_1e-4 \
      "${FULL_FINETUNE_ARGS[@]}" max_steps=200 save_every=0 eval_every=0 \
      wandb.enabled=false "output_dir=${run_dir}" hydra/job_logging=stdout \
      2>&1 | tee "${run_dir}/train.log"
done
```

该示例是待在训练机器运行的测试方法，不是已完成的 GPU benchmark。首次使用空的测试目录；重复实验换一个目录，避免覆盖日志和最终 checkpoint。最好交换 on/off 顺序再重复一次。用相同窗口比较端到端耗时，同时观察运行期间的显存，而不是只看初始化打印；如需定位差距，再分别测量 VAE encode、DiT forward、backward/optimizer 和 NCCL 通信。接受新默认值的依据应是实际稳定性、吞吐与显存结果。
