# 离线融合 LoRA checkpoint

训练保存行为不变：`checkpoints/weights/step_XXXXXX.pt` 是一个完整文件，包含 base、未融合的 LoRA A/B、LoRA 配置，以及 KV-fusion 和可选 proprio encoder 权重。base 与 adapter 是同一 `mot` state dict 中的不同参数键，并非两个独立文件。

例如一个已注入 LoRA 的 Linear 保存为：

```text
mixtures.action.blocks.0.self_attn.q.base_layer.weight
mixtures.action.blocks.0.self_attn.q.base_layer.bias
mixtures.action.blocks.0.self_attn.q.lora_A.default.weight
mixtures.action.blocks.0.self_attn.q.lora_B.default.weight
```

融合后只留下该层的 `weight` 和原有 `bias`。新训练中的 `action_encoder`、`head` 本来就是全参训练的普通 Linear，它们的已训练权重直接保留。

## 使用方法

在仓库根目录、已激活的 Python 环境中执行。CentOS7 私有运行时沿用 `source .runtime/centos7/core/activate.sh`。

```bash
python scripts/merge_lora_checkpoint.py \
  --input runs/your_experiment/checkpoints/weights/step_001000.pt \
  --output runs/your_experiment/checkpoints/weights/step_001000_merged.pt
```

替换为实际 checkpoint 路径。输出父目录必须已存在；输出文件必须尚不存在。脚本不覆盖输入或已有输出，检查和保存失败时不会留下半成品目标文件。

脚本只依赖当前环境中的 PyTorch，在 CPU 上执行，不构建 DiT、不下载 Wan 权重、不加载数据集，也不需要 GPU 或 PEFT 运行时。它仍需要足够的主机内存和磁盘空间容纳完整模型权重。输入支持 PyTorch archive 的内存映射；非融合层的张量不复制，FP32 临时计算按层执行，不构建整份 FP32 模型。

输入必须是 FasterWAM 已汇总的完整 `mot` 权重 `.pt`，不能直接传 `checkpoints/state/step_XXXXXX/` 或 DeepSpeed 单 rank 分片。ZeRO-3 正常保存的 `checkpoints/weights/step_XXXXXX.pt` 可直接使用。缺少 base 的 adapter-only 文件、缺少 LoRA metadata 的文件和已经融合的 dense 文件会被拒绝。

## 融合规则与检查

对于本仓库支持的普通 Linear LoRA，逐层执行：

\[
W_{\mathrm{merged}} = W_{\mathrm{base}} + \frac{\alpha}{r} B A.
\]

`r` 与 `alpha` 来自 checkpoint 中各 expert 的配置，不从矩阵形状猜测 scaling。脚本支持 video/action 单独或同时开启 LoRA，也支持旧版本在 action 输入/输出层添加过 LoRA 的 checkpoint；旧增量会完整参与融合。

FP32、FP16、BF16 权重均在 CPU 上以 FP32 计算乘积、缩放和加法，最后转换回各层原 base dtype。公式与 PEFT 的普通 Linear LoRA 相同；PEFT 0.14 的 CPU 低精度 merge 会先将 delta 转回低精度再加 base，因此其舍入顺序与本脚本不同。低精度下不保证逐位一致，融合前后推理也可能有浮点舍入差异。比较输出时应使用 `eval()` 关闭 dropout，并采用相应 dtype 的数值容差。

预检覆盖 metadata 版本、expert 开关、目标层匹配、base/A/B 是否齐全、rank/shape、参数键冲突和非有限值；融合后再次检查 FP32 结果及转换后的权重，拒绝溢出。只支持本仓库的标准单 `default` adapter，不支持 DoRA、RS-LoRA、量化权重或 `modules_to_save` 等其它格式。没有模型架构配置时，无法判定某个非 adapter 层是否被整层删除，最终评测模型的严格加载仍负责检查完整架构。

输出保留 `mot`、可选 `proprio_encoder`、`step` 和 `torch_dtype`，移除 LoRA 配置和参数键、版本化 LoRA 标记，以及 optimizer 等训练状态。该文件是 dense 权重导出，不是完整训练状态。

官方依据：[PEFT 0.14 LoRA 配置](https://huggingface.co/docs/peft/v0.14.0/en/package_reference/lora)及 [Linear.get_delta_weight / merge 源码](https://github.com/huggingface/peft/blob/v0.14.0/src/peft/tuners/lora/layer.py)。

## 评测和后续训练

使用当前分支评测融合后的文件时，将两个 LoRA 开关都关闭，以构建普通 dense DiT。否则旧 dense 加载兼容逻辑会保留零增量 adapter，虽然可加载，但没有去除 adapter 的推理开销。例如沿用已有 RoboTwin 评测环境和数据统计文件：

```bash
TASK_NAME=robotwin_fasterwam_3cam_384_1e-4 \
CKPT_PATH=runs/your_experiment/checkpoints/weights/step_001000_merged.pt \
DATASET_STATS_PATH=/path/to/dataset_stats.json \
NUM_GPUS=8 \
bash scripts/eval_fasterwam_robotwin.sh \
  model.video_dit_config.lora.enabled=false \
  model.action_dit_config.lora.enabled=false
```

同一模型架构下，无 PEFT 的原版 dense 加载器也可读取该权重格式。数据预处理、统计文件、任务配置及推理设置仍须与实验一致。

精确继续原 LoRA 实验时，使用原未融合 checkpoint 或完整训练 state。若从融合后的 dense `.pt` 使用 `resume=...` 开始 LoRA 训练，代码将其作为新的 base，初始化零增量 adapter，并重新开始 optimizer、scheduler 和 step；不等价于恢复原 LoRA 优化过程。旧版 action I/O 带 adapter 的实验如需切换到新的 I/O 全参训练规则，可采用这一 warm start 路径。
