# FasterWAM PEFT LoRA 实验

使用 `peft==0.14.0` 的 `inject_adapter_in_model`，在原生 video/action DiT 内替换选定的 Linear。不会给专家套上改变 `blocks`、`pre_dit`、`post_dit` 路径的 PeftModel 外层。先加载 Wan 视频权重及预处理好的 SparseActionDiT dense backbone，完成原模型初始化，再注入 adapter。

## 依赖与开关

`pyproject.toml` 和 `uv.lock` 固定 PEFT 0.14.0，保留其它依赖版本。已在 Python 3.10、Torch 2.7.1 CPU、Transformers 4.49.0、Hub 0.29.2、Accelerate 1.12.0 的组合中执行实际 PEFT 测试。尚未执行 GPU/DeepSpeed 多卡 LoRA benchmark。

已有训练环境中可只安装这一新增依赖，避免升级现有 Torch 等包。CentOS7 私有运行时先按原有方式激活 `source .runtime/centos7/core/activate.sh`，再运行：

```bash
uv pip install --python "$(command -v python)" --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple peft==0.14.0
python -c 'import peft, transformers, torch; print(peft.__version__, transformers.__version__, torch.__version__)'
```

配置就在 `configs/model/fasterwam.yaml` 的 `video_dit_config.lora` 和 `action_dit_config.lora` 下：

| 配置 | video 默认值 | action 默认值 |
| --- | --- | --- |
| `enabled` | `false`，保留全参训练 | `true`，只训练该专家的 LoRA A/B |
| `r` | 16 | 16 |
| `lora_alpha` | 16 | 16 |
| `lora_dropout` | 0.0 | 0.0 |
| `target_modules` | self/cross attention 的 q/k/v/o、FFN 的两个 Linear | 同左，另加 `action_encoder`、`head` |

`enabled=false` 的含义是这个专家不使用 LoRA，按原来方式全参训练，**不是冻结整个专家**。开启时 base 的所有原始参数（包括 bias、norm 和 modulation）冻结，只训练 LoRA A/B；固定使用 `bias="none"`、普通 LoRA、identity 初始化，不使用 DoRA、RS-LoRA 或 `modules_to_save`。标准初始化的 B 为零，因此注入本身不改变初始模型函数。

Action encoder/head 原本随机初始化，并不来自 Wan 的初始化 backbone。它们也加入 LoRA，使其低秩增量可以学习；这些层的随机 base 仍然冻结。是否能达到全参微调的效果需要实验。MoT 的 KV-fusion 参数和独立 proprio encoder 不属于两个 DiT 内部的 base，继续按原训练策略学习。Trainer 在每次恢复训练模式时重新施加冻结规则，优化器只接收 `requires_grad=True` 的参数。

Adapter dtype 跟随 base。PEFT Linear 保持输出 dtype；DeepSpeed BF16 初始化也会转换模型 dtype。本实现不在 ZeRO 分片后强行转换 adapter 参数。

## 启动示例

在仓库根目录运行。下例与用户已有八卡日志一样使用每卡 batch 16、梯度累积 1，即全局 batch 128；这不是论文 RoboTwin 的全局 batch 1,024。需要保持论文 batch 时调整每卡 batch 与累积次数，使其乘以 GPU 数得到 1,024。

```bash
# 默认：video 全参训练，action LoRA
bash scripts/train_zero2.sh 8 \
  task=robotwin_fasterwam_3cam_384_1e-4 \
  batch_size=16 gradient_accumulation_steps=1

# 两个专家都使用 LoRA
bash scripts/train_zero2.sh 8 \
  task=robotwin_fasterwam_3cam_384_1e-4 \
  model.video_dit_config.lora.enabled=true \
  model.action_dit_config.lora.enabled=true \
  model.video_dit_config.lora.r=16 \
  model.video_dit_config.lora.lora_alpha=16 \
  model.action_dit_config.lora.r=32 \
  model.action_dit_config.lora.lora_alpha=32

# 原 full-parameter baseline，用于与原实验比较
bash scripts/train_zero2.sh 8 \
  task=robotwin_fasterwam_3cam_384_1e-4 \
  model.video_dit_config.lora.enabled=false \
  model.action_dit_config.lora.enabled=false
```

选择 ZeRO-3 时，把脚本换成 `train_zero3.sh` 并保持 `eval_every=0`。日志仍写入每个实验的 `output_dir/train.log`。日志还会打印两个专家各自的 LoRA 开关、可训练和总参数数目。

自定义目标层使用模块名后缀列表，例如：

```bash
model.action_dit_config.lora.target_modules='[self_attn.q,self_attn.k,self_attn.v,self_attn.o,action_encoder,head]'
```

PEFT 0.14.0 的 `all-linear` 简写不适用于这两个原生 `nn.Module`；本实现会明确拒绝字符串、空列表、拼错的目标或非 Linear 目标。两个 backbone 预处理脚本会剥离 `lora` 配置，继续生成 dense 初始化权重，不必重新生成已有 backbone。

## 参数量与训练成本

根据当前完整 RoboTwin 配置在 meta device 上构建并统计（不是显存实测）：

| 专家 | 原 base 参数 | 默认 rank 16 的 adapter 参数 |
| --- | ---: | ---: |
| video | 4,999,787,712 | 40,304,640 |
| action | 558,896,142 | 12,026,304 |

默认只对 action 开启 LoRA 时，约 50 亿 video 参数仍参与全参训练。两个专家都开启时，它们合计约 5,233 万 adapter 参数参与训练，另有 fusion/proprio 参数。冻结 base 可减少参数梯度和优化器状态，但 base 权重仍存在，前向计算和许多中间激活也仍需要；训练吞吐不会按可训练参数的缩小比例提升。

## 保存、恢复和评测

普通 `.pt` 仍保留 `mot`、可选 `proprio_encoder`、step/dtype。存在 LoRA 时同时保存**完整 base + adapter 权重**和版本化 LoRA 配置。它不是 adapter-only 文件；随机 action 输入/输出 base 也在文件中，不依赖重新随机初始化恢复。ZeRO-3 使用所有 rank 汇总后的完整 state，保存路径与 ZeRO-1/2 使用同一 metadata 规则。

- **训练恢复完整状态**：`resume=/path/to/checkpoints/state/step_XXXXXX`。保持相同 ZeRO stage、GPU 拓扑、batch/累积及 LoRA 配置。Trainer 在 prepare 前检查 LoRA enabled/rank/alpha/dropout/targets，以及可训练参数名称、形状和优化器顺序。即使改 alpha 不改变张量形状，也会被拒绝。
- **从 LoRA `.pt` 开始新训练**：`resume=/path/to/checkpoints/weights/step_XXXXXX.pt`。使用与 checkpoint 相同的 LoRA 设置；base 和 adapter 都恢复，optimizer/scheduler/step 重新开始。配置不匹配会报错，不会静默忽略 adapter。
- **从旧 dense `.pt` 开始 LoRA 训练**：保持所需的 LoRA 配置，传入旧 `.pt`。加载器将选定层的 dense 键映射到 `base_layer`，加载全部 base 并将 adapter 重置为零增量；只允许新增 adapter 参数不在旧文件中。旧完整 state 不能直接转换为 LoRA 优化器状态，改用 `.pt` warm start。
- **评测 LoRA `.pt`**：使用本分支的现有评测入口。默认 `load_checkpoint` 根据文件中的 metadata 配置两个专家，再严格加载全部权重；可恢复与 YAML 默认值不同的 rank/targets。缺失 metadata 的 LoRA 权重或不完整 base 会报错。旧版本代码不知道 PEFT 参数键，不能用它加载新 LoRA checkpoint。
- **评测旧 dense `.pt`**：仍可加载。若要保持原始 dense 推理结构和延迟，关闭两个 LoRA 开关；启用状态下加载旧 dense 权重会保留零增量 adapter。

训练 checkpoint 中 adapter 不会被原地 merge/unload。该实现也不提供合并后的 dense 导出；需要精确继续 LoRA 训练时使用上述完整 checkpoint。

官方实现依据：[PEFT 0.14 低层注入 API](https://huggingface.co/docs/peft/v0.14.0/en/developer_guides/low_level_api)、[注入源码](https://github.com/huggingface/peft/blob/v0.14.0/src/peft/mapping.py)、[LoRA Linear/初始化/dtype](https://github.com/huggingface/peft/blob/v0.14.0/src/peft/tuners/lora/layer.py)。
