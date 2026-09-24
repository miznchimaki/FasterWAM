# FasterWAM：CentOS 7 + 私有 glibc + uv

适用于 `miznchimaki/FasterWAM`，核对的仓库提交为
`83667817df0d4f823f39d90700e61ea2f432ac45`。新增适配脚本；原仓库源码、
`pyproject.toml`、四份 `uv.lock`、原安装脚本均未修改。没有执行 git add、commit 或 push。

## 1. 这套方案做什么

沿用此前 `lerobot310` 的成功原则：**对真实 Python ELF 设置 PT_INTERP 和
DT_RPATH，而不是只在 shell wrapper 中临时运行 loader。**

1. 从现有可运行的 Python 3.10 ELF 复制一个 FasterWAM 专用基底。
2. 在 `.runtime/centos7/<profile>/python-base` 复制标准库，排除旧 site-packages。
3. `patchelf --set-interpreter` 指向你的 glibc，`--force-rpath --set-rpath`
   指定私有库路径。原 Python 和现有 wrappers 不变。
4. 显式 `uv venv --python <私有基底>`，再 `uv sync --locked --python <私有基底>`。
   uv 自动检测该解释器的 glibc；不伪造 manylinux 平台，也不下载另一套 Python。
5. 保留 build isolation。核心安装阶段设 `DS_BUILD_OPS=0`，避免意外预编译
   DeepSpeed 扩展；这不意味着任意后续 CUDA JIT 都无需编译器。
6. 修补虚拟环境中带 PT_INTERP 的原生可执行文件，例如 torch_shm_manager、
   ptxas、wandb-core、TensorBoard server。保留原有 `$ORIGIN`，采用 copy-on-write，
   防止影响 uv 缓存或其他环境。
7. 运行诊断；没有 GPU 时明确显示跳过 GPU 检查，而不是声称 CUDA 已验证。

不向 LD_LIBRARY_PATH 导出新 glibc；生成的启动器清理 LD_LIBRARY_PATH、
LD_PRELOAD、PYTHONHOME、PYTHONPATH。系统 bash/git 等子进程继续使用系统运行时。

**仍依赖原 Conda prefix 中的 libpython、SSL/FFI 等动态库及头文件**；这不是一个
可以脱离原 prefix 单独搬走的 Python 发行版。不要删除/移动原 lerobot310、私有
glibc 或 FasterWAM 仓库。新虚拟环境不会继承原环境的 Python 包。

## 2. 放置文件与配置

把压缩包解压到现有 FasterWAM 仓库根目录；压缩包只包含本文档和新增脚本。

```bash
cd /你的路径/FasterWAM
tar -xzf /你的下载路径/fasterwam-centos7-setup.tar.gz

# 在当前 shell 清理曾经为 glibc wrapper 设置的全局变量。
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH

mkdir -p .runtime
cp scripts/setup/centos7/config.example.sh .runtime/centos7.env
```

编辑 `.runtime/centos7.env`。对你目前的服务器，默认路径应对应：

```bash
FASTERWAM_BASE_PREFIX="/mnt/lustre/lizongshu/depends/anaconda3/envs/lerobot310"
FASTERWAM_BASE_PYTHON="${FASTERWAM_BASE_PREFIX}/bin/python3.10-glibc235"
FASTERWAM_GLIBC_ROOT="/mnt/lustre/lizongshu/depends/glibc-2.35"
FASTERWAM_FFMPEG_ROOT="/mnt/lustre/lizongshu/depends/ffmpeg-7.1.1"
FASTERWAM_DRIVER_LIBDIR="/mnt/lustre/lizongshu/depends/nvidia-driver-libs"
FASTERWAM_GCC_ROOT="/mnt/lustre/lizongshu/depends/gcc-12.1.0"
```

`FASTERWAM_BASE_PYTHON` 必须是可运行的 **ELF 文件**，不是 `lerobot310-python`
这个 shell wrapper。原始 Conda Python 3.10 若能在干净 shell 中运行，也可用作源。

`patchelf` 优先从 PATH 查找，然后尝试 `$HOME/depends/local/bin/patchelf`；如不在
这些位置，设置 `FASTERWAM_PATCHELF`。它和 uv、编译器都必须能在不依赖全局
LD_LIBRARY_PATH 的情况下启动。

uv 优先使用现有可执行文件，也可通过 `FASTERWAM_UV_BIN` 指定。没有 uv 时，
脚本用旧 Python 的 pip 把 `uv==0.12.15` 安装到仓库独立工具目录，**不会向旧
Conda 环境安装包**。此 bootstrap 默认用清华源；项目依赖仍按锁文件中的
PyPI/PyTorch 下载 URL 获取，不能只换一个 pip 镜像参数就改写锁内 URL。

如果 uv 版本太旧而不识别锁格式或配置，可手动更新独立工具目录：

```bash
lerobot310-python -m pip install --upgrade --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --target "$PWD/.runtime/centos7/uv-tools" 'uv==0.12.15'
```

然后将 `FASTERWAM_UV_BIN` 设为该目录下实际生成的 `bin/uv` 或 `uv/uv`。

脚本优先从 Conda lib 选择含 GLIBCXX_3.4.30 的 libstdc++，否则尝试 GCC 12
的 lib64/lib；也可以指定 `FASTERWAM_CXX_LIBDIR`。只在私有目录链接选定的
libstdc++/libgcc，不把整套 GCC/Make 库全局加入运行时。

## 3. 安装核心训练环境

建议先跑轻量检查，不下载项目依赖：

```bash
bash scripts/setup/install_centos7.sh core --prepare-only
```

成功后正式安装：

```bash
set -o pipefail
bash scripts/setup/install_centos7.sh core 2>&1 | tee setup-centos7-core.log
source .runtime/centos7/core/activate.sh
```

`--prepare-only` 可跳过，但它能在下载大体积 PyTorch wheels 前发现路径、
libpython、glibc 或 Python 子进程问题。两次执行是可重复的，不会删除环境。

已有 `.venv` 且不是本适配脚本创建时，脚本主动退出。若要保留后重建，请自行
重命名，例如 `mv .venv .venv.before-centos7`。脚本不执行 rm -rf。
如变更 Python 基底、glibc prefix 或仓库位置，应同时重命名对应 `.venv`
和 `.runtime/centos7/core` 后重建，不能混用旧状态。

核心依赖保持仓库锁定版本：Python 3.10、torch 2.7.1+cu128、
torchvision 0.22.1+cu128、torchcodec 0.5+cu128、DeepSpeed 0.18.5。
不额外安装 flash-attn，不把之前 lerobot 环境的 Python 包复制进来。

额外处理了原 RoboTwin lock 的 PyPI 地址差异：锁中是 `https://pypi.org/simple/`，
而 pyproject 中末尾没有 `/`，会使本次核验的 uv 报 `--locked` 失败。适配器通过
`UV_INDEX` 选择与原锁完全相同的地址拼写，不更新依赖版本或重新生成锁文件。

## 4. 正常使用与验收

以后进入仓库先运行：

```bash
source .runtime/centos7/core/activate.sh
python -c 'import sys,os,torch; print(sys.executable); print(os.confstr("CS_GNU_LIBC_VERSION")); print(torch.__version__)'
```

输出应指向本仓库 `.venv/bin/python`，glibc 为 2.35（或你配置的更高版本），
torch 为 `2.7.1+cu128`。使用这里的专用 activate 比只 source `.venv/bin/activate`
多了干净运行时、uv 路由和私有 FFmpeg 的配置。

在实际 GPU 节点执行完整核心验收，用已下载数据中的一个真实视频路径：

```bash
python scripts/setup/centos7/doctor.py --profile core \
  --require-cuda --video /你的数据目录/某个真实视频.mp4
```

它检查解释器/实际 libc/ELF、Python 子进程、spawn、锁定包导入、原生工具启动、
Torch 共享内存、CUDA matmul/SDPA，以及 PyAV 和 TorchCodec **CPU 解码**。
没有实际视频时可先省略 `--video`，但这不等于验证了视频解码。
沿用此前 AV1 硬解不可用的处理，不要求 AV1 CUDA 硬件解码。

也可不激活，使用一条命令启动：

```bash
.runtime/centos7/core/run python scripts/setup/centos7/doctor.py --profile core --require-cuda
.runtime/centos7/core/run bash scripts/train_fasterwam_libero.sh batch_size=1
```

第二条仍使用上游默认进程数。实际训练时按分配到的 GPU 设置，例如：

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"  # 改为你的实际模型目录
NPROC_PER_NODE=1 bash scripts/train_fasterwam_libero.sh batch_size=1
```

这只示范启动方式，**不表示 batch_size=1 就保证单卡显存足够**。SparseActionDiT
初始化、T5 embedding 预计算、数据/权重路径继续使用原 README 的流程。
环境脚本不会重新下载模型或训练数据。

后续若执行 `uv sync` 或 `uv pip install`，新安装的原生程序可能需要补丁：

```bash
# 完整同步 + 重做补丁 + 诊断（推荐）。
bash scripts/setup/install_centos7.sh core

# 自己加完实验依赖后，仅重做原生程序补丁，避免 sync 删除未声明的实验包。
python scripts/setup/centos7/runtime.py patch-native \
  --state .runtime/centos7/core/state.json
python scripts/setup/centos7/doctor.py --profile core --require-cuda
```

原 `install_core.sh`/`install_libero.sh` 等仍保留原行为。CentOS 7 请使用新增的
入口；不要在这套环境上重新运行未经适配的原安装脚本。

## 5. 可选评测环境

只训练时无需安装下列环境。按你要跑的 benchmark 单独安装：

| profile | 上游目录 | 锁定 Torch |
|---|---|---|
| core | `.venv` | 2.7.1+cu128 |
| libero | `.venvs/libero` | 2.7.1+cu128 |
| libero-plus | `.venvs/libero-plus` | 2.7.1+cu128 |
| robotwin | `.venvs/robotwin` | 2.4.1+cu121 |

```bash
bash scripts/setup/install_centos7.sh libero
source .runtime/centos7/libero/activate.sh
python scripts/setup/centos7/doctor.py --profile libero --require-cuda --render
```

之后原 `scripts/eval_fasterwam_libero.sh` 及其 CKPT_PATH、DATASET_STATS_PATH、
NUM_GPUS 参数不变。LIBERO-Plus 对应把 profile 改成 `libero-plus`。
固定 clone revision、editable install、配置生成及缺失仿真资产的下载流程
与原脚本相同；仿真资产与已下载训练数据不是一回事。

LIBERO 默认使用 OSMesa，需要机器有可加载的 OSMesa 库。若改用 NVIDIA EGL，
将 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl` 一起加入配置并重新执行安装，
诊断与正式评测要使用相同设置。

LIBERO-Plus 还会直接加载 ImageMagick 的 MagickWand。只有 pip 的 `wand`
包不够；若此前已安装 ImageMagick，可在配置设置 `MAGICK_HOME`，并通过
`FASTERWAM_EXTRA_LIB_DIRS` 加入对应 lib 路径。doctor 会实际导入 wand.image。
uv 和私有 glibc 不能凭空提供机器缺失的图形/图像系统库。

RoboTwin 的原锁文件刻意使用 CUDA 12.1，不能跟核心环境合并。cuRobo 必须编译
CUDA 扩展：

```bash
# 将这项写入 .runtime/centos7.env；按本机真实路径修改。
# FASTERWAM_CUDA_HOME=/usr/local/cuda-12.1

# A100 的示例；其他 GPU 请使用对应架构。可见 GPU 时通常无需手填架构。
TORCH_CUDA_ARCH_LIST=8.0 bash scripts/setup/install_centos7.sh robotwin
source .runtime/centos7/robotwin/activate.sh
python scripts/setup/centos7/doctor.py --profile robotwin --require-cuda
```

RoboTwin 还依赖实际 Vulkan/驱动/ICD；doctor 的导入和 CUDA 检查不替代完整
仿真渲染/任务评测。PyTorch wheel 含 CUDA 运行库，不代表已安装 nvcc/toolkit。

## 6. 报错时定位

- `GLIBC_2.xx not found`：检查报错的是 Python 还是另一个 ELF。运行 doctor，
  并查看 `.runtime/centos7/<profile>/patched-native.json`。不要回到全局
  LD_LIBRARY_PATH 指向新 glibc 的做法。
- `GLIBCXX_* not found`：设置含所需符号的 `FASTERWAM_CXX_LIBDIR` 后重跑；
  libstdc++ 与 glibc 是两类不同问题。
- TorchCodec 缺 `libavcodec.so.61` 等：检查 `FASTERWAM_FFMPEG_ROOT` 指向此前
  已验证的 shared FFmpeg 7 构建。核心脚本复用它的 dav1d/x264，以及旧环境中
  必要 NPP/NVRTC/runtime/nvjitlink 库作为低优先级 fallback；不加入旧
  Torch/cuDNN/cuBLAS/NCCL 的 package library 路径。
- `Permission denied`：检查报错库文件及父目录权限、symlink 目标、挂载 noexec；
  修改 loader 无法解决文件权限问题。
- `CUDA is unavailable`：在分配到 GPU 的节点跑 `nvidia-smi` 和 doctor
  `--require-cuda`；检查驱动与作业 GPU 可见性。
- 下载失败：保留 setup log 的具体 URL/错误。不要为修网络问题更新整个锁文件。
- 不同路径/挂载后解释器无法启动：ELF interpreter/RPATH 使用绝对路径，按第 3
  节重建对应 profile。

日志中保留 **第一个失败命令及完整 traceback**。如只想先完成安装再查某个 import，
可以显式 `--skip-check`；此选项不会把未通过诊断的环境变成已验证环境。

## 7. 验证范围和依据

已在独立 Linux 测试夹具使用 Python 3.10 和复制到私有目录的本机 glibc 验证：
私有 prefix、uv venv、父子进程及 spawn、PEP 517 隔离构建及其子进程使用私有
libc、原生 ELF 补丁的 copy-on-write。另做 Bash/Python 语法与锁文件不变检查，
以及四个项目的 `uv sync --locked --dry-run`（RoboTwin 应用上述 index 拼写修复）。
这不是在你的 CentOS 7 服务器上的完整复现；你的 glibc 2.35、CUDA 驱动、FFmpeg
构建和真实视频还需要执行上述 doctor。未声称已跑通训练或仿真 benchmark。

依据：

- [FasterWAM README](https://github.com/miznchimaki/FasterWAM/blob/83667817df0d4f823f39d90700e61ea2f432ac45/README.md)
- [核心 uv 配置](https://github.com/miznchimaki/FasterWAM/blob/83667817df0d4f823f39d90700e61ea2f432ac45/pyproject.toml)
- [FasterWAM Hugging Face 模型](https://huggingface.co/hustvl/FasterWAM)
- [uv Python 选择](https://docs.astral.sh/uv/concepts/python-versions/)
- [uv libc 检测代码](https://github.com/astral-sh/uv/blob/main/crates/uv-python/python/get_interpreter_info.py)
- [uv 命令、locked 与 link-mode](https://docs.astral.sh/uv/reference/cli/)
- [patchelf 手册](https://github.com/NixOS/patchelf/blob/master/patchelf.1)
- [DeepSpeed 0.18.5 setup](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.5/setup.py)
- [cuRobo v0.7.8 setup](https://github.com/NVlabs/curobo/blob/v0.7.8/setup.py)
