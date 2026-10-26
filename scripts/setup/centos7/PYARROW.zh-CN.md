# CentOS 7：PyArrow 独立导入与 librt 搜索路径修复

适用于当前三个环境的 Python 3.10 / pyarrow==23.0.0：core、libero、libero-plus。
无需重新安装环境、下载模型数据、重建 EGL 或占用 GPU。

## 原因与边界

官方 cp310 manylinux_2_28_x86_64 PyArrow 23.0.0 wheel 中的 37 个共享库
使用 `DT_RUNPATH=$ORIGIN`；其中多个库依赖 `librt.so.1`。
glibc 加载器处理带 RUNPATH 的请求库时不会按无 RUNPATH 的方式继承
主 Python 的 RPATH。因此只修补 Python ELF，不保证 PyArrow 能找到私有 librt。

此前 doctor 在同一进程中先 import torch，再 import pyarrow；前者可能提前加载
librt，从而掩盖后者独立导入的问题。新的 doctor 会先启动独立子进程验证 PyArrow，
不改变原有检查项数量。`Permission denied` 本身不能说明被拒绝的是哪条候选路径；
修复前必须在另一子进程中用绝对路径验证私有 librt。该步骤失败则不改 PyArrow。

参考：glibc 2.35 `elf/dl-load.c` 的 `_dl_map_object` RUNPATH/RPATH 分支：
<https://github.com/bminor/glibc/blob/release/2.35/master/elf/dl-load.c>。

## 已安装环境的修复

应用包含本文件的补丁后，在 FasterWAM 根目录执行：

```bash
mkdir -p centos7_env_adaptation/logs/pyarrow
(
  set -o pipefail
  bash scripts/setup/repair_centos7_pyarrow.sh \
    2>&1 | tee centos7_env_adaptation/logs/pyarrow/repair-pyarrow.log
)
```

不需要先 activate。脚本分别使用各环境自己的 run 启动器，不受当前激活环境影响。
也可只修复一个环境：

```bash
bash scripts/setup/repair_centos7_pyarrow.sh libero-plus
```

每个环境都应看到 `PASS PyArrow repair [profile]` 和 `PASS profile profile`。
`BEFORE fresh PyArrow (diagnostic): exit=1` 是修复前的原始错误；判断最终是否成功
应看 AFTER / PASS 和整个命令的退出码。脚本会继续检查其他 profile，任一失败则最终非零退出。

独立验证，仍不占用 GPU：

```bash
bash scripts/setup/repair_centos7_pyarrow.sh --check

for profile in core libero libero-plus; do
  .runtime/centos7/"${profile}"/run python -I -c \
    'import pyarrow; print(pyarrow.__version__); import datasets; print("datasets", datasets.__version__)' \
    || break
done
```

`--check` 在新进程首先 import pyarrow，验证 compute、dataset、IPC 和
Parquet/Snappy 往返，以及 `/proc/self/maps` 中 libc、librt 等的来源。
如果仍失败，请保留完整日志；若修复后的 probe 失败，还会生成
`.runtime/centos7/<profile>/pyarrow-loader-debug.log` 供进一步定位。

## 修改范围与重跑

- 每个 profile 只修改其 site-packages/pyarrow 下的 ELF 共享库；保留包版本和锁文件。
- RPATH 顺序：私有 glibc、原有 `$ORIGIN` 路径、该 profile 的 native-libs、其余原有路径。
  不额外加入 torch/NVIDIA/FFmpeg/旧 Conda 的整组路径。
- 对临时副本运行 patchelf，确认 DT_NEEDED 不变、DT_RUNPATH 消失，再原子替换。
  不原地改写 uv 缓存硬链接，不改共享库 PT_INTERP，不修改系统文件或系统权限。
- 原始 ELF 按 SHA256 备份到 `.runtime/centos7/<profile>/pyarrow-repair-backups/`，
  进度记录到 `pyarrow-repair.json`。第一次每个环境约需额外 145 MiB 备份空间。
  重跑已修好的文件显示 UNCHANGED；中断后可重新运行。执行期间不要同时 uv/pip 修改同一环境。
- 正常 `install_centos7.sh` 已接入此修复；后续通过其他命令重装 PyArrow 后，
  再运行本修复脚本即可。不需要重新执行全套环境安装。
- 新启动的 Python 会使用修复后的文件；已有 Python/Jupyter 进程应重启再验证。

## 本地开发验证

```bash
python scripts/setup/centos7/test_pyarrow_doctor.py
python scripts/setup/centos7/test_pyarrow_repair.py
```

ELF 测试需要 gcc 和 patchelf；通过 `PATCHELF=/path/to/patchelf` 指定测试工具。
它在临时目录编译最小共享库，验证 RUNPATH 继承问题、copy-on-write、幂等性及备份，
不修改当前运行环境。
