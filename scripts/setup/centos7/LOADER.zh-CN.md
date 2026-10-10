# LIBERO：私有 glibc 与 NVIDIA EGL 修复

适用于 Python 已使用私有 glibc 2.35、MuJoCo 固定为 3.3.2 的 CentOS 7 环境。
本脚本只操作当前 repo 的 `.venvs/libero` 和 `.runtime/centos7/libero`。
在没有进程使用这个 LIBERO 环境时应用；独立 core 环境的实验不需要停止。

## 对应的故障

1. `_specs` 的 `RUNPATH=$ORIGIN` 绕过 Python 的 RPATH，导致迁移后的 loader
   去旧构建目录寻找 `librt.so.1`。修复保留 `$ORIGIN`，给 MuJoCo DSO 补齐
   私有 glibc 的 RPATH，并用全新进程、`MUJOCO_GL=disable` 验证独立导入。
2. `libEGL.so.1` 能被 `ctypes.CDLL` 加载，但实际是旧 Mesa，随后出现
   `DRI2: failed to open glapi provider` / `eglQueryDevicesEXT is not available`。
   NVIDIA vendor 库存在，并不能证明当前 EGL 前端能分派到它。
3. GLVND 构建到 XML 代码生成时报 `pyexpat: undefined symbol:
   XML_SetAllocTrackerActivationThreshold`。第一版脚本复制到 `native-libs`
   的系统 `libexpat.so.1` 可能抢先于基底 Conda 的新版库被加载；迁移后的
   标准库扩展自身 RUNPATH 也可能绕过 Python 主程序的搜索路径。
   ElementTree 随后报的 `No module named expat` 是二次报错，不表示需要 pip
   安装一个 expat 包。

修复器会在 GLVND 编译之前检查历史记录里的 Expat/zlib 副本，先在独立进程
验证基底库确实能支持 XML 解析和压缩，再将哈希匹配的旧受管副本移入备份。
接着通过普通新进程（不预加载）验证 Python 的 XML/zlib，并打印实际库路径。
若普通导入仍失败、但基底库预检成功，才对私有 `pyexpat`/`zlib` 扩展做
带备份的 RPATH 修复；保留 NEEDED，不修改基底 Conda 文件或系统库。
图形依赖若仍需要 Expat/zlib，会优先采用基底 Python 的同 SONAME 库，
并在图形库部署后再次检查普通 XML/zlib 导入，防止旧系统副本被重新引入。

## 应用

从 repo 根目录执行。`--build-glvnd` 下载固定版本的官方 libglvnd v1.7.0
源码、Meson 1.5.2 的纯 Python wheel 和 Ninja 1.11.1.4 的 Linux x86_64 wheel，
验证 SHA256，在 LIBERO 私有目录构建。不向虚拟环境安装 Meson/Ninja；
不下载/安装 NVIDIA 驱动。需要已有 C 编译器、系统 C 开发头文件及 nm。
构建禁用 X11/GLX，不需要 X11 开发包。

LIBERO 的依赖锁不包含 Ninja，因此不要求 `.venvs/libero/bin/ninja` 存在。
脚本从已验证的 wheel 中提取原始 Ninja，放在
`.runtime/centos7/libero/glvnd-build/ninja-*/`。此 wheel 的平台下限为 glibc 2.12；
先检查直接执行，失败时尝试显式私有 loader，再验证其 shell 子进程构建。
不对 Ninja 原始 ELF 执行 patchelf，也不向子进程导出 `LD_LIBRARY_PATH`。

```bash
(
  set -o pipefail
  .runtime/centos7/libero/run python -u \
    scripts/setup/centos7/repair_libero_loader.py --apply --build-glvnd \
    2>&1 | tee .runtime/centos7/libero/repair-loader-glvnd.log
)
```

已有正确的 GLVND 时，可不编译，显式提供同一次安装的三个库所在目录：

```bash
.runtime/centos7/libero/run python scripts/setup/centos7/repair_libero_loader.py \
  --apply --glvnd-dir /actual/glvnd/lib
```

该目录必须同时含 `libEGL.so.1`、`libOpenGL.so.0`、`libGLdispatch.so.0`
及其有效目标。脚本验证 SONAME、GLdispatch 依赖、EGL vendor loader 标记，
不会静默回退到系统旧 Mesa。未指定目录且没有历史记录时，会自动寻找合适的
完整 GLVND 组合，包括被旧软链接遮住的版本化文件。

NVIDIA 库从当前服务器读取：要求 `libEGL_nvidia.so.0` 及匹配版本的
`libnvidia-eglcore.so.VERSION`，递归处理它们的 ELF 依赖。可读到驱动版本时，
拒绝文件名版本不匹配的 NVIDIA 库。缺库时可以补充已有驱动库目录：

```bash
.runtime/centos7/libero/run python scripts/setup/centos7/repair_libero_loader.py \
  --apply --build-glvnd --graphics-dir /actual/host/nvidia/lib64
```

若无法联网，先将下列三个固定文件传至服务器，再添加
`--glvnd-source /path/v1.7.0.tar.gz --meson-wheel /path/meson-1.5.2-py3-none-any.whl`
以及 `--ninja-wheel /path/ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl`。
它们只与 `--build-glvnd` 一起使用，SHA256 不匹配会停止：

- 源码：<https://codeload.github.com/NVIDIA/libglvnd/tar.gz/refs/tags/v1.7.0>
  SHA256 `073e7292788d4d3eeb45ea6c7bdcce9bfdb3b3eef8d7dbd47f2f30dce046ef98`。
- Meson：<https://pypi.org/project/meson/1.5.2/#files>
  wheel SHA256 `77706e2368a00d789c097632ccf4fc39251fba56d03e1e1b262559a3c7a08f5b`。
- Ninja：<https://pypi.org/project/ninja/1.11.1.4/#files>
  选择 Linux x86_64 manylinux_2_12 wheel，SHA256
  `096487995473320de7f65d622c3f1d16c3ad174797602218ca8c967f51ec38a0`。

## 安全重跑与检查

无 `--apply` 时只显示计划，并以独立子进程检查私有 librt 可加载性。
ELF 修改采用备份、复制及原子替换，保留原始文件，不改共享 uv 缓存，
不向 DSO 添加 PT_INTERP。不会修改系统库、原始驱动、权限或 core 环境，
不会全局设置 `LD_LIBRARY_PATH`。

记录位于 `.runtime/centos7/libero/loader-repair.json`，原始 ELF 备份位于
`loader-repair-backups`。新版会将旧脚本复制、但不再需要的图形依赖移到
该备份目录；只移动记录内且哈希匹配的文件，保留其他文件和既有 C++ 运行库。
检测到未知修改会停止。中断后可重跑同一命令，已构建并验证的 GLVND 会复用。
CentOS7 安装入口在之后 uv 同步时会调用修复器，使用记录中的 GLVND 目录。

检查依次包括：私有 librt、禁用图形的 MuJoCo 独立导入、普通 Python XML/zlib
解析和实际加载路径、图形库 dlopen、
实际 EGL dispatcher 的 `1.5 libglvnd` 版本标记、`eglQueryDevicesEXT` 函数及
设备数量、加载路径、全新进程中的 MuJoCo/robosuite EGL 导入。
设备数为 0 会明确报错，需要检查作业 GPU 分配、设备权限和驱动可见性；
这与 EGL 函数地址缺失不同。该步骤不创建渲染上下文。

成功后脚本只刷新 LIBERO 的 `run`、`bin/uv`、`activate.sh`，持久化
`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl` 和私有 NVIDIA vendor JSON 路径。
此 profile 的修复记录优先于通用配置中的图形后端；不修改通用配置文件。
已经激活的终端需重新 source `.runtime/centos7/libero/activate.sh`；
直接使用下面的 `run` 不需要重新激活。

```bash
(
  set -o pipefail
  .runtime/centos7/libero/run python -u scripts/setup/centos7/doctor.py \
    --profile libero --require-cuda --render \
    2>&1 | tee .runtime/centos7/libero/doctor-egl.log
)
```

仅在 doctor 的实际渲染也通过后，才继续 LIBERO 评测。仍需用一个实际任务
检查资产、相机及 GPU 分配；库导入或设备枚举成功不等于完整评测成功。
