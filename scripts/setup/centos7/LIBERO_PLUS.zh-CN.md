# CentOS 7：恢复 LIBERO-Plus 原生依赖

适用状态：`install_centos7.sh libero-plus` 已完成 Python/uv、包安装、
LIBERO-Plus clone、assets.zip 下载/解压及配置生成，但最后 doctor 报错：

- MuJoCo：`NoneType ... glGetError`；
- robosuite：`libGL.so.1` 缺失；
- Wand：`MagickWand shared library not found`。

这三个问题分别涉及离屏图形后端、普通/无界面 OpenCV 共享 `cv2`
命名空间、以及 Python Wand 之外的原生 ImageMagick。已经下载的资产无需重下。

## 直接恢复

应用补丁后，在仓库根目录和有 GPU 的计算节点执行：

```bash
(
  set -o pipefail
  MAX_JOBS=2 bash scripts/setup/repair_centos7_libero_plus.sh \
    2>&1 | tee .runtime/centos7/libero-plus/repair-plus.log
)
```

入口会依次完成：

1. 在 `.venvs/libero-plus` 中修复 OpenCV 为唯一的
   `opencv-python-headless==4.11.0.86`；重装只涉及 OpenCV，使用清华镜像。
2. 按 `--profile libero-plus` 修补该环境自己的 MuJoCo DSO，配置私有
   GLVND/NVIDIA EGL 库及 vendor JSON。默认只读验证已有 LIBERO GLVND
   构建及成功修复后的 NVIDIA 库副本，核对 SHA256、来源和当前驱动版本，
   再复制到 Plus 自己的 `native-libs` 并设置 Plus 的 RPATH。
   不复制 LIBERO 的整个 runtime 或修复记录，不修改其动态库。
3. 构建 Plus 专用 ImageMagick/MagickWand，支持 LIBERO-Plus 实际使用的
   `PNG -> MagickMotionBlurImage -> PNG/OpenCV`，并运行该操作的 smoke test。
4. 刷新 Plus 的启动器，持久化 EGL、vendor JSON、`MAGICK_HOME` 和
   `WAND_MAGICK_LIBRARY_SUFFIX`；最后运行 doctor CUDA/实际 MuJoCo 渲染检查。

缺少可复用的 LIBERO GLVND 构建时，会为 Plus 单独构建。若已有源构建的
校验失败，入口会停止；不要用未经验证的目录绕过校验。如确实要重新构建
独立的 Plus GLVND，执行：

```bash
MAX_JOBS=2 bash scripts/setup/repair_centos7_libero_plus.sh --build-glvnd
```

重复执行会校验并复用成功的原生构建。若仅最后 doctor 失败，保留日志，
不要删除 `.runtime`、`.venvs`、`assets.zip` 或整套 LIBERO-Plus 源码。

## 恢复日志中的 libGL 和 NVIDIA EGL 错误

如果开头的 `DIAGNOSTIC OpenCV before repair` 有 `libGL.so.1: ... Permission denied`，
而随后出现 `PASS OpenCV: ... GUI=NONE, resize OK`，表示 OpenCV 已经修好。
不要为这条历史诊断去 chmod 系统库或再次安装 GUI OpenCV。

若最后是 `GRAPHICS NOT READY` / `MISSING libEGL_nvidia.so.0`，则是另一项
NVIDIA EGL 库发现失败。GLVND 构建通过并不代表厂商驱动库已经找到。
旧恢复脚本只复用了 GLVND 三个分发库和手工指定的搜索目录；当原系统路径
不可读、SONAME 链接缺失或不在搜索范围时，会漏掉原 LIBERO 已修好的副本。
具体系统路径失败原因不能仅由 `MISSING` 判断。

现在 `--reuse-glvnd-from libero` 会读取源 profile 成功的 `loader-repair.json`，
验证已管理库副本的 SHA256，并结合原始文件名记录核对 NVIDIA 驱动版本。
即使原始系统文件路径已不可用，也能只读复用可验证的私有副本。复制后的
NVIDIA 库使用目标环境的 RPATH；不会把 LIBERO 的整个 native-libs 加入搜索路径。
目标记录会保存来源，后续不带 reuse 参数的 `--profile libero-plus --apply`
也会继续验证和使用该来源。显式 `--graphics-dir` 会优先采用指定目录。

应用增量修复后，重新执行同一恢复入口即可；推荐换日志文件名：

```bash
(
  set -o pipefail
  MAX_JOBS=2 bash scripts/setup/repair_centos7_libero_plus.sh \
    2>&1 | tee .runtime/centos7/libero-plus/repair-plus-vendor.log
)
```

预期新增 `REUSE verified graphics from libero:`，随后有
`PASS fresh dlopen libEGL_nvidia.so.0` 和 `PASS fresh EGL MuJoCo and robosuite imports`。
入口会继续 ImageMagick 构建/验证及最终 doctor。库副本被改动、源修复未成功，
或驱动版本不一致时会停止并给出错误；保留完整日志，不跳过这些检查。
本修复不下载或安装 NVIDIA 驱动，也不修改源 LIBERO 环境。

## ImageMagick 的隔离方式

固定 ImageMagick 6.9.13-57、libpng 1.6.44、zlib 1.3.1；三个源包均校验
SHA256。需要现有可用的 C 编译器、make、binutils 和 C 开发头文件，
默认使用启动器中配置的 GCC。`MAX_JOBS` 默认 2。

libpng/zlib 构建为私有 PIC 静态库，符号不向外导出，运行时不新放置
`libz.so` 或 `libexpat.so`，避免重现之前 Python XML/zlib 的库冲突。
最终仅在 Plus runtime 下提供带私有 RPATH 的 MagickWand/MagickCore；
不修改系统 ImageMagick、系统 glibc 或已有 Conda 环境。

该构建面向 benchmark 的 PNG/运动模糊 C API，不提供通用 `convert` 命令，
也不启用 JPEG/PDF 等 benchmark 此路径不需要的 delegates。
无需设置全局 `LD_LIBRARY_PATH`，无需为此重新 prepare Python。

源码下载由新脚本开头的固定 HTTPS URL 指定。若计算节点无法下载，可先在
能联网的机器下载相同文件，上传后直接使用 builder 的离线参数：

```bash
.runtime/centos7/libero-plus/run python \
  scripts/setup/centos7/build_libero_plus_imagemagick.py --apply \
  --source /path/to/ImageMagick-6.9.13-57.tar.xz \
  --png-source /path/to/libpng-1.6.44.tar.xz \
  --zlib-source /path/to/zlib-1.3.1.tar.gz
```

必须使用确切版本与匹配 SHA256，不能换其他版本文件。

## 成功标准和后续使用

日志应包括：

- `PASS OpenCV: ... GUI=NONE`；
- `PASS fresh EGL MuJoCo and robosuite imports`；
- `PASS private Wand PNG -> motion blur -> PNG/OpenCV; Python XML/zlib intact`；
- doctor 的 `PASS pinned packages / native imports`、`PASS PyTorch execution`、
  `PASS MuJoCo rendering`，以及 `Doctor finished: 7 passed, 0 failed.`。

这里验证基础环境和 benchmark 使用的 ImageMagick 操作，尚不代表全部
LIBERO-Plus 任务、资产以及多 GPU 设备映射均已验收。

```bash
source .runtime/centos7/libero-plus/activate.sh
# 或不修改当前 shell：
.runtime/centos7/libero-plus/run bash scripts/eval_fasterwam_libero_plus.sh
```

评测脚本仍需要正常提供 `CKPT_PATH`、`DATASET_STATS_PATH` 等参数。
`PYTHON_BIN` 应继续使用 `.venvs/libero-plus/bin/python`。
已激活的终端需要重新 source；新 `run` 调用会直接采用更新后的配置。

以后重新执行 `install_centos7.sh libero-plus` 时，更新后的 lock 会持续排除
GUI OpenCV；已选择的图形修复会重新应用，私有 ImageMagick 会重新验证复用。
自动恢复只对对应 profile 生效，默认的 `repair_libero_loader.py` 和
`repair_libero_opencv.py` 不传 `--profile` 时仍处理原 LIBERO。
