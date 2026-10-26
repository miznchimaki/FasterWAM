# LIBERO 的 OpenCV headless 修复

旧 `environments/libero/uv.lock` 同时包含 `opencv-python` 和
`opencv-python-headless` 4.11.0.86。前者由 robosuite 1.4.0 间接引入。
二者共同写入 `cv2`，GUI 版本可能覆盖 headless 版本并在导入时要求
`libGL.so.1`。OpenCV 官方要求同一环境只安装一个变体。

修复在 LIBERO 项目中明确排除 `opencv-python`，保留原本显式要求的
`opencv-python-headless==4.11.0.86`；锁文件仅删除 GUI 包及对应依赖边，
其余包版本、下载 URL 和哈希保持不变。此策略针对无窗口的 LIBERO 评测，
不支持 OpenCV 的 `imshow` 窗口；MuJoCo 离屏渲染有独立的 OSMesa/EGL 依赖。
LIBERO doctor 的普通导入检查现在也采用与评测脚本一致的默认 OSMesa 后端，
避免普通检查隐式用 EGL、正式评测却用 OSMesa；显式配置的后端保持优先。

已经完成大部分安装、仅在最终导入检查失败时，应用补丁后直接执行：

```bash
.runtime/centos7/libero/run python scripts/setup/centos7/repair_libero_opencv.py --repair
.runtime/centos7/libero/run python scripts/setup/centos7/doctor.py --profile libero --require-cuda
```

修复器只接受当前仓库的 private LIBERO Python，已有正确 headless 模块时不
重复安装。否则先卸载两个冲突包，再以 `--no-deps` 从清华源安装固定 headless
版本，验证 import、GUI=NONE、resize，以及其他包的版本保持不变。它不会
启动 CUDA 或仿真，也不调用完整 `uv sync`。下载失败时可重跑修复命令。

正常安装入口也会自动验证和修复旧环境迁移时被卸载器删除的共享 cv2 文件。
修改锁文件是必要的：只手动卸载 GUI 包，旧锁下次同步时仍会把它装回来。

注意 robosuite 发布的 wheel 元数据仍声明 `opencv-python`；通用的
`pip check`/`uv pip check` 可能据此提示缺少该包。这是已明确配置的 headless
替代，不应为了消除这个提示重新混装 GUI 包。保留上游 wheel 元数据，不伪造
包名或修改 dist-info。请以 cv2 的实际构建检查和 robosuite 导入、渲染验收判断。

导入检查通过后，在有空闲 GPU 的评测节点进一步执行：

```bash
.runtime/centos7/libero/run python scripts/setup/centos7/doctor.py \
  --profile libero --require-cuda --render
```

默认渲染后端为 OSMesa；如果出现缺少 `libOSMesa`、EGL 初始化失败等信息，
需根据服务器实际图形库继续处理，headless OpenCV 本身不提供 MuJoCo 渲染器。
不要把 `libEGL.so`/`libOSMesa.so` 伪装成 `libGL.so.1`，也不要全局导出包含
私有 libc 的 `LD_LIBRARY_PATH`。若仍失败，保留完整 traceback 以区分 cv2 与
渲染后端的加载链。
