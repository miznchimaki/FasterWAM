# Copy to .runtime/centos7.env, then edit only paths that differ on your server.
# This is a trusted shell configuration, sourced by install_centos7.sh.
FASTERWAM_BASE_PREFIX="${HOME}/depends/anaconda3/envs/lerobot310"
FASTERWAM_BASE_PYTHON="${FASTERWAM_BASE_PREFIX}/bin/python3.10-glibc235"
FASTERWAM_GLIBC_ROOT="${HOME}/depends/glibc-2.35"
FASTERWAM_FFMPEG_ROOT="${HOME}/depends/ffmpeg-7.1.1"
FASTERWAM_DRIVER_LIBDIR="${HOME}/depends/nvidia-driver-libs"
FASTERWAM_GCC_ROOT="${HOME}/depends/gcc-12.1.0"
# FASTERWAM_PATCHELF="${HOME}/depends/local/bin/patchelf"
# FASTERWAM_UV_BIN="${HOME}/.local/bin/uv"
# FASTERWAM_CXX_LIBDIR="${HOME}/depends/gcc-12.1.0/lib64"
# Extra SHARED LIBRARY directories, colon-separated; no CUDA stubs or system libc.
# FASTERWAM_EXTRA_LIB_DIRS="${HOME}/depends/dav1d-1.5.3/lib:${HOME}/depends/x264/lib"
# Needed only for compiling CUDA extensions; RoboTwin requires CUDA Toolkit 12.1.
# FASTERWAM_CUDA_HOME="/usr/local/cuda-12.1"
# MAX_JOBS=4
# Optional LIBERO-Plus native ImageMagick installation:
# MAGICK_HOME="${HOME}/depends/imagemagick"
# FASTERWAM_EXTRA_LIB_DIRS="${MAGICK_HOME}/lib"
# LIBERO launchers default to osmesa; to use NVIDIA EGL, set BOTH consistently:
# MUJOCO_GL=egl
# PYOPENGL_PLATFORM=egl
