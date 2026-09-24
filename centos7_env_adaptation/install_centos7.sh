#!/usr/bin/env bash
# Preserve upstream uv projects/locks; replace only the Python/runtime bootstrap.
set -euo pipefail
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SETUP_DIR}/../.." && pwd)"
for owned in "${ROOT}/.runtime" "${ROOT}/.runtime/centos7" "${ROOT}/.venvs"; do
  [[ ! -L "${owned}" ]] || { echo "Refusing runtime directory symlink: ${owned}" >&2; exit 1; }
done
PROFILE="${1:-core}"
if [[ "${PROFILE}" == --help || "${PROFILE}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/setup/install_centos7.sh [core|libero|libero-plus|robotwin] [--prepare-only|--skip-check]
Default: core. Reads .runtime/centos7.env (or FASTERWAM_CONFIG).
--prepare-only: build/verify private Python + uv venv, without package sync.
--skip-check: install, but leave full import/compute checks to the doctor command.
Never removes an existing environment, edits dependency locks, or changes system libc.
HELP
  exit 0
fi
[[ $# -eq 0 ]] || shift
PREPARE_ONLY=0
SKIP_CHECK=0
for arg in "$@"; do
  case "${arg}" in
    --prepare-only) PREPARE_ONLY=1 ;;
    --skip-check) SKIP_CHECK=1 ;;
    *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
  esac
done
case "${PROFILE}" in
  core) PROJECT="${ROOT}"; ENV_DIR="${ROOT}/.venv" ;;
  libero|libero-plus|robotwin) PROJECT="${ROOT}/environments/${PROFILE}"; ENV_DIR="${ROOT}/.venvs/${PROFILE}" ;;
  *) echo "Unknown profile: ${PROFILE}" >&2; exit 2 ;;
esac
CONFIG="${FASTERWAM_CONFIG:-${ROOT}/.runtime/centos7.env}"
if [[ -n "${FASTERWAM_CONFIG:-}" && ! -f "${CONFIG}" ]]; then
  echo "Explicit FASTERWAM_CONFIG does not exist: ${CONFIG}" >&2; exit 1
fi
if [[ -f "${CONFIG}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG}"
  set +a
fi
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
export FASTERWAM_BASE_PREFIX="${FASTERWAM_BASE_PREFIX:-${HOME}/depends/anaconda3/envs/lerobot310}"
BASE_PYTHON="${FASTERWAM_BASE_PYTHON:-${FASTERWAM_BASE_PREFIX}/bin/python3.10-glibc235}"
if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "Working base Python missing: ${BASE_PYTHON}" >&2
  echo "Set FASTERWAM_BASE_PYTHON to the actual working Python 3.10 ELF, not lerobot310-python (a shell wrapper)." >&2
  exit 1
fi
PATCHELF="${FASTERWAM_PATCHELF:-}"
if [[ -z "${PATCHELF}" ]]; then
  PATCHELF="$(command -v patchelf || true)"
fi
if [[ -z "${PATCHELF}" && -x "${HOME}/depends/local/bin/patchelf" ]]; then
  PATCHELF="${HOME}/depends/local/bin/patchelf"
fi
if [[ ! -x "${PATCHELF}" ]]; then
  echo "Set FASTERWAM_PATCHELF to your working patchelf executable (0.17+ preferred)." >&2; exit 1
fi
PATCHELF="$(readlink -f -- "${PATCHELF}")"
"${PATCHELF}" --version
UV="${FASTERWAM_UV_BIN:-}"
if [[ -z "${UV}" ]]; then UV="$(command -v uv || true)"; fi
if [[ -z "${UV}" && -x "${HOME}/.local/bin/uv" ]]; then UV="${HOME}/.local/bin/uv"; fi
if [[ -z "${UV}" ]]; then
  TOOLS_DIR="${ROOT}/.runtime/centos7/uv-tools"
  if [[ ! -x "${TOOLS_DIR}/bin/uv" && ! -x "${TOOLS_DIR}/uv/uv" ]]; then
    # A dedicated --target keeps the working Conda environment untouched.
    "${BASE_PYTHON}" -I -B -m pip install --no-deps \
      --index-url "${FASTERWAM_BOOTSTRAP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
      --target "${TOOLS_DIR}" 'uv==0.12.15'
  fi
  if [[ -x "${TOOLS_DIR}/bin/uv" ]]; then UV="${TOOLS_DIR}/bin/uv"; else UV="${TOOLS_DIR}/uv/uv"; fi
fi
[[ -x "${UV}" ]] || { echo "uv executable missing: ${UV}" >&2; exit 1; }
UV="$(readlink -f -- "${UV}")"
# Only these four paths are generated profile shims. In particular,
# .runtime/centos7/uv-tools/bin/uv is the real bootstrapped executable.
case "${UV}" in
  "${ROOT}/.runtime/centos7/core/bin/uv"|\
  "${ROOT}/.runtime/centos7/libero/bin/uv"|\
  "${ROOT}/.runtime/centos7/libero-plus/bin/uv"|\
  "${ROOT}/.runtime/centos7/robotwin/bin/uv")
    UV_STATE="$(dirname -- "$(dirname -- "${UV}")")/state.json"
    if [[ ! -f "${UV_STATE}" || ! -r "${UV_STATE}" ]]; then
      echo "Generated uv wrapper is missing its state file: ${UV_STATE}" >&2
      echo "Set FASTERWAM_UV_BIN to the real uv executable (for example .runtime/centos7/uv-tools/bin/uv)." >&2
      exit 1
    fi
    UV="$("${BASE_PYTHON}" -I -B -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["uv"])' "${UV_STATE}")"
    [[ -x "${UV}" ]] || { echo "Saved uv executable is missing: ${UV}" >&2; exit 1; }
    ;;
esac
"${UV}" --version
STATE_DIR="${ROOT}/.runtime/centos7/${PROFILE}"
STATE="${STATE_DIR}/state.json"
PRIVATE_PYTHON="${STATE_DIR}/python-base/bin/python3.10"
# All commands below use this repository's exact paths; run from any directory.
cd "${ROOT}"
[[ -f "${PROJECT}/uv.lock" ]] || { echo "Missing ${PROJECT}/uv.lock" >&2; exit 1; }
LOCK_BEFORE="$(sha256sum "${PROJECT}/uv.lock")"
PROJECT_BEFORE="$(sha256sum "${PROJECT}/pyproject.toml")"
"${BASE_PYTHON}" -I -B -S "${SETUP_DIR}/centos7/runtime.py" prepare \
  --root "${ROOT}" --profile "${PROFILE}" --patchelf "${PATCHELF}" --uv "${UV}"
export UV_PYTHON="${PRIVATE_PYTHON}" UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
if [[ ! -d "${ENV_DIR}" ]]; then
  "${UV}" venv --python "${PRIVATE_PYTHON}" "${ENV_DIR}"
fi
# Prefix check detects an unrelated/replaced venv before uv can modify it.
"${ENV_DIR}/bin/python" -I -c \
  'import sys; from pathlib import Path; assert Path(sys.prefix)==Path(sys.argv[1]); assert Path(sys.base_prefix)==Path(sys.argv[2]), (sys.prefix,sys.base_prefix)' \
  "${ENV_DIR}" "${STATE_DIR}/python-base"
# shellcheck disable=SC1090
source "${STATE_DIR}/activate.sh"
if [[ -n "${CC:-}" ]]; then "${CC}" --version >/dev/null; fi
if [[ -n "${CXX:-}" ]]; then "${CXX}" --version >/dev/null; fi
export FASTERWAM_GLIBC_ROOT="${FASTERWAM_GLIBC_ROOT:-${HOME}/depends/glibc-2.35}"
"${ENV_DIR}/bin/python" "${SETUP_DIR}/centos7/doctor.py" --profile "${PROFILE}" --runtime-only
if [[ "${PREPARE_ONLY}" == 1 ]]; then
  echo "Private runtime ready. Re-run without --prepare-only to install locked dependencies."
  exit 0
fi
if [[ "${PROFILE}" == robotwin ]]; then
  if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "RoboTwin cuRobo build needs CUDA Toolkit 12.1; set FASTERWAM_CUDA_HOME (CUDA wheels do not provide nvcc)." >&2; exit 1
  fi
  NVCC_VERSION="$("${CUDA_HOME}/bin/nvcc" --version)"
  if [[ "${NVCC_VERSION}" != *'release 12.1,'* ]]; then
    echo "RoboTwin lock uses torch 2.4.1+cu121; point FASTERWAM_CUDA_HOME to CUDA Toolkit 12.1." >&2; exit 1
  fi
fi
# DeepSpeed 0.18.5 supports its normal isolated build without precompiled ops.
# Keep isolation enabled; forcing torch into that build can require nvcc.
DS_BUILD_OPS=0 "${UV}" sync --project "${PROJECT}" --python "${PRIVATE_PYTHON}" --locked
"${PRIVATE_PYTHON}" -I "${SETUP_DIR}/centos7/runtime.py" patch-native --state "${STATE}"
if [[ "${PROFILE}" != core ]]; then
  # Same pinned clones, editable installs and post-processing as upstream.
  # shellcheck source=centos7/benchmark_post.sh
  source "${SETUP_DIR}/centos7/benchmark_post.sh"
  "${PRIVATE_PYTHON}" -I "${SETUP_DIR}/centos7/runtime.py" patch-native --state "${STATE}"
fi
[[ "$(sha256sum "${PROJECT}/uv.lock")" == "${LOCK_BEFORE}" ]]
[[ "$(sha256sum "${PROJECT}/pyproject.toml")" == "${PROJECT_BEFORE}" ]]
if [[ "${SKIP_CHECK}" == 0 ]]; then
  "${ENV_DIR}/bin/python" "${SETUP_DIR}/centos7/doctor.py" --profile "${PROFILE}"
else
  echo "Full diagnostics skipped explicitly (--skip-check)."
fi
echo "Ready: source ${STATE_DIR}/activate.sh"
echo "One-command launcher: ${STATE_DIR}/run <command> [args...]"
