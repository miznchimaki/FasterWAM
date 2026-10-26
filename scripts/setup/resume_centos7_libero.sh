#!/usr/bin/env bash
# Recover the specific clone-success / old-Git checkout-failure installation.
set -euo pipefail
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH

if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/setup/resume_centos7_libero.sh" >&2
  exit 2
fi
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SETUP_DIR}/../.." && pwd)"
SOURCE_DIR="${ROOT}/third_party/LIBERO"

if [[ ! -f "${ROOT}/.runtime/centos7/libero/state.json" || ! -x "${ROOT}/.venvs/libero/bin/python" ]]; then
  echo "Prepared LIBERO runtime missing. Run install_centos7.sh libero first." >&2
  exit 1
fi
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  echo "Expected the completed LIBERO clone at ${SOURCE_DIR}; refusing recovery." >&2
  exit 1
fi

(
  cd -- "${SOURCE_DIR}"
  [[ "$(git rev-parse --show-toplevel)" == "$(pwd -P)" ]] || {
    echo "Unexpected Git checkout root; refusing recovery." >&2; exit 1;
  }
  ORIGIN="$(git config --get remote.origin.url)"
  case "${ORIGIN}" in
    https://github.com/Lifelong-Robot-Learning/LIBERO.git|https://github.com/Lifelong-Robot-Learning/LIBERO|git@github.com:Lifelong-Robot-Learning/LIBERO.git) ;;
    *) echo "Unexpected LIBERO origin: ${ORIGIN}; refusing recovery." >&2; exit 1 ;;
  esac
  CHANGES="$(git status --porcelain)"
  if [[ -n "${CHANGES}" ]]; then
    echo "LIBERO has local changes or untracked files; preserve them before switching revision:" >&2
    printf '%s\n' "${CHANGES}" >&2
    exit 1
  fi
  # No -C, forced checkout, reset or clean. A commit hash gives detached HEAD
  # on the old Git shipped with CentOS 7 as well as current Git.
  EXPECTED="$(git rev-parse --verify '8f1084e^{commit}')"
  git checkout "${EXPECTED}"
  [[ "$(git rev-parse HEAD)" == "${EXPECTED}" ]]
)

echo "LIBERO source is at the pinned revision; resuming the normal installer."
exec bash "${SETUP_DIR}/install_centos7.sh" libero
