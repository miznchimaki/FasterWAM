#!/usr/bin/env bash
# Repair the existing core environment without running uv sync again.
set -euo pipefail
ulimit -c 0 || true
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SETUP_DIR}/../.." && pwd)"
cd "${ROOT}"
if [[ ! -f .runtime/centos7/core/state.json || ! -x .venv/bin/python ]]; then
  echo "An existing prepared core environment is required; run install_centos7.sh core first." >&2
  exit 1
fi
# Preserve the Python ELF whose private-glibc checks already passed.
CONFIG="${FASTERWAM_CONFIG:-${ROOT}/.runtime/centos7.env}"
if [[ -n "${FASTERWAM_CONFIG:-}" && ! -f "${CONFIG}" ]]; then
  echo "Explicit FASTERWAM_CONFIG does not exist: ${CONFIG}" >&2
  exit 1
fi
if [[ -f "${CONFIG}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG}"
  set +a
fi
# shellcheck disable=SC1091
source "${ROOT}/.runtime/centos7/core/activate.sh"
"${ROOT}/.venv/bin/python" -I -u "${SETUP_DIR}/centos7/doctor.py" --profile core --runtime-only
"${ROOT}/.venv/bin/python" -I -u "${SETUP_DIR}/centos7/repair_core.py"
"${ROOT}/.venv/bin/python" -I -u "${SETUP_DIR}/centos7/doctor.py" --profile core
