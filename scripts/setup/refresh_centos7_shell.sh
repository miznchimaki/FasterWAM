#!/usr/bin/env bash
# Regenerate shell entry points only. Never patch Python or synchronize packages.
set -euo pipefail
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SETUP_DIR}/../.." && pwd)"
PROFILE="${1:-core}"
if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/setup/refresh_centos7_shell.sh [core|libero|libero-plus|robotwin]" >&2
  exit 2
fi
case "${PROFILE}" in
  core) ENV_DIR="${ROOT}/.venv" ;;
  libero|libero-plus|robotwin) ENV_DIR="${ROOT}/.venvs/${PROFILE}" ;;
  *) echo "Unknown profile: ${PROFILE}" >&2; exit 2 ;;
esac
for owned in "${ROOT}/.runtime" "${ROOT}/.runtime/centos7" "${ROOT}/.runtime/centos7/${PROFILE}" "${ENV_DIR}"; do
  [[ ! -L "${owned}" ]] || { echo "Refusing runtime directory symlink: ${owned}" >&2; exit 1; }
done
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
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH
STATE="${ROOT}/.runtime/centos7/${PROFILE}/state.json"
if [[ ! -f "${STATE}" || ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Prepared environment missing: ${PROFILE}. Run install_centos7.sh first." >&2
  exit 1
fi
"${ENV_DIR}/bin/python" -I -B "${SETUP_DIR}/centos7/runtime.py" refresh-shell --state "${STATE}"
