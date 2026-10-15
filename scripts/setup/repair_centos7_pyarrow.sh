#!/usr/bin/env bash
# Repair installed PyArrow only; do not uv sync, rebuild graphics or run a GPU job.
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SETUP_DIR}/../.." && pwd)"
MODE=apply
if [[ "${1:-}" == --check ]]; then MODE=check; shift; fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash scripts/setup/repair_centos7_pyarrow.sh [--check] [core|libero|libero-plus ...]'
  echo 'Default: repair all three installed environments; --check changes no ELF files.'
  exit 0
fi
if [[ $# -eq 0 ]]; then set -- core libero libero-plus; fi
for profile in "$@"; do
  case "${profile}" in
    core|libero|libero-plus) ;;
    *) echo "Unsupported profile: ${profile}" >&2; exit 2 ;;
  esac
done
cd "${ROOT}"
failed=0
for profile in "$@"; do
  run="${ROOT}/.runtime/centos7/${profile}/run"
  echo "=== PyArrow ${MODE}: ${profile} ==="
  if [[ ! -x "${run}" ]]; then
    echo "Missing profile launcher: ${run}" >&2
    failed=1
    continue
  fi
  args=(--root "${ROOT}" --profile "${profile}")
  if [[ "${MODE}" == apply ]]; then args+=(--apply); fi
  if "${run}" python -I -B -u "${SETUP_DIR}/centos7/repair_pyarrow.py" "${args[@]}"; then
    echo "PASS profile ${profile}"
  else
    echo "FAIL profile ${profile}; see the preceding traceback" >&2
    failed=1
  fi
done
exit "${failed}"
