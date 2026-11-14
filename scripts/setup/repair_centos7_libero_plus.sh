#!/usr/bin/env bash
# Resume native-library repair for an already prepared LIBERO-Plus environment.
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SETUP_DIR}/../.." && pwd)"
cd "${ROOT}"

BUILD_GLVND=0
RENDER=1
for arg in "$@"; do
  case "${arg}" in
    --build-glvnd) BUILD_GLVND=1 ;;
    --skip-render) RENDER=0 ;;
    --help|-h)
      cat <<'HELP'
Usage: bash scripts/setup/repair_centos7_libero_plus.sh [--build-glvnd] [--skip-render]
Repair an existing private LIBERO-Plus environment, then run its doctor.
By default reuse the verified LIBERO GLVND build as a read-only source;
--build-glvnd builds a separate GLVND bundle for LIBERO-Plus instead.
ImageMagick with PNG support is built privately and reused on later runs.
--skip-render omits the final GPU/render check (not a full acceptance test).
Existing asset downloads and LIBERO/core environments are not modified.
HELP
      exit 0 ;;
    *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
  esac
done

RUN="${ROOT}/.runtime/centos7/libero-plus/run"
[[ -x "${RUN}" ]] || {
  echo "Missing ${RUN}; first prepare/install the CentOS 7 libero-plus profile." >&2
  exit 1
}

"${RUN}" python -u "${SETUP_DIR}/centos7/repair_libero_opencv.py" \
  --profile libero-plus --repair

GLVND_ARGS=(--build-glvnd)
if [[ "${BUILD_GLVND}" == 0 && -f "${ROOT}/.runtime/centos7/libero/private-glvnd/build.json" ]]; then
  GLVND_ARGS=(--reuse-glvnd-from libero)
fi
"${RUN}" python -u "${SETUP_DIR}/centos7/repair_libero_loader.py" \
  --root "${ROOT}" --profile libero-plus --apply "${GLVND_ARGS[@]}"

# Each invocation re-enters the refreshed launcher; the builder and doctor
# therefore see the private EGL vendor and, later, the managed MAGICK_HOME.
"${RUN}" python -u "${SETUP_DIR}/centos7/build_libero_plus_imagemagick.py" --apply

DOCTOR_ARGS=(--profile libero-plus)
if [[ "${RENDER}" == 1 ]]; then
  DOCTOR_ARGS+=(--require-cuda --render)
fi
"${RUN}" python -u "${SETUP_DIR}/centos7/doctor.py" "${DOCTOR_ARGS[@]}"
echo "Ready: source ${ROOT}/.runtime/centos7/libero-plus/activate.sh"
