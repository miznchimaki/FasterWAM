# Sourced only by install_centos7.sh after the corresponding uv sync.
# The benchmark-specific operations match upstream scripts/setup/install_*.sh.
# shellcheck source=../_common.sh
source "${SETUP_DIR}/_common.sh"
case "${PROFILE}" in
  libero|libero-plus)
    if [[ "${PROFILE}" == libero ]]; then
      SOURCE_DIR="${ROOT}/third_party/LIBERO"
      fasterwam_clone_pinned "https://github.com/Lifelong-Robot-Learning/LIBERO.git" "8f1084e" "${SOURCE_DIR}"
    else
      SOURCE_DIR="${ROOT}/third_party/LIBERO-plus"
      fasterwam_clone_pinned "https://github.com/sylvestf/LIBERO-plus.git" "4976dc3" "${SOURCE_DIR}"
    fi
    "${UV}" pip install --python "${ENV_DIR}/bin/python" --no-deps -e "${SOURCE_DIR}"
    if [[ "${PROFILE}" == libero-plus ]]; then
      ASSETS_DIR="${SOURCE_DIR}/libero/libero/assets"
      NESTED_ASSETS_DIR="${SOURCE_DIR}/libero/libero/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
      if [[ ! -d "${ASSETS_DIR}" ]]; then
        ARCHIVE="${SOURCE_DIR}/libero/libero/assets.zip"
        if [[ ! -f "${ARCHIVE}" ]]; then
          "${ENV_DIR}/bin/huggingface-cli" download Sylvest/LIBERO-plus assets.zip \
            --repo-type dataset --local-dir "${SOURCE_DIR}/libero/libero"
        fi
        unzip -q -o "${ARCHIVE}" -d "${SOURCE_DIR}/libero/libero"
        if [[ -d "${NESTED_ASSETS_DIR}" ]]; then mv "${NESTED_ASSETS_DIR}" "${ASSETS_DIR}"; fi
      fi
    fi
    "${ENV_DIR}/bin/python" "${SETUP_DIR}/configure_libero.py" \
      --source-root "${SOURCE_DIR}" --config-dir "${ROOT}/.runtime/${PROFILE}"
    ;;
  robotwin)
    SOURCE_DIR="${ROOT}/third_party/RoboTwin"
    [[ -d "${SOURCE_DIR}" ]] || { echo "Missing RoboTwin source: ${SOURCE_DIR}" >&2; exit 1; }
    "${ENV_DIR}/bin/python" "${SETUP_DIR}/patch_robotwin_env.py"
    CUROBO_ROOT="${SOURCE_DIR}/envs/curobo"
    fasterwam_clone_pinned "https://github.com/NVlabs/curobo.git" "v0.7.8" "${CUROBO_ROOT}"
    if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]] && ! "${ENV_DIR}/bin/python" -c \
      'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
      echo "Set TORCH_CUDA_ARCH_LIST for the target GPU when no build GPU is visible (A100: 8.0)." >&2; exit 1
    fi
    MAX_JOBS="${MAX_JOBS:-4}" CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS:-4}" \
      "${UV}" pip install --python "${ENV_DIR}/bin/python" --no-deps --no-build-isolation -e "${CUROBO_ROOT}"
    if [[ ! -d "${SOURCE_DIR}/assets/background_texture" || ! -d "${SOURCE_DIR}/assets/embodiments" || ! -d "${SOURCE_DIR}/assets/objects" ]]; then
      (cd "${SOURCE_DIR}" && PATH="${ENV_DIR}/bin:${PATH}" bash script/_download_assets.sh)
    fi
    ln -sfn "${ROOT}/experiments/robotwin/fasterwam_policy" "${SOURCE_DIR}/policy/fasterwam_policy"
    ;;
esac
