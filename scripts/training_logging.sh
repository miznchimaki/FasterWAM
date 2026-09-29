#!/usr/bin/env bash
# Shared by the ZeRO launchers. TRAIN_ARGS contains the Hydra training arguments.

run_training_with_log() {
  local resolve_dir output_dir log_file resolve_status arg
  local -a pipeline_status
  # One distributed launcher owns one experiment directory. A Hydra sweep
  # needs a separate launcher per run, otherwise its workers share log paths.
  for arg in "${TRAIN_ARGS[@]}"; do
    case "${arg}" in
      -m|--multirun|hydra.mode=MULTIRUN)
        echo "Error: use one training launcher invocation per experiment; Hydra multirun is not supported here." >&2
        return 2
        ;;
    esac
  done
  resolve_dir="$(mktemp -d)" || return

  # Let Hydra resolve the final output_dir (including user overrides and
  # interpolations). Do not create Hydra logs/config snapshots for this probe.
  if FASTERWAM_RESOLVE_OUTPUT_DIR_FILE="${resolve_dir}/output_dir" \
    python scripts/train.py "${TRAIN_ARGS[@]}" \
      hydra.run.dir=. hydra.job.chdir=false hydra.output_subdir=null \
      hydra/job_logging=stdout \
      >"${resolve_dir}/resolve.log" 2>&1; then
    if [[ ! -f "${resolve_dir}/output_dir" ]]; then
      # Hydra --help / --cfg / --info modes do not invoke the training function.
      cat -- "${resolve_dir}/resolve.log"
      rm -rf -- "${resolve_dir}"
      return 0
    fi
  else
    resolve_status=$?
    cat -- "${resolve_dir}/resolve.log" >&2
    rm -rf -- "${resolve_dir}"
    return "${resolve_status}"
  fi

  output_dir="$(cat -- "${resolve_dir}/output_dir")"
  if ! mkdir -p -- "${output_dir}"; then
    rm -rf -- "${resolve_dir}"
    return 1
  fi
  log_file="${output_dir}/train.log"
  # Each node captures all of its local workers. Separate files avoid competing
  # writers when multi-node jobs use a shared output directory.
  if (( MACHINE_RANK != 0 )); then
    log_file="${output_dir}/train.node${MACHINE_RANK}.log"
  fi
  if ! touch -- "${log_file}"; then
    rm -rf -- "${resolve_dir}"
    return 1
  fi

  export PYTHONUNBUFFERED=1
  export FASTERWAM_RUN_OUTPUT_DIR="${output_dir}"
  export FASTERWAM_LOG_CAPTURE=1

  # Merge at the launcher boundary: Python/native stdout, stderr, all local
  # ranks, DataLoader children, and Accelerate/DeepSpeed startup/error output.
  # Append so restarting a run never truncates its existing transcript.
  if {
    cat -- "${resolve_dir}/resolve.log"
    rm -rf -- "${resolve_dir}"
    if [[ -n "${RUN_ID_SYNC_MESSAGE:-}" ]]; then
      echo "${RUN_ID_SYNC_MESSAGE}"
    fi
    echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID}"
    echo "[launch] output_dir=${output_dir}"
    echo "[launch] log_file=${log_file}"
    "$@" "${TRAIN_ARGS[@]}" \
      'output_dir=${oc.env:FASTERWAM_RUN_OUTPUT_DIR}' \
      hydra/job_logging=stdout
  } 2>&1 | tee -a -- "${log_file}"; then
    return 0
  else
    pipeline_status=("${PIPESTATUS[@]}")
    # A successful tee must not hide a failed training process.
    if (( pipeline_status[0] != 0 )); then
      return "${pipeline_status[0]}"
    fi
    return "${pipeline_status[1]}"
  fi
}
