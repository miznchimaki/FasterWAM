import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from fasterwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    if os.environ.get("FASTERWAM_LOG_CAPTURE"):
        # Persist a self-contained path in config.yaml, not an env interpolation.
        cfg.output_dir = str(cfg.output_dir)

    # Resolve the same Hydra overrides as training, before the launcher starts.
    # This mode must not import Torch, initialize CUDA, or load a dataset/model.
    output_dir_file = os.environ.get("FASTERWAM_RESOLVE_OUTPUT_DIR_FILE")
    if output_dir_file:
        from hydra.utils import to_absolute_path

        Path(output_dir_file).write_text(to_absolute_path(str(cfg.output_dir)), encoding="utf-8")
        return

    from fasterwam.runtime import run_training

    run_training(cfg)


if __name__ == "__main__":
    main()
