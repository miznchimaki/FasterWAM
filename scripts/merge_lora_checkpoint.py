"""Export a complete FasterWAM LoRA .pt as dense inference weights on CPU."""

import argparse
from pathlib import Path
import sys


_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge one complete FasterWAM LoRA checkpoint into dense inference weights (CPU only).",
        epilog="The source remains unmerged for training/resume. ZeRO state directories must first be consolidated "
               "to a full mot .pt. Merging uses FP32 accumulation and a final cast to each original base dtype.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Complete mot .pt with LoRA metadata and base weights")
    parser.add_argument("--output", type=Path, required=True, help="New dense .pt path; existing files are never overwritten")
    args = parser.parse_args(argv)
    from fasterwam.utils.lora_merge import merge_lora_checkpoint

    try:
        output = merge_lora_checkpoint(args.input, args.output)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"LoRA export failed: {error}\n")
    print(f"Saved dense inference checkpoint: {output}")


if __name__ == "__main__":
    main()
