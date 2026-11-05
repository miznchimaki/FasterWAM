"""Offline conversion of complete FasterWAM LoRA weights to dense weights.

Only PyTorch is required: no PEFT import, model construction, YAML, or dataset.
The format describes standard, unmerged Linear LoRA with alpha/r scaling. Each
layer is accumulated in CPU FP32 and cast once to its original base dtype. This
can differ by rounding from PEFT 0.14's CPU half-precision merge, which rounds
the delta before adding it to the base weight.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
import tempfile
import zipfile

import torch

from .lora_checkpoint import _read_lora_metadata


_MERGE_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
_LAYER_KEYS = {"base_layer.weight", "base_layer.bias", "lora_A.default.weight", "lora_B.default.weight"}
_REQUIRED_LAYER_KEYS = _LAYER_KEYS - {"base_layer.bias"}


def _matches(path: str, suffix: str) -> bool:
    return path == suffix or path.endswith("." + suffix)


def _check_tensor(value, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized or value.is_complex():
        raise ValueError(f"{name} must be a real, unquantized, strided CPU tensor.")
    if value.numel() == 0 or hasattr(value, "ds_id"):
        raise ValueError(f"{name} is empty or partitioned; consolidate ZeRO weights to a complete .pt first.")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values.")


def _check_mapping(state, name: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} must be a nonempty state-dict mapping.")
    for key, value in state.items():
        if not isinstance(key, str) or not key or any(not part for part in key.split(".")):
            raise ValueError(f"{name} contains an invalid tensor key: {key!r}.")
        _check_tensor(value, f"{name}.{key}")


def _validate_payload(payload):
    if not isinstance(payload, Mapping) or "mot" not in payload:
        raise ValueError("Expected one complete FasterWAM 'mot' .pt; consolidate ZeRO shards before exporting.")
    configs = _read_lora_metadata(payload)
    if configs is None:
        raise ValueError("LoRA metadata is required; adapter scaling must not be inferred from tensor shapes.")
    state = payload["mot"]
    _check_mapping(state, "mot")
    layers = {}
    dense_keys = set()
    branches = set()
    fusion_indices = set()
    for key in state:
        parts = key.split(".")
        branch = parts[1] if len(parts) > 2 and parts[0] == "mixtures" else None
        if branch in ("video", "action"):
            branches.add(branch)
        elif re.fullmatch(r"video_kv_fusion_logits\.[0-9]+", key):
            fusion_indices.add(int(parts[-1]))
            if state[key].ndim != 1:
                raise ValueError(f"Fusion logits must be vectors: {key}.")
            dense_keys.add(key)
            continue
        else:
            raise ValueError(f"Unsupported mot tensor namespace: {key}.")

        special = next((index for index, part in enumerate(parts)
                        if part == "base_layer" or part == "modules_to_save" or part.startswith("lora_")), None)
        if special is None:
            dense_keys.add(key)
            continue
        path, leaf = ".".join(parts[:special]), ".".join(parts[special:])
        if special < 3 or leaf not in _LAYER_KEYS:
            raise ValueError(f"Unsupported adapter layout or adapter name: {key}.")
        layers.setdefault(path, {})[leaf] = state[key]

    if branches != {"video", "action"}:
        raise ValueError("A complete mot checkpoint must contain both video and action experts.")
    if fusion_indices and fusion_indices != set(range(max(fusion_indices) + 1)):
        raise ValueError("Fusion logits have missing or duplicate layer indices.")
    matches = {branch: set() for branch in configs}
    for path, weights in layers.items():
        branch, relative = path.split(".", 2)[1:]
        config = configs[branch]
        if not config["enabled"]:
            raise ValueError(f"Adapter tensors occur in disabled {branch} expert: {path}.")
        suffixes = {suffix for suffix in config["target_modules"] if _matches(relative, suffix)}
        if not suffixes:
            raise ValueError(f"Adapter layer is absent from metadata target_modules: {path}.")
        matches[branch].update(suffixes)
        missing = _REQUIRED_LAYER_KEYS - weights.keys()
        if missing:
            raise ValueError(f"Incomplete adapter layer {path}: missing {sorted(missing)}.")
        base, a, b = (weights[name] for name in
                      ("base_layer.weight", "lora_A.default.weight", "lora_B.default.weight"))
        if any(value.dtype not in _MERGE_DTYPES for value in (base, a, b)):
            raise ValueError(f"Only FP32, FP16, and BF16 Linear LoRA weights are supported: {path}.")
        if base.ndim != 2 or a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"LoRA weights must be 2D Linear matrices: {path}.")
        rank = config["r"]
        if tuple(a.shape) != (rank, base.shape[1]) or tuple(b.shape) != (base.shape[0], rank):
            raise ValueError(f"LoRA rank/shape mismatch at {path}: base={tuple(base.shape)}, "
                             f"A={tuple(a.shape)}, B={tuple(b.shape)}, metadata r={rank}.")
        if "base_layer.bias" in weights and tuple(weights["base_layer.bias"].shape) != (base.shape[0],):
            raise ValueError(f"Linear base bias shape mismatch at {path}.")
        for name in ("weight", "bias"):
            if path + "." + name in dense_keys:
                raise ValueError(f"Dense/adapter key collision at {path}.{name}.")

    for branch, config in configs.items():
        if not config["enabled"]:
            continue
        missing = set(config["target_modules"]) - matches[branch]
        if missing:
            raise ValueError(f"Metadata targets have no complete adapters in {branch}: {sorted(missing)}.")
        # A second layer matching the same suffix must not silently stay dense.
        prefix = f"mixtures.{branch}."
        for key in dense_keys:
            if key.startswith(prefix) and key.endswith((".weight", ".bias")):
                path = key[len(prefix):].rsplit(".", 1)[0]
                if any(_matches(path, suffix) for suffix in config["target_modules"]):
                    raise ValueError(f"Metadata-targeted layer lacks adapter structure: {key}.")

    if "proprio_encoder" in payload:
        proprio = payload["proprio_encoder"]
        _check_mapping(proprio, "proprio_encoder")
        if set(proprio) != {"weight", "bias"}:
            raise ValueError("proprio_encoder must contain the complete Linear weight and bias.")
        if proprio["weight"].ndim != 2 or tuple(proprio["bias"].shape) != (proprio["weight"].shape[0],):
            raise ValueError("proprio_encoder Linear weight/bias shapes do not match.")
    return configs, layers, dense_keys


@torch.no_grad()
def merge_lora_checkpoint_payload(payload: Mapping) -> dict:
    """Return dense inference weights without modifying the supplied payload.

    Validate the serialized layout and every adapter before merging. Without a
    model configuration this cannot detect a non-adapter layer deleted in its
    entirety; the evaluation model's strict loader performs that final check.
    Unchanged tensors are shared with the input rather than copied. Optimizer,
    scheduler, and other training state are deliberately absent from the result.
    """
    configs, layers, dense_keys = _validate_payload(payload)
    state = {key: value for key, value in payload["mot"].items() if key in dense_keys}
    for path, weights in layers.items():
        config = configs[path.split(".")[1]]
        base = weights["base_layer.weight"]
        # Allocate FP32 intermediates for one layer, never a full FP32 model.
        merged = weights["lora_B.default.weight"].float() @ weights["lora_A.default.weight"].float()
        merged.mul_(config["lora_alpha"] / config["r"])
        merged.add_(base.float())
        _check_tensor(merged, f"merged {path}.weight (FP32)")
        merged = merged.to(dtype=base.dtype)
        _check_tensor(merged, f"merged {path}.weight ({base.dtype})")
        state[path + ".weight"] = merged
        if "base_layer.bias" in weights:
            state[path + ".bias"] = weights["base_layer.bias"]
    result = {"mot": state}
    for key in ("step", "torch_dtype", "proprio_encoder"):
        if key in payload:
            result[key] = payload[key]
    return result


def merge_lora_checkpoint(input_path: str | os.PathLike, output_path: str | os.PathLike) -> Path:
    """Export a single complete .pt on CPU; never overwrite an existing path.

    The result is published only after a successful save. DeepSpeed state
    directories and raw shards must first be consolidated to FasterWAM's full
    weights format. No training checkpoint or resume state is changed.
    """
    source = Path(input_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().absolute()
    if not source.is_file():
        raise ValueError("Input must be one complete .pt file, not a ZeRO checkpoint directory.")
    if source == output.resolve():
        raise ValueError("Input and output must be different files.")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    # mmap limits input RAM for modern torch.save archives; older full-weight
    # files remain supported without mmap. Neither path initializes a model.
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(source))
    dense = merge_lora_checkpoint_payload(payload)
    del payload  # Release optimizer/adapter/source weights no longer referenced by the dense result.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(dense, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace/rename, link atomically fails if a concurrent writer
        # created the requested output after our initial existence check.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
