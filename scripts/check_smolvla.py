#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


EXPECTED_MODEL_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
SMOLVLA_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
SMOLVLM_PROCESSOR_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def parse_args() -> argparse.Namespace:
    data_root = Path(
        os.environ.get(
            "M20PRO_VLA_DATA_ROOT",
            "/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA",
        )
    )
    parser = argparse.ArgumentParser(description="Load and run the pinned SmolVLA base checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "M20PRO_SMOLVLA_CHECKPOINT",
                data_root / "models/lerobot_smolvla_base_c83c316",
            )
        ),
    )
    parser.add_argument(
        "--processor",
        type=Path,
        default=Path(
            os.environ.get(
                "M20PRO_SMOLVLM_PROCESSOR",
                data_root / "models/smolvlm2_500m_processor_7b375e1",
            )
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=data_root / "logs/smolvla_base_smoke_v1.json",
    )
    parser.add_argument("--task-text", default="Go to the red object.")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model_file = args.checkpoint / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"SmolVLA checkpoint not found: {model_file}")
    if not (args.processor / "config.json").is_file():
        raise FileNotFoundError(f"SmolVLM processor/config not found: {args.processor}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SmolVLA smoke test")

    model_sha256 = None
    if not args.skip_sha256:
        model_sha256 = sha256_file(model_file)
        if model_sha256 != EXPECTED_MODEL_SHA256:
            raise RuntimeError(
                f"SmolVLA SHA-256 mismatch: expected {EXPECTED_MODEL_SHA256}, got {model_sha256}"
            )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.vlm_model_name = str(args.processor)
    config.load_vlm_weights = False
    config.device = "cuda"

    load_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint,
        config=config,
        local_files_only=True,
        strict=True,
    )
    load_seconds = time.perf_counter() - load_started

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    encoded = tokenizer(
        f"{args.task_text}\n",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.tokenizer_max_length,
    )
    batch = {
        "observation.state": torch.zeros(1, 6, device="cuda"),
        OBS_LANGUAGE_TOKENS: encoded.input_ids.to("cuda"),
        OBS_LANGUAGE_ATTENTION_MASK: encoded.attention_mask.to("cuda").bool(),
    }
    for key in config.image_features:
        batch[key] = torch.full((1, 3, 256, 256), 0.5, device="cuda")

    inference_started = time.perf_counter()
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    metrics = {
        "schema": "m20pro_smolvla_smoke_v1",
        "smolvla_repo": "lerobot/smolvla_base",
        "smolvla_revision": SMOLVLA_REVISION,
        "smolvlm_processor_repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "smolvlm_processor_revision": SMOLVLM_PROCESSOR_REVISION,
        "checkpoint": str(args.checkpoint),
        "model_sha256": model_sha256,
        "parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
        "task_text": args.task_text,
        "image_features": list(config.image_features),
        "action_shape": list(action_chunk.shape),
        "action_finite": bool(torch.isfinite(action_chunk).all().item()),
        "action_min": float(action_chunk.min().item()),
        "action_max": float(action_chunk.max().item()),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "cuda_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "success": bool(torch.isfinite(action_chunk).all().item()),
        "scope": "Checkpoint integrity and consumer-GPU inference only; not a navigation result.",
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
