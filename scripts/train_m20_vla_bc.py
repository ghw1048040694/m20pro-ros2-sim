"""Train a compact language-conditioned M20 action-chunk behavior policy.

This is an imitation-learning/VLA baseline, not a reward optimizer. It learns
from the released native M20 expert trajectories and consumes the same sensor
modalities that the eventual navigation policy will see: front/rear RGB,
LiDAR, proprioception and a natural-language task string.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from m20_vla_model import M20VLAActionChunk


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_DATASETS = [
    DATA_ROOT / "datasets/public_m20_native_v1",
    DATA_ROOT / "datasets/public_m20_native_backward_v1",
    DATA_ROOT / "datasets/public_m20_native_turn_v2",
]
DEFAULT_OUTPUT = DATA_ROOT / "checkpoints/m20_vla_bc_v1"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, action="append", default=None, help="Dataset directory; repeatable.")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--horizon", type=int, default=8, help="Predicted action chunk length.")
parser.add_argument("--stride", type=int, default=2, help="Window stride in expert frames.")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Episode:
    path: Path
    task_text: str
    front: np.ndarray
    rear: np.ndarray
    lidar: np.ndarray
    proprio: np.ndarray
    action: np.ndarray


def downsample_rgb(images: np.ndarray) -> np.ndarray:
    return np.stack([cv2.resize(frame, (80, 48), interpolation=cv2.INTER_AREA) for frame in images], axis=0)


def load_episodes(dataset_dirs: list[Path]) -> list[Episode]:
    episodes: list[Episode] = []
    for directory in dataset_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
        paths = sorted(directory.glob("episode_*.h5"))
        if not paths:
            raise FileNotFoundError(f"No episode_*.h5 files in {directory}")
        for path in paths:
            with h5py.File(path, "r") as h5:
                if not bool(h5.attrs.get("success", False)):
                    print(f"[M20PRO-BC] skip diagnostic episode={path}")
                    continue
                obs = h5["observation"]
                front = downsample_rgb(np.asarray(obs["front_rgb"], dtype=np.uint8))
                rear = downsample_rgb(np.asarray(obs["rear_rgb"], dtype=np.uint8))
                lidar = np.asarray(obs["lidar"], dtype=np.float32)
                proprio = np.asarray(obs["proprio"], dtype=np.float32)
                action = np.asarray(h5["action"], dtype=np.float32)
                if not (len(front) == len(rear) == len(lidar) == len(proprio) == len(action)):
                    raise ValueError(f"Length mismatch in {path}")
                if len(action) < args.horizon:
                    continue
                if not all(np.isfinite(x).all() for x in (lidar, proprio, action)):
                    raise ValueError(f"Non-finite numeric data in {path}")
                episodes.append(
                    Episode(
                        path=path,
                        task_text=str(h5.attrs.get("task_text", "")),
                        front=front,
                        rear=rear,
                        lidar=lidar,
                        proprio=proprio,
                        action=action,
                    )
                )
    if not episodes:
        raise RuntimeError("No successful episodes available for BC training")
    return episodes


def encode_text(text: str, max_length: int = 32) -> np.ndarray:
    encoded = np.frombuffer(text.encode("utf-8")[:max_length], dtype=np.uint8).astype(np.int64) + 1
    tokens = np.zeros(max_length, dtype=np.int64)
    tokens[: len(encoded)] = encoded
    return tokens


class ChunkDataset(Dataset):
    def __init__(self, episodes: list[Episode], horizon: int, stride: int):
        self.episodes = episodes
        self.horizon = horizon
        self.windows: list[tuple[int, int]] = []
        for episode_id, episode in enumerate(episodes):
            self.windows.extend((episode_id, start) for start in range(0, len(episode.action) - horizon + 1, stride))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, start = self.windows[index]
        episode = self.episodes[episode_id]
        end = start + self.horizon
        rgb = np.concatenate((episode.front[start], episode.rear[start]), axis=-1).transpose(2, 0, 1)
        return {
            "rgb": torch.from_numpy(rgb.copy()).float().div_(255.0),
            "lidar": torch.from_numpy(episode.lidar[start].copy()).float().div_(20.0),
            "proprio": torch.from_numpy(episode.proprio[start].copy()).float(),
            "language": torch.from_numpy(encode_text(episode.task_text)),
            "target": torch.from_numpy(episode.action[start:end].copy()).float(),
        }


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loss_for(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    prediction = model(batch["rgb"], batch["lidar"], batch["proprio"], batch["language"])
    action_loss = torch.nn.functional.smooth_l1_loss(prediction, batch["target"])
    smooth_loss = torch.nn.functional.smooth_l1_loss(prediction[:, 1:], prediction[:, :-1])
    return action_loss + 0.02 * smooth_loss


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            loss = loss_for(model, batch)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach().item()) * batch["target"].shape[0]
        count += batch["target"].shape[0]
    return total / max(count, 1)


def main() -> None:
    seed_everything(args.seed)
    dataset_dirs = args.dataset or DEFAULT_DATASETS
    episodes = load_episodes(dataset_dirs)
    random.Random(args.seed).shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * args.val_fraction))) if len(episodes) > 1 else 0
    val_episodes = episodes[:val_count]
    train_episodes = episodes[val_count:] or episodes
    train_set = ChunkDataset(train_episodes, args.horizon, args.stride)
    val_set = ChunkDataset(val_episodes, args.horizon, args.stride) if val_episodes else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True) if val_set else None
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = M20VLAActionChunk(args.horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "m20_vla_action_chunk_bc_v1",
        "horizon": args.horizon,
        "stride": args.stride,
        "image_shape": [6, 48, 80],
        "lidar_shape": [72],
        "proprio_shape": [57],
        "action_shape": [16],
        "languages": sorted({episode.task_text for episode in episodes}),
        "datasets": [str(directory) for directory in dataset_dirs],
        "train_episodes": [str(episode.path) for episode in train_episodes],
        "val_episodes": [str(episode.path) for episode in val_episodes],
        "device": str(device),
        "seed": args.seed,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False) if val_loader else train_loss
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(
            f"[M20PRO-BC] epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} lr={row['lr']:.3e}",
            flush=True,
        )
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "horizon": args.horizon,
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                args.output_dir / "best.pt",
            )
    torch.save(
        {"model_state_dict": model.state_dict(), "horizon": args.horizon, "config": config, "epoch": args.epochs},
        args.output_dir / "last.pt",
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"[M20PRO-BC] episodes={len(episodes)} train_windows={len(train_set)} val_windows={len(val_set) if val_set else 0}")
    print(f"[M20PRO-BC] best_val_loss={best_val:.6f} checkpoint={args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
