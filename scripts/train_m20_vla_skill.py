"""Train the high-level M20 VLA skill selector from public expert data.

This is supervised imitation learning.  The labels are the command stream of
the released M20 ONNX expert (or the target-bearing command used to generate a
successful M20 demonstration); no simulator reward is used.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from m20_vla_skill_model import COMMAND_SCALE, SKILL_NAMES, M20VLASkillPolicy


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_OUTPUT = DATA_ROOT / "checkpoints/m20_vla_skill_v1"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, action="append", default=None, help="Dataset directory; repeatable.")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--architecture", choices=["global_v1", "spatial_v2"], default="spatial_v2")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--command-loss-weight", type=float, default=1.0)
parser.add_argument("--skill-loss-weight", type=float, default=0.5)
parser.add_argument("--post-reach-steps", type=int, default=20, help="Keep at most this many stop frames after a successful target reach.")
parser.add_argument("--init-checkpoint", type=Path, default=None)
parser.add_argument("--freeze-backbone", action="store_true", help="Train only skill_head; used to add skills without forgetting navigation.")
parser.add_argument("--search-head", action="store_true", help="Add an isolated language search-intent head.")
parser.add_argument("--search-loss-weight", type=float, default=1.0)
args = parser.parse_args()
if args.freeze_backbone and args.init_checkpoint is None:
    parser.error("--freeze-backbone requires --init-checkpoint")


@dataclass
class Episode:
    path: Path
    task_text: str
    front: np.ndarray
    rear: np.ndarray
    lidar: np.ndarray
    proprio: np.ndarray
    command: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_text(text: str, max_length: int = 32) -> np.ndarray:
    values = np.frombuffer(text.encode("utf-8")[:max_length], dtype=np.uint8).astype(np.int64) + 1
    tokens = np.zeros(max_length, dtype=np.int64)
    tokens[: len(values)] = values
    return tokens


def default_dataset_dirs() -> list[Path]:
    names = ["public_m20_native_v1", "public_m20_native_backward_v1", "public_m20_native_turn_v2"]
    names.extend(sorted(path.name for path in (DATA_ROOT / "datasets").glob("m20_vla_sync_*")))
    names.extend(sorted(path.name for path in (DATA_ROOT / "datasets").glob("m20_skill_expert_*")))
    names.extend(sorted(path.name for path in (DATA_ROOT / "datasets").glob("m20_search_expert_*")))
    return [DATA_ROOT / "datasets" / name for name in names]


def downsample_rgb(images: np.ndarray) -> np.ndarray:
    return np.stack([cv2.resize(frame, (80, 48), interpolation=cv2.INTER_AREA) for frame in images], axis=0)


def load_episodes(dataset_dirs: list[Path]) -> list[Episode]:
    episodes: list[Episode] = []
    for directory in dataset_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
        for path in sorted(directory.glob("episode_*.h5")):
            with h5py.File(path, "r") as h5:
                task_text = str(h5.attrs.get("task_text", ""))
                stable_search = (
                    any(token in task_text for token in ("寻找", "搜索", "扫描"))
                    and int(h5.attrs.get("terminated_steps", 1)) == 0
                    and float(h5.attrs.get("min_root_height", 0.0)) >= 0.45
                )
                if not bool(h5.attrs.get("success", False)) and not stable_search:
                    print(f"[M20PRO-SKILL] skip unsuccessful episode={path}")
                    continue
                obs = h5["observation"]
                front = downsample_rgb(np.asarray(obs["front_rgb"], dtype=np.uint8))
                rear = downsample_rgb(np.asarray(obs["rear_rgb"], dtype=np.uint8))
                lidar = np.asarray(obs["lidar"], dtype=np.float32)
                proprio = np.asarray(obs["proprio"], dtype=np.float32)
                # Do not expose either the expert's command or its previous
                # low-level action to the high-level policy.  At replay time
                # those values come from the learned two-layer controller.
                proprio[:, 6:9] = 0.0
                proprio[:, 41:57] = 0.0
                if "expert_command" in h5:
                    command = np.asarray(h5["expert_command"], dtype=np.float32)
                else:
                    command = np.repeat(np.asarray(h5.attrs.get("command", (0.0, 0.0, 0.0)), dtype=np.float32)[None], len(front), axis=0)
                target_step = int(h5.attrs.get("target_reached_step", -1))
                if target_step >= 0 and bool(h5.attrs.get("stop_on_target", False)):
                    end = min(len(command), target_step + 1 + args.post_reach_steps)
                    front, rear, lidar, proprio, command = (
                        front[:end], rear[:end], lidar[:end], proprio[:end], command[:end]
                    )
                    command[target_step + 1 :] = 0.0
                if not (len(front) == len(rear) == len(lidar) == len(proprio) == len(command)):
                    raise ValueError(f"Length mismatch in {path}")
                if len(command) == 0 or not all(np.isfinite(x).all() for x in (lidar, proprio, command)):
                    raise ValueError(f"Non-finite or empty episode: {path}")
                episodes.append(Episode(path, task_text, front, rear, lidar, proprio, command))
    if not episodes:
        raise RuntimeError("No successful M20 expert episodes available")
    return episodes


def skill_id(command: np.ndarray, task_text: str = "") -> int:
    if any(token in task_text for token in ("寻找", "搜索", "扫描")):
        return SKILL_NAMES.index("search")
    forward, _, yaw = command
    # Slow approach commands (for example 0.045 m/s) are still forward
    # locomotion.  Only an explicit zero command is the stop skill.
    if float(np.max(np.abs(command))) < 1e-3:
        return SKILL_NAMES.index("stop")
    if yaw > 0.08:
        return SKILL_NAMES.index("left")
    if yaw < -0.08:
        return SKILL_NAMES.index("right")
    if forward < -0.05:
        return SKILL_NAMES.index("backward")
    return SKILL_NAMES.index("forward")


class FrameDataset(Dataset):
    def __init__(self, episodes: list[Episode], stride: int):
        self.episodes = episodes
        self.frames: list[tuple[int, int]] = []
        for episode_id, episode in enumerate(episodes):
            self.frames.extend((episode_id, step) for step in range(0, len(episode.command), stride))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, step = self.frames[index]
        episode = self.episodes[episode_id]
        rgb = np.concatenate((episode.front[step], episode.rear[step]), axis=-1).transpose(2, 0, 1)
        command = np.clip(episode.command[step], -COMMAND_SCALE.numpy(), COMMAND_SCALE.numpy())
        return {
            "rgb": torch.from_numpy(rgb.copy()).float().div_(255.0),
            "lidar": torch.from_numpy(episode.lidar[step].copy()).float().div_(20.0),
            "proprio": torch.from_numpy(episode.proprio[step].copy()).float(),
            "language": torch.from_numpy(encode_text(episode.task_text)),
            "command": torch.from_numpy(command / COMMAND_SCALE.numpy()).float(),
            "skill": torch.tensor(skill_id(episode.command[step], episode.task_text), dtype=torch.long),
            "search_target": torch.tensor(
                1.0 if any(token in episode.task_text for token in ("寻找", "搜索", "扫描")) else 0.0,
                dtype=torch.float32,
            ),
        }


def run_epoch(
    model: M20VLASkillPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    skill_weight: torch.Tensor,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    command_total = skill_total = correct = count = 0.0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            outputs = model(batch["rgb"], batch["lidar"], batch["proprio"], batch["language"])
            if args.search_head:
                command_pred, skill_logits, search_logit = outputs
            else:
                command_pred, skill_logits = outputs
            command_loss = nn.functional.smooth_l1_loss(command_pred, batch["command"])
            skill_loss = nn.functional.cross_entropy(skill_logits, batch["skill"], weight=skill_weight)
            search_loss = (
                nn.functional.binary_cross_entropy_with_logits(search_logit, batch["search_target"])
                if args.search_head else torch.zeros((), device=device)
            )
            loss = (
                args.command_loss_weight * command_loss
                + args.skill_loss_weight * skill_loss
                + args.search_loss_weight * search_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        size = float(batch["skill"].shape[0])
        command_total += float(command_loss.detach()) * size
        skill_total += float(skill_loss.detach()) * size
        correct += float((skill_logits.argmax(-1) == batch["skill"]).sum())
        count += size
    return command_total / max(count, 1.0), skill_total / max(count, 1.0), correct / max(count, 1.0)


def main() -> None:
    seed_everything(args.seed)
    dataset_dirs = args.dataset or default_dataset_dirs()
    episodes = load_episodes(dataset_dirs)
    grouped: dict[str, list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.task_text, []).append(episode)
    split_rng = random.Random(args.seed)
    train_episodes: list[Episode] = []
    val_episodes: list[Episode] = []
    for group in grouped.values():
        split_rng.shuffle(group)
        val_count = max(1, int(round(len(group) * args.val_fraction))) if len(group) > 1 else 0
        val_count = min(val_count, len(group) - 1)
        val_episodes.extend(group[:val_count])
        train_episodes.extend(group[val_count:])
    train_set = FrameDataset(train_episodes, args.stride)
    val_set = FrameDataset(val_episodes, args.stride) if val_episodes else None
    if not train_set:
        raise RuntimeError("Training split is empty")
    train_skills = torch.tensor([skill_id(ep.command[step], ep.task_text) for ep_id, step in train_set.frames for ep in [train_set.episodes[ep_id]]])
    counts = torch.bincount(train_skills, minlength=len(SKILL_NAMES)).float()
    skill_weight = (counts.sum() / counts.clamp_min(1.0)).sqrt()
    skill_weight = (skill_weight / skill_weight.mean()).to(args.device if torch.cuda.is_available() else "cpu")
    sample_weights = (1.0 / counts.clamp_min(1.0))[train_skills].double()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True) if val_set else None
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    skill_weight = skill_weight.to(device)
    model = M20VLASkillPolicy(args.architecture, search_head=args.search_head).to(device)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["model_state_dict"], strict=False)
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("search_head.") if args.search_head else (
                name.startswith("language.") or name.startswith("skill_head.")
            )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "m20_vla_skill_bc_v1",
        "architecture": args.architecture,
        "skills": list(SKILL_NAMES),
        "command_scale": COMMAND_SCALE.tolist(),
        "sensor_shapes": {"rgb": [6, 48, 80], "lidar": [72], "proprio": [57], "language": [32]},
        "proprio_mask": [6, 7, 8] + list(range(41, 57)),
        "post_reach_steps": args.post_reach_steps,
        "datasets": [str(directory) for directory in dataset_dirs],
        "train_episodes": [str(ep.path) for ep in train_episodes],
        "val_episodes": [str(ep.path) for ep in val_episodes],
        "expert": "AI-DA-STC/M20-autonomy-sim policy.onnx plus successful target-bearing command labels",
        "reward_used": False,
        "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        "freeze_backbone": args.freeze_backbone,
        "search_head": args.search_head,
        "search_loss_weight": args.search_loss_weight,
        "seed": args.seed,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_cmd, train_skill, train_acc = run_epoch(model, train_loader, optimizer, device, skill_weight)
        if val_loader is not None:
            val_cmd, val_skill, val_acc = run_epoch(model, val_loader, None, device, skill_weight)
        else:
            val_cmd, val_skill, val_acc = train_cmd, train_skill, train_acc
        scheduler.step()
        total_val = args.command_loss_weight * val_cmd + args.skill_loss_weight * val_skill
        row = {"epoch": epoch, "train_command_loss": train_cmd, "train_skill_loss": train_skill, "train_skill_accuracy": train_acc,
               "val_command_loss": val_cmd, "val_skill_loss": val_skill, "val_skill_accuracy": val_acc, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(f"[M20PRO-SKILL] epoch={epoch:03d}/{args.epochs} train_cmd={train_cmd:.5f} train_acc={train_acc:.3f} val_cmd={val_cmd:.5f} val_acc={val_acc:.3f}", flush=True)
        if total_val <= best:
            best = total_val
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch, "val_loss": total_val}, args.output_dir / "best.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": args.epochs}, args.output_dir / "last.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    coverage = {name: int((train_skills == index).sum()) for index, name in enumerate(SKILL_NAMES)}
    (args.output_dir / "label_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n")
    print(f"[M20PRO-SKILL] episodes={len(episodes)} train_frames={len(train_set)} val_frames={len(val_set) if val_set else 0}")
    print(f"[M20PRO-SKILL] label_coverage={coverage}")
    print(f"[M20PRO-SKILL] checkpoint={args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
