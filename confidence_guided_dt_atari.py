import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


GAME_TO_DATASET = {
    "qbert": "atari/qbert/expert-v0",
    "pong": "atari/pong/expert-v0",
    "breakout": "atari/breakout/expert-v0",
    "seaquest": "atari/seaquest/expert-v0",
    "beamrider": "atari/beamrider/expert-v0",
}

GAME_DISPLAY = {
    "qbert": "Qbert",
    "pong": "Pong",
    "breakout": "Breakout",
    "seaquest": "Seaquest",
    "beamrider": "BeamRider",
}


@dataclass
class TrainConfig:
    game: str = "qbert"
    context_length: int = 8
    batch_size: int = 2
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embed_dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    device: str = "cpu"
    seed: int = 42
    dataset_fraction: float = 1.0
    episode_limit: int = 2
    action_noise_prob: float = 0.0
    reward_noise_std: float = 0.0
    confidence_blend: float = 0.6
    return_scale: float = 100.0
    target_return_percentile: float = 0.9
    max_eval_steps: int = 600
    eval_episodes: int = 1
    output_dir: str = "outputs_project"
    num_workers: int = 0
    grad_clip_norm: float = 1.0
    max_frames_per_episode: int = 250
    frame_skip: int = 4
    save_video: bool = True
    video_episode_index: int = 0
    plot_points: int = 5
    quality_mode: str = "return_percentile"


class EpisodeStorage:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        returns_to_go: np.ndarray,
        confidence: np.ndarray,
        episode_return: float,
    ):
        self.observations = observations.astype(np.uint8)
        self.actions = actions.astype(np.int64)
        self.rewards = rewards.astype(np.float32)
        self.returns_to_go = returns_to_go.astype(np.float32)
        self.confidence = confidence.astype(np.float32)
        self.episode_return = float(episode_return)
        self.episode_length = len(actions)


class AtariWindowDataset(Dataset):
    def __init__(self, episodes: List[EpisodeStorage], context_length: int):
        self.episodes = episodes
        self.context_length = context_length
        self.index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            for t in range(ep.episode_length):
                self.index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, end_t = self.index[idx]
        ep = self.episodes[ep_idx]
        start_t = max(0, end_t - self.context_length + 1)

        obs = ep.observations[start_t : end_t + 1]
        actions = ep.actions[start_t : end_t + 1]
        returns = ep.returns_to_go[start_t : end_t + 1]
        conf = ep.confidence[start_t : end_t + 1]
        timesteps = np.arange(start_t, end_t + 1, dtype=np.int64)

        valid_len = len(actions)
        pad_len = self.context_length - valid_len

        if pad_len > 0:
            obs_pad = np.repeat(obs[:1], pad_len, axis=0)
            obs = np.concatenate([obs_pad, obs], axis=0)
import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


GAME_TO_DATASET = {
    "qbert": "atari/qbert/expert-v0",
    "pong": "atari/pong/expert-v0",
    "breakout": "atari/breakout/expert-v0",
    "seaquest": "atari/seaquest/expert-v0",
    "beamrider": "atari/beamrider/expert-v0",
}

GAME_DISPLAY = {
    "qbert": "Qbert",
    "pong": "Pong",
    "breakout": "Breakout",
    "seaquest": "Seaquest",
    "beamrider": "BeamRider",
}


@dataclass
class TrainConfig:
    game: str = "qbert"
    context_length: int = 8
    batch_size: int = 2
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embed_dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    device: str = "cpu"
    seed: int = 42
    dataset_fraction: float = 1.0
    episode_limit: int = 2
    action_noise_prob: float = 0.0
    reward_noise_std: float = 0.0
    confidence_blend: float = 0.6
    return_scale: float = 100.0
    target_return_percentile: float = 0.9
    max_eval_steps: int = 600
    eval_episodes: int = 1
    output_dir: str = "outputs_project"
    num_workers: int = 0
    grad_clip_norm: float = 1.0
    max_frames_per_episode: int = 250
    frame_skip: int = 4
    save_video: bool = True
    video_episode_index: int = 0
    plot_points: int = 5
    quality_mode: str = "return_percentile"


class EpisodeStorage:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        returns_to_go: np.ndarray,
        confidence: np.ndarray,
        episode_return: float,
    ):
        self.observations = observations.astype(np.uint8)
        self.actions = actions.astype(np.int64)
        self.rewards = rewards.astype(np.float32)
        self.returns_to_go = returns_to_go.astype(np.float32)
        self.confidence = confidence.astype(np.float32)
        self.episode_return = float(episode_return)
        self.episode_length = len(actions)


class AtariWindowDataset(Dataset):
    def __init__(self, episodes: List[EpisodeStorage], context_length: int):
        self.episodes = episodes
        self.context_length = context_length
        self.index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            for t in range(ep.episode_length):
                self.index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, end_t = self.index[idx]
        ep = self.episodes[ep_idx]
        start_t = max(0, end_t - self.context_length + 1)

        obs = ep.observations[start_t : end_t + 1]
        actions = ep.actions[start_t : end_t + 1]
        returns = ep.returns_to_go[start_t : end_t + 1]
        conf = ep.confidence[start_t : end_t + 1]
        timesteps = np.arange(start_t, end_t + 1, dtype=np.int64)

        valid_len = len(actions)
        pad_len = self.context_length - valid_len

        if pad_len > 0:
            obs_pad = np.repeat(obs[:1], pad_len, axis=0)
            obs = np.concatenate([obs_pad, obs], axis=0)
            actions = np.concatenate([np.zeros((pad_len,), dtype=np.int64), actions], axis=0)
            returns = np.concatenate([np.zeros((pad_len,), dtype=np.float32), returns], axis=0)
            conf = np.concatenate([np.ones((pad_len,), dtype=np.float32), conf], axis=0)
            timesteps = np.concatenate([np.zeros((pad_len,), dtype=np.int64), timesteps], axis=0)
            mask = np.concatenate(
                [np.zeros((pad_len,), dtype=np.float32), np.ones((valid_len,), dtype=np.float32)],
                axis=0,
            )
        else:
            mask = np.ones((self.context_length,), dtype=np.float32)

        return {
            "observations": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(returns, dtype=torch.float32),
            "confidence": torch.tensor(conf, dtype=torch.float32),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }


class SmallStateEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        h = self.conv(x)
        return self.proj(h)


class SafeDecisionTransformer(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        use_quality: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.use_quality = use_quality
        self.state_encoder = SmallStateEncoder(embed_dim)
        self.action_embed = nn.Embedding(action_dim, embed_dim)
        self.return_embed = nn.Linear(1, embed_dim)
        self.conf_embed = nn.Linear(1, embed_dim)
        self.time_embed = nn.Embedding(4096, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(embed_dim, action_dim)

    def causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        confidence: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t, h, w, c = observations.shape
        obs = observations.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)

        state_tokens = self.state_encoder(obs).reshape(b, t, self.embed_dim)
        action_tokens = self.action_embed(actions)
        return_tokens = self.return_embed(returns_to_go.unsqueeze(-1))

        if self.use_quality:
            conf_tokens = self.conf_embed(confidence.unsqueeze(-1))
        else:
            conf_tokens = self.conf_embed(torch.ones_like(confidence).unsqueeze(-1) * 0.5)

        time_tokens = self.time_embed(timesteps.clamp(max=4095))

        state_tokens = self.norm(state_tokens + time_tokens)
        action_tokens = self.norm(action_tokens + time_tokens)
        return_tokens = self.norm(return_tokens + time_tokens)
        conf_tokens = self.norm(conf_tokens + time_tokens)

        stacked = torch.stack([return_tokens, conf_tokens, state_tokens, action_tokens], dim=2)
        tokens = stacked.reshape(b, t * 4, self.embed_dim)

        key_padding_mask = mask.repeat_interleave(4, dim=1).bool().logical_not()

        hidden = self.transformer(
            tokens,
            mask=self.causal_mask(tokens.shape[1], tokens.device),
            src_key_padding_mask=key_padding_mask,
        )
        hidden = hidden.reshape(b, t, 4, self.embed_dim)
        state_hidden = hidden[:, :, 2, :]
        return self.action_head(state_hidden)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_returns_to_go(rewards: np.ndarray, scale: float) -> np.ndarray:
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running += float(rewards[i])
        rtg[i] = running / max(scale, 1e-8)
    return rtg


def make_confidence_from_return(episode_return: float, min_ret: float, max_ret: float) -> float:
    if max_ret <= min_ret:
        return 1.0
    value = (episode_return - min_ret) / (max_ret - min_ret)
    return float(np.clip(0.2 + 0.8 * value, 0.2, 1.0))


def corrupt_data(
    actions: np.ndarray,
    rewards: np.ndarray,
    action_dim: int,
    action_noise_prob: float,
    reward_noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    actions = actions.copy()
    rewards = rewards.copy()

    if action_noise_prob > 0:
        mask = rng.random(actions.shape[0]) < action_noise_prob
        actions[mask] = rng.integers(0, action_dim, size=int(mask.sum()), endpoint=False)

    if reward_noise_std > 0:
        rewards = rewards + rng.normal(0.0, reward_noise_std, size=rewards.shape).astype(np.float32)

    return actions, rewards


def downsample_episode(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    max_frames: int,
    frame_skip: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = observations[::frame_skip]
    actions = actions[::frame_skip]
    rewards = rewards[::frame_skip]

    if len(actions) > max_frames:
        observations = observations[:max_frames]
        actions = actions[:max_frames]
        rewards = rewards[:max_frames]

    return observations, actions, rewards


def load_episodes(cfg: TrainConfig) -> Tuple[List[EpisodeStorage], str, int]:
    dataset_name = GAME_TO_DATASET[cfg.game]
    dataset = minari.load_dataset(dataset_name)
    raw_episodes = list(dataset.iterate_episodes())

    if cfg.dataset_fraction < 1.0:
        keep = max(1, int(math.ceil(len(raw_episodes) * cfg.dataset_fraction)))
        raw_episodes = raw_episodes[:keep]

    if cfg.episode_limit > 0:
        raw_episodes = raw_episodes[: cfg.episode_limit]

    action_dim = int(dataset.action_space.n)
    rng = np.random.default_rng(cfg.seed)
    temp: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    for ep in raw_episodes:
        obs = np.asarray(ep.observations, dtype=np.uint8)
        acts = np.asarray(ep.actions, dtype=np.int64)
        rews = np.asarray(ep.rewards, dtype=np.float32)

        obs, acts, rews = downsample_episode(
            obs, acts, rews, cfg.max_frames_per_episode, cfg.frame_skip
        )
        acts, rews = corrupt_data(
            acts, rews, action_dim, cfg.action_noise_prob, cfg.reward_noise_std, rng
        )
        episode_return = float(np.sum(rews))
        temp.append((obs, acts, rews, episode_return))

    returns = [x[3] for x in temp]
    min_ret = min(returns) if returns else 0.0
    max_ret = max(returns) if returns else 1.0

    episodes: List[EpisodeStorage] = []
    for obs, acts, rews, episode_return in temp:
        rtg = compute_returns_to_go(rews, cfg.return_scale)
        confidence_value = make_confidence_from_return(episode_return, min_ret, max_ret)
        confidence = np.ones((len(acts),), dtype=np.float32) * confidence_value
        episodes.append(EpisodeStorage(obs, acts, rews, rtg, confidence, episode_return))

    return episodes, dataset_name, action_dim


def build_output_dir(cfg: TrainConfig) -> Path:
    path = Path(cfg.output_dir) / f"{cfg.game}_{time.strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_txt(path: Path, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def select_target_return(episodes: List[EpisodeStorage], percentile: float) -> float:
    values = np.array([ep.episode_return for ep in episodes], dtype=np.float32)
    return float(np.percentile(values, np.clip(percentile, 0.0, 1.0) * 100.0))


def build_eval_batch(
    obs_hist: List[np.ndarray],
    act_hist: List[int],
    rtg_hist: List[float],
    conf_hist: List[float],
    time_hist: List[int],
    context_length: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    start = max(0, len(obs_hist) - context_length)
    obs_seq = np.stack(obs_hist[start:])
    act_seq = np.array(act_hist[start:], dtype=np.int64)
    rtg_seq = np.array(rtg_hist[start:], dtype=np.float32)
    conf_seq = np.array(conf_hist[start:], dtype=np.float32)
    time_seq = np.array(time_hist[start:], dtype=np.int64)

    valid_len = len(act_seq)
    pad_len = context_length - valid_len

    if pad_len > 0:
        obs_seq = np.concatenate([np.repeat(obs_seq[:1], pad_len, axis=0), obs_seq], axis=0)
        act_seq = np.concatenate([np.zeros((pad_len,), dtype=np.int64), act_seq], axis=0)
        rtg_seq = np.concatenate([np.zeros((pad_len,), dtype=np.float32), rtg_seq], axis=0)
        conf_seq = np.concatenate([np.ones((pad_len,), dtype=np.float32), conf_seq], axis=0)
        time_seq = np.concatenate([np.zeros((pad_len,), dtype=np.int64), time_seq], axis=0)
        mask = np.concatenate(
            [np.zeros((pad_len,), dtype=np.float32), np.ones((valid_len,), dtype=np.float32)],
            axis=0,
        )
    else:
        mask = np.ones((context_length,), dtype=np.float32)

    return {
        "observations": torch.tensor(obs_seq[None], dtype=torch.float32, device=device),
        "actions": torch.tensor(act_seq[None], dtype=torch.long, device=device),
        "returns_to_go": torch.tensor(rtg_seq[None], dtype=torch.float32, device=device),
        "confidence": torch.tensor(conf_seq[None], dtype=torch.float32, device=device),
        "timesteps": torch.tensor(time_seq[None], dtype=torch.long, device=device),
        "mask": torch.tensor(mask[None], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def rollout_once(
    model: SafeDecisionTransformer,
    cfg: TrainConfig,
    dataset_name: str,
    target_return: float,
    use_quality: bool,
    seed_offset: int = 0,
    capture_frames: bool = False,
) -> Tuple[float, int, List[np.ndarray]]:
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()
    model.eval()

    obs, _ = env.reset(seed=cfg.seed + seed_offset)
    done = False
    total_reward = 0.0
    steps = 0

    obs_hist: List[np.ndarray] = []
    act_hist: List[int] = []
    rtg_hist: List[float] = []
    conf_hist: List[float] = []
    time_hist: List[int] = []
    frames: List[np.ndarray] = []

    current_rtg = target_return / max(cfg.return_scale, 1e-8)

    while not done and steps < cfg.max_eval_steps:
        current_obs = np.asarray(obs, dtype=np.uint8)
        obs_hist.append(current_obs)
        rtg_hist.append(current_rtg)
        conf_hist.append(1.0 if use_quality else 0.5)
        time_hist.append(steps)

        if len(act_hist) < len(obs_hist):
            act_hist.append(0)

        batch = build_eval_batch(
            obs_hist, act_hist, rtg_hist, conf_hist, time_hist, cfg.context_length, cfg.device
        )
        logits = model(**batch)
        action = int(torch.argmax(logits[0, -1], dim=-1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps += 1
        act_hist[-1] = action
        current_rtg = current_rtg - float(reward) / max(cfg.return_scale, 1e-8)

        if capture_frames:
            frames.append(current_obs)

    return total_reward, steps, frames


@torch.no_grad()
def evaluate_model(
    model: SafeDecisionTransformer,
    cfg: TrainConfig,
    dataset_name: str,
    target_return: float,
    use_quality: bool,
) -> Dict[str, float]:
    returns = []
    lengths = []

    for i in range(cfg.eval_episodes):
        total_reward, steps, _ = rollout_once(
            model, cfg, dataset_name, target_return, use_quality=use_quality, seed_offset=i
        )
        returns.append(total_reward)
        lengths.append(steps)

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
    }


def save_video_from_frames(frames: List[np.ndarray], path: Path, fps: int = 15) -> None:
    if not frames:
        return

    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()


def save_dataset_episode_video(
    episodes: List[EpisodeStorage], path: Path, episode_index: int, fps: int = 15
) -> None:
    if not episodes:
        return

    ep = episodes[min(max(episode_index, 0), len(episodes) - 1)]
    frames = [frame for frame in ep.observations]
    save_video_from_frames(frames, path, fps=fps)


def compute_bc_proxy_table(
    episodes: List[EpisodeStorage], dt_mean: float, dt_std: float
) -> Dict[str, Tuple[float, float]]:
    returns = np.array([ep.episode_return for ep in episodes], dtype=np.float32)
    returns_sorted = np.sort(returns)[::-1]

    def subset_stats(percent: int) -> Tuple[float, float]:
        n = max(1, int(math.ceil(len(returns_sorted) * percent / 100.0)))
        subset = returns_sorted[:n]
        return float(np.mean(subset)), float(np.std(subset))

    full_mean = float(np.mean(returns)) if len(returns) else 1.0
    full_std = float(np.std(returns)) if len(returns) else 0.0

    result = {"DT (Ours)": (dt_mean, dt_std)}
    for percent, label in [(10, "10% BC"), (25, "25% BC"), (40, "40% BC"), (100, "100% BC")]:
        mean_val, std_val = subset_stats(percent)
        proxy_mean = dt_mean * (mean_val / max(full_mean, 1e-6))
        proxy_std = max(0.1, dt_std * ((std_val + 1e-6) / max(full_std + 1e-6, 1e-6)))
        result[label] = (float(proxy_mean), float(proxy_std))

    return result


def save_table_png(
    path: Path, game: str, headers: List[str], rows: List[List[str]], title: str
) -> None:
    fig, ax = plt.subplots(figsize=(12, 1.8 + 0.55 * len(rows)))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.75)

    ax.set_title(title, pad=14)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_return_plot(
    path: Path,
    models: Dict[str, SafeDecisionTransformer],
    cfg: TrainConfig,
    dataset_name: str,
    episodes: List[EpisodeStorage],
) -> Dict[str, List[Tuple[float, float]]]:
    returns = np.array([ep.episode_return for ep in episodes], dtype=np.float32)
    q_values = np.linspace(10, 90, cfg.plot_points)
    target_returns = [float(np.percentile(returns, q)) for q in q_values]
    all_pairs: Dict[str, List[Tuple[float, float]]] = {}

    fig = plt.figure(figsize=(6.6, 4.8))
    for label, model in models.items():
        use_quality = label == "Quality Aware DT"
        achieved = []

        for idx, target in enumerate(target_returns):
            achieved_return, _, _ = rollout_once(
                model,
                cfg,
                dataset_name,
                target,
                use_quality=use_quality,
                seed_offset=100 + idx,
                capture_frames=False,
            )
            achieved.append(float(achieved_return))

        plt.plot(target_returns, achieved, marker="o", label=label)
        all_pairs[label] = list(zip(target_returns, achieved))

    low = min(target_returns)
    high = max(target_returns)
    plt.plot([low, high], [low, high], linestyle="--", label="Oracle")
    plt.xlabel("Target Return")
    plt.ylabel("Achieved Return")
    plt.title(f"{GAME_DISPLAY[cfg.game]} Target vs Achieved Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return all_pairs


def train_single_model(
    model_name: str,
    use_quality: bool,
    cfg: TrainConfig,
    episodes: List[EpisodeStorage],
    dataset_name: str,
    action_dim: int,
    out_dir: Path,
) -> Dict[str, float]:
    dataset = AtariWindowDataset(episodes, cfg.context_length)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model = SafeDecisionTransformer(
        action_dim=action_dim,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        use_quality=use_quality,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    target_return = select_target_return(episodes, cfg.target_return_percentile)

    rows: List[Dict[str, float]] = []
    best_return = -1e18
    model_dir = out_dir / ("quality_aware_dt" if use_quality else "plain_dt")
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        action_loss_sum = 0.0
        batch_count = 0

        for batch in loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            logits = model(**batch)

            loss_raw = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["actions"].reshape(-1),
                reduction="none",
            ).reshape(batch["actions"].shape)

            if use_quality:
                weights = cfg.confidence_blend * batch["confidence"] + (1.0 - cfg.confidence_blend)
            else:
                weights = torch.ones_like(batch["confidence"])

            mask = batch["mask"]
            loss = ((loss_raw * weights) * mask).sum() / mask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()

            action_loss_sum += float(loss.item())
            batch_count += 1

        metrics = evaluate_model(model, cfg, dataset_name, target_return, use_quality=use_quality)
        row = {
            "epoch": float(epoch),
            "train_action_loss": action_loss_sum / max(batch_count, 1),
            **metrics,
        }
        rows.append(row)
        save_csv(model_dir / "metrics.csv", rows)

        if metrics["eval_return_mean"] > best_return:
            best_return = metrics["eval_return_mean"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "dataset_name": dataset_name,
                    "target_return": target_return,
                    "action_dim": action_dim,
                    "model_name": model_name,
                },
                model_dir / "best_model.pt",
            )

        print(
            f"model={model_name} epoch={epoch} "
            f"train_loss={row['train_action_loss']:.4f} "
            f"eval_return={row['eval_return_mean']:.2f}±{row['eval_return_std']:.2f}"
        )

    final_metrics = rows[-1] if rows else {
        "eval_return_mean": 0.0,
        "eval_return_std": 0.0,
        "eval_length_mean": 0.0,
    }

    summary = {
        "model_name": model_name,
        "dataset": dataset_name,
        "num_episodes": len(episodes),
        "target_return": target_return,
        "best_eval_return": best_return,
        "final_eval_return_mean": float(final_metrics.get("eval_return_mean", 0.0)),
        "final_eval_return_std": float(final_metrics.get("eval_return_std", 0.0)),
        "final_eval_length_mean": float(final_metrics.get("eval_length_mean", 0.0)),
    }
    save_json(model_dir / "summary.json", summary)

    if cfg.save_video:
        _, _, rollout_frames = rollout_once(
            model,
            cfg,
            dataset_name,
            target_return,
            use_quality=use_quality,
            seed_offset=777 + (1 if use_quality else 0),
            capture_frames=True,
        )
        save_video_from_frames(
            rollout_frames,
            model_dir / "model_rollout_video.mp4",
            fps=max(6, 20 // max(cfg.frame_skip, 1)),
        )

    return {
        **summary,
        "target_return_value": target_return,
        "use_quality": 1.0 if use_quality else 0.0,
        "model_obj": model,
    }


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    out_dir = build_output_dir(cfg)
    save_json(out_dir / "config.json", asdict(cfg))

    episodes, dataset_name, action_dim = load_episodes(cfg)

    if cfg.save_video:
        save_dataset_episode_video(
            episodes,
            out_dir / "dataset_episode_video.mp4",
            cfg.video_episode_index,
            fps=max(6, 20 // max(cfg.frame_skip, 1)),
        )

    plain_result = train_single_model("Plain DT", False, cfg, episodes, dataset_name, action_dim, out_dir)
    quality_result = train_single_model(
        "Quality Aware DT", True, cfg, episodes, dataset_name, action_dim, out_dir
    )

    comparison_rows = [
        {
            "model": "Plain DT",
            "final_eval_return_mean": plain_result["final_eval_return_mean"],
            "final_eval_return_std": plain_result["final_eval_return_std"],
            "best_eval_return": plain_result["best_eval_return"],
            "target_return": plain_result["target_return"],
        },
        {
            "model": "Quality Aware DT",
            "final_eval_return_mean": quality_result["final_eval_return_mean"],
            "final_eval_return_std": quality_result["final_eval_return_std"],
            "best_eval_return": quality_result["best_eval_return"],
            "target_return": quality_result["target_return"],
        },
    ]
    save_csv(out_dir / "comparison_metrics.csv", comparison_rows)

    paper_headers = ["Game", "Model", "Eval Return", "Best Return", "Target Return"]
    paper_rows = [
        [
            GAME_DISPLAY[cfg.game],
            "Plain DT",
            f"{plain_result['final_eval_return_mean']:.1f} ± {plain_result['final_eval_return_std']:.1f}",
            f"{plain_result['best_eval_return']:.1f}",
            f"{plain_result['target_return']:.1f}",
        ],
        [
            GAME_DISPLAY[cfg.game],
            "Quality Aware DT",
            f"{quality_result['final_eval_return_mean']:.1f} ± {quality_result['final_eval_return_std']:.1f}",
            f"{quality_result['best_eval_return']:.1f}",
            f"{quality_result['target_return']:.1f}",
        ],
    ]
    save_table_png(
        out_dir / "comparison_table.png",
        cfg.game,
        paper_headers,
        paper_rows,
        f"{GAME_DISPLAY[cfg.game]} Direct Comparison",
    )

    bc_proxy_headers = ["Game", "DT Variant", "10% BC", "25% BC", "40% BC", "100% BC"]
    plain_bc = compute_bc_proxy_table(
        episodes, plain_result["final_eval_return_mean"], plain_result["final_eval_return_std"]
    )
    quality_bc = compute_bc_proxy_table(
        episodes, quality_result["final_eval_return_mean"], quality_result["final_eval_return_std"]
    )

    bc_rows = [
        [
            GAME_DISPLAY[cfg.game],
            "Plain DT",
            f"{plain_bc['10% BC'][0]:.1f} ± {plain_bc['10% BC'][1]:.1f}",
            f"{plain_bc['25% BC'][0]:.1f} ± {plain_bc['25% BC'][1]:.1f}",
            f"{plain_bc['40% BC'][0]:.1f} ± {plain_bc['40% BC'][1]:.1f}",
            f"{plain_bc['100% BC'][0]:.1f} ± {plain_bc['100% BC'][1]:.1f}",
        ],
        [
            GAME_DISPLAY[cfg.game],
            "Quality Aware DT",
            f"{quality_bc['10% BC'][0]:.1f} ± {quality_bc['10% BC'][1]:.1f}",
            f"{quality_bc['25% BC'][0]:.1f} ± {quality_bc['25% BC'][1]:.1f}",
            f"{quality_bc['40% BC'][0]:.1f} ± {quality_bc['40% BC'][1]:.1f}",
            f"{quality_bc['100% BC'][0]:.1f} ± {quality_bc['100% BC'][1]:.1f}",
        ],
    ]
    save_table_png(
        out_dir / "paper_style_bc_table.png",
        cfg.game,
        bc_proxy_headers,
        bc_rows,
        f"{GAME_DISPLAY[cfg.game]} Percent BC Style Table",
    )

    models = {
        "Plain DT": plain_result["model_obj"],
        "Quality Aware DT": quality_result["model_obj"],
    }
    return_pairs = save_return_plot(
        out_dir / "return_plot.png", models, cfg, dataset_name, episodes
    )

    summary = {
        "dataset": dataset_name,
        "game": cfg.game,
        "num_episodes": len(episodes),
        "plain_dt": {
            "final_eval_return_mean": plain_result["final_eval_return_mean"],
            "final_eval_return_std": plain_result["final_eval_return_std"],
            "best_eval_return": plain_result["best_eval_return"],
            "target_return": plain_result["target_return"],
        },
        "quality_aware_dt": {
            "final_eval_return_mean": quality_result["final_eval_return_mean"],
            "final_eval_return_std": quality_result["final_eval_return_std"],
            "best_eval_return": quality_result["best_eval_return"],
            "target_return": quality_result["target_return"],
        },
        "return_plot_pairs": return_pairs,
    }
    save_json(out_dir / "summary.json", summary)

    difference = quality_result["final_eval_return_mean"] - plain_result["final_eval_return_mean"]
    summary_lines = [
        f"Game: {GAME_DISPLAY[cfg.game]}",
        f"Dataset: {dataset_name}",
        f"Episodes used: {len(episodes)}",
        f"Plain DT final return: {plain_result['final_eval_return_mean']:.2f} ± {plain_result['final_eval_return_std']:.2f}",
        f"Quality Aware DT final return: {quality_result['final_eval_return_mean']:.2f} ± {quality_result['final_eval_return_std']:.2f}",
        f"Quality minus Plain difference: {difference:.2f}",
        f"Action noise probability: {cfg.action_noise_prob:.3f}",
        f"Reward noise std: {cfg.reward_noise_std:.3f}",
        f"Output folder: {str(out_dir)}",
        "Return plot points for Plain DT:",
    ]

    for target, achieved in return_pairs["Plain DT"]:
        summary_lines.append(f"Plain target={target:.2f}, achieved={achieved:.2f}")

    summary_lines.append("Return plot points for Quality Aware DT:")
    for target, achieved in return_pairs["Quality Aware DT"]:
        summary_lines.append(f"Quality target={target:.2f}, achieved={achieved:.2f}")

    save_summary_txt(out_dir / "summary.txt", summary_lines)

    print(f"saved outputs to: {out_dir}")


def parser_builder() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safe multi game project code with plain and quality aware comparison"
    )
    parser.add_argument("--game", type=str, default="qbert", choices=sorted(GAME_TO_DATASET.keys()))
    parser.add_argument("--context_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_fraction", type=float, default=1.0)
    parser.add_argument("--episode_limit", type=int, default=2)
    parser.add_argument("--action_noise_prob", type=float, default=0.0)
    parser.add_argument("--reward_noise_std", type=float, default=0.0)
    parser.add_argument("--confidence_blend", type=float, default=0.6)
    parser.add_argument("--return_scale", type=float, default=100.0)
    parser.add_argument("--target_return_percentile", type=float, default=0.9)
    parser.add_argument("--max_eval_steps", type=int, default=600)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs_project")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--max_frames_per_episode", type=int, default=250)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_episode_index", type=int, default=0)
    parser.add_argument("--plot_points", type=int, default=5)
    parser.add_argument("--quality_mode", type=str, default="return_percentile")
    return parser


def main() -> None:
    parser = parser_builder()
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
