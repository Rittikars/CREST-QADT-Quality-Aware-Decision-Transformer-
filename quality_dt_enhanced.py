

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


GAME_TO_DATASET = {
    "qbert":      "atari/qbert/expert-v0",
    "pong":       "atari/pong/expert-v0",
    "breakout":   "atari/breakout/expert-v0",
    "seaquest":   "atari/seaquest/expert-v0",
    "beamrider":  "atari/beamrider/expert-v0",
}

GAME_DISPLAY = {k: k.capitalize() for k in GAME_TO_DATASET}
GAME_DISPLAY["qbert"] = "Qbert"
GAME_DISPLAY["beamrider"] = "BeamRider"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    game: str = "qbert"
    context_length: int = 8
    batch_size: int = 2
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embed_dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    device: str = "cpu"
    seed: int = 42
    episode_limit: int = 10
    action_noise_prob: float = 0.0
    reward_noise_std: float = 0.0
    confidence_blend: float = 0.6
    # NEW: confidence_mode — "linear" (original) or "rank" (rank-normalized)
    confidence_mode: str = "linear"
    # NEW: confidence_decay — 0.0 means no decay; 0.5 means steps at end
    #      of episode get ~60% of the confidence of steps at the start.
    confidence_decay: float = 0.0
    return_scale: float = 100.0
    target_return_percentile: float = 0.9
    max_eval_steps: int = 600
    eval_episodes: int = 1
    output_dir: str = "outputs_enhanced"
    num_workers: int = 0
    grad_clip_norm: float = 1.0
    max_frames_per_episode: int = 250
    frame_skip: int = 4
    save_video: bool = False
    plot_points: int = 5
    # NEW: experiment flags
    run_ablation: bool = False
    run_noise: bool = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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
        self.observations   = observations.astype(np.uint8)
        self.actions        = actions.astype(np.int64)
        self.rewards        = rewards.astype(np.float32)
        self.returns_to_go  = returns_to_go.astype(np.float32)
        self.confidence     = confidence.astype(np.float32)
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

        obs     = ep.observations[start_t: end_t + 1]
        actions = ep.actions[start_t: end_t + 1]
        returns = ep.returns_to_go[start_t: end_t + 1]
        conf    = ep.confidence[start_t: end_t + 1]
        times   = np.arange(start_t, end_t + 1, dtype=np.int64)

        valid_len = len(actions)
        pad_len   = self.context_length - valid_len

        if pad_len > 0:
            obs     = np.concatenate([np.repeat(obs[:1], pad_len, 0), obs], 0)
            actions = np.concatenate([np.zeros((pad_len,), np.int64),   actions], 0)
            returns = np.concatenate([np.zeros((pad_len,), np.float32), returns], 0)
            conf    = np.concatenate([np.ones( (pad_len,), np.float32), conf],    0)
            times   = np.concatenate([np.zeros((pad_len,), np.int64),   times],   0)
            mask    = np.concatenate([np.zeros((pad_len,), np.float32),
                                      np.ones((valid_len,), np.float32)], 0)
        else:
            mask = np.ones((self.context_length,), np.float32)

        return {
            "observations":  torch.tensor(obs,     dtype=torch.float32),
            "actions":       torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(returns, dtype=torch.float32),
            "confidence":    torch.tensor(conf,    dtype=torch.float32),
            "timesteps":     torch.tensor(times,   dtype=torch.long),
            "mask":          torch.tensor(mask,    dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SmallStateEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 8, 4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x / 255.0))


class QualityDecisionTransformer(nn.Module):
    """
    Decision Transformer with optional quality/confidence token.

    Token sequence per timestep (4 tokens):
        [return-to-go, confidence, state, action]

    When use_quality=False the confidence token is a constant 0.5,
    making the model functionally identical to plain DT except for
    having one extra (uninformative) token — a fair comparison.
    """

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
        self.embed_dim  = embed_dim
        self.action_dim = action_dim
        self.use_quality = use_quality

        self.state_encoder = SmallStateEncoder(embed_dim)
        self.action_embed  = nn.Embedding(action_dim, embed_dim)
        self.return_embed  = nn.Linear(1, embed_dim)
        self.conf_embed    = nn.Linear(1, embed_dim)
        self.time_embed    = nn.Embedding(4096, embed_dim)
        self.norm          = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers)
        self.action_head  = nn.Linear(embed_dim, action_dim)

    def causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(
        self,
        observations: torch.Tensor,
        actions:      torch.Tensor,
        returns_to_go: torch.Tensor,
        confidence:   torch.Tensor,
        timesteps:    torch.Tensor,
        mask:         torch.Tensor,
    ) -> torch.Tensor:
        b, t, h, w, c = observations.shape
        obs_flat = observations.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)

        s_tok = self.state_encoder(obs_flat).reshape(b, t, self.embed_dim)
        a_tok = self.action_embed(actions)
        r_tok = self.return_embed(returns_to_go.unsqueeze(-1))

        # KEY DIFFERENCE: when use_quality=True the actual per-episode
        # confidence score is injected; otherwise a neutral 0.5 constant.
        if self.use_quality:
            c_tok = self.conf_embed(confidence.unsqueeze(-1))
        else:
            c_tok = self.conf_embed(torch.full_like(confidence, 0.5).unsqueeze(-1))

        t_emb = self.time_embed(timesteps.clamp(max=4095))

        s_tok = self.norm(s_tok + t_emb)
        a_tok = self.norm(a_tok + t_emb)
        r_tok = self.norm(r_tok + t_emb)
        c_tok = self.norm(c_tok + t_emb)

        tokens = torch.stack([r_tok, c_tok, s_tok, a_tok], dim=2).reshape(b, t * 4, self.embed_dim)
        kpm = mask.repeat_interleave(4, dim=1).bool().logical_not()

        hidden = self.transformer(
            tokens,
            mask=self.causal_mask(tokens.shape[1], tokens.device),
            src_key_padding_mask=kpm,
        )
        state_hidden = hidden.reshape(b, t, 4, self.embed_dim)[:, :, 2, :]
        return self.action_head(state_hidden)


# ---------------------------------------------------------------------------
# Confidence functions  (NEW: linear vs rank-normalized, timestep decay)
# ---------------------------------------------------------------------------

def make_confidence_linear(
    episode_return: float, min_ret: float, max_ret: float
) -> float:
    """Original method: linear rescale of return to [0.2, 1.0]."""
    if max_ret <= min_ret:
        return 1.0
    val = (episode_return - min_ret) / (max_ret - min_ret)
    return float(np.clip(0.2 + 0.8 * val, 0.2, 1.0))


def make_confidence_rank(rank: int, total: int) -> float:
    """
    NEW — Rank-normalized confidence.
    Best episode (rank=0) → 1.0, worst (rank=total-1) → 0.2.
    Robust to outliers: the actual return values don't matter,
    only the ordering does.
    """
    if total <= 1:
        return 1.0
    return float(np.clip(1.0 - 0.8 * (rank / (total - 1)), 0.2, 1.0))


def apply_timestep_decay(
    base_confidence: float, episode_length: int, decay: float
) -> np.ndarray:
    """
    NEW — Timestep-decaying confidence array.
    Step 0 gets base_confidence; the last step gets
    base_confidence * exp(-decay).  decay=0 means no change.
    Higher decay = more credit to early causal actions.
    """
    if decay <= 0.0 or episode_length <= 1:
        return np.full((episode_length,), base_confidence, dtype=np.float32)
    t = np.arange(episode_length, dtype=np.float32) / max(episode_length - 1, 1)
    return (base_confidence * np.exp(-decay * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_rtg(rewards: np.ndarray, scale: float) -> np.ndarray:
    rtg = np.zeros_like(rewards, np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running += float(rewards[i])
        rtg[i] = running / max(scale, 1e-8)
    return rtg


def corrupt(actions, rewards, action_dim, action_noise_prob, reward_noise_std, rng):
    actions, rewards = actions.copy(), rewards.copy()
    if action_noise_prob > 0:
        m = rng.random(len(actions)) < action_noise_prob
        actions[m] = rng.integers(0, action_dim, int(m.sum()))
    if reward_noise_std > 0:
        rewards += rng.normal(0, reward_noise_std, rewards.shape).astype(np.float32)
    return actions, rewards


def downsample(obs, acts, rews, max_frames, frame_skip):
    obs, acts, rews = obs[::frame_skip], acts[::frame_skip], rews[::frame_skip]
    if len(acts) > max_frames:
        obs, acts, rews = obs[:max_frames], acts[:max_frames], rews[:max_frames]
    return obs, acts, rews


def load_episodes(cfg: TrainConfig) -> Tuple[List[EpisodeStorage], str, int]:
    dataset_name = GAME_TO_DATASET[cfg.game]
    dataset      = minari.load_dataset(dataset_name)
    raw          = list(dataset.iterate_episodes())
    if cfg.episode_limit > 0:
        raw = raw[: cfg.episode_limit]

    action_dim = int(dataset.action_space.n)
    rng        = np.random.default_rng(cfg.seed)

    temp: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for ep in raw:
        obs  = np.asarray(ep.observations, np.uint8)
        acts = np.asarray(ep.actions,      np.int64)
        rews = np.asarray(ep.rewards,      np.float32)
        obs, acts, rews = downsample(obs, acts, rews, cfg.max_frames_per_episode, cfg.frame_skip)
        acts, rews      = corrupt(acts, rews, action_dim, cfg.action_noise_prob, cfg.reward_noise_std, rng)
        temp.append((obs, acts, rews, float(np.sum(rews))))

    returns = [x[3] for x in temp]
    min_ret, max_ret = min(returns, default=0.0), max(returns, default=1.0)

    # For rank-normalized mode: sort indices by return descending
    if cfg.confidence_mode == "rank":
        sorted_indices = np.argsort(returns)[::-1].tolist()
        rank_of = {orig_idx: rank for rank, orig_idx in enumerate(sorted_indices)}
    else:
        rank_of = {}

    episodes: List[EpisodeStorage] = []
    for idx, (obs, acts, rews, ep_return) in enumerate(temp):
        rtg = compute_rtg(rews, cfg.return_scale)

        if cfg.confidence_mode == "rank":
            base_conf = make_confidence_rank(rank_of[idx], len(temp))
        else:
            base_conf = make_confidence_linear(ep_return, min_ret, max_ret)

        # NEW: apply optional timestep decay
        conf = apply_timestep_decay(base_conf, len(acts), cfg.confidence_decay)
        episodes.append(EpisodeStorage(obs, acts, rews, rtg, conf, ep_return))

    return episodes, dataset_name, action_dim


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def select_target_return(episodes: List[EpisodeStorage], percentile: float) -> float:
    vals = np.array([ep.episode_return for ep in episodes], np.float32)
    return float(np.percentile(vals, np.clip(percentile, 0, 1) * 100))


def build_eval_batch(obs_h, act_h, rtg_h, conf_h, time_h, ctx, device):
    start   = max(0, len(obs_h) - ctx)
    obs_s   = np.stack(obs_h[start:])
    act_s   = np.array(act_h[start:], np.int64)
    rtg_s   = np.array(rtg_h[start:], np.float32)
    conf_s  = np.array(conf_h[start:], np.float32)
    time_s  = np.array(time_h[start:], np.int64)
    vlen    = len(act_s)
    pad     = ctx - vlen
    if pad > 0:
        obs_s  = np.concatenate([np.repeat(obs_s[:1], pad, 0),  obs_s])
        act_s  = np.concatenate([np.zeros((pad,), np.int64),    act_s])
        rtg_s  = np.concatenate([np.zeros((pad,), np.float32),  rtg_s])
        conf_s = np.concatenate([np.ones( (pad,), np.float32),  conf_s])
        time_s = np.concatenate([np.zeros((pad,), np.int64),    time_s])
        mask   = np.concatenate([np.zeros((pad,), np.float32), np.ones((vlen,), np.float32)])
    else:
        mask = np.ones((ctx,), np.float32)
    return {
        "observations":  torch.tensor(obs_s[None],  dtype=torch.float32, device=device),
        "actions":       torch.tensor(act_s[None],  dtype=torch.long,    device=device),
        "returns_to_go": torch.tensor(rtg_s[None],  dtype=torch.float32, device=device),
        "confidence":    torch.tensor(conf_s[None], dtype=torch.float32, device=device),
        "timesteps":     torch.tensor(time_s[None], dtype=torch.long,    device=device),
        "mask":          torch.tensor(mask[None],   dtype=torch.float32, device=device),
    }


@torch.no_grad()
def rollout_once(model, cfg, dataset_name, target_return, use_quality, seed_offset=0):
    dataset = minari.load_dataset(dataset_name)
    env     = dataset.recover_environment()
    model.eval()
    obs, _  = env.reset(seed=cfg.seed + seed_offset)
    done    = False
    total_r = 0.0
    steps   = 0
    obs_h, act_h, rtg_h, conf_h, time_h = [], [], [], [], []
    cur_rtg = target_return / max(cfg.return_scale, 1e-8)

    while not done and steps < cfg.max_eval_steps:
        obs_h.append(np.asarray(obs, np.uint8))
        rtg_h.append(cur_rtg)
        conf_h.append(1.0 if use_quality else 0.5)
        time_h.append(steps)
        if len(act_h) < len(obs_h):
            act_h.append(0)

        batch  = build_eval_batch(obs_h, act_h, rtg_h, conf_h, time_h, cfg.context_length, cfg.device)
        logits = model(**batch)
        action = int(torch.argmax(logits[0, -1], dim=-1).item())
        obs, reward, term, trunc, _ = env.step(action)
        done     = bool(term or trunc)
        total_r += float(reward)
        steps   += 1
        act_h[-1] = action
        cur_rtg  -= float(reward) / max(cfg.return_scale, 1e-8)

    return total_r


@torch.no_grad()
def evaluate_model(model, cfg, dataset_name, target_return, use_quality):
    returns = [
        rollout_once(model, cfg, dataset_name, target_return, use_quality, seed_offset=i)
        for i in range(cfg.eval_episodes)
    ]
    return float(np.mean(returns)), float(np.std(returns))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    label: str,
    use_quality: bool,
    cfg: TrainConfig,
    episodes: List[EpisodeStorage],
    dataset_name: str,
    action_dim: int,
    out_dir: Path,
) -> Dict:
    loader = DataLoader(
        AtariWindowDataset(episodes, cfg.context_length),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    )
    model = QualityDecisionTransformer(
        action_dim=action_dim, embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers, num_heads=cfg.num_heads,
        dropout=cfg.dropout, use_quality=use_quality,
    ).to(cfg.device)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    target_ret   = select_target_return(episodes, cfg.target_return_percentile)
    best_return  = -1e18
    epoch_rows   = []

    model_dir = out_dir / label.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0

        for batch in loader:
            batch  = {k: v.to(cfg.device) for k, v in batch.items()}
            logits = model(**batch)

            loss_raw = F.cross_entropy(
                logits.reshape(-1, action_dim),
                batch["actions"].reshape(-1),
                reduction="none",
            ).reshape(batch["actions"].shape)

            # CORE QUALITY-AWARE LOSS:
            # When use_quality=True, each timestep's loss is scaled by
            # its per-episode confidence (high-return → higher weight).
            # confidence_blend interpolates between uniform (0.0) and
            # fully quality-weighted (1.0).
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

            loss_sum  += float(loss.item())
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        mean_r, std_r = evaluate_model(model, cfg, dataset_name, target_ret, use_quality)
        epoch_rows.append({"epoch": epoch, "loss": avg_loss, "return_mean": mean_r, "return_std": std_r})

        if mean_r > best_return:
            best_return = mean_r
            torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg)}, model_dir / "best_model.pt")

        print(f"  [{label}] epoch={epoch:3d}  loss={avg_loss:.4f}  return={mean_r:.1f}±{std_r:.1f}")

    # Save per-epoch CSV
    with open(model_dir / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss", "return_mean", "return_std"])
        w.writeheader(); w.writerows(epoch_rows)

    return {
        "label":        label,
        "use_quality":  use_quality,
        "target_ret":   target_ret,
        "best_return":  best_return,
        "final_mean":   epoch_rows[-1]["return_mean"] if epoch_rows else 0.0,
        "final_std":    epoch_rows[-1]["return_std"]  if epoch_rows else 0.0,
        "epoch_rows":   epoch_rows,
        "model":        model,
    }


# ---------------------------------------------------------------------------
# NEW: confidence_blend ablation sweep
# ---------------------------------------------------------------------------

def run_ablation_sweep(cfg: TrainConfig, out_dir: Path) -> None:
    """
    Train Quality-Aware DT with blend in {0.0, 0.3, 0.6, 1.0} and
    Plain DT (baseline). Save results to ablation_results.csv and
    ablation_chart.png.

    blend=0.0 → uniform weights (same as Plain DT in effect)
    blend=1.0 → fully quality-weighted loss
    """
    print("\n=== Ablation sweep: confidence_blend ===")
    set_seed(cfg.seed)
    episodes, dataset_name, action_dim = load_episodes(cfg)

    blends    = [0.0, 0.3, 0.6, 1.0]
    labels    = [f"Quality DT (blend={b})" for b in blends]
    results   = []
    abl_dir   = out_dir / "ablation_blend"
    abl_dir.mkdir(parents=True, exist_ok=True)

    # Plain DT baseline
    plain_cfg         = TrainConfig(**asdict(cfg))
    plain_cfg.output_dir = str(abl_dir)
    plain_result      = train_model("Plain DT", False, plain_cfg, episodes, dataset_name, action_dim, abl_dir)
    results.append({"label": "Plain DT", "blend": "N/A",
                    "final_mean": plain_result["final_mean"], "best_return": plain_result["best_return"]})

    # Quality DT at each blend value
    for blend, label in zip(blends, labels):
        cfg_b               = TrainConfig(**asdict(cfg))
        cfg_b.confidence_blend = blend
        r = train_model(label, True, cfg_b, episodes, dataset_name, action_dim, abl_dir)
        results.append({"label": label, "blend": blend,
                        "final_mean": r["final_mean"], "best_return": r["best_return"]})

    # Save CSV
    abl_csv = abl_dir / "ablation_results.csv"
    with open(abl_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "blend", "final_mean", "best_return"])
        w.writeheader(); w.writerows(results)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names   = [r["label"] for r in results]
    means   = [r["final_mean"] for r in results]
    colors  = ["#888888"] + ["#1a7abf", "#1a9e6b", "#e07b20", "#c0392b"]
    bars    = ax.bar(range(len(names)), means, color=colors[: len(names)], edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Final eval return")
    ax.set_title(f"{GAME_DISPLAY[cfg.game]} — confidence_blend ablation")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(abl_dir / "ablation_chart.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(f"\nAblation results saved to {abl_dir}")
    for r in results:
        print(f"  {r['label']:35s}  final={r['final_mean']:.1f}  best={r['best_return']:.1f}")


# ---------------------------------------------------------------------------
# NEW: noise robustness experiment
# ---------------------------------------------------------------------------

def run_noise_experiment(cfg: TrainConfig, out_dir: Path) -> None:
    """
    Train Plain DT and Quality-Aware DT on datasets with injected noise
    (action_noise_prob=0.1, reward_noise_std=0.05) and compare.

    Hypothesis: Quality-Aware DT is more robust because noisy episodes
    tend to have lower returns and therefore receive lower confidence
    weights — automatically de-emphasising corrupted data.
    """
    print("\n=== Noise robustness experiment ===")
    noise_dir = out_dir / "noise_experiment"
    noise_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for noise_level, action_noise, reward_noise in [
        ("clean",   0.00, 0.00),
        ("noisy",   0.10, 0.05),
    ]:
        for label, use_quality in [("Plain DT", False), ("Quality-Aware DT", True)]:
            exp_cfg                  = TrainConfig(**asdict(cfg))
            exp_cfg.action_noise_prob = action_noise
            exp_cfg.reward_noise_std  = reward_noise

            set_seed(cfg.seed)
            episodes, dataset_name, action_dim = load_episodes(exp_cfg)
            result = train_model(
                f"{label}_{noise_level}", use_quality,
                exp_cfg, episodes, dataset_name, action_dim, noise_dir,
            )
            rows.append({
                "noise_level":  noise_level,
                "model":        label,
                "action_noise": action_noise,
                "reward_noise": reward_noise,
                "final_mean":   result["final_mean"],
                "best_return":  result["best_return"],
            })
            print(f"  [{noise_level}] {label}: final={result['final_mean']:.1f}  best={result['best_return']:.1f}")

    noise_csv = noise_dir / "noise_results.csv"
    with open(noise_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Bar chart: 2 groups (clean, noisy) × 2 models
    fig, ax   = plt.subplots(figsize=(7, 4))
    x         = np.arange(2)
    width     = 0.35
    clean_p   = next(r["final_mean"] for r in rows if r["noise_level"] == "clean" and r["model"] == "Plain DT")
    clean_q   = next(r["final_mean"] for r in rows if r["noise_level"] == "clean" and r["model"] == "Quality-Aware DT")
    noisy_p   = next(r["final_mean"] for r in rows if r["noise_level"] == "noisy" and r["model"] == "Plain DT")
    noisy_q   = next(r["final_mean"] for r in rows if r["noise_level"] == "noisy" and r["model"] == "Quality-Aware DT")

    b1 = ax.bar(x - width / 2, [clean_p, noisy_p], width, label="Plain DT",        color="#888888")
    b2 = ax.bar(x + width / 2, [clean_q, noisy_q], width, label="Quality-Aware DT", color="#1a7abf")
    ax.set_xticks(x); ax.set_xticklabels(["Clean dataset", "Noisy dataset"])
    ax.set_ylabel("Final eval return")
    ax.set_title(f"{GAME_DISPLAY[cfg.game]} — noise robustness")
    ax.legend()
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(noise_dir / "noise_chart.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(f"\nNoise experiment results saved to {noise_dir}")


# ---------------------------------------------------------------------------
# Standard train run
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / f"{cfg.game}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Special experiment modes
    if cfg.run_ablation:
        run_ablation_sweep(cfg, out_dir)
        return
    if cfg.run_noise:
        run_noise_experiment(cfg, out_dir)
        return

    # Standard Plain DT vs Quality-Aware DT comparison
    print(f"\n=== {GAME_DISPLAY[cfg.game]} — standard run ===")
    episodes, dataset_name, action_dim = load_episodes(cfg)

    plain_r   = train_model("Plain DT",         False, cfg, episodes, dataset_name, action_dim, out_dir)
    quality_r = train_model("Quality-Aware DT", True,  cfg, episodes, dataset_name, action_dim, out_dir)

    diff = quality_r["final_mean"] - plain_r["final_mean"]

    summary = {
        "game":           cfg.game,
        "confidence_mode":  cfg.confidence_mode,
        "confidence_decay": cfg.confidence_decay,
        "confidence_blend": cfg.confidence_blend,
        "plain_dt":       {"final_mean": plain_r["final_mean"],   "best_return": plain_r["best_return"]},
        "quality_aware_dt": {"final_mean": quality_r["final_mean"], "best_return": quality_r["best_return"]},
        "improvement":    diff,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Learning curve plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for result, color in [(plain_r, "#888888"), (quality_r, "#1a7abf")]:
        epochs  = [r["epoch"]       for r in result["epoch_rows"]]
        losses  = [r["loss"]        for r in result["epoch_rows"]]
        returns = [r["return_mean"] for r in result["epoch_rows"]]
        axes[0].plot(epochs, losses,  color=color, marker="o", markersize=3, label=result["label"])
        axes[1].plot(epochs, returns, color=color, marker="o", markersize=3, label=result["label"])
    axes[0].set_title("Training loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].set_title("Eval return");   axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.suptitle(f"{GAME_DISPLAY[cfg.game]} — Plain DT vs Quality-Aware DT")
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*55}")
    print(f"  Game:               {GAME_DISPLAY[cfg.game]}")
    print(f"  Confidence mode:    {cfg.confidence_mode}")
    print(f"  Confidence decay:   {cfg.confidence_decay}")
    print(f"  Confidence blend:   {cfg.confidence_blend}")
    print(f"  Plain DT:           {plain_r['final_mean']:.1f} (best {plain_r['best_return']:.1f})")
    print(f"  Quality-Aware DT:   {quality_r['final_mean']:.1f} (best {quality_r['best_return']:.1f})")
    print(f"  Improvement:        {diff:+.1f}")
    print(f"  Output:             {out_dir}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Quality-Aware Decision Transformer — Enhanced")
    p.add_argument("--game",               type=str,   default="qbert",   choices=sorted(GAME_TO_DATASET))
    p.add_argument("--context_length",     type=int,   default=8)
    p.add_argument("--batch_size",         type=int,   default=2)
    p.add_argument("--epochs",             type=int,   default=10)
    p.add_argument("--learning_rate",      type=float, default=3e-4)
    p.add_argument("--weight_decay",       type=float, default=1e-4)
    p.add_argument("--embed_dim",          type=int,   default=64)
    p.add_argument("--num_layers",         type=int,   default=2)
    p.add_argument("--num_heads",          type=int,   default=2)
    p.add_argument("--dropout",            type=float, default=0.1)
    p.add_argument("--device",             type=str,   default="cpu")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--episode_limit",      type=int,   default=10)
    p.add_argument("--action_noise_prob",  type=float, default=0.0)
    p.add_argument("--reward_noise_std",   type=float, default=0.0)
    p.add_argument("--confidence_blend",   type=float, default=0.6)
    p.add_argument("--confidence_mode",    type=str,   default="linear",
                   choices=["linear", "rank"],
                   help="linear=original rescale, rank=rank-normalized (NEW)")
    p.add_argument("--confidence_decay",   type=float, default=0.0,
                   help="Timestep decay factor (0=none, 0.5=moderate decay). NEW.")
    p.add_argument("--return_scale",       type=float, default=100.0)
    p.add_argument("--target_return_percentile", type=float, default=0.9)
    p.add_argument("--max_eval_steps",     type=int,   default=600)
    p.add_argument("--eval_episodes",      type=int,   default=1)
    p.add_argument("--output_dir",         type=str,   default="outputs_enhanced")
    p.add_argument("--num_workers",        type=int,   default=0)
    p.add_argument("--grad_clip_norm",     type=float, default=1.0)
    p.add_argument("--max_frames_per_episode", type=int, default=250)
    p.add_argument("--frame_skip",         type=int,   default=4)
    p.add_argument("--plot_points",        type=int,   default=5)
    p.add_argument("--run_ablation",       action="store_true",
                   help="Run confidence_blend ablation sweep (NEW)")
    p.add_argument("--run_noise",          action="store_true",
                   help="Run noise robustness experiment (NEW)")

    args = p.parse_args()
    cfg  = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
