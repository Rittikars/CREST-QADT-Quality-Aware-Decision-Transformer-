# quality_dt_halfcheetah.py
import d4rl.locomotion
import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import os

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

try:
    import gym
    import d4rl
    import d4rl.locomotion
except ImportError:
    gym = None
    d4rl = None

@dataclass
class TrainConfig:
    env_name: str = "halfcheetah-medium-v2"
    context_length: int = 20
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    embed_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    device: str = "cpu"
    seed: int = 42
    episode_limit: int = 0
    confidence_blend: float = 0.2
    confidence_mode: str = "rank"   # "linear" or "rank"
    confidence_decay: float = 0.0
    return_scale: float = 1000.0
    target_return_percentile: float = 0.9
    max_eval_steps: int = 1000
    eval_episodes: int = 3
    output_dir: str = "halfcheetah_results"
    grad_clip_norm: float = 1.0


class EpisodeStorage:
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        returns_to_go: np.ndarray,
        confidence: np.ndarray,
        episode_return: float,
    ):
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.rewards = rewards.astype(np.float32)
        self.returns_to_go = returns_to_go.astype(np.float32)
        self.confidence = confidence.astype(np.float32)
        self.episode_return = float(episode_return)
        self.episode_length = len(actions)


class ContinuousWindowDataset(Dataset):
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

        states = ep.states[start_t:end_t + 1]
        actions = ep.actions[start_t:end_t + 1]
        returns = ep.returns_to_go[start_t:end_t + 1]
        conf = ep.confidence[start_t:end_t + 1]
        timesteps = np.arange(start_t, end_t + 1, dtype=np.int64)

        valid_len = len(actions)
        pad_len = self.context_length - valid_len

        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]

        if pad_len > 0:
            states = np.concatenate(
                [np.zeros((pad_len, state_dim), dtype=np.float32), states], axis=0
            )
            actions = np.concatenate(
                [np.zeros((pad_len, action_dim), dtype=np.float32), actions], axis=0
            )
            returns = np.concatenate(
                [np.zeros((pad_len,), dtype=np.float32), returns], axis=0
            )
            conf = np.concatenate(
                [np.ones((pad_len,), dtype=np.float32), conf], axis=0
            )
            timesteps = np.concatenate(
                [np.zeros((pad_len,), dtype=np.int64), timesteps], axis=0
            )
            mask = np.concatenate(
                [np.zeros((pad_len,), dtype=np.float32), np.ones((valid_len,), dtype=np.float32)],
                axis=0,
            )
        else:
            mask = np.ones((self.context_length,), dtype=np.float32)

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "returns_to_go": torch.tensor(returns, dtype=torch.float32),
            "confidence": torch.tensor(conf, dtype=torch.float32),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }


class VectorDecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
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

        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(action_dim, embed_dim)
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
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        confidence: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = states.shape

        s_tok = self.state_embed(states)
        a_tok = self.action_embed(actions)
        r_tok = self.return_embed(returns_to_go.unsqueeze(-1))

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


def make_confidence_linear(episode_return: float, min_ret: float, max_ret: float) -> float:
    if max_ret <= min_ret:
        return 1.0
    value = (episode_return - min_ret) / (max_ret - min_ret)
    return float(np.clip(0.2 + 0.8 * value, 0.2, 1.0))


def make_confidence_rank(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return float(np.clip(1.0 - 0.8 * (rank / (total - 1)), 0.2, 1.0))


def apply_timestep_decay(base_conf: float, episode_length: int, decay: float) -> np.ndarray:
    if decay <= 0.0 or episode_length <= 1:
        return np.full((episode_length,), base_conf, dtype=np.float32)
    t = np.arange(episode_length, dtype=np.float32) / max(episode_length - 1, 1)
    return (base_conf * np.exp(-decay * t)).astype(np.float32)


def load_halfcheetah_episodes(cfg: TrainConfig) -> Tuple[List[EpisodeStorage], int, int]:
    if gym is None or d4rl is None:
        raise ImportError("Please install gym and d4rl first.")

    env = gym.make(cfg.env_name)
    dataset = d4rl.qlearning_dataset(env)

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]
    timeouts = dataset.get("timeouts", np.zeros_like(terminals))

    state_dim = observations.shape[1]
    action_dim = actions.shape[1]

    episodes_raw = []
    start = 0
    for i in range(len(rewards)):
        if terminals[i] or timeouts[i]:
            obs_ep = observations[start:i + 1]
            act_ep = actions[start:i + 1]
            rew_ep = rewards[start:i + 1]
            ep_return = float(np.sum(rew_ep))
            episodes_raw.append((obs_ep, act_ep, rew_ep, ep_return))
            start = i + 1

    if start < len(rewards):
        obs_ep = observations[start:]
        act_ep = actions[start:]
        rew_ep = rewards[start:]
        ep_return = float(np.sum(rew_ep))
        episodes_raw.append((obs_ep, act_ep, rew_ep, ep_return))

    if cfg.episode_limit > 0:
        episodes_raw = episodes_raw[:cfg.episode_limit]

    returns = [x[3] for x in episodes_raw]
    min_ret = min(returns) if returns else 0.0
    max_ret = max(returns) if returns else 1.0

    if cfg.confidence_mode == "rank":
        sorted_indices = np.argsort(returns)[::-1].tolist()
        rank_of = {orig_idx: rank for rank, orig_idx in enumerate(sorted_indices)}
    else:
        rank_of = {}

    episodes = []
    for idx, (obs_ep, act_ep, rew_ep, ep_return) in enumerate(episodes_raw):
        rtg = compute_returns_to_go(rew_ep, cfg.return_scale)

        if cfg.confidence_mode == "rank":
            base_conf = make_confidence_rank(rank_of[idx], len(episodes_raw))
        else:
            base_conf = make_confidence_linear(ep_return, min_ret, max_ret)

        conf = apply_timestep_decay(base_conf, len(act_ep), cfg.confidence_decay)
        episodes.append(EpisodeStorage(obs_ep, act_ep, rew_ep, rtg, conf, ep_return))

    return episodes, state_dim, action_dim


def select_target_return(episodes: List[EpisodeStorage], percentile: float) -> float:
    values = np.array([ep.episode_return for ep in episodes], dtype=np.float32)
    return float(np.percentile(values, np.clip(percentile, 0.0, 1.0) * 100.0))


def build_eval_batch(
    state_hist: List[np.ndarray],
    action_hist: List[np.ndarray],
    rtg_hist: List[float],
    conf_hist: List[float],
    time_hist: List[int],
    context_length: int,
    state_dim: int,
    action_dim: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    start = max(0, len(state_hist) - context_length)

    state_seq = np.stack(state_hist[start:])
    action_seq = np.stack(action_hist[start:])
    rtg_seq = np.array(rtg_hist[start:], dtype=np.float32)
    conf_seq = np.array(conf_hist[start:], dtype=np.float32)
    time_seq = np.array(time_hist[start:], dtype=np.int64)

    valid_len = len(action_seq)
    pad_len = context_length - valid_len

    if pad_len > 0:
        state_seq = np.concatenate([np.zeros((pad_len, state_dim), dtype=np.float32), state_seq], axis=0)
        action_seq = np.concatenate([np.zeros((pad_len, action_dim), dtype=np.float32), action_seq], axis=0)
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
        "states": torch.tensor(state_seq[None], dtype=torch.float32, device=device),
        "actions": torch.tensor(action_seq[None], dtype=torch.float32, device=device),
        "returns_to_go": torch.tensor(rtg_seq[None], dtype=torch.float32, device=device),
        "confidence": torch.tensor(conf_seq[None], dtype=torch.float32, device=device),
        "timesteps": torch.tensor(time_seq[None], dtype=torch.long, device=device),
        "mask": torch.tensor(mask[None], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def rollout_once(
    model: VectorDecisionTransformer,
    cfg: TrainConfig,
    target_return: float,
    state_dim: int,
    action_dim: int,
    use_quality: bool,
    seed_offset: int = 0,
) -> float:
    env = gym.make(cfg.env_name)
    model.eval()

    try:
        state = env.reset(seed=cfg.seed + seed_offset)
        if isinstance(state, tuple):
            state = state[0]
    except TypeError:
        state = env.reset()

    done = False
    total_reward = 0.0
    steps = 0

    state_hist = []
    action_hist = []
    rtg_hist = []
    conf_hist = []
    time_hist = []

    current_rtg = target_return / max(cfg.return_scale, 1e-8)

    while not done and steps < cfg.max_eval_steps:
        current_state = np.asarray(state, dtype=np.float32)
        state_hist.append(current_state)
        rtg_hist.append(current_rtg)
        conf_hist.append(1.0 if use_quality else 0.5)
        time_hist.append(steps)

        if len(action_hist) < len(state_hist):
            action_hist.append(np.zeros((action_dim,), dtype=np.float32))

        batch = build_eval_batch(
            state_hist, action_hist, rtg_hist, conf_hist, time_hist,
            cfg.context_length, state_dim, action_dim, cfg.device
        )
        action_pred = model(**batch)[0, -1].cpu().numpy()
        action = np.clip(action_pred, env.action_space.low, env.action_space.high)

        step_out = env.step(action)
        if len(step_out) == 5:
            state, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            state, reward, done, _ = step_out

        total_reward += float(reward)
        steps += 1
        action_hist[-1] = action.astype(np.float32)
        current_rtg = current_rtg - float(reward) / max(cfg.return_scale, 1e-8)

    return total_reward


@torch.no_grad()
def evaluate_model(
    model: VectorDecisionTransformer,
    cfg: TrainConfig,
    target_return: float,
    state_dim: int,
    action_dim: int,
    use_quality: bool,
) -> Dict[str, float]:
    returns = []
    for i in range(cfg.eval_episodes):
        r = rollout_once(model, cfg, target_return, state_dim, action_dim, use_quality, seed_offset=i)
        returns.append(r)
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
    }


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


def train_model(
    label: str,
    use_quality: bool,
    cfg: TrainConfig,
    episodes: List[EpisodeStorage],
    state_dim: int,
    action_dim: int,
    out_dir: Path,
) -> Dict:
    loader = DataLoader(
        ContinuousWindowDataset(episodes, cfg.context_length),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = VectorDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        use_quality=use_quality,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    target_return = select_target_return(episodes, cfg.target_return_percentile)

    rows = []
    best_return = -1e18
    model_dir = out_dir / label.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0

        for batch in loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            action_pred = model(**batch)

            mse_raw = ((action_pred - batch["actions"]) ** 2).mean(dim=-1)

            if use_quality:
                weights = cfg.confidence_blend * batch["confidence"] + (1.0 - cfg.confidence_blend)
            else:
                weights = torch.ones_like(batch["confidence"])

            mask = batch["mask"]
            loss = ((mse_raw * weights) * mask).sum() / mask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            loss_sum += float(loss.item())
            batch_count += 1

        metrics = evaluate_model(model, cfg, target_return, state_dim, action_dim, use_quality)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batch_count, 1),
            **metrics,
        }
        rows.append(row)

        if metrics["eval_return_mean"] > best_return:
            best_return = metrics["eval_return_mean"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "target_return": target_return,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                },
                model_dir / "best_model.pt",
            )

        print(
            f"[{label}] epoch={epoch:3d} "
            f"loss={row['train_loss']:.4f} "
            f"return={row['eval_return_mean']:.1f}±{row['eval_return_std']:.1f}"
        )

    save_csv(model_dir / "training_log.csv", rows)

    return {
        "label": label,
        "use_quality": use_quality,
        "target_return": target_return,
        "best_return": best_return,
        "final_mean": rows[-1]["eval_return_mean"] if rows else 0.0,
        "final_std": rows[-1]["eval_return_std"] if rows else 0.0,
        "epoch_rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--context_length", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode_limit", type=int, default=0)
    parser.add_argument("--confidence_blend", type=float, default=0.2)
    parser.add_argument("--confidence_mode", type=str, default="rank", choices=["linear", "rank"])
    parser.add_argument("--confidence_decay", type=float, default=0.0)
    parser.add_argument("--return_scale", type=float, default=1000.0)
    parser.add_argument("--target_return_percentile", type=float, default=0.9)
    parser.add_argument("--max_eval_steps", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="halfcheetah_results")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    set_seed(cfg.seed)

    out_dir = Path(cfg.output_dir) / f"halfcheetah_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", asdict(cfg))

    episodes, state_dim, action_dim = load_halfcheetah_episodes(cfg)

    print(f"\n=== {cfg.env_name} ===")
    plain_r = train_model("Plain DT", False, cfg, episodes, state_dim, action_dim, out_dir)
    quality_r = train_model("Quality Aware DT", True, cfg, episodes, state_dim, action_dim, out_dir)

    diff = quality_r["final_mean"] - plain_r["final_mean"]

    summary = {
        "env_name": cfg.env_name,
        "confidence_mode": cfg.confidence_mode,
        "confidence_decay": cfg.confidence_decay,
        "confidence_blend": cfg.confidence_blend,
        "plain_dt": {
            "final_mean": plain_r["final_mean"],
            "final_std": plain_r["final_std"],
            "best_return": plain_r["best_return"],
        },
        "quality_aware_dt": {
            "final_mean": quality_r["final_mean"],
            "final_std": quality_r["final_std"],
            "best_return": quality_r["best_return"],
        },
        "improvement": diff,
    }
    save_json(out_dir / "summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for result, color in [(plain_r, "#888888"), (quality_r, "#1a7abf")]:
        epochs = [r["epoch"] for r in result["epoch_rows"]]
        losses = [r["train_loss"] for r in result["epoch_rows"]]
        returns = [r["eval_return_mean"] for r in result["epoch_rows"]]
        axes[0].plot(epochs, losses, color=color, marker="o", markersize=3, label=result["label"])
        axes[1].plot(epochs, returns, color=color, marker="o", markersize=3, label=result["label"])

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].set_title("Eval return")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(f"{cfg.env_name} — Plain DT vs Quality Aware DT")
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves.png", dpi=220, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 55)
    print(f"Environment:         {cfg.env_name}")
    print(f"Confidence mode:    {cfg.confidence_mode}")
    print(f"Confidence decay:   {cfg.confidence_decay}")
    print(f"Confidence blend:   {cfg.confidence_blend}")
    print(f"Plain DT:           {plain_r['final_mean']:.1f} (best {plain_r['best_return']:.1f})")
    print(f"Quality Aware DT:   {quality_r['final_mean']:.1f} (best {quality_r['best_return']:.1f})")
    print(f"Improvement:        {diff:+.1f}")
    print(f"Output:             {out_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
