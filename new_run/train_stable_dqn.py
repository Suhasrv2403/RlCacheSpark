#!/usr/bin/env python3
"""
train_stable_dqn.py

Stable offline Double DQN training with robust stabilizers, optimized
for the cache eviction problem with a focus on stability (DDQN, Huber Loss)
and utilizing static state normalization.

This is the canonical/current training script referenced by the README
(as opposed to the top-level `dqn_training.py`, a vanilla-DQN precursor —
see that file's module docstring for the comparison). Both scripts now
share one DQN architecture and state dimension via `dqn_model.py`
(STATE_DIM=38); see that module's docstring for how the previous
34-vs-38 mismatch was reconciled.

Inputs: a replay-buffer CSV (default path `../replay_buffer_cost_aware.csv`,
    relative to this file), with columns [state features..., action,
    next-state features..., reward] and an optional trailing "policy"
    column — expected to hold the *cost-aware* reward, not a raw
    hit/miss signal (see `DEFAULT_REWARD_SCALE`/`DEFAULT_CLIP_REWARD`
    comments below). Optionally a `--resume` checkpoint path.
Outputs: `<model>` (final policy-network state_dict), periodic
    `<model>.ckpt_epoch<N>.pth` full checkpoints (policy + target net +
    optimizer state, every 500 epochs), and `results/training_loss.png`.

All defaults below are loaded from config.yaml's `train_stable_dqn`
section (see `dqn_model.load_config`); CLI flags override them when
passed explicitly, and hardcoded fallbacks below apply if config.yaml
or a given key is missing, so this still runs with zero args.

Usage:
    python train_stable_dqn.py --csv replay_buffer_cost_aware.csv --model dqn_cost_aware_net.pth
"""

import os
import sys
import argparse
import random
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional

# dqn_model.py currently lives at the repo root (one level up from this
# new_run/ script); add it to sys.path so this script works whether it's
# run from new_run/ or from the repo root. NOTE: once the planned folder
# restructure lands (dqn_model.py -> src/dqn_model.py, this script ->
# scripts/train_stable_dqn.py), this path needs updating to point at src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dqn_model import DQN, load_config  # noqa: E402

_FULL_CONFIG = load_config()
_CONFIG = _FULL_CONFIG.get("train_stable_dqn", {})
_SEED = _FULL_CONFIG.get("seed", 42)

# Seed every RNG this script touches, so a fixed replay buffer gives a
# fully reproducible run (previously nothing here was seeded at all).
torch.manual_seed(_SEED)
np.random.seed(_SEED)
random.seed(_SEED)

# -----------------------------
# Config defaults (tunable) — sourced from config.yaml, falling back to
# these hardcoded values if config.yaml or a key is missing.
# -----------------------------
DEFAULT_BATCH = _CONFIG.get("batch_size", 1024)
DEFAULT_EPOCHS = _CONFIG.get("num_epochs", 3000)
DEFAULT_LR = _CONFIG.get("learning_rate", 5e-4)  # Lowered LR for increased stability
DEFAULT_GAMMA = _CONFIG.get("gamma", 0.99)
DEFAULT_TAU = _CONFIG.get("tau", 0.005)  # Soft update coefficient (small value for stability)
DEFAULT_REPLAY_CAP = _CONFIG.get("replay_capacity", 500_000)

# --- CRITICAL CHANGE: REWARD SCALING ADJUSTMENT ---
# The new cost-aware reward is already normalized and small.
# We set scale to 1.0 to prevent gradient explosion.
DEFAULT_CLIP_REWARD = _CONFIG.get("reward_clip", 5.0)  # Clip rewards to a reasonable range for cost-based penalties
DEFAULT_REWARD_SCALE = _CONFIG.get("reward_scale", 1.0)  # Set to 1.0 to preserve the magnitude of the cost signal
# ----------------------------------------------------

DEFAULT_GRAD_CLIP = _CONFIG.get("grad_clip", 1.0)
DEFAULT_DEVICE = _CONFIG.get("device", "cpu")
RESULTS_DIR = "results"
DEFAULT_LR_SCHEDULER_FACTOR = _CONFIG.get("lr_scheduler", {}).get("factor", 0.5)
DEFAULT_LR_SCHEDULER_PATIENCE = _CONFIG.get("lr_scheduler", {}).get("patience", 50)
DEFAULT_CHECKPOINT_EVERY = _CONFIG.get("checkpoint_every_epochs", 500)
DEFAULT_CSV = _CONFIG.get("replay_buffer_csv", "../replay_buffer_cost_aware.csv")
DEFAULT_MODEL_PATH = _CONFIG.get("model_path", "dqn_cost_aware_net.pth")


# -----------------------------
# Model: imported from dqn_model.DQN (see that module's docstring for
# the history of why this used to be defined separately in three places).
# STATE_DIM (38) / NUM_ACTIONS (6) are the canonical dimensions; this
# script still infers the actual state_dim from the CSV at load time
# (see extend_from_csv_stream) rather than assuming STATE_DIM, so it
# keeps working if pointed at a differently-shaped buffer.
# -----------------------------


# -----------------------------
# Replay buffer (simple)
# -----------------------------
class ReplayBuffer:
    """Fixed-capacity FIFO buffer of (state, action, reward, next_state) transitions,
    populated by streaming a replay-buffer CSV rather than by live interaction."""

    def __init__(self, capacity: int = DEFAULT_REPLAY_CAP):
        """
        Args:
            capacity: Max transitions retained (oldest dropped once full).
        """
        self.buf = deque(maxlen=capacity)

    def add(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray) -> None:
        """Append one transition to the buffer."""
        self.buf.append((s, a, r, s_next))

    def extend_from_csv_stream(self, csv_file: str, max_rows: Optional[int] = None, chunksize: int = 100000) -> None:
        """
        Stream a replay-buffer CSV in chunks to avoid loading it entirely
        into memory, inferring the state dimension `n` from the column
        count rather than hardcoding it (contrast with `dqn_training.py`,
        which hardcodes fixed column offsets).

        Expected column layout: `n` state features, 1 action column,
        `n` next-state features, 1 reward column, and optionally a
        trailing "policy" label column (detected by name, case-insensitive).

        Args:
            csv_file: Path to the replay buffer CSV.
            max_rows: Optional cap on total transitions loaded.
            chunksize: Rows read per `pandas.read_csv` chunk.
        """
        print(f"Loading data from {csv_file}...")
        read = 0
        for chunk in pd.read_csv(csv_file, chunksize=chunksize):
            cols = list(chunk.columns)
            # Infer state dimension N
            # Assumes columns are: N state features, 1 action, N next state features, 1 reward, [1 policy]
            has_policy = cols[-1].lower() == "policy"
            n = (len(cols) - 2) // 2 if not has_policy else (len(cols) - 3) // 2

            # --- INFORMATIONAL CHECK ---
            if read == 0 and n != 38:
                print(f"[WARN] Inferred state dimension is {n}. Expected 38 features (5*6 + 8).")
            # ---------------------------

            for _, row in chunk.iterrows():
                s = row.iloc[0:n].to_numpy(dtype=float)
                a = int(row.iloc[n])
                s_next = row.iloc[n + 1:n + 1 + n].to_numpy(dtype=float)
                r = float(row.iloc[n + 1 + n])  # This is the Cost-Aware Reward
                self.add(s, a, r, s_next)
                read += 1
                if max_rows and read >= max_rows:
                    return
        return

    def sample(self, batch_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(np.array(a), dtype=torch.int64),
            torch.tensor(np.array(r), dtype=torch.float32),
            torch.tensor(np.array(s_next), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# -----------------------------
# Helpers: normalization stats (Unchanged)
# -----------------------------
def compute_state_stats(replay: ReplayBuffer) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes static mean and standard deviation for state normalization.
    """
    # stack a representative subset if buffer very large (to save memory)
    N = min(len(replay.buf), 20000)
    idxs = np.linspace(0, len(replay.buf) - 1, N, dtype=int)
    arr = np.vstack([replay.buf[i][0] for i in idxs])
    # Add a small epsilon to std to prevent division by zero for constant features
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    return mean, std


def normalize(tensor, mean, std):
    """ Applies static normalization. """
    return (tensor - mean) / std


# -----------------------------
# Training loop (Unchanged core logic, excellent setup!)
# -----------------------------
def train_offline_dqn(csv_file: str,
                      model_path: str,
                      device: str = DEFAULT_DEVICE,
                      batch_size: int = DEFAULT_BATCH,
                      num_epochs: int = DEFAULT_EPOCHS,
                      lr: float = DEFAULT_LR,
                      gamma: float = DEFAULT_GAMMA,
                      tau: float = DEFAULT_TAU,
                      reward_clip: float = DEFAULT_CLIP_REWARD,
                      reward_scale: float = DEFAULT_REWARD_SCALE,
                      grad_clip: float = DEFAULT_GRAD_CLIP,
                      replay_capacity: int = DEFAULT_REPLAY_CAP,
                      max_rows: int = None,
                      resume_checkpoint: str = None) -> Tuple[list, int, int]:
    """Train an offline Double DQN policy on a streamed replay-buffer CSV.

    Stabilizers applied (why each is needed for this noisy, offline,
    Spark-like-workload setting):
      - Double DQN target (`policy_net` selects the next action,
        `target_net` evaluates it): decouples action selection from
        value evaluation to counter the max-operator's overestimation
        bias that plain DQN suffers from (see `dqn_training.py` for the
        vanilla-DQN comparison).
      - Huber loss (`nn.SmoothL1Loss`): quadratic near zero, linear for
        large errors, so occasional reward/Q outliers don't dominate the
        gradient the way squared error would.
      - Soft (Polyak) target updates (`tau`-weighted blend every step,
        not a periodic hard copy): keeps the bootstrap target slowly
        moving instead of jumping, which is gentler on an offline buffer
        that provides no fresh on-policy correction signal.
      - Static state normalization (precomputed mean/std over a buffer
        sample): keeps input feature scales comparable regardless of
        which sub-features (e.g. size in bytes vs. row counts vs. small
        ratios) dominate raw magnitude.
      - Reward clipping + scaling, and gradient-norm clipping: bound the
        magnitude of both the training signal and the resulting gradient
        step, since a single bad/rare transition can otherwise destabilize
        a Q-network's rolling estimates.

    Args:
        csv_file: Replay buffer CSV path.
        model_path: Where to save the final policy-network state_dict.
        device: torch device string ("cpu" or "cuda").
        batch_size: Transitions per gradient step.
        num_epochs: Number of gradient steps (misleadingly named — each
            "epoch" here is one sampled minibatch update, not a full
            pass over the buffer).
        lr: Adam learning rate.
        gamma: Discount factor for the DDQN target.
        tau: Soft target-update coefficient (fraction of policy_net
            copied into target_net's running average each step).
        reward_clip: Clamp raw rewards to [-reward_clip, reward_clip]
            before scaling.
        reward_scale: Multiplier applied to clipped rewards.
        grad_clip: Max gradient norm for `clip_grad_norm_`.
        replay_capacity: Max transitions retained in the buffer.
        max_rows: Optional cap on transitions loaded from the CSV.
        resume_checkpoint: Optional path to a full checkpoint (policy +
            target + optimizer state) to resume from.

    Returns:
        Tuple of (recent_losses, state_dim, num_actions), where
        recent_losses is the last up-to-200 loss values (for plotting).
    """
    device = torch.device(device)

    # 1) load replay (stream)
    print("[TRAIN] Loading replay data (stream)...")
    replay = ReplayBuffer(capacity=replay_capacity)
    replay.extend_from_csv_stream(csv_file, max_rows=max_rows)
    print(f"[TRAIN] Loaded {len(replay)} transitions.")

    if len(replay) == 0:
        raise RuntimeError("No transitions loaded. Check CSV.")

    # infer dims
    s0, a0, r0, s1 = replay.buf[0]
    state_dim = len(s0)
    num_actions = max(t[1] for t in replay.buf) + 1
    print(f"[TRAIN] State Dimension: {state_dim}, Num Actions: {num_actions}")

    # compute static state normalization stats
    state_mean_np, state_std_np = compute_state_stats(replay)
    state_mean = torch.tensor(state_mean_np, dtype=torch.float32, device=device)
    state_std = torch.tensor(state_std_np, dtype=torch.float32, device=device)
    print("[TRAIN] Computed static state normalization statistics.")

    # build networks
    policy_net = DQN(state_dim, num_actions).to(device)
    target_net = DQN(state_dim, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Scheduler to automatically reduce LR if loss plateaus (good stabilizer).
    # BUG FIX: this used to pass verbose=True, which current PyTorch
    # (2.x, verbose was removed from ReduceLROnPlateau) rejects outright
    # with a TypeError — training could never even start. Dropped it;
    # LR changes are still visible via the printed `lr_now` in the
    # epoch log line below.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=DEFAULT_LR_SCHEDULER_FACTOR,
                                                     patience=DEFAULT_LR_SCHEDULER_PATIENCE)

    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[TRAIN] Loading checkpoint {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        policy_net.load_state_dict(ckpt['policy_state'])
        target_net.load_state_dict(ckpt['target_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"[TRAIN] Resumed from epoch {start_epoch}")

    # training state
    losses = deque(maxlen=200)
    # Use Huber Loss (SmoothL1Loss) for improved stability against outliers
    huber = nn.SmoothL1Loss()

    # training loop
    for epoch in range(start_epoch, num_epochs):
        if len(replay) < batch_size:
            print("[TRAIN] Not enough samples for batch - waiting")
            break

        states, actions, rewards_raw, next_states = replay.sample(batch_size)

        # push to device and normalize states
        states = states.to(device)
        next_states = next_states.to(device)
        actions = actions.to(device)

        # Reward Preprocessing: Clip then Scale (R is now small and negative)
        rewards = rewards_raw.clamp(-reward_clip, reward_clip) * reward_scale
        rewards = rewards.to(device)

        # Static State Normalization: Apply fixed mean/std
        states = normalize(states, state_mean, state_std)
        next_states = normalize(next_states, state_mean, state_std)

        # Q(s, a) from policy (online network)
        q_values = policy_net(states)  # shape [B, A]
        q_val = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN targets: Decouple selection and evaluation
        with torch.no_grad():
            # 1. Actions chosen by policy_net (online) - Selection
            next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)  # [B,1]
            # 2. Values evaluated by target_net - Evaluation
            next_q = target_net(next_states).gather(1, next_actions).squeeze(1)  # [B]
            target = rewards + gamma * next_q

        # loss (Huber)
        loss = huber(q_val, target)

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
        optimizer.step()

        # soft update target network (tau update)
        for p, tp in zip(policy_net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        losses.append(loss.item())

        # scheduler step (use mean of recent losses for stability)
        if (epoch + 1) % 10 == 0:
            scheduler.step(np.mean(losses) if len(losses) >= 1 else loss.item())

        # logging
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"[TRAIN] Epoch {epoch:04d} | Loss: {loss.item():.6f} | LR: {lr_now:.6e}")

        # checkpoint occasionally
        if (epoch + 1) % DEFAULT_CHECKPOINT_EVERY == 0 or epoch == num_epochs - 1:
            ckpt = {
                'epoch': epoch + 1,
                'policy_state': policy_net.state_dict(),
                'target_state': target_net.state_dict(),
                'opt_state': optimizer.state_dict()
            }
            ckpt_path = model_path + f".ckpt_epoch{epoch + 1}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"[TRAIN] Saved checkpoint {ckpt_path}")

    # final save (policy only)
    torch.save(policy_net.state_dict(), model_path)
    print(f"[TRAIN] Training complete. Model saved to {model_path}")

    return list(losses), state_dim, num_actions


# -----------------------------------
# PLOT FUNCTIONS (Unchanged)
# -----------------------------------
def plot_training_loss(losses: list) -> None:
    """Save a raw + rolling-mean(50) Huber loss curve to results/training_loss.png.

    Args:
        losses: Sequence of per-step loss values to plot.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    plt.figure(figsize=(7, 4))
    plt.plot(losses, alpha=0.3, label="loss")
    plt.plot(pd.Series(losses).rolling(50).mean(), label="smooth", linewidth=2)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Huber Loss")
    plt.title("DQN Training Loss Curve")
    plt.savefig(f"{RESULTS_DIR}/training_loss.png", dpi=200)
    plt.close()
    print("[PLOT] training_loss.png")


def plot_distribution(values, name: str) -> None:
    """Save a histogram+KDE of `values` to results/<name>.png (spaces replaced with underscores).

    Args:
        values: 1D array-like of numeric values to plot.
        name: Plot title and (sanitized) output filename stem.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    plt.figure(figsize=(6, 4))
    sns.histplot(values, bins=50, kde=True)
    plt.title(name)
    plt.savefig(f"{RESULTS_DIR}/{name.replace(' ', '_')}.png", dpi=200)
    plt.close()
    print(f"[PLOT] {name}.png")


# ... (Other plotting functions remain unchanged) ...


def quick_test_model(model_path: str, state_dim: int, num_actions: int, device: str = "cpu") -> None:
    """
    Load and run a quick forward pass to ensure model loads and outputs Q-values.

    Handles two checkpoint shapes: a plain policy-network `state_dict`
    (final save) or a full checkpoint dict with a `'policy_state'` key
    (periodic training checkpoints).

    Args:
        model_path: Path to the checkpoint to sanity-check.
        state_dim: Expected input feature count (must match the trained model).
        num_actions: Expected output action count.
        device: torch device to map the checkpoint onto.
    """
    model = DQN(state_dim, num_actions)
    if os.path.exists(model_path):
        sd = torch.load(model_path, map_location=device)
        try:
            # try direct load (policy net only)
            model.load_state_dict(sd)
        except Exception:
            # maybe checkpoint-style saved -> try key
            if isinstance(sd, dict) and 'policy_state' in sd:
                model.load_state_dict(sd['policy_state'])
            else:
                raise
        model.eval()
        sample = torch.randn(1, state_dim)
        with torch.no_grad():
            out = model(sample)
        print("[TEST] Model forward OK. Q-values shape:", out.shape)
    else:
        print("[TEST] Model not found at", model_path)


# -----------------------------
# CLI (Updated default CSV)
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help="replay buffer file (should contain cost-aware reward)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="file to save policy network")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    # --- CRITICAL CHANGE: DEFAULT REWARD SCALING ---
    parser.add_argument("--reward_clip", type=float, default=DEFAULT_CLIP_REWARD)
    parser.add_argument("--reward_scale", type=float, default=DEFAULT_REWARD_SCALE)
    # -----------------------------------------------
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--replay_capacity", type=int, default=DEFAULT_REPLAY_CAP)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # train
    losses, sdim, nactions = train_offline_dqn(csv_file=args.csv,
                                               model_path=args.model,
                                               device=args.device,
                                               batch_size=args.batch,
                                               num_epochs=args.epochs,
                                               lr=args.lr,
                                               gamma=args.gamma,
                                               tau=args.tau,
                                               reward_clip=args.reward_clip,
                                               reward_scale=args.reward_scale,
                                               grad_clip=args.grad_clip,
                                               replay_capacity=args.replay_capacity,
                                               max_rows=args.max_rows,
                                               resume_checkpoint=args.resume)

    # final test forward pass and plotting
    quick_test_model(args.model, sdim, nactions, device=args.device)
    plot_training_loss(losses)
