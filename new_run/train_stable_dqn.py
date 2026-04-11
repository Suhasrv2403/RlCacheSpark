#!/usr/bin/env python3
"""
train_stable_dqn.py

Stable offline Double DQN training with robust stabilizers, optimized
for the cache eviction problem with a focus on stability (DDQN, Huber Loss)
and utilizing static state normalization.

Usage:
    python train_stable_dqn.py --csv replay_buffer_cost_aware.csv --model dqn_cost_aware_net.pth
"""

import os
import argparse
import random
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

# -----------------------------
# Config defaults (tunable)
# -----------------------------
DEFAULT_BATCH = 1024
DEFAULT_EPOCHS = 3000
DEFAULT_LR = 5e-4  # Lowered LR for increased stability
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005  # Soft update coefficient (small value for stability)
DEFAULT_REPLAY_CAP = 500_000

# --- CRITICAL CHANGE: REWARD SCALING ADJUSTMENT ---
# The new cost-aware reward is already normalized and small.
# We set scale to 1.0 to prevent gradient explosion.
DEFAULT_CLIP_REWARD = 5.0  # Clip rewards to a reasonable range for cost-based penalties
DEFAULT_REWARD_SCALE = 1.0  # Set to 1.0 to preserve the magnitude of the cost signal
# ----------------------------------------------------

DEFAULT_GRAD_CLIP = 1.0
DEFAULT_DEVICE = "cpu"
RESULTS_DIR = "results"


# -----------------------------
# Model
# -----------------------------
class DQN(nn.Module):
    """
    DQN Network Architecture. Uses LayerNorm (LN) for stability during inference.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Use a slightly larger hidden dimension for the new 38-feature input
        hidden_dim = 256  # Keeping it at 256 is fine
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # LayerNorm implementation
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Replay buffer (simple)
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=DEFAULT_REPLAY_CAP):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s_next):
        self.buf.append((s, a, r, s_next))

    def extend_from_csv_stream(self, csv_file, max_rows=None, chunksize=100000):
        """
        Stream CSV in chunks to avoid memory explosion.
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
                      resume_checkpoint: str = None):
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
    # Scheduler to automatically reduce LR if loss plateaus (good stabilizer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=50, verbose=True)

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
        if (epoch + 1) % 500 == 0 or epoch == num_epochs - 1:
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
def plot_training_loss(losses):
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


def plot_distribution(values, name):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    plt.figure(figsize=(6, 4))
    sns.histplot(values, bins=50, kde=True)
    plt.title(name)
    plt.savefig(f"{RESULTS_DIR}/{name.replace(' ', '_')}.png", dpi=200)
    plt.close()
    print(f"[PLOT] {name}.png")


# ... (Other plotting functions remain unchanged) ...


def quick_test_model(model_path, state_dim, num_actions, device="cpu"):
    """
    Load and run a quick forward pass to ensure model loads and outputs Q-values.
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
    # --- CRITICAL CHANGE: DEFAULT CSV NAME ---
    parser.add_argument("--csv", type=str, default="../replay_buffer_cost_aware.csv",
                        help="replay buffer file (should contain cost-aware reward)")
    # -----------------------------------------
    parser.add_argument("--model", type=str, default="dqn_cost_aware_net.pth", help="file to save policy network")
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