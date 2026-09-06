"""
Offline DQN Training with Model Reuse, Checkpoints, and Logging
---------------------------------------------------------------
- Loads existing model if available
- Trains on offline replay buffer (state, action, reward, next_state)
- Periodically saves updated model weights
- Logs per-epoch loss and Q-value stats to CSV for visualization

NOTE: This is the original (vanilla) DQN trainer — it uses a plain
single-network DQN target (`max_next_q` from the same target net) rather
than Double DQN, and has no Huber loss / soft target updates / state
normalization. `new_run/train_stable_dqn.py` is the newer, stabilized
DDQN trainer described in the README and should be treated as the
canonical training script; this file appears to be superseded by it and
is a candidate for removal or archival (see review summary). It now
shares its network architecture and state dimension with that script
via `dqn_model.py` (see that module's docstring for the 34-vs-38
reconciliation), so checkpoints trained by either script are
architecture-compatible.

Inputs: replay_buffer_multi_policy.csv (path configurable via
    config.yaml's dqn_training.replay_buffer_csv), a flat CSV with N
    state columns, 1 action column, N next-state columns, 1 reward
    column, where N is inferred from the CSV's column count (see
    `generate_offline_data_from_cache`) rather than hardcoded.
Outputs: dqn_policy_net.pth (model checkpoint, saved every 10 epochs and
    at the end), training_log.csv (per-epoch loss/Q-value log).

Hyperparameters are loaded from config.yaml's `dqn_training` section
(see `dqn_model.load_config`), falling back to the hardcoded defaults
below if the file or a given key is missing, so this still runs with
zero args.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import pandas as pd
import os
import csv
from math import ceil

from dqn_model import DQN, STATE_DIM, NUM_ACTIONS, load_config

_FULL_CONFIG = load_config()
_CONFIG = _FULL_CONFIG.get("dqn_training", {})
_SEED = _FULL_CONFIG.get("seed", 42)

# Seed every RNG this script touches, so a fixed replay buffer gives a
# fully reproducible run (previously nothing here was seeded at all).
torch.manual_seed(_SEED)
np.random.seed(_SEED)
random.seed(_SEED)

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    """Fixed-capacity FIFO buffer of (state, action, reward, next_state) transitions."""

    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maximum number of transitions retained; oldest
                entries are dropped once full (deque maxlen).
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        """Append one offline transition to the buffer."""
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniformly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample without replacement.

        Returns:
            Tuple of (states, actions, rewards, next_states) tensors, with
            states/next_states as float32 and actions as int64.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

# -----------------------------
# Load Offline Data
# -----------------------------
def generate_offline_data_from_cache(log_file: str) -> list[tuple[np.ndarray, int, float, np.ndarray]]:
    """Load a replay buffer CSV into (s, a, r, s') tuples.

    The state dimension N is inferred from the CSV's column count
    (columns are N state features, 1 action, N next-state features, 1
    reward — optionally with a trailing "policy" label column, matching
    the layout `Executor.write_replay_buffer_simple` writes and the
    convention `new_run/train_stable_dqn.py::extend_from_csv_stream`
    uses), rather than hardcoded to a fixed width. This is what lets
    this script train on the same 38-feature cost-aware buffers as the
    newer trainer without silently misaligning columns.

    Args:
        log_file: Path to the replay buffer CSV.

    Returns:
        List of (state, action, reward, next_state) tuples.
    """
    df = pd.read_csv(log_file)
    cols = list(df.columns)
    has_policy = len(cols) > 0 and str(cols[-1]).lower() == "policy"
    n = (len(cols) - 2) // 2 if not has_policy else (len(cols) - 3) // 2
    if n != STATE_DIM:
        print(f"[WARN] Inferred state dimension is {n}, expected canonical STATE_DIM={STATE_DIM}.")

    data = []
    for _, row in df.iterrows():
        state = np.array(row.iloc[0:n])
        action = int(row.iloc[n])
        next_state = np.array(row.iloc[n + 1:n + 1 + n])
        reward = float(row.iloc[n + 1 + n])
        data.append((state, action, reward, next_state))
    return data

# -----------------------------
# Logging Utilities
# -----------------------------
def log_training(epoch: int, avg_loss: float, avg_q: float, csv_file: str = "training_log.csv") -> None:
    """Append one epoch's stats as a row to a CSV log, writing a header on first use.

    Args:
        epoch: Epoch index.
        avg_loss: Mean training loss (MSE) over the epoch's batches.
        avg_q: Mean chosen-action Q-value over the epoch's batches.
        csv_file: Path to the log CSV; created with a header if absent.
    """
    header = ["epoch", "avg_loss", "avg_q_value"]
    row = [epoch, avg_loss, avg_q]
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

# -----------------------------
# Offline DQN Training (with resume + logging)
# -----------------------------
def train_offline_dqn() -> None:
    """Train (or resume) a vanilla offline DQN on a fixed replay buffer CSV.

    All hyperparameters below are hardcoded rather than sourced from a
    config file or CLI args (unlike `new_run/train_stable_dqn.py`, which
    exposes them via argparse) — see review summary for a proposed
    config.yaml consolidating these values.
    """
    # 1️⃣ Hyperparameters (from config.yaml's `dqn_training` section,
    # falling back to these defaults if the file/key is missing)
    # num_actions: cache_length (5) candidate evictions + 1 "no eviction" no-op
    num_actions = _CONFIG.get("num_actions", NUM_ACTIONS)
    gamma = _CONFIG.get("gamma", 0.98)  # discount factor for future reward
    batch_size = _CONFIG.get("batch_size", 128)
    lr = _CONFIG.get("learning_rate", 5e-4)
    num_epochs = _CONFIG.get("num_epochs", 50)
    # hard target-network sync period, in epochs (see soft updates in train_stable_dqn.py)
    update_target_every = _CONFIG.get("target_update_frequency_epochs", 5)
    model_path = _CONFIG.get("model_path", "dqn_policy_net.pth")
    log_path = _CONFIG.get("training_log_csv", "training_log.csv")
    replay_buffer_csv = _CONFIG.get("replay_buffer_csv", "replay_buffer_multi_policy.csv")

    # 2️⃣ Load replay buffer (state_dim is inferred from the CSV itself,
    # not assumed, so this trainer stays correct even if the state
    # vector's feature count changes upstream in executor.py)
    offline_data = generate_offline_data_from_cache(replay_buffer_csv)
    state_dim = len(offline_data[0][0])
    replay_buffer = ReplayBuffer(capacity=len(offline_data))
    for s, a, r, s_next in offline_data:
        replay_buffer.add(s, a, r, s_next)

    # 3️⃣ Initialize or load model
    policy_net = DQN(state_dim, num_actions)
    target_net = DQN(state_dim, num_actions)

    if os.path.exists(model_path):
        print(f"🔄 Found existing model: {model_path}, resuming training...")
        policy_net.load_state_dict(torch.load(model_path))
        target_net.load_state_dict(policy_net.state_dict())
    else:
        print("🚀 Starting fresh training...")
        target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    n_samples = len(replay_buffer)
    num_batches_per_epoch = ceil(n_samples / batch_size)
    print(f"Starting Offline Training with {n_samples} samples...\n")

    # 4️⃣ Training Loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        avg_q_values = []

        for _ in range(num_batches_per_epoch):
            if len(replay_buffer) < batch_size:
                continue

            states, actions, rewards, next_states = replay_buffer.sample(batch_size)

            # Current Q-values
            q_pred = policy_net(states)
            chosen_q_values = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)
            avg_q_values.append(chosen_q_values.mean().item())

            # Target Q-values: vanilla DQN target (target_net both selects
            # and evaluates the best next action). This is the standard
            # DQN max-operator target, which is known to overestimate
            # Q-values; new_run/train_stable_dqn.py addresses this with a
            # Double DQN target (policy_net selects, target_net evaluates).
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                targets = rewards + gamma * max_next_q

            # MSE loss is sensitive to reward/Q-value outliers; the newer
            # trainer switches to Huber (SmoothL1) loss for this reason.
            loss = F.mse_loss(chosen_q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Periodic hard target sync: copy policy_net's weights into
        # target_net wholesale every `update_target_every` epochs, as
        # opposed to the soft (Polyak-averaged) update used in
        # train_stable_dqn.py.
        if epoch % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save a checkpoint every `checkpoint_every_epochs` epochs
        checkpoint_every = _CONFIG.get("checkpoint_every_epochs", 10)
        if epoch % checkpoint_every == 0 or epoch == num_epochs - 1:
            torch.save(policy_net.state_dict(), model_path)
            print(f"💾 Model checkpoint saved at epoch {epoch}")

        avg_loss = total_loss / num_batches_per_epoch
        avg_q = np.mean(avg_q_values) if avg_q_values else 0
        log_training(epoch, avg_loss, avg_q, log_path)
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.6f} | Avg Q: {avg_q:.4f}")

    print("\n✅ Training complete!")
    torch.save(policy_net.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

    # 5️⃣ Quick Test
    sample_state = torch.tensor(np.random.rand(state_dim), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(sample_state)
        best_action = q_values.argmax().item()
    print("\nSample State Test:")
    print("Predicted Q-values:", q_values.numpy())
    print("Best action:", best_action)
    print(f"\n📈 Training log saved to: {log_path}")

# -----------------------------
# Run Training
# -----------------------------
if __name__ == "__main__":
    train_offline_dqn()
