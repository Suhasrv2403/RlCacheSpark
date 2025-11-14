"""
Offline DQN Training with Model Reuse, Checkpoints, and Logging
---------------------------------------------------------------
- Loads existing model if available
- Trains on offline replay buffer (state, action, reward, next_state)
- Periodically saves updated model weights
- Logs per-epoch loss and Q-value stats to CSV for visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import pandas as pd
import os
import csv
from math import ceil

# -----------------------------
# DQN Model
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Load Offline Data
# -----------------------------
def generate_offline_data_from_cache(log_file):
    df = pd.read_csv(log_file)
    data = []
    for _, row in df.iterrows():
        state = np.array(row.iloc[0:34])
        next_state = np.array(row.iloc[35:69])
        action = int(row.iloc[34])
        reward = float(row.iloc[69])
        data.append((state, action, reward, next_state))
    return data

# -----------------------------
# Logging Utilities
# -----------------------------
def log_training(epoch, avg_loss, avg_q, csv_file="training_log.csv"):
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
def train_offline_dqn():
    # 1️⃣ Hyperparameters
    state_dim = 34
    num_actions = 6
    gamma = 0.98
    batch_size = 128
    lr = 5e-4
    num_epochs = 50
    update_target_every = 5
    model_path = "dqn_policy_net.pth"
    log_path = "training_log.csv"

    # 2️⃣ Initialize or load model
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

    # 3️⃣ Load replay buffer
    offline_data = generate_offline_data_from_cache("replay_buffer_multi_policy.csv")
    replay_buffer = ReplayBuffer(capacity=len(offline_data))
    for s, a, r, s_next in offline_data:
        replay_buffer.add(s, a, r, s_next)

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

            # Target Q-values
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                targets = rewards + gamma * max_next_q

            # Compute loss + backprop
            loss = F.mse_loss(chosen_q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Periodic target sync
        if epoch % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
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
