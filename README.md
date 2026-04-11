# RlCacheSpark

Simulation and **offline reinforcement learning** for **cache eviction** in a Spark-style partitioned workload. The project compares classical policies (LRU, LFU, FIFO, RANDOM) with a **DQN-trained eviction policy** using a replay buffer of `(state, action, reward, next_state)` transitions.

## What’s in the repo

| Area | Role |
|------|------|
| `DataSetProcessing.py` | `Partition` abstraction: typed column storage, size estimates, metadata for queries. |
| `executor.py` | `Executor`: simulates a size-bounded cache over partitions, runs queries, supports eviction policies including **RL** (loads `dqn_policy_net.pth` when `eviction_policy="RL"`). |
| `evaluate_policies.py` | Runs each policy on the same workload and records hit ratio, latency, and timelines for comparison. |
| `dqn_training.py` | Offline DQN training on a CSV replay buffer; checkpoints and `training_log.csv` (ignored by git—regenerate locally). |
| `new_run/train_stable_dqn.py` | Double DQN variant with stabilizers (Huber loss, soft updates, reward clipping) for cost-aware rewards. |
| `plot_results.py` | Builds poster-style figures from `policy_evaluation_summary.csv` and `policy_timeline_data.csv`. |

Large artifacts (multi‑GB replay buffers, all `.pth` weights, PNG exports) are **not** tracked; clone the repo, install dependencies, then reproduce outputs locally.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Typical workflow

1. **Simulate and log transitions** — Run your pipeline so the executor writes a replay buffer CSV (see `dqn_training.py` for the expected column layout: state, action, next state, reward).
2. **Train the DQN** — `python dqn_training.py` (or `python new_run/train_stable_dqn.py` with your buffer path) to produce `dqn_policy_net.pth` in the project root (or the path you pass to the executor).
3. **Evaluate policies** — `python evaluate_policies.py` to refresh summary and timeline CSVs.
4. **Plots** — `python plot_results.py` to regenerate figures under `evaluation_plots_poster/` (ignored by git).

## Citation / context

Use this repo as research code for RL-based cache management experiments; adapt partition counts, cache size, and state dimensions in `executor.py` / `dqn_training.py` if you change the MDP.
