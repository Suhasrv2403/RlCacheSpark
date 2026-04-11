# The Cost of a Miss: RL-Driven Cache Eviction (RlCacheSpark)

**Suhas Ramesh Vittal** · Department of Computer Science · Golisano College of Computing and Information Sciences · Rochester Institute of Technology · Rochester, NY

Distributed data processing frameworks such as **Apache Spark** rely on the cache to cut query latency. Spark partitions are **heterogeneous** in recomputation cost, but the default policy **LRU** treats every cached block alike—so it can evict “heavy” partitions (expensive to rebuild) while keeping “light” ones. That **cost blindness** hurts **tail latency** (e.g. P99) even when average hit ratio looks fine.

This repository is a **Spark-inspired executor simulator** plus **offline reinforcement learning** that learns **cost-aware** evictions: a **Double DQN (DDQN)** trained on logged transitions, with rewards based on **recomputation penalties** rather than a simple hit/miss bit. The full write-up is in **`report.docx`** (same title as this project).

---

## At a glance

| | |
|--|--|
| **Goal** | Dynamic eviction that adapts to workload shifts and protects expensive partitions. |
| **Training** | **Offline RL** on a large replay buffer (states from LRU/LFU/FIFO/RANDOM mixes). |
| **Algorithm** | **DDQN**, **Huber loss**, **soft target updates**, **feature normalization** (stability under noisy Spark-like workloads). |
| **Reported gains (see paper)** | ~**35%** improvement in **P99 latency** vs LRU (avg.); ~**12%** hit-ratio improvement; **~12%** avg. improvement over LRU on **unseen** query patterns. |

*Exact numbers depend on your simulator settings and seeds; reproduce with `evaluate_policies.py` and your trained weights.*

---

## Why LRU is not enough

- **Static policy:** LRU does not adapt when access patterns change.
- **Equal miss cost:** A miss on a cheap partition (~10 ms) and a miss on an expensive one (~1000 ms+) are not the same in production, but LRU does not distinguish them.
- **Hit ratio alone** can miss the point: a small fraction of misses on heavy partitions can dominate tail latency.

This work shifts from **discrete hit/miss rewards** toward **normalized, cost-aware penalties** tied to eviction and recomputation cost, so the agent can preferentially retain partitions that matter for worst-case latency.

---

## System overview (from the report)

### Simulator instead of a live Spark cluster

RL needs many interactions. Running inside a real Spark/JVM stack is slow (startup, network). This repo implements a **lightweight executor model** that captures:

- Fixed executor memory and a **cache of partitions** (sizes tunable).
- **Partition metadata** (size, recomputation cost, access stats, recency, query match / match ratio).
- **Filter pushdown–style** matching: predicates are checked against metadata before touching row data.
- **Eviction cycle:** on a full-cache miss, an eviction is chosen; the **evicted partition’s recomputation cost** feeds cost and analysis (e.g. P99).

### MDP formulation

- **State** combines (1) **per-partition** features for cached partitions, (2) a **workload / query-context** vector (e.g. temporal vs categorical vs numerical intent), and (3) **global cache** signals (utilization, rolling eviction rate / thrashing pressure).
- **Actions:** discrete choice of **which cached partition to evict** (or **no eviction** when the design allows)—the code uses a small fixed cache footprint (e.g. five slots in the paper) → **six** actions including “no op.”
- **Reward:** **cost-aware**—penalties tied to **recomputation cost** of evicted or missed work, not ±1 hit/miss. This encourages **weighted** behavior: accept smaller penalties on light partitions to avoid large ones on heavy partitions.

### Offline data and network

- A **replay buffer** of **(state, action, reward, next_state)** is built by driving the simulator with **multiple** eviction policies so the value function sees both good and bad cache states.
- The **DQN** in early work showed **overestimation** and instability; the report describes moving to **DDQN**, **Huber loss** (outliers), and **soft target networks** for smoother learning.

---

## Experiments (high level)

The report evaluates **LRU, LFU, FIFO, RANDOM**, and **RL** on shared query traces:

- **Trained / mixed** workloads — check convergence.
- **Phase shift** — e.g. temporal-heavy → categorical-heavy mid-run; LRU **thrashes** while the RL agent uses **query context** to adapt faster.
- **Unseen** workloads — new ranges/categories; tests **generalization**, not memorization.

Metrics emphasize **cache hit ratio**, **P50**, and especially **P99 latency** as the tail-sensitive target.

### Limitations (from the report)

- Simulator **collapses** real costs (I/O, skew, serialization) into **scalars**—production costs are messier.
- **Action/state size** grows with cache width; at **cluster** scale (millions of partitions), new designs (top-K, hierarchical policies) would be needed.
- **Offline** training is tied to the buffer distribution; **distribution shift** may require **retraining** or **online** RL later.

### Future directions (from the report)

- **Online RL** for continual adaptation.
- **Proactive** cache management (use idle time), not only reactive evictions.

---

## This repository: code map

| Artifact | Purpose |
|----------|---------|
| `DataSetProcessing.py` | `Partition` objects: typed storage, metadata, size estimates. |
| `executor.py` | Cache simulator, query execution, eviction policies including **RL** (loads `dqn_policy_net.pth` by default). |
| `evaluate_policies.py` | Compare policies on the same workload; timelines and summaries for analysis/plots. |
| `dqn_training.py` | Offline DQN training from CSV replay data; checkpoints + logs (generated locally). |
| `new_run/train_stable_dqn.py` | **DDQN**-style training with stabilizers aligned with the report (Huber, soft updates, etc.). |
| `plot_results.py` | Figures from `policy_evaluation_summary.csv` / `policy_timeline_data.csv`. |
| `data_explore.py` | Small exploratory plots for datasets. |

Checkpoints (`.pth`), large `replay_buffer*.csv` files, PNG outputs, and IDE/OS junk are **gitignored** so the repo stays clone-friendly. After cloning, install deps, regenerate buffers and weights locally, then run evaluation and plotting.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Typical workflow

1. Run the simulator to produce **replay** data in the format expected by `dqn_training.py` / `train_stable_dqn.py`.
2. **Train** → obtain `dqn_policy_net.pth` (or a name you pass into the executor).
3. **Evaluate** → `python evaluate_policies.py`.
4. **Plot** → `python plot_results.py`.

---

## References (from the report)

1. M. Zaharia et al., *Spark: The Definitive Guide*, O’Reilly, 2018.  
2. R. Chen et al., “Improving Spark Cache Hit Ratio Through Learned Policies,” IEEE Big Data, 2022.  
3. Z. Jia et al., “Optimizing Caching in Distributed Systems with Reinforcement Learning,” TKDE, 2021.  
4. J. L. Ba et al., “Layer Normalization,” arXiv:1607.06450.  
5. R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.

---

## Repository hygiene (for contributors)

The GitHub-oriented cleanup added **`.gitignore`**, **`requirements.txt`**, and stopped tracking **large CSVs**, **weights**, **generated plots**, and **IDE/caches**. Full narrative PDF/DOCX content lives in **`report.docx`**; this README is the **landing page** summary.
