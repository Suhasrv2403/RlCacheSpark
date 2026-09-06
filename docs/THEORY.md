# Theory & Design Notes

This document holds the deep-dive material that used to live in the
README: the motivation for cost-aware eviction, the full MDP
formulation, the offline RL data pipeline, experiment design,
limitations, future directions, and references. See [`README.md`](../README.md)
for the project pitch and how to run things.

## Why LRU is not enough

- **Static policy:** LRU does not adapt when access patterns change.
- **Equal miss cost:** A miss on a cheap partition (~10 ms) and a miss on an expensive one (~1000 ms+) are not the same in production, but LRU does not distinguish them.
- **Hit ratio alone** can miss the point: a small fraction of misses on heavy partitions can dominate tail latency.

This work shifts from **discrete hit/miss rewards** toward **normalized, cost-aware penalties** tied to eviction and recomputation cost, so the agent can preferentially retain partitions that matter for worst-case latency.

## System overview

### Simulator instead of a live Spark cluster

RL needs many interactions. Running inside a real Spark/JVM stack is slow (startup, network). [`src/executor.py`](../src/executor.py) implements a **lightweight executor model** that captures:

- Fixed executor memory and a **cache of partitions** (sizes tunable).
- **Partition metadata** ([`src/DataSetProcessing.py`](../src/DataSetProcessing.py): size, recomputation cost, access stats, recency, query match / match ratio).
- **Filter pushdown–style** matching: predicates are checked against metadata before touching row data.
- **Eviction cycle:** on a full-cache miss, an eviction is chosen; the **evicted partition's recomputation cost** feeds cost and analysis (e.g. P99).

### MDP formulation

- **State** (38-dim, `dqn_model.STATE_DIM`): for each of the 5 cached partitions, 6 features (access count, seconds since last access, size in MB, row count, column count, is-cached), plus 8 global features — cached-partition count, instant hits/misses/hit-ratio for the triggering query, cache utilization, rolling eviction rate, and a 2-dim temporal/categorical query-intent flag. The full named layout is documented as a comment directly above `Executor._get_cache_state_vector`.
- **Actions:** discrete choice of **which cached partition to evict** (or **no eviction**) — the cache holds 5 slots → **6** actions including the no-op.
- **Reward:** two reward functions exist in `Executor`. `_simulate_multi_query_reward` is **hit-ratio-based** and is what currently gets logged into the replay buffer during data generation. `compute_cost_aware_reward` implements the **cost-aware** penalty described below (a strictly negative penalty proportional to `Partition.recomputation_cost_ms`) and is unit-tested, but **is not yet wired into the replay-buffer-generation path** — that integration (so training data reflects cost-aware rewards end to end) is the next step for this reward model, tracked as a follow-up rather than done silently.

### Offline data and network

- A **replay buffer** of **(state, action, reward, next_state)** is built by driving the simulator with **multiple** eviction policies so the value function sees both good and bad cache states.
- Early **DQN** runs showed **overestimation** and instability; training uses **DDQN**, **Huber loss** (outliers), and **soft target networks** for smoother learning. See [`scripts/train_stable_dqn.py`](../scripts/train_stable_dqn.py)'s `train_offline_dqn` docstring for exactly why each stabilizer is needed.

## Experiments (high level)

**LRU, LFU, FIFO, RANDOM**, and **RL** are compared on shared query traces via [`scripts/evaluate_policies.py`](../scripts/evaluate_policies.py):

- **Trained_Workload** — a stationary mix; check convergence.
- **Moving_Workload** — a phase shift partway through (temporal-heavy → categorical-heavy); LRU **thrashes** while the RL agent's query-intent state features let it adapt faster.
- **Unseen_Workload** — compound filters and edge ranges not exercised elsewhere; tests **generalization**, not memorization.

Metrics emphasize **cache hit ratio**, **P50**, and especially **P99 latency** as the tail-sensitive target — see `results/policy_evaluation_summary.csv` for per-workload numbers and `scripts/plot_results.py` for the charts.

### Limitations

- Simulator **collapses** real costs (I/O, skew, serialization) into **scalars**—production costs are messier. `Partition.recomputation_cost_ms` is a deterministic row/size-proportional placeholder, not a measured cost.
- **Action/state size** grows with cache width; at **cluster** scale (millions of partitions), new designs (top-K, hierarchical policies) would be needed.
- **Offline** training is tied to the buffer distribution; **distribution shift** may require **retraining** or **online** RL later.
- The cost-aware reward (`compute_cost_aware_reward`) is implemented and tested but not yet wired into offline data generation, so trained checkpoints still reflect the hit-ratio-based reward until that integration lands.

### Future directions

- Wire `compute_cost_aware_reward` into `run_query_and_record_rl_data` so replay buffers carry true cost-aware rewards.
- **Online RL** for continual adaptation.
- **Proactive** cache management (use idle time), not only reactive evictions.

## References

1. M. Zaharia et al., *Spark: The Definitive Guide*, O'Reilly, 2018.
2. R. Chen et al., "Improving Spark Cache Hit Ratio Through Learned Policies," IEEE Big Data, 2022.
3. Z. Jia et al., "Optimizing Caching in Distributed Systems with Reinforcement Learning," TKDE, 2021.
4. J. L. Ba et al., "Layer Normalization," arXiv:1607.06450.
5. R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
