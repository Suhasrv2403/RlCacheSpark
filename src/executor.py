"""
Spark-inspired cache/executor simulator with pluggable eviction policies.

`Executor` owns a fixed-size in-memory cache of `Partition` objects
(`DataSetProcessing.Partition`) and simulates running filter queries
against a partitioned dataset, tracking hits/misses and triggering
eviction on cache-full misses. Supports LRU, LFU, FIFO, RANDOM, and a
trained-DQN ("RL") eviction policy, and can generate offline RL
training data (state/action/next_state/reward transitions) by driving
the simulator with each of the non-RL policies.

MDP formulation (see README for the full write-up):
  - State: a per-partition feature vector for every cached partition,
    concatenated with global cache signals (see `_get_cache_state_vector`).
  - Action: index of the cached partition to evict, or a "no eviction"
    action when the index is out of range of the current cache size.
  - Reward: two reward functions exist. `_simulate_multi_query_reward`
    is hit-ratio-based and is what `run_query_and_record_rl_data`
    currently logs into the replay buffer for offline training.
    `compute_cost_aware_reward` is the "cost-aware" recomputation-cost
    penalty the README describes for the DDQN pipeline — implemented
    and unit-tested, but not yet wired into
    `run_query_and_record_rl_data` (see that method's docstring and the
    review summary for the integration follow-up).

Inputs: a pandas DataFrame (via `create_partitions`), and optionally a
    trained DQN checkpoint path for the "RL" policy (default
    "dqn_policy_net.pth").
Outputs: when run as a script, writes replay_buffer_multi_policy.csv
    (state/action/next_state/reward transitions for offline training).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict
from DataSetProcessing import Partition  # use your Partition class here
import csv
import random
from datetime import timedelta, datetime
from tqdm import tqdm
import torch
import os

from dqn_model import DQN, STATE_DIM, NUM_ACTIONS

class Executor:
    """Simulates a Spark-executor-style partition cache under a chosen eviction policy.

    Owns two partition maps: `all_partitions` (the full dataset, keyed by
    partition id) and `cache` (the subset currently resident in memory,
    an `OrderedDict` so LRU/FIFO can use insertion/access order directly).
    """

    def __init__(self, max_cache_size_mb: int = 512, eviction_policy: str = "LRU"):
        """
        Args:
            max_cache_size_mb: Cache capacity in megabytes (converted to
                bytes and stored as `max_cache_size`; not currently
                enforced as a hard byte limit — eviction is instead
                driven by `cache_length`, a partition *count*, set later
                via `initialize_cache`).
            eviction_policy: One of "LRU", "LFU", "FIFO", "RANDOM", "RL"
                (case-insensitive). "RL" eagerly loads a DQN checkpoint
                named "dqn_policy_net.pth" from the working directory.
        """
        self.max_cache_size = max_cache_size_mb * 1024 * 1024
        self.cache: Dict[int, Partition] = OrderedDict()
        self.all_partitions: Dict[int, Partition] = {}
        self.eviction_policy = eviction_policy.upper()
        self.total_cache_used = 0
        self.cache_length = 0
        # Stats
        self.total_hits = 0
        self.total_misses = 0
        self.total_evictions = 0  # incremented in evict_partition; feeds the rolling-eviction-rate state feature
        self.query_log: List[Dict[str, Any]] = []
        self.flag = False

        if self.eviction_policy == "RL":
            self.load_rl_model("dqn_policy_net.pth")
    # -----------------------------
    # Partitioning and Initialization
    # -----------------------------

    def create_partitions(self, data: pd.DataFrame, n_partitions: int = 4) -> None:
        """Split `data` into `n_partitions` roughly-equal Partitions and register them.

        Note: calling this again with new data adds to (does not clear)
        `all_partitions`, re-keyed from 0 — if called more than once,
        later partitions with the same id overwrite earlier ones (see
        the `__main__` block, which calls this twice with two different
        DataFrames and 20/10 partitions, overwriting ids 0-9).

        Args:
            data: DataFrame to split.
            n_partitions: Number of partitions to create.
        """
        # BUG FIX: np.array_split(dataframe, n) used to return a list of
        # DataFrames, but with the numpy/pandas versions pinned by
        # requirements.txt (numpy 2.4.x / pandas 3.0.x) it silently
        # returns raw ndarrays instead, which crashes
        # Partition.__init__ (expects a DataFrame). Splitting the row
        # *index* instead and slicing with .iloc keeps this correct
        # regardless of that numpy/pandas behavior change.
        row_groups = np.array_split(np.arange(len(data)), n_partitions)
        split_data = [data.iloc[idx] for idx in row_groups]
        for i, df_part in enumerate(split_data):
            self.all_partitions[i] = Partition(df_part, partition_id=i)
        print(f"[INIT] Created {len(self.all_partitions)} partitions.")

    def initialize_cache(self, k: int) -> None:
        """Set the cache capacity to `k` partitions and pre-load the first `k`.

        Args:
            k: Number of partitions the cache may hold at once (also
                defines the RL action space size: `k` eviction choices
                plus one "no eviction" action).
        """
        self.cache_length = k
        for i in range(min(k, len(self.all_partitions))):
            self.load_partition(self.all_partitions[i])
        print(f"[CACHE INIT] Pre-loaded {k} partitions into cache.")

    def load_rl_model(self, model_path: str = "dqn_policy_net.pth") -> None:
        """
        Load a trained DQN checkpoint for the "RL" eviction policy.

        If the file doesn't exist, leaves `self.rl_model = None`, in
        which case `_rl_decide_eviction` falls back to a uniform-random
        choice among cached partitions rather than failing.

        Args:
            model_path: Path to a `state_dict` saved via `torch.save`.
        """
        if not os.path.exists(model_path):
            print(f"[RL] Model file not found at {model_path}. RL will use random decisions.")
            self.rl_model = None
            return

        # Architecture and dimensions come from dqn_model.py (the single
        # canonical definition shared by this class and both training
        # scripts): STATE_DIM=38 matches initialize_cache(k=5) (5
        # partition slots -> 5*6 per-partition features + 8 global
        # features, see _get_cache_state_vector) and NUM_ACTIONS=6 = 5
        # evictable slots + 1 "no eviction" no-op.
        self.rl_model = DQN(STATE_DIM, NUM_ACTIONS)
        self.rl_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.rl_model.eval()
        print(f"[RL] Loaded DQN model from {model_path} ✅")

    def _rl_decide_eviction(self, query_filters: Dict[str, Any]) -> Optional[int]:
        """
        Pick which cached partition to evict via a forward pass of the
        trained DQN, or None to signal "no eviction".

        Action space: index into the *current* list of cached partition
        ids (`list(self.cache.keys())`) — an action index at or beyond
        the current cache size is interpreted as the "no eviction"
        no-op action, so the effective action space shrinks/grows with
        how full the cache currently is.

        Args:
            query_filters: The filters of the query that triggered this
                eviction decision; folded into the state vector via
                `_get_cache_state_vector` (global hit/miss signal for
                this specific query).

        Returns:
            The partition id to evict, or None for "no eviction".
        """
        if not hasattr(self, "rl_model") or self.rl_model is None:
            print("[RL] No model loaded — using random eviction.")
            return np.random.choice(list(self.cache.keys()))

        # Build current cache state vector (same layout used at training time)
        state,order = self._get_cache_state_vector(query_filters)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = self.rl_model(state_tensor)
            print("[RL] Q-values:", q_values)
            best_action = q_values.argmax().item()

        cache_keys = list(self.cache.keys())

        # "No eviction" action
        if best_action >= len(cache_keys):
            print("[RL] Chose 'no eviction' action.")
            return None

        chosen_partition = cache_keys[best_action]
        print(f"[RL] Selected eviction action {best_action} → Partition {chosen_partition}")

        return chosen_partition

    # -----------------------------
    # Core Cache Logic
    # -----------------------------

    def load_partition(self, partition: Partition, query_filters: Optional[Dict[str, Any]] = None) -> None:
        """Insert `partition` into the cache, evicting first if at capacity.

        Args:
            partition: The partition to load (on a cache miss).
            query_filters: Filters of the triggering query; forwarded to
                `evict_partition` for the RL policy's state vector.

        Behavior note: if the RL policy's eviction decision is "no
        eviction" (`self.flag` set True by `evict_partition`), the new
        partition is deliberately *not* cached — the whole point of the
        no-op action is to leave the cache exactly as-is even though the
        query just missed. `self.flag` is a one-shot signal consumed
        here and reset immediately; it assumes `evict_partition` is
        always called (via the `while` loop below) immediately before
        this check on the same call to `load_partition`.
        """
        pid = partition.partition_id
        size = partition.size_in_memory

        # Evict partitions if full
        while len(self.cache) >= self.cache_length:
            evicted_id = self.evict_partition(query_filters)
            if evicted_id is None:
                break

        if self.eviction_policy == "RL" and self.flag:

            self.flag = False


        else:
            self.cache[pid] = partition
            partition.is_cached = True
            self.total_cache_used += size
        #print(f"[LOAD] Cached Partition {pid} ({size / 1024**2:.2f} MB)")

    def evict_partition(self, query_filters: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Evict one partition from the cache according to `self.eviction_policy`.

        Args:
            query_filters: Only used by the "RL" policy, to build the
                state vector fed to the DQN.

        Returns:
            For LRU/LFU/FIFO/RANDOM: the id of the *next* partition that
            would be evicted after this one (informational only, used in
            a log line — not acted on by callers). For "RL": always None
            (the "next in line" concept doesn't apply to a value-based
            single-step decision). Also returns None if the cache is
            already empty, or if the RL policy chooses "no eviction" (in
            which case `self.flag` is set so `load_partition` skips
            caching the new partition — see `load_partition`).

        Raises:
            ValueError: If `self.eviction_policy` is not one of
                LRU/LFU/FIFO/RANDOM/RL.
        """
        if not self.cache:
            return None

        evict_id = None
        next_evict_id = None

        if self.eviction_policy == "LRU":
            sorted_keys = sorted(self.cache, key=lambda pid: self.cache[pid].last_accessed)
            evict_id = sorted_keys[0]
            next_evict_id = sorted_keys[1] if len(sorted_keys) > 1 else None

        elif self.eviction_policy == "LFU":
            sorted_keys = sorted(self.cache, key=lambda pid: self.cache[pid].access_count)
            evict_id = sorted_keys[0]
            next_evict_id = sorted_keys[1] if len(sorted_keys) > 1 else None

        elif self.eviction_policy == "FIFO":
            keys = list(self.cache.keys())
            evict_id = keys[0]
            next_evict_id = keys[1] if len(keys) > 1 else None

        elif self.eviction_policy == "RANDOM":
            keys = list(self.cache.keys())
            evict_id = np.random.choice(keys)
            remaining = [k for k in keys if k != evict_id]
            next_evict_id = remaining[0] if remaining else None

        elif self.eviction_policy == "RL":
            evict_id = self._rl_decide_eviction(query_filters)
            next_evict_id = None
            if evict_id is None:
                self.flag = True
                print("[RL] Model chose 'no eviction' action. Cache unchanged.")
                return None

        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

        # Evict the partition
        evicted = self.cache.pop(evict_id)
        self.total_cache_used -= evicted.size_in_memory
        evicted.is_cached = False
        self.total_evictions += 1

        # Optional logging
        print(f"[EVICT] Partition {evict_id} evicted using {self.eviction_policy}. Next in line: {next_evict_id}")

        return next_evict_id



    # -----------------------------
    # Query Execution
    # -----------------------------

    def execute_query(self, query_filters: Dict[str, Any]) -> List[Partition]:
        """
        Execute a query against all partitions, updating cache/hit-miss
        state as a side effect:
          - Prune partitions that provably can't match (via `Partition.can_prune`).
          - For each non-pruned partition: cache hit if already loaded,
            otherwise a miss that triggers `load_partition` (and possibly
            an eviction).
          - Appends a per-query summary dict to `self.query_log`.

        Args:
            query_filters: Filter dict, same shape accepted by
                `Partition.can_prune` / `Partition.filter_rows`.

        Returns:
            The list of Partition objects that matched (were not pruned),
            in partition-id order — this returns full Partition objects,
            not the filtered rows; call `.filter_rows(query_filters)` on
            each if row-level data is needed.
        """
        hits, misses = 0, 0
        matched_partitions = []

        #print(f"\n[QUERY START] Filters: {query_filters}")

        for pid, partition in self.all_partitions.items():
            if not partition.can_prune(query_filters):
                matched_partitions.append(pid)

                #print(f"Partition {pid} matches filters: ", end="")
                if pid in self.cache:
                    hits += 1
                    self.total_hits += 1
                    partition.access_count += 1
                    partition.last_accessed = pd.Timestamp.now()
                    #print(f"  [HIT IN CACHE] Partition {pid}")

                else:
                    misses += 1
                    self.total_misses += 1
                    #print(f"  [HIT OUTSIDE CACHE] Partition {pid}")
                    # Insert code to calculate input feature vector
                    self.load_partition(partition,query_filters)
                    # get next two partitions to be evicted
                    # compute feature vector
                    # calculate reward
                    partition.access_count += 1
                    partition.last_accessed = pd.Timestamp.now()
                    # store data input vector1 , action1 , reward1
                    # store data input vector2 , action2 , reward2

                #print("-" * 50)
                #temp = partition.partition_stats(query_filters)
                #for k, v in temp.items():
                    #print(f"  {k}: {v}")
                #print("-" * 50)

                # Move recently used partition to end for LRU/FIFO
                if self.eviction_policy in ["LRU", "FIFO"]:
                    self.cache.move_to_end(pid)

        hit_ratio = self.get_cache_hit_ratio()
        self.query_log.append({
            "filters": query_filters,
            "hits": hits,
            "misses": misses,
            "hit_ratio": hit_ratio
        })

        #print(f"[QUERY END] Hits: {hits}, Misses: {misses}, Hit Ratio: {hit_ratio:.2f}")
        return [self.all_partitions[pid] for pid in matched_partitions]

    # -----------------------------
    # Metrics and Utilities
    # -----------------------------

    def get_instant_cache_hit_ratio(self, query_filters: Dict[str, Any]) -> Tuple[int, int]:
        """Count hits/misses this specific query would produce against the current cache.

        Read-only: unlike `execute_query`, does not mutate cache state,
        access counts, or `self.total_hits`/`self.total_misses`. Used to
        build the RL state vector and reward simulation without disturbing
        real cache statistics.

        FIXED (previously a bug): this used to return the bare int `0`
        when no partition matched `query_filters`, instead of a
        `(0, 0)` tuple, which crashed every `hits, miss = ...` caller
        (`_rl_decide_eviction` -> `_get_cache_state_vector`,
        `_simulate_multi_query_reward`) the first time a query matched
        zero partitions. Now always returns a 2-tuple.

        Args:
            query_filters: Filter dict identifying which partitions are relevant.

        Returns:
            `(total_hits, total_miss)` tuple; `(0, 0)` if no partition matches.
        """
        total_hits = 0
        total_miss = 0
        for pid, partition in self.all_partitions.items():
            if not partition.can_prune(query_filters):

                # Cache hit — update metadata only
                if pid in self.cache:
                    total_hits += 1

                else:
                    total_miss += 1
        return (total_hits, total_miss)

    def get_cache_hit_ratio(self) -> float:
        """Return the cumulative hit ratio in [0, 1] across all `execute_query` calls so far."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def get_cache_summary(self) -> Dict[str, Any]:
        """Return a snapshot dict of policy name, cache occupancy (MB/count), and hit ratio."""
        return {
            "policy": self.eviction_policy,
            "cached_partitions": len(self.cache),
            "total_cache_used_MB": self.total_cache_used / 1024**2,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "cache_hit_ratio": self.get_cache_hit_ratio()
        }

    def show_cache_state(self) -> None:
        """Print every cached partition and the running hit ratio (debugging aid)."""
        print("\n[CACHE STATE]")
        for pid, part in self.cache.items():
            print(str(part))
        print(f"Total Cache Used: {self.total_cache_used / 1024**2:.2f} MB | "
              f"Hit Ratio: {self.get_cache_hit_ratio():.2f}")

    @staticmethod
    def compute_cost_aware_reward(partition: Partition) -> float:
        """Cost-aware eviction penalty: how expensive it is to lose `partition`.

        This is the reward signal the README's MDP formulation describes
        ("penalties tied to recomputation cost of evicted or missed
        work, not ±1 hit/miss") — distinct from `_simulate_multi_query_reward`
        below, which is hit-ratio-based and is what
        `run_query_and_record_rl_data` currently logs into the replay
        buffer. Wiring this into that data-generation path (so training
        data actually reflects cost-aware rewards end to end) is a
        follow-up; this method exists and is tested independently so
        that work has a correct building block to start from.

        Args:
            partition: The partition being evicted or missed.

        Returns:
            A strictly negative penalty equal to
            `-partition.recomputation_cost_ms` — larger/heavier
            partitions (higher recomputation cost) produce a more
            negative (worse) reward.
        """
        return -partition.recomputation_cost_ms

    def _simulate_multi_query_reward(
        self,
        simulated_cache: Dict[int, Partition],
        future_queries: List[Dict[str, Any]],
        weights: List[float],
    ) -> float:
        """
        Reward shaping for offline RL data generation: rather than a
        single-step ±1 hit/miss signal, this estimates how a *candidate*
        post-eviction cache would perform over several *hypothetical*
        future queries, so an eviction is rewarded/penalized by its
        effect on near-future hit ratio rather than just the immediate
        query.

        NOTE: this reward is purely hit-ratio-based (no recomputation
        cost is used) — it does not match the "cost-aware" reward the
        README describes for the DDQN training pipeline
        (`scripts/train_stable_dqn.py` consumes a `replay_buffer_cost_aware.csv`
        that this function does not produce). See review summary.

        Method: computes a `baseline` hit ratio over `future_queries`
        using the *current* (pre-eviction) cache, then for each future
        query computes the hit ratio the `simulated_cache` (post-eviction
        candidate) would achieve, and accumulates
        `weight[i] * (hit_ratio_i - baseline)` — i.e. a weighted sum of
        hit-ratio improvement over doing nothing. Weights let later
        future queries count more (see `_sample_future_queries` callers,
        which pass increasing weights) as a crude recency/discounting
        proxy.

        Args:
            simulated_cache: Candidate cache contents after a hypothetical eviction.
            future_queries: Synthetic queries to evaluate the candidate against.
            weights: Per-future-query weight, same length as `future_queries`.

        Returns:
            Weighted sum of hit-ratio improvement over the baseline
            (can be negative if the candidate performs worse).
        """
        reward = 0.0

        # For reward baseline
        before_hits, before_miss = 0, 0
        for q in future_queries:
            h, m = self.get_instant_cache_hit_ratio(q)
            before_hits += h
            before_miss += m
        if before_hits + before_miss == 0:
            baseline = 0
        else:
            baseline = before_hits / (before_hits + before_miss)

        # Temporary executor-like structure
        temp_cache = {k: v for k, v in simulated_cache.items()}

        for i, q in enumerate(future_queries):
            h, m = 0, 0
            # simulate hits
            for pid, partition in self.all_partitions.items():
                if not partition.can_prune(q):
                    if pid in temp_cache:
                        h += 1
                    else:
                        m += 1

            if h + m > 0:
                hit_ratio = h / (h + m)
                reward += weights[i] * (hit_ratio - baseline)

        return reward

    def _sample_future_queries(self, N: int = 5) -> List[Dict[str, Any]]:
        """
        Generate N synthetic future queries for multi-step reward simulation.
        You can replace this with real workload traces later.

        Query "shape" is chosen uniformly at random from 5 fixed types
        (amount/category/date/state_segment/customer); the categories,
        states, and segments sampled from are hardcoded here (magic
        values) and duplicated from the `__main__` block below and from
        `evaluate_policies.py` — consider centralizing them in a shared
        workload-generation module or config.

        Args:
            N: Number of synthetic queries to generate.

        Returns:
            List of N query filter dicts.
        """
        categories = ["Electronics", "Clothing", "Books", "Toys"]
        states = ["CA", "NY", "TX", "FL", "WA"]
        segments = ["Regular", "Premium", "VIP"]

        queries = []
        for _ in range(N):
            q_type = random.choice(["amount", "category", "date", "state_segment", "customer"])
            if q_type == "amount":
                queries.append({"order_amount": (">", np.random.uniform(100, 900))})
            elif q_type == "category":
                queries.append({"product_category": ("=", random.choice(categories))})
            elif q_type == "date":
                random_day = pd.Timestamp("2025-01-01") + timedelta(days=random.randint(0, 364))
                queries.append({"order_date": (">=", random_day)})
            elif q_type == "state_segment":
                queries.append({
                    "state": ("=", random.choice(states)),
                    "segment": ("=", random.choice(segments))
                })
            else:
                queries.append({"customer_id": (">", random.randint(1, 4800))})

        return queries

    def run_query_and_record_rl_data(
        self,
        query_filters: Dict[str, Any],
        replay_buffer: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run one query and, for every resulting cache miss, log one offline
        RL transition per *candidate* eviction (plus one "do nothing"
        transition), for later use in `write_replay_buffer_simple`.

        For each miss:
          1. Capture the current cache state vector (before eviction).
          2. Shortlist candidate partitions to evict — either all cached
             partitions, or (if `top_k` is set) the `top_k` least-recently-used.
          3. For each candidate: simulate evicting it and inserting the
             new (missed) partition, score that hypothetical cache with
             `_simulate_multi_query_reward`, and log a transition with
             that reward.
          4. Also log a "no-op" transition (new_cache == current_state,
             reward 0.0) using the next available action index.
          5. Actually apply the real cache update via `load_partition`
             (using the *real* eviction policy, not the simulated candidates).

        NOTE: docstring header says "reward is not computed here" but
        `reward` is in fact computed per candidate via
        `_simulate_multi_query_reward` and stored in each transition dict
        — the header comment appears stale.

        Args:
            query_filters: Filters for this query.
            replay_buffer: List to append new transition dicts to (mutated
                and also returned).
            top_k: If set, restricts eviction candidates to the `top_k`
                least-recently-used cached partitions (candidate action
                space is then smaller than `initialize_cache`'s `k`).

        Returns:
            The same `replay_buffer` list, with new transitions appended.
            Each transition dict has keys: current_cache (np.ndarray),
            action (int), new_cache (np.ndarray), reward (float).
        """
        print(f"\n[RL QUERY RUN] Filters: {query_filters}")
        matched_partitions = []

        # 1️⃣ Identify partitions relevant to query
        for pid, partition in self.all_partitions.items():
            if not partition.can_prune(query_filters):
                matched_partitions.append(pid)

                # Cache hit — update metadata only
                if pid in self.cache:
                    self.total_hits += 1
                    partition.access_count += 1
                    partition.last_accessed = pd.Timestamp.now()
                    continue

                # Cache miss — simulate candidate evictions
                self.total_misses += 1
                new_partition = partition

                # 2️⃣ Capture current cache state
                current_state = self._get_cache_state_vector(query_filters)

                # 3️⃣ Determine candidate partitions for eviction
                candidates = list(self.cache.keys())
                if top_k and len(candidates) > top_k:
                    # Example: least recently used (LRU) as shortlist
                    candidates = sorted(candidates, key=lambda pid: self.cache[pid].last_accessed)[:top_k]

                # 4️⃣ For each candidate, simulate eviction and cache update
                i = 0
                future_queries = self._sample_future_queries(N=5)
                weights = [1.0, 1.1, 1.2, 1.25, 1.3]

                for evict_id in candidates:
                    simulated_cache = dict(self.cache)
                    simulated_cache.pop(evict_id)
                    simulated_cache[new_partition.partition_id] = new_partition

                    # compute multi-step reward
                    reward = self._simulate_multi_query_reward(
                        simulated_cache,
                        future_queries,
                        weights
                    )

                    new_state = self._get_cache_state_vector(query_filters, simulated_cache)[0]

                    replay_buffer.append({
                        "current_cache": current_state[0],
                        "action": i,
                        "new_cache": new_state,
                        "reward": reward
                    })

                    i += 1
                    print(f"[RL-MULTI] Evict {evict_id} → {reward:.4f} multi-query reward")

                    replay_buffer.append({
                        "current_cache": current_state[0],
                        "action": i,
                        "new_cache": current_state[0],
                        "reward": 0.0
                    })

                # 5️⃣ Apply actual cache update (based on your policy)
                self.load_partition(new_partition)

        return replay_buffer

    @staticmethod
    def _classify_query_intent(query_filters: Dict[str, Any]) -> Tuple[float, float]:
        """Classify a query's filter values as temporal / categorical / numerical intent.

        This is the "workload / query-context" signal described in the
        README's MDP formulation. Rather than inspecting partition
        column dtypes, it inspects the filter *values* directly (a
        date/datetime value -> temporal; a string value -> categorical;
        anything else, e.g. int/float -> numerical), which is cheap and
        needs no partition lookups. A query can combine multiple filters
        (e.g. `{"state": ("=", "CA"), "segment": ("=", "Premium")}`), so
        both flags can be 1.0 at once; numerical-only intent is implicitly
        represented as (0.0, 0.0).

        Args:
            query_filters: Filter dict as accepted by `Partition.can_prune`
                (values are `(operator, value)` tuples or lists).

        Returns:
            `(is_temporal, is_categorical)` as 0.0/1.0 floats.
        """
        is_temporal = 0.0
        is_categorical = 0.0
        for condition in query_filters.values():
            if isinstance(condition, list):
                values = condition
            elif isinstance(condition, tuple) and len(condition) == 2:
                values = [condition[1]]
            else:
                values = []
            for v in values:
                if isinstance(v, (pd.Timestamp, datetime)):
                    is_temporal = 1.0
                elif isinstance(v, str):
                    is_categorical = 1.0
        return is_temporal, is_categorical

    # Canonical 38-feature state vector (STATE_DIM in dqn_model.py):
    #   Per cached partition (6 features x up to 5 slots = 30 features),
    #   in `cache.items()` iteration order:
    #     [0] access_count               - int, times this partition has been read
    #     [1] seconds_since_last_access  - float, recency signal
    #     [2] size_in_memory_mb          - float, partition size in MB
    #     [3] row_count                  - int, rows in this partition
    #     [4] column_count               - int, columns in this partition
    #     [5] is_cached                  - bool (always True for a cached partition)
    #   Global features (8), appended after all partition blocks:
    #     [30] num_cached_partitions     - int, len(cache)
    #     [31] instant_hits              - int, partitions in `cache` matching `queryFilters`
    #     [32] instant_misses            - int, matching partitions NOT in `cache`
    #     [33] instant_hit_ratio         - float in [0, 1], instant_hits / (instant_hits + instant_misses)
    #     [34] cache_utilization         - float in [0, 1], total_cache_used / max_cache_size
    #     [35] rolling_eviction_rate     - float, total_evictions / total queries served so far
    #     [36] query_is_temporal         - 0.0/1.0, from _classify_query_intent
    #     [37] query_is_categorical      - 0.0/1.0, from _classify_query_intent
    def _get_cache_state_vector(self, queryFilters: Dict[str, Any], cache_state: Optional[Dict[int, Partition]] = None):
        """Flatten cache + query context into the fixed-length feature vector fed to the DQN.

        See the feature-list comment directly above this method for the
        full, named 38-feature layout (STATE_DIM in dqn_model.py).

        Partition order (and hence feature order) follows `cache.items()`
        iteration order, which is insertion/access order for the
        OrderedDict-backed real cache — i.e. the state vector's layout
        shifts depending on cache history, not a fixed partition-id slot
        assignment. `pid_list` (only returned for the "RL" policy) records
        which partition id occupies each per-partition feature block, so
        `_rl_decide_eviction` can map the DQN's chosen action index back
        to a concrete partition id.

        Args:
            queryFilters: The triggering query's filters — used for the
                global hit/miss/ratio features and the query-intent flags.
            cache_state: Optional cache dict to compute the vector over,
                for scoring hypothetical/simulated caches instead of the
                live `self.cache`.

        Returns:
            If `self.eviction_policy == "RL"`: a tuple
            `(feature_vector: np.ndarray, pid_list: list[int])`.
            Otherwise: just the `feature_vector` (breaking the otherwise
            consistent tuple return — callers must branch on policy).
        """
        cache = cache_state if cache_state is not None else self.cache
        features = []
        pid_list = []
        for pid, part in cache.items():
            pid_list.append(pid)
            features.extend([
                part.access_count,
                (pd.Timestamp.now() - part.last_accessed).total_seconds(),
                part.size_in_memory / 1024 ** 2,
                part.row_count,
                len(part.data.columns),
                part.is_cached
            ])
        # Global cache stats
        hits, miss = self.get_instant_cache_hit_ratio(queryFilters)
        ratio = hits / (hits + miss) if (hits + miss) > 0 else 0.0
        cache_utilization = self.total_cache_used / self.max_cache_size if self.max_cache_size > 0 else 0.0
        queries_served = self.total_hits + self.total_misses
        rolling_eviction_rate = self.total_evictions / queries_served if queries_served > 0 else 0.0
        is_temporal, is_categorical = self._classify_query_intent(queryFilters)
        features.extend([
            len(cache),  # number of cached partitions
            hits,
            miss,
            ratio,
            cache_utilization,
            rolling_eviction_rate,
            is_temporal,
            is_categorical,
        ])
        if self.eviction_policy == "RL":
            return (np.array(features, dtype=float), pid_list)
        return np.array(features, dtype=float)

    def warmup_cache(self, queries: List[Dict[str, Any]], policy: str = "LRU") -> None:
        """Run `queries` under a temporarily-overridden policy to warm the cache before evaluation.

        Args:
            queries: Sequence of query filter dicts to execute in order.
            policy: Eviction policy to use only for this warm-up (restored
                to `self.eviction_policy` afterward).
        """
        print(f"[WARMUP] Starting warm-up with {len(queries)} queries using {policy}")
        old_policy = self.eviction_policy
        self.eviction_policy = policy
        for q in queries:
            self.execute_query(q)
        self.eviction_policy = old_policy
        print(f"[WARMUP] Completed. Cache hit ratio baseline: {self.get_cache_hit_ratio():.3f}")


    def write_replay_buffer_simple(
        self,
        replay_buffer: List[Dict[str, Any]],
        csv_file: str,
        policy_label: Optional[str] = None,
    ) -> None:
        """
        Write replay buffer to CSV with only state, action, next_state, reward.
        Automatically appends if file already exists.

        Each item in replay_buffer should be a dict:
            {
                "current_cache": np.array,
                "action": int,
                "new_cache": np.array,
                "reward": float
            }
        Optionally, add 'policy_label' to tag which eviction policy generated the data.
        """
        if not replay_buffer:
            print("[WARN] Replay buffer is empty. Nothing to write.")
            return

        # Column headers
        state_len = len(replay_buffer[0]["current_cache"])
        state_cols = [f"s{i}" for i in range(state_len)]
        next_state_cols = [f"next_s{i}" for i in range(state_len)]
        all_cols = state_cols + ["action"] + next_state_cols + ["reward"]
        if policy_label:
            all_cols.append("policy")

        file_exists = os.path.exists(csv_file)

        # Write or append to file
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header only if file didn't exist
            if not file_exists:
                writer.writerow(all_cols)

            for item in replay_buffer:
                row = [
                    *item["current_cache"].tolist(),
                    item["action"],
                    *item["new_cache"].tolist(),
                    item["reward"]
                ]
                if policy_label:
                    row.append(policy_label)
                writer.writerow(row)

        mode = "Appended" if file_exists else "Created"
        print(f"[SAVE] {mode} {len(replay_buffer)} rows to {csv_file} ({'+' if file_exists else ''}header added).")


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # -------------------------------
    # Create Synthetic Data
    # -------------------------------
    n_rows = 100_000
    df_sales = pd.DataFrame({
        "order_id": np.arange(n_rows),
        "customer_id": np.random.randint(1, 5000, n_rows),
        "product_category": np.random.choice(["Electronics", "Clothing", "Books", "Toys"], n_rows,
                                             p=[0.4, 0.3, 0.2, 0.1]),
        "order_amount": np.random.uniform(5, 1000, n_rows),
        "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit="d")
    })

    df_customer = pd.DataFrame({
        "customer_id": np.arange(5000),
        "state": np.random.choice(["CA", "NY", "TX", "FL", "WA"], 5000),
        "segment": np.random.choice(["Regular", "Premium", "VIP"], 5000, p=[0.6, 0.3, 0.1])
    })

    # -------------------------------
    # Parameters
    # -------------------------------
    POLICIES = ["LRU", "LFU", "FIFO", "RANDOM"]  # RL excluded: this script *generates* its training data
    N_WORKLOADS_PER_POLICY = 1500  # 1500 * 4 ≈ 6000 queries total
    replay_buffer = []

    # Query pools
    categories = ["Electronics", "Clothing", "Books", "Toys"]
    states = ["CA", "NY", "TX", "FL", "WA"]
    segments = ["Regular", "Premium", "VIP"]

    # -------------------------------
    # Generate Data
    # -------------------------------
    for policy in POLICIES:
        print(f"\n🔁 Generating data for policy: {policy}")

        # Reinitialize executor for each policy
        executor = Executor(max_cache_size_mb=20, eviction_policy=policy)
        executor.create_partitions(df_sales, n_partitions=20)
        executor.create_partitions(df_customer, n_partitions=10)
        executor.initialize_cache(k=5)

        # Simulate random workloads
        for i in tqdm(range(N_WORKLOADS_PER_POLICY), desc=f"Policy {policy}"):
            q_type = random.choice(["amount", "category", "date", "state_segment", "customer"])
            if q_type == "amount":
                query = {"order_amount": (">", np.random.uniform(100, 900))}
            elif q_type == "category":
                query = {"product_category": ("=", random.choice(categories))}
            elif q_type == "date":
                random_day = pd.Timestamp("2025-01-01") + timedelta(days=random.randint(0, 364))
                query = {"order_date": (">=", random_day)}
            elif q_type == "state_segment":
                query = {"state": ("=", random.choice(states)), "segment": ("=", random.choice(segments))}
            else:
                query = {"customer_id": (">", random.randint(1, 4800))}

            replay_buffer = executor.run_query_and_record_rl_data(query, replay_buffer)

    # -------------------------------
    # Save Combined Experience Buffer
    # -------------------------------
        print(f"\n✅ Total generated samples: {len(replay_buffer)}")
        executor.write_replay_buffer_simple(replay_buffer, "replay_buffer_multi_policy.csv")
