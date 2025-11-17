import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from DataSetProcessing import Partition  # use your Partition class here
import csv
import random
from datetime import timedelta
from tqdm import tqdm
import torch
import numpy as np
from typing import Optional
import os

class Executor:
    def __init__(self, max_cache_size_mb: int = 512, eviction_policy: str = "LRU"):
        self.max_cache_size = max_cache_size_mb * 1024 * 1024
        self.cache: Dict[int, Partition] = OrderedDict()
        self.all_partitions: Dict[int, Partition] = {}
        self.eviction_policy = eviction_policy.upper()
        self.total_cache_used = 0
        self.cache_length = 0
        # Stats
        self.total_hits = 0
        self.total_misses = 0
        self.query_log: List[Dict[str, Any]] = []
        self.flag = False

        if self.eviction_policy == "RL":
            self.load_rl_model("dqn_policy_net.pth")
    # -----------------------------
    # Partitioning and Initialization
    # -----------------------------

    def create_partitions(self, data: pd.DataFrame, n_partitions: int = 4):
        """Split the dataset into partitions"""
        split_data = np.array_split(data, n_partitions)
        for i, df_part in enumerate(split_data):
            self.all_partitions[i] = Partition(df_part, partition_id=i)
        print(f"[INIT] Created {len(self.all_partitions)} partitions.")

    def initialize_cache(self, k: int):
        """Pre-load first k partitions into cache"""
        self.cache_length = k
        for i in range(min(k, len(self.all_partitions))):
            self.load_partition(self.all_partitions[i])
        print(f"[CACHE INIT] Pre-loaded {k} partitions into cache.")

    def load_rl_model(self, model_path: str = "dqn_policy_net.pth"):
        """
        Load the trained DQN model from file for RL eviction policy.
        """
        if not os.path.exists(model_path):
            print(f"[RL] Model file not found at {model_path}. RL will use random decisions.")
            self.rl_model = None
            return

        # Infer input/output dimensions (modify if you changed them)
        state_dim = 34  # length of cache state vector
        num_actions = 6  # number of possible eviction actions (4 partitions + 1 'no eviction')

        class DQN(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, output_dim)
                )

            def forward(self, x):
                return self.net(x)

        self.rl_model = DQN(state_dim, num_actions)
        self.rl_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.rl_model.eval()
        print(f"[RL] Loaded DQN model from {model_path} ✅")

    def _rl_decide_eviction(self,query_filters) -> Optional[int]:
        """
        Decide which partition to evict using trained DQN model.
        If the action index >= len(cache), interpret as 'no eviction'.
        """
        if not hasattr(self, "rl_model") or self.rl_model is None:
            print("[RL] No model loaded — using random eviction.")
            return np.random.choice(list(self.cache.keys()))

        # Build current cache state vector (same as training)
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

    def load_partition(self, partition: Partition,query_filters = None):
        """Load a new partition into cache (with eviction if needed)."""
        pid = partition.partition_id
        size = partition.size_in_memory

        # Evict partitions if full
        while len(self.cache) >= self.cache_length:
            evicted_id = self.evict_partition(query_filters)
            if evicted_id is None:
                break

        if self.eviction_policy == "RL" and self.flag == True:

            self.flag = False


        else:
            self.cache[pid] = partition
            partition.is_cached = True
            self.total_cache_used += size
        #print(f"[LOAD] Cached Partition {pid} ({size / 1024**2:.2f} MB)")

    import numpy as np
    from typing import Optional

    def evict_partition(self,query_filters = None) -> Optional[int]:
        """Evict one partition based on policy."""
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

        # Optional logging
        print(f"[EVICT] Partition {evict_id} evicted using {self.eviction_policy}. Next in line: {next_evict_id}")

        return next_evict_id



    # -----------------------------
    # Query Execution
    # -----------------------------

    def execute_query(self, query_filters: Dict[str, Any]) -> List[Partition]:
        """
        Execute query:
          - Check which partitions are relevant (non-pruned)
          - Cache hit if already loaded
          - Cache miss if not cached
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

    def get_instant_cache_hit_ratio(self,query_filters: Dict[str, Any]) -> float:
        total_hits = 0
        total_miss = 0
        for pid, partition in self.all_partitions.items():
            if not partition.can_prune(query_filters):

                # Cache hit — update metadata only
                if pid in self.cache:
                    total_hits += 1

                else:
                    total_miss += 1
        if total_hits + total_miss == 0:
            return 0
        return (total_hits, total_miss)

    def get_cache_hit_ratio(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def get_cache_summary(self) -> Dict[str, Any]:
        return {
            "policy": self.eviction_policy,
            "cached_partitions": len(self.cache),
            "total_cache_used_MB": self.total_cache_used / 1024**2,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "cache_hit_ratio": self.get_cache_hit_ratio()
        }

    def show_cache_state(self):
        print("\n[CACHE STATE]")
        for pid, part in self.cache.items():
            print(str(part))
        print(f"Total Cache Used: {self.total_cache_used / 1024**2:.2f} MB | "
              f"Hit Ratio: {self.get_cache_hit_ratio():.2f}")

    def _simulate_multi_query_reward(self, simulated_cache, future_queries, weights):
        """
        Simulate next K queries and compute weighted reward.
        Uses hit_ratio improvement.
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

    def _sample_future_queries(self, N=5):
        """
        Generate N synthetic future queries for multi-step reward simulation.
        You can replace this with real workload traces later.
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

    def run_query_and_record_rl_data(self, query_filters: Dict[str, Any], replay_buffer: List[Dict[str, Any]],
                                     top_k: int = None):
        """
        Run query and record transitions for RL:
        (query, current_cache_state, action, new_cache_state)

        Each cache miss triggers simulation of possible evictions.
        Reward is not computed here — only state transitions are logged.
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

    def _get_cache_state_vector(self, queryFilters ,cache_state=None):
        """Flatten cache-level features into a vector for DQN input."""
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
        # Add global cache stats
        hits,miss = self.get_instant_cache_hit_ratio(queryFilters)
        if hits + miss == 0:
            ratio = 0
        else:
            ratio = hits / (hits + miss)
        features.extend([
            len(cache),  # number of cached partitions
            hits,
            miss,
            ratio
        ])
        if self.eviction_policy == "RL":
            return (np.array(features, dtype=float), pid_list)
        return np.array(features, dtype=float)

    def warmup_cache(self, queries: List[Dict[str, Any]], policy="LRU"):
        print(f"[WARMUP] Starting warm-up with {len(queries)} queries using {policy}")
        old_policy = self.eviction_policy
        self.eviction_policy = policy
        for q in queries:
            self.execute_query(q)
        self.eviction_policy = old_policy
        print(f"[WARMUP] Completed. Cache hit ratio baseline: {self.get_cache_hit_ratio():.3f}")


    def write_replay_buffer_simple(self, replay_buffer, csv_file, policy_label=None):
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
    POLICIES = ["LRU", "LFU", "FIFO", "RANDOM"]
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
