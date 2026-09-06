"""
Evaluate Cache Eviction Policies (LRU, LFU, FIFO, RANDOM, RL)
with Time-Series Tracking
-------------------------------------------------------------
Compares each policy on:
  - Cache hit ratio over time (per query)
  - Query latency over time
  - Total cache hit ratio
  - Eviction count
  - RL overhead

Inputs: none from disk (workload is synthetic, generated in-process);
    the RL policy additionally loads a trained checkpoint (default
    "dqn_policy_net.pth") via `Executor.load_rl_model`.
Outputs: policy_evaluation_summary.csv (one row per policy) and
    policy_timeline_data.csv (one row per policy per query), both written
    to the current working directory by `run_evaluation()`.
"""

import time
from typing import Any, Optional

import pandas as pd
import numpy as np
import torch
from executor import Executor  # import your executor class with RL integrated


# -----------------------------
# Helper: Evaluate One Policy
# -----------------------------
def evaluate_policy(
    policy_name: str,
    data: pd.DataFrame,
    queries: list[dict[str, Any]],
    model_path: Optional[str] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one eviction policy over a fixed query sequence and collect metrics.

    Args:
        policy_name: One of "LRU", "LFU", "FIFO", "RANDOM", "RL" (matched
            against `Executor.eviction_policy`).
        data: Source DataFrame to partition and cache.
        queries: Ordered list of filter dicts, each applied as one query.
        model_path: Path to a trained DQN checkpoint; only used when
            `policy_name == "RL"`.

    Returns:
        A tuple of (overall_metrics, query_timeline):
          - overall_metrics: dict with final hit ratio (%), average query
            latency (ms), and total eviction count for this policy.
          - query_timeline: list of per-query dicts (hit ratio %, latency
            ms) suitable for concatenation into a timeline DataFrame.
    """
    print(f"\n=== Evaluating Policy: {policy_name} ===")

    executor = Executor(max_cache_size_mb=10, eviction_policy=policy_name)
    executor.create_partitions(data, n_partitions=10)
    executor.initialize_cache(k=5)

    if policy_name == "RL":
        executor.load_rl_model(model_path)

    total_evictions = 0
    total_latency = 0.0
    query_timeline = []

    # ---- Run each query in sequence ----
    for i, q in enumerate(queries):
        start_time = time.time()
        executor.execute_query(q)
        latency = (time.time() - start_time) * 1000  # wall-clock time in milliseconds
        total_latency += latency

        # Track timeline metrics
        cache_summary = executor.get_cache_summary()
        total_evictions += len([e for e in executor.query_log if "evict" in str(e).lower()])

        query_timeline.append({
            "Query_Index": i + 1,
            "Policy": policy_name,
            "Query": str(q),
            "Cache_Hit_Ratio": round(cache_summary["cache_hit_ratio"] * 100, 2),
            "Query_Latency_ms": round(latency, 2)
        })

        print(f"Query {i+1}/{len(queries)} ({policy_name}) -> "
              f"HitRatio={cache_summary['cache_hit_ratio']:.2f}, Latency={latency:.2f}ms")

    # ---- Compute overall summary ----
    avg_latency = total_latency / len(queries)
    summary = executor.get_cache_summary()

    overall_metrics = {
        "Policy": policy_name,
        "Final_Cache_Hit_Ratio(%)": round(summary["cache_hit_ratio"] * 100, 2),
        "Avg_Query_Latency(ms)": round(avg_latency, 2),
        "Total_Evictions": total_evictions,
    }

    print(f"\n[SUMMARY] {overall_metrics}")
    return overall_metrics, query_timeline


# -----------------------------
# Generate Synthetic Data
# -----------------------------
def generate_sample_data(n_rows: int = 100_000) -> pd.DataFrame:
    """Generate a synthetic e-commerce orders table for the simulator.

    Args:
        n_rows: Number of synthetic order rows to generate.

    Returns:
        DataFrame with columns order_id, customer_id, product_category,
        order_amount, order_date.
    """
    np.random.seed(42)  # fixed seed so evaluation runs are reproducible across policies
    df = pd.DataFrame({
        "order_id": np.arange(n_rows),
        "customer_id": np.random.randint(1, 5000, n_rows),
        "product_category": np.random.choice(["Electronics", "Clothing", "Books", "Toys"], n_rows,
                                             p=[0.4, 0.3, 0.2, 0.1]),
        "order_amount": np.random.uniform(5, 1000, n_rows),
        "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit="d")
    })
    return df


# -----------------------------
# Main Evaluation Routine
# -----------------------------
def run_evaluation() -> None:
    """Run every configured eviction policy over the same fixed query set,
    print a comparison table, and write the summary/timeline CSVs consumed
    by `plot_results.py`.
    """
    data = generate_sample_data(50_000)
    queries = [
        {"order_amount": (">", 900)},
        {"product_category": ("=", "Toys")},
        {"customer_id": (">", 4500)},
        {"order_date": (">=", pd.Timestamp("2025-12-01"))},
        {"product_category": ("=", "Electronics")},
        {"product_category": ("=", "Books")},
        {"order_amount": ("<", 50)},
        {"customer_id": ("<", 2000)},
        {"product_category": ("=", "Clothing")},
        {"order_amount": (">", 500)},
    ]

    all_metrics = []
    all_timeline_data = []

    # NOTE: "RANDOM" is omitted here even though plot_results.py and the
    # README both list it as one of the compared policies — see review
    # summary for details (not fixed here per review scope).
    policies = ["LRU", "LFU", "FIFO",  "RL"]

    for policy in policies:
        model_path = "dqn_policy_net.pth" if policy == "RL" else None
        metrics, timeline = evaluate_policy(policy, data, queries, model_path)
        all_metrics.append(metrics)
        all_timeline_data.extend(timeline)

    # Convert to DataFrame
    summary_df = pd.DataFrame(all_metrics)
    timeline_df = pd.DataFrame(all_timeline_data)

    summary_df.to_csv("policy_evaluation_summary.csv", index=False)
    timeline_df.to_csv("policy_timeline_data.csv", index=False)

    print("\n=== Final Policy Comparison ===")
    print(summary_df)

    print("\n=== Timeline Data (Head) ===")
    print(timeline_df.head())

    print("\nResults saved to:")
    print(" - policy_evaluation_summary.csv (overall summary)")
    print(" - policy_timeline_data.csv (per-query timeline) ✅")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    run_evaluation()
