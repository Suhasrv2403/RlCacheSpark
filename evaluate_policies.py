"""
Evaluate Cache Eviction Policies (LRU, LFU, FIFO, RANDOM, RL)
with Time-Series Tracking, Across Multiple Workload Scenarios
---------------------------------------------------------------------
Compares each policy across three workload scenarios (matching the
README's experiment design):
  - Trained_Workload: a stationary mix of query types/columns.
  - Moving_Workload: a phase shift partway through, from temporal-heavy
    to categorical-heavy queries, to test adaptability (see
    `plot_results.py::plot_hit_ratio_timeline_separated`, which draws a
    "Phase Shift" marker specifically for a workload named
    "Moving_Workload").
  - Unseen_Workload: compound/edge-range filters not exercised by the
    other two scenarios, to test generalization.

For each (workload, policy) pair, tracks:
  - Cache hit ratio over time (per query) and final hit ratio
  - Query latency over time, plus P50/P99 latency and total wall-clock runtime
  - Eviction count

Inputs: none from disk (workloads are synthetic, generated in-process);
    the RL policy additionally loads a trained checkpoint (default
    "dqn_policy_net.pth") via `Executor.load_rl_model`.
Outputs: policy_evaluation_summary.csv (one row per workload x policy)
    and policy_timeline_data.csv (one row per workload x policy x query),
    both written to the current working directory by `run_evaluation()`.
    These columns are what `plot_results.py` expects to read — the two
    scripts were previously out of sync (see git history / review
    summary); keep them consistent if you change either.
"""

import time
from typing import Any, Optional

import pandas as pd
import numpy as np
from executor import Executor  # import your executor class with RL integrated


# -----------------------------
# Helper: Evaluate One Policy
# -----------------------------
def evaluate_policy(
    policy_name: str,
    data: pd.DataFrame,
    queries: list[dict[str, Any]],
    model_path: Optional[str] = None,
    workload_name: str = "Trained_Workload",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one eviction policy over a fixed query sequence and collect metrics.

    Args:
        policy_name: One of "LRU", "LFU", "FIFO", "RANDOM", "RL" (matched
            against `Executor.eviction_policy`).
        data: Source DataFrame to partition and cache.
        queries: Ordered list of filter dicts, each applied as one query.
        model_path: Path to a trained DQN checkpoint; only used when
            `policy_name == "RL"`.
        workload_name: Label identifying which workload scenario this run
            belongs to (e.g. "Trained_Workload", "Moving_Workload",
            "Unseen_Workload"); stamped onto every output row so
            `plot_results.py` can facet/group by it.

    Returns:
        A tuple of (overall_metrics, query_timeline):
          - overall_metrics: dict with workload/policy labels, final hit
            ratio (%), P50/P99/avg query latency (ms), total wall-clock
            runtime (s), and total eviction count for this run.
          - query_timeline: list of per-query dicts (workload, hit ratio
            %, latency ms) suitable for concatenation into a timeline
            DataFrame.
    """
    print(f"\n=== Evaluating Policy: {policy_name} ({workload_name}) ===")

    executor = Executor(max_cache_size_mb=10, eviction_policy=policy_name)
    executor.create_partitions(data, n_partitions=10)
    executor.initialize_cache(k=5)

    if policy_name == "RL":
        executor.load_rl_model(model_path)

    latencies_ms: list[float] = []
    query_timeline = []

    # ---- Run each query in sequence ----
    for i, q in enumerate(queries):
        start_time = time.time()
        executor.execute_query(q)
        latency = (time.time() - start_time) * 1000  # wall-clock time in milliseconds
        latencies_ms.append(latency)

        # Track timeline metrics
        cache_summary = executor.get_cache_summary()

        query_timeline.append({
            "Workload": workload_name,
            "Query_Index": i + 1,
            "Policy": policy_name,
            "Query": str(q),
            "Cache_Hit_Ratio": round(cache_summary["cache_hit_ratio"] * 100, 2),
            "Query_Latency_ms": round(latency, 2)
        })

        print(f"Query {i+1}/{len(queries)} ({policy_name}) -> "
              f"HitRatio={cache_summary['cache_hit_ratio']:.2f}, Latency={latency:.2f}ms")

    # ---- Compute overall summary ----
    summary = executor.get_cache_summary()
    latencies_arr = np.array(latencies_ms)

    overall_metrics = {
        "Workload_Set": workload_name,
        "Policy": policy_name,
        "Final_Cache_Hit_Ratio(%)": round(summary["cache_hit_ratio"] * 100, 2),
        "Avg_Query_Latency(ms)": round(float(latencies_arr.mean()), 2),
        "P50_Latency(ms)": round(float(np.percentile(latencies_arr, 50)), 2),
        "P99_Latency(ms)": round(float(np.percentile(latencies_arr, 99)), 2),
        "Total_Runtime(s)": round(float(latencies_arr.sum()) / 1000.0, 4),
        # executor.total_evictions is incremented directly inside
        # Executor.evict_partition, unlike the previous implementation
        # here which scanned query_log for the substring "evict" — those
        # log entries never actually contained that word, so evictions
        # were silently always counted as 0.
        "Total_Evictions": executor.total_evictions,
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
# Workload Scenarios
# -----------------------------
def trained_workload_queries() -> list[dict[str, Any]]:
    """Stationary mix of query types/columns — the "seen during training" baseline scenario."""
    return [
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


def moving_workload_queries() -> list[dict[str, Any]]:
    """Phase-shift scenario: temporal-heavy queries, then an abrupt shift to categorical-heavy queries.

    Named "Moving_Workload" (not just descriptive — `plot_results.py`
    matches on this exact name to draw its "Phase Shift" marker at query
    index 10.5). Tests whether a policy adapts once the query mix
    changes mid-run, per the README's phase-shift experiment.
    """
    temporal_phase = [
        {"order_date": (">=", pd.Timestamp("2025-01-01") + pd.Timedelta(days=30 * i))}
        for i in range(10)
    ]
    categorical_phase = [
        {"product_category": ("=", cat)}
        for cat in ["Electronics", "Clothing", "Books", "Toys"] * 3
    ][:10]
    return temporal_phase + categorical_phase


def unseen_workload_queries() -> list[dict[str, Any]]:
    """Compound / edge-range filters not exercised by the other scenarios — a generalization test.

    Combines two conditions per query (a shape not used in
    `trained_workload_queries`) and targets narrower or more extreme
    ranges, so a policy that merely memorized the trained query mix
    doesn't get credit here.
    """
    return [
        {"product_category": ("=", "Books"), "order_amount": (">", 800)},
        {"product_category": ("=", "Toys"), "customer_id": ("<", 500)},
        {"order_amount": (">", 970)},
        {"customer_id": (">", 4950)},
        {"product_category": ("=", "Electronics"), "order_amount": ("<", 20)},
        {"order_date": (">=", pd.Timestamp("2025-12-20"))},
        {"product_category": ("=", "Clothing"), "customer_id": (">", 4800)},
        {"order_amount": (">", 990)},
        {"product_category": ("=", "Books"), "customer_id": ("<", 100)},
        {"customer_id": ("<", 50)},
    ]


# -----------------------------
# Main Evaluation Routine
# -----------------------------
def run_evaluation() -> None:
    """Run every configured eviction policy over three workload scenarios,
    print a comparison table, and write the summary/timeline CSVs consumed
    by `plot_results.py`.
    """
    data = generate_sample_data(50_000)

    workloads = {
        "Trained_Workload": trained_workload_queries(),
        "Moving_Workload": moving_workload_queries(),
        "Unseen_Workload": unseen_workload_queries(),
    }

    policies = ["LRU", "LFU", "FIFO", "RANDOM", "RL"]

    all_metrics = []
    all_timeline_data = []

    for workload_name, queries in workloads.items():
        for policy in policies:
            model_path = "dqn_policy_net.pth" if policy == "RL" else None
            metrics, timeline = evaluate_policy(policy, data, queries, model_path, workload_name)
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
