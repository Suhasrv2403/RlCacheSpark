"""
Tests for core Executor cache-simulator mechanics: eviction, cache-full
triggering, cache-hit no-ops, and the cost-aware reward function.
"""

import pandas as pd
import pytest

from executor import Executor
from DataSetProcessing import Partition


@pytest.fixture
def small_dataframe() -> pd.DataFrame:
    # 10 rows, split into 5 partitions of 2 rows each by create_partitions:
    # partition 0 -> value in [0,1], partition 1 -> [2,3], ..., partition 4 -> [8,9]
    return pd.DataFrame({"value": range(10)})


@pytest.fixture
def executor_with_full_cache(small_dataframe: pd.DataFrame) -> Executor:
    ex = Executor(max_cache_size_mb=10, eviction_policy="LRU")
    ex.create_partitions(small_dataframe, n_partitions=5)
    ex.initialize_cache(k=3)  # caches partitions 0, 1, 2
    return ex


def test_eviction_reduces_cache_size_by_exactly_one(executor_with_full_cache: Executor):
    ex = executor_with_full_cache
    assert len(ex.cache) == 3

    ex.evict_partition()

    assert len(ex.cache) == 2


def test_full_cache_triggers_eviction_on_load(executor_with_full_cache: Executor):
    ex = executor_with_full_cache
    assert len(ex.cache) == 3
    assert ex.total_evictions == 0

    new_partition = ex.all_partitions[3]  # not yet cached
    ex.load_partition(new_partition)

    assert len(ex.cache) == 3  # one evicted, one loaded -> size unchanged
    assert ex.total_evictions == 1
    assert 3 in ex.cache


def test_cache_hit_does_not_trigger_eviction(executor_with_full_cache: Executor):
    ex = executor_with_full_cache
    cached_ids_before = set(ex.cache.keys())

    # Matches only partitions 0/1/2 (values 0-5), all already cached —
    # every match should be a hit, so no eviction should fire.
    ex.execute_query({"value": ("<", 6)})

    assert ex.total_evictions == 0
    assert ex.total_misses == 0
    assert ex.total_hits == 3  # one hit per matching cached partition
    assert set(ex.cache.keys()) == cached_ids_before


def test_cache_miss_on_uncached_partition_triggers_eviction(executor_with_full_cache: Executor):
    ex = executor_with_full_cache

    # Matches partition 3 (values 6-7), which is not cached -> a miss,
    # which (with a full cache) must trigger exactly one eviction.
    ex.execute_query({"value": (">=", 6), })  # matches partitions 3 and 4 (values 6-9)

    assert ex.total_misses > 0
    assert ex.total_evictions >= 1
    assert len(ex.cache) == 3  # cache size stays at capacity


def test_cost_aware_reward_is_negative_and_scales_with_recomputation_cost():
    cheap = Partition(pd.DataFrame({"a": range(5)}), partition_id=0)
    expensive = Partition(pd.DataFrame({"a": range(5000)}), partition_id=1)

    reward_cheap = Executor.compute_cost_aware_reward(cheap)
    reward_expensive = Executor.compute_cost_aware_reward(expensive)

    assert reward_cheap < 0
    assert reward_expensive < 0
    # Losing the more expensive-to-recompute partition must be penalized more.
    assert reward_expensive < reward_cheap
    assert reward_cheap == -cheap.recomputation_cost_ms
    assert reward_expensive == -expensive.recomputation_cost_ms
