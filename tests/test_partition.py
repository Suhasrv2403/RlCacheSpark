"""
Tests for DataSetProcessing.Partition.

Converted from a commented-out manual smoke test that used to live at
the bottom of DataSetProcessing.py (never executed automatically).
"""

import numpy as np
import pandas as pd
import pytest

from DataSetProcessing import Partition


@pytest.fixture
def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id": np.arange(1, 21),
        "age": rng.integers(18, 65, 20),
        "salary": rng.uniform(30000, 120000, 20).round(2),
        "department": rng.choice(["HR", "Engineering", "Finance", "Marketing"], 20),
        "city": rng.choice(["New York", "San Francisco", "Chicago", "Austin"], 20),
        "active": rng.choice([True, False], 20),
        "notes": rng.choice([
            "Excellent performer", "Needs improvement", "Promoted recently",
            "On probation", "Team player", "Works remotely"
        ], 20),
    })


@pytest.fixture
def partition(sample_data: pd.DataFrame) -> Partition:
    return Partition(sample_data, partition_id=1)


def test_partition_basic_info(partition: Partition, sample_data: pd.DataFrame):
    assert partition.partition_id == 1
    assert partition.row_count == len(sample_data)
    assert len(partition.data.columns) == len(sample_data.columns)
    assert partition.size_in_memory > 0


def test_partition_column_stats_numeric(partition: Partition):
    stats = partition.get_column_stats("salary")
    assert stats  # non-empty for a known column
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["unique_count"] <= partition.row_count


def test_partition_column_stats_unknown_column_returns_empty(partition: Partition):
    assert partition.get_column_stats("does_not_exist") == {}


def test_partition_can_prune_out_of_range_numeric_filter(partition: Partition):
    # age's max is at most 64 (rng.integers(18, 65, ...) is exclusive of 65),
    # so a filter requiring age > 64 can never match this partition.
    assert partition.can_prune({"age": (">", 64)}) is True


def test_partition_cannot_prune_in_range_numeric_filter(partition: Partition):
    # age's range spans [18, 64], so a filter of age > 17 must match every row.
    assert partition.can_prune({"age": (">", 17)}) is False


def test_partition_filter_rows_callable(partition: Partition):
    filtered = partition.filter_rows(lambda df: df["salary"] > 80000)
    assert (filtered["salary"] > 80000).all()


def test_partition_filter_rows_dict(partition: Partition):
    filtered = partition.filter_rows({"age": (">", 30)})
    assert (filtered["age"] > 30).all()


def test_partition_project_columns(partition: Partition):
    projected = partition.project_columns(["id", "age", "salary"])
    assert list(projected.data.columns) == ["id", "age", "salary"]
    assert projected.row_count == partition.row_count
    assert projected.partition_id == partition.partition_id


def test_partition_recomputation_cost_scales_with_size():
    small = Partition(pd.DataFrame({"a": range(10)}), partition_id=0)
    big = Partition(pd.DataFrame({"a": range(10_000)}), partition_id=1)
    assert small.recomputation_cost_ms > 0
    assert big.recomputation_cost_ms > small.recomputation_cost_ms
