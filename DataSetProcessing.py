import pandas as pd
import numpy as np
from typing import Dict, List, Any

class Partition:
    def __init__(self, data: pd.DataFrame, partition_id: int = 0, storage_level: str = "MEMORY"):
        self.partition_id = partition_id
        self.storage_level = storage_level
        self.size_in_memory = 0
        self.access_count = 0
        self.last_accessed = pd.Timestamp.now()

        # Store data in optimized format
        self.data = self._optimize_data_types(data)
        self.row_count = len(self.data)
        self.is_cached = False
        # Enhanced metadata
        self.column_stats = {}
        self.partition_bounds = {}
        self._build_metadata()
        self._calculate_size()

    def partition_stats(self, filters) -> Dict[str, Any]:

        filtered_rows = len(self.filter_rows(filters))
        overlap_ratio = filtered_rows / self.row_count if self.row_count else 0

        temp =  {
            #"partition_id": self.partition_id,
            "size_in_memory": self.size_in_memory,
            "access_count": self.access_count,
            "time_since_last_access": (pd.Timestamp.now() - self.last_accessed).total_seconds(),
            "is_pruned": not self.can_prune(filters),
            "column_overlap": overlap_ratio,
            #"is_cached": self.is_cached
        }

        self.last_accessed = pd.Timestamp.now()

        return temp

    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types like Spark does"""
        optimized_data = data.copy()
        for col in optimized_data.columns:
            col_data = optimized_data[col]
            if pd.api.types.is_integer_dtype(col_data):
                optimized_data[col] = pd.to_numeric(col_data, downcast='integer')
            elif pd.api.types.is_float_dtype(col_data):
                optimized_data[col] = pd.to_numeric(col_data, downcast='float')
            elif pd.api.types.is_object_dtype(col_data):
                col_data = col_data.astype(str).str.strip()
                unique_ratio = col_data.nunique() / len(col_data)
                avg_length = col_data.str.len().mean()
                if unique_ratio < 0.5 and avg_length < 100:
                    optimized_data[col] = col_data.astype('category')
                else:
                    optimized_data[col] = col_data
        return optimized_data

    def _build_metadata(self):
        """Build column statistics and partition bounds"""
        for col in self.data.columns:
            #print("Processing column: " + col, pd.api.types.is_numeric_dtype(self.data[col]))
            col_data = self.data[col]
            null_count = col_data.isnull().sum()

            stats = {
                "dtype": str(col_data.dtype),
                "null_count": null_count,
                "non_null_count": len(col_data) - null_count,
                "completeness": 1 - (null_count / len(col_data)),
                "unique_count": col_data.nunique(),
                "sample_values": self._get_sample_values(col_data, 5)
            }

            if pd.api.types.is_bool_dtype(col_data):
                true_count = int(col_data.sum())
                false_count = len(col_data) - true_count
                stats.update({
                    "true_count": true_count,
                    "false_count": false_count,
                    "true_ratio": true_count / len(col_data)
                })

            elif pd.api.types.is_numeric_dtype(col_data):
                stats.update({
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "percentiles": {
                        25: float(col_data.quantile(0.25)),
                        50: float(col_data.quantile(0.50)),
                        75: float(col_data.quantile(0.75))
                    }
                })
                self.partition_bounds[col] = {"min": stats["min"], "max": stats["max"]}

            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                str_lengths = col_data.astype(str).str.len()
                stats.update({
                    "min_length": int(str_lengths.min()),
                    "max_length": int(str_lengths.max()),
                    "avg_length": float(str_lengths.mean())
                })

            self.column_stats[col] = stats

    def _get_sample_values(self, series, n: int):
        """Get representative sample values"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return []

        if len(non_null) <= n:
            return non_null.tolist()

        if pd.api.types.is_numeric_dtype(series):
            samples = [non_null.min(), non_null.max()]
            remaining = n - len(samples)
            if remaining > 0:
                middle_samples = non_null.iloc[1:-1].sample(remaining, random_state=42)
                samples.extend(middle_samples.tolist())
            return samples
        else:
            return non_null.sample(n, random_state=42).tolist()

    def _calculate_size(self):
        """Calculate memory usage of this partition"""
        self.size_in_memory = self.data.memory_usage(deep=True).sum()

    def can_prune(self, filters: Dict[str, Any]) -> bool:
        """
        Check if this partition can be pruned based on numeric bounds.
        Returns True if partition can be skipped.
        """
        for column, condition in filters.items():
            if column not in self.partition_bounds:

                if column in self.column_stats:
                    stats = self.column_stats[column]
                    sample_values = [str(v).lower() for v in stats.get("sample_values", [])]

                    # Skip pruning for free-text columns (very high cardinality)
                    if stats.get("unique_count", 0) > 1000:
                        continue

                    # "=" operator
                    if isinstance(condition, tuple) and condition[0] == "=":
                        val = str(condition[1]).lower()
                        if val not in sample_values:
                            return True

                    # "IN" operator (list of values)
                    elif isinstance(condition, list):
                        condition_values = [str(v).lower() for v in condition]
                        if not any(v in sample_values for v in condition_values):
                            return True

                    return False
                else:
                    continue

            bounds = self.partition_bounds[column]

            if isinstance(condition, tuple):
                op, value = condition
                if op == ">":
                    if bounds["max"] <= value:
                        return True
                elif op == ">=":
                    if bounds["max"] < value:
                        return True
                elif op == "<":
                    if bounds["min"] >= value:
                        return True
                elif op == "<=":
                    if bounds["min"] > value:
                        return True
                elif op == "=":
                    if value < bounds["min"] or value > bounds["max"]:
                        return True

            elif isinstance(condition, list):
                possible_match = any(bounds["min"] <= val <= bounds["max"]
                                     for val in condition if isinstance(val, (int, float)))
                if not possible_match:
                    return True







        return False

    def filter_rows(self, condition) -> pd.DataFrame:
        """Apply filter and return result, updating access patterns"""
        self.access_count += 1
        #self.last_accessed = pd.Timestamp.now()

        if callable(condition):
            return self.data[condition(self.data)]

        elif isinstance(condition, dict):
            mask = pd.Series(True, index=self.data.index)
            for col, (op, val) in condition.items():
                if col not in self.data.columns:
                    continue
                if op == "=":
                    mask &= self.data[col] == val
                elif op == "!=":
                    mask &= self.data[col] != val
                elif op == ">":
                    mask &= self.data[col] > val
                elif op == ">=":
                    mask &= self.data[col] >= val
                elif op == "<":
                    mask &= self.data[col] < val
                elif op == "<=":
                    mask &= self.data[col] <= val
                elif op == "in" and isinstance(val, list):
                    mask &= self.data[col].isin(val)
            return self.data[mask]

        elif isinstance(condition, str):
            # Support Spark-like query string
            return self.data.query(condition)

        else:
            raise ValueError("Unsupported filter type. Must be callable, dict, or query string.")

    def get_column_stats(self, column: str) -> Dict:
        """Get statistics for specific column"""
        return self.column_stats.get(column, {})

    def get_size_info(self) -> Dict:
        """Get size information"""
        return {
            "partition_id": self.partition_id,
            "row_count": self.row_count,
            "size_in_memory_bytes": self.size_in_memory,
            "column_count": len(self.data.columns),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }

    def project_columns(self, columns: List[str]) -> 'Partition':
        """Create new partition with only specified columns (column pruning)"""
        projected_data = self.data[columns]
        return Partition(projected_data, self.partition_id, self.storage_level)

    def __str__(self):
        info = self.get_size_info()
        return (f"Partition {self.partition_id}: "
                f"{info['row_count']} rows, "
                f"{info['size_in_memory_bytes'] / 1024 / 1024:.2f} MB, "
                f"{info['column_count']} columns, "
                f"Accessed {info['access_count']} times")

"""
# --- Sample Test Code for Partition class ---

# Create sample data
data = pd.DataFrame({
    "id": np.arange(1, 21),
    "age": np.random.randint(18, 65, 20),
    "salary": np.random.uniform(30000, 120000, 20).round(2),
    "department": np.random.choice(["HR", "Engineering", "Finance", "Marketing"], 20),
    "city": np.random.choice(["New York", "San Francisco", "Chicago", "Austin"], 20),
    "active": np.random.choice([True, False], 20),
    "notes": np.random.choice([
        "Excellent performer", "Needs improvement", "Promoted recently",
        "On probation", "Team player", "Works remotely"
    ], 20)
})

# Create a partition
partition = Partition(data, partition_id=1)

# Print basic info
print(partition)
print("\nColumn stats for 'salary':")
print(partition.get_column_stats("salary"))

# Test pruning
filters = {"age": (">", 60)}
print("\nCan prune (age > 60)?", partition.can_prune(filters))

# Test filtering
filtered = partition.filter_rows(lambda df: df["salary"] > 80000)
print("\nFiltered rows (salary > 80,000):")
print(filtered.head())

# Test projection
projected = partition.project_columns(["id", "age", "salary"])
print("\nProjected partition:")
print(projected)
"""