"""
Canonical DQN network used by every training/inference path in this repo.

Previously this class was defined three times, independently, with two
incompatible architectures and two incompatible state dimensions:
  - dqn_training.py: 128-hidden, no LayerNorm, 34-dim state (5 cached
    partitions x 6 per-partition features + 4 global features).
  - executor.py (Executor.load_rl_model): identical 128-hidden copy of
    the above, also assuming a 34-dim state.
  - new_run/train_stable_dqn.py: 256-hidden with LayerNorm, sized for a
    38-dim state (5 x 6 + 8 global features) per its own inline comment
    ("Expected 38 features (5*6 + 8)") — but nothing in the repo actually
    built an 8-feature global block; the state vector everywhere else
    only ever produced 4.

This module is the single source of truth going forward: the 256-hidden
LayerNorm architecture (more stable for offline training, per the
DDQN/Huber/soft-update pipeline in new_run/train_stable_dqn.py) is kept
as canonical, and the state vector is reconciled to genuinely produce 38
features (see `executor.py::_get_cache_state_vector` for the completed
8-feature global block: cache utilization and rolling eviction rate were
added, plus a 2-dim query-intent one-hot for temporal/categorical
filters, matching the README's MDP description of the state as
per-partition + workload/query-context + global cache signals).

STATE_DIM (38) and NUM_ACTIONS (6 = 5 cache slots + 1 "no eviction") are
defined here so every consumer (training scripts, Executor) references
the same numbers instead of re-hardcoding them.
"""

from pathlib import Path

import torch
import torch.nn as nn
import yaml

STATE_DIM = 38
NUM_ACTIONS = 6
HIDDEN_DIM = 256

# config.yaml currently lives next to this file at the repo root. If the
# planned folder restructure moves this module to src/dqn_model.py and
# the config to configs/config.yaml, update this path accordingly.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the repo's YAML hyperparameter config.

    Args:
        path: Path to a config.yaml (defaults to the repo-root one next
            to this module).

    Returns:
        Parsed config dict. Callers should read via `dict.get(key,
        <hardcoded fallback>)` so scripts keep working with zero args
        even if a key is missing or the file itself is absent (see each
        training script's use of this function).
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class DQN(nn.Module):
    """Feed-forward Q-network: state vector -> per-action Q-values.

    Two hidden layers with a LayerNorm after the first (stabilizes
    offline training against noisy/unnormalized input scales — see
    new_run/train_stable_dqn.py's training docstring for the full
    stabilizer rationale).
    """

    def __init__(self, input_dim: int = STATE_DIM, output_dim: int = NUM_ACTIONS, hidden_dim: int = HIDDEN_DIM):
        """
        Args:
            input_dim: Size of the (ideally normalized) state vector.
            output_dim: Number of discrete eviction actions.
            hidden_dim: Width of both hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of states, shape [B, input_dim] -> [B, output_dim]."""
        return self.net(x)
