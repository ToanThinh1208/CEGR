from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import collections
import numpy as np


@dataclass
class Step:
    t: int
    s: Any
    a: Any
    s2: Any
    r: float
    comps: Dict[str, float]
    goal: bool


@dataclass
class Trajectory:
    steps: List[Step]
    success: bool
    ret: float

    def comp_totals(self, gamma: float = 1.0) -> Dict[str, float]:
        tot = collections.defaultdict(float)
        for st in self.steps:
            for k, v in st.comps.items():
                tot[k] += (gamma ** st.t) * v
        return dict(tot)

    def action_names(self) -> List[str]:
        return [str(st.a) for st in self.steps]

    def __len__(self) -> int:
        return len(self.steps)


class BaseEnv(ABC):
    """Abstract Base Class for all reinforcement learning environments in CEGR."""

    def __init__(self, horizon: int):
        self.horizon = horizon

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and return initial observation/state and info."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step environment with an action and return (obs, reward, terminated, truncated, info)."""
        pass

    @abstractmethod
    def extract_state_dict(self, obs: Any, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert raw observation into a semantic state dictionary for LLM reward evaluation."""
        pass

    @abstractmethod
    def gt_score(self, trajectory: List[Any]) -> float:
        """Ground-truth progress/success score used by Gate G4 for Spearman rank correlation."""
        pass
