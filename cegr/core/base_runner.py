from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRunner(ABC):
    """
    Abstract Base Class for all experiment runners:
      - ORIG (No-repair baseline)
      - EUREKA (Evolutionary search with reflection)
      - CARD / CARD_REFL (Trajectory preference evaluation)
      - CEGR (Multi-tier GateKeeper synthesis)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_rounds = config.get("max_rounds", config.get("budget", {}).get("n_rounds", 4))
        self.model_id = config.get("llm", {}).get("model_id", "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")

    @abstractmethod
    def run_single_job(self, reward_record: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """Runs the method on a single seed/job starting from reward_record."""
        pass

    @abstractmethod
    def run_batch(self, reward_records: List[Dict[str, Any]], n_samples: int = 1) -> List[Dict[str, Any]]:
        """Runs the method across a batch of reward functions."""
        pass
