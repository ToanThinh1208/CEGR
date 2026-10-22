from typing import Any, Dict, List, Optional, Tuple


class TrajectoryBuffer:
    """
    Fixed Trajectory Buffer for CARD & CARD_REFL.
    Maintains static reference trajectory preference pairs P_0 = {(tau_k+, tau_k-)}.
    Strictly static: does NOT dynamically mine or add attacker counterexamples during search.
    """

    def __init__(self, initial_pairs: Optional[List[Dict[str, Any]]] = None):
        self.pairs: List[Dict[str, Any]] = list(initial_pairs or [])

    def add_pair(self, better: Any, worse: Any, tag: str = "p0"):
        self.pairs.append({
            "better": better,
            "worse": worse,
            "tag": tag,
            "type": "p0",
        })

    def get_pairs(self) -> List[Dict[str, Any]]:
        return list(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)
