from typing import Any, Dict, List, Optional, Tuple


class TrajectoryBuffer:
    """
    Fixed Trajectory Buffer for CARD & CARD_REFL.
    Maintains static reference trajectory preference pairs P_0 = {(tau_k+, tau_k-)}.
    Strictly static: does NOT dynamically mine or add attacker counterexamples during search.
    Preserves all reference pair attributes (better, worse, tag, type, better_steps, worse_steps, progress_pair)
    to support discrete (DoorKey) and continuous (Meta-World) rollouts.
    """

    def __init__(self, initial_pairs: Optional[List[Dict[str, Any]]] = None):
        self.pairs: List[Dict[str, Any]] = []
        if initial_pairs:
            for p in initial_pairs:
                if isinstance(p, dict):
                    self.add_pair(
                        better=p.get("better"),
                        worse=p.get("worse"),
                        tag=p.get("tag", "p0"),
                        better_steps=p.get("better_steps"),
                        worse_steps=p.get("worse_steps"),
                        type=p.get("type", "p0"),
                        progress_pair=p.get("progress_pair"),
                        **{k: v for k, v in p.items() if k not in ("better", "worse", "tag", "better_steps", "worse_steps", "type", "progress_pair")},
                    )

    def add_pair(
        self,
        better: Any,
        worse: Any = None,
        tag: str = "p0",
        better_steps: Optional[Any] = None,
        worse_steps: Optional[Any] = None,
        type: str = "p0",
        progress_pair: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ):
        """
        Adds a static reference preference pair (tau+, tau-) to P_0.
        Accepts explicit kwargs or a pair dict as the first argument.
        """
        if isinstance(better, dict) and worse is None and "better" in better:
            p = better
            self.add_pair(
                better=p.get("better"),
                worse=p.get("worse"),
                tag=p.get("tag", "p0"),
                better_steps=p.get("better_steps"),
                worse_steps=p.get("worse_steps"),
                type=p.get("type", "p0"),
                progress_pair=p.get("progress_pair"),
                **{k: v for k, v in p.items() if k not in ("better", "worse", "tag", "better_steps", "worse_steps", "type", "progress_pair")},
            )
            return

        pair_data: Dict[str, Any] = {
            "better": better,
            "worse": worse,
            "tag": tag,
            "type": type,
        }
        if better_steps is not None:
            pair_data["better_steps"] = better_steps
        if worse_steps is not None:
            pair_data["worse_steps"] = worse_steps
        if progress_pair is not None:
            pair_data["progress_pair"] = progress_pair

        for k, v in kwargs.items():
            if k not in pair_data:
                pair_data[k] = v

        self.pairs.append(pair_data)

    def get_pairs(self) -> List[Dict[str, Any]]:
        return list(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.pairs[idx]
