import collections
import hashlib
from typing import Any, Dict, List, Optional, Set
import numpy as np


ACTION_NAMES_MAP = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}


def _extract_signature(worse: Any, worse_steps: Optional[List[Any]], tag: str) -> str:
    """Extracts behavior signature of the exploit trajectory (e.g. toggle_loop, pickup_loop, gripper_spam)."""
    if worse_steps and len(worse_steps) > 0 and hasattr(worse_steps[0], "a"):
        acts = [st.a for st in worse_steps]
        # Check continuous gripper spam or hover
        if len(acts) > 0 and hasattr(acts[0], "__iter__"):
            grippers = [float(a[-1]) if len(a) > 0 else 0.0 for a in acts]
            if len(grippers) > 4:
                flips = sum(1 for i in range(len(grippers) - 1) if (grippers[i] > 0) != (grippers[i + 1] > 0))
                if flips >= 4:
                    return "gripper_spam"
            return "continuous_exploit"
    if isinstance(worse, (list, tuple)) and len(worse) > 0:
        named_acts = [ACTION_NAMES_MAP.get(a, str(a)) for a in worse if isinstance(a, (int, np.integer))]
        if named_acts:
            cnt = collections.Counter(named_acts)
            top_act, top_count = cnt.most_common(1)[0]
            if top_act in ("toggle", "pickup", "drop", "left", "right"):
                return f"{top_act}_loop"
            return f"{top_act}_spam"
    if "toggle" in tag.lower():
        return "toggle_loop"
    if "pickup" in tag.lower():
        return "pickup_loop"
    return "exploit_policy"


def _extract_progress_pair(tag: str, better: Any, worse: Any) -> str:
    """Extracts milestone progress pair string (e.g. T3>T2)."""
    for pair_name in ("T3>T2", "T2>T1", "T1>T0", "T3>T0", "T3>T1", "T2>T0"):
        if pair_name in tag:
            return pair_name
    return tag or "unknown_pair"


def make_ce(
    better: Any,
    worse: Any,
    typ: str,
    rnd: int,
    tag: str = "",
    better_steps: Optional[List[Any]] = None,
    worse_steps: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Creates a standardized counterexample dictionary with content MD5 hash and signature key."""
    # Preserve step objects for continuous domain re-evaluation
    if better_steps is None and isinstance(better, (list, tuple)) and len(better) > 0 and hasattr(better[0], "s"):
        better_steps = list(better)
        better = [tuple(st.a) if hasattr(st.a, "__iter__") else st.a for st in better_steps]
    if worse_steps is None and isinstance(worse, (list, tuple)) and len(worse) > 0 and hasattr(worse[0], "s"):
        worse_steps = list(worse)
        worse = [tuple(st.a) if hasattr(st.a, "__iter__") else st.a for st in worse_steps]

    better_str = ",".join(map(str, better)) if hasattr(better, "__iter__") else str(better)
    worse_str = ",".join(map(str, worse)) if hasattr(worse, "__iter__") else str(worse)
    raw_key = f"{better_str}|{worse_str}|{typ}"
    key = hashlib.md5(raw_key.encode()).hexdigest()[:12]

    behavior_sig = _extract_signature(worse, worse_steps, tag)
    progress_pair = _extract_progress_pair(tag, better, worse)
    sig_key = (typ, behavior_sig, progress_pair)

    out = {
        "key": key,
        "sig_key": sig_key,
        "behavior_signature": behavior_sig,
        "progress_pair": progress_pair,
        "better": list(better) if hasattr(better, "__iter__") else better,
        "worse": list(worse) if hasattr(worse, "__iter__") else worse,
        "type": typ,
        "round": int(rnd),
        "tag": tag,
    }
    if better_steps is not None:
        out["better_steps"] = better_steps
    if worse_steps is not None:
        out["worse_steps"] = worse_steps
    return out


class CEPool:
    """
    Cumulative CounterExample Pool for CEGR.
    Features:
      - Bounded FIFO memory (Capacity = 40 items)
      - Double deduplication: content hash (MD5) AND signature key (type, behavior_sig, progress_pair)
      - Categorized tracking ('hack' vs 'under_signal' vs 'p0')
    """

    def __init__(self, max_size: int = 40):
        self.items: List[Dict[str, Any]] = []
        self.keys: Set[str] = set()
        self.sig_keys: Set[Tuple[str, str, str]] = set()
        self.max_size = max_size

    def add(self, ce: Dict[str, Any]) -> bool:
        # Deduplication: drop if content hash or signature key already exists
        if ce.get("key") in self.keys:
            return False
        sig_k = ce.get("sig_key")
        if sig_k and sig_k in self.sig_keys:
            return False

        self.items.append(ce)
        if "key" in ce:
            self.keys.add(ce["key"])
        if sig_k:
            self.sig_keys.add(sig_k)

        if len(self.items) > self.max_size:
            old = self.items.pop(0)
            if "key" in old:
                self.keys.discard(old["key"])
            if "sig_key" in old:
                self.sig_keys.discard(old["sig_key"])
        return True

    def counts(self) -> Dict[str, int]:
        return dict(collections.Counter(c["type"] for c in self.items))

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.items)

    def __len__(self) -> int:
        return len(self.items)


Counterexample = Dict[str, Any]
