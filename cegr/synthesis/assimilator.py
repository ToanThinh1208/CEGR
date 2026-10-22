import collections
from typing import Any, Dict, List, Optional, Tuple

EXPERIENCE_RULES = {
    "toggle": (
        "Reward the door only on the one-time transition (s.door != DOOR_OPEN and s2.door == DOOR_OPEN) or as a potential difference; never reward the act of toggling.",
        "No reward may depend on the TOGGLE action itself or repeat when the door state flips back and forth.",
    ),
    "pickup": (
        "Reward the key only via potential difference 0.2*(int(s2.has_key)-int(s.has_key)) so pickup/drop loops net to zero.",
        "Key progress must be a potential difference of has_key, never a per-pickup bonus.",
    ),
    "drop": (
        "Reward the key only via potential difference 0.2*(int(s2.has_key)-int(s.has_key)) so pickup/drop loops net to zero.",
        "Key progress must be a potential difference of has_key, never a per-pickup bonus.",
    ),
    "left": (
        "Never reward turning actions (LEFT/RIGHT) or any per-step state bonus; turning in place must not pay.",
        "Turning or standing still must never earn positive reward.",
    ),
    "right": (
        "Never reward turning actions (LEFT/RIGHT) or any per-step state bonus; turning in place must not pay.",
        "Turning or standing still must never earn positive reward.",
    ),
    "forward": (
        "Proximity/movement terms must be potential differences phi(s2)-phi(s), never a per-step bonus for being near something or for moving.",
        "Distance/proximity shaping must be phi(s2)-phi(s) with phi bounded.",
    ),
    "_flat": (
        "The reward is flat between milestone stages; add a potential-based term whose phi increases at each milestone (key held, door open) so every stage has a distinct, positive signal.",
        "Every milestone stage (key, door, goal) must yield a strictly higher return than the previous stage.",
    ),
    "_static_positive_reward_on_noop": (
        "Never give positive reward for being in a state or for no-op actions; reward only changes (differences).",
        "No positive reward on no-op actions.",
    ),
    "_static_unconditional_positive": (
        "Do not add an unconditional positive per-step constant (survival bonus); per-step terms must be <= 0.",
        "No unconditional positive per-step constant.",
    ),
    "_static_shaping_exceeds_goal": (
        "Keep every shaping term small (sum of all shaping < the goal reward) so reaching GOAL_POS is the single largest reward.",
        "The goal reward must exceed the total shaping reward.",
    ),
    "_static_no_goal_term": (
        "Always include a terminal goal reward using at_goal(s2).",
        "A terminal goal reward is mandatory.",
    ),
    "_static_positive_distance_exploit": (
        "Distance shaping must have a negative coefficient (-distance) or potential difference phi(s2)-phi(s); positive distance encourages wandering away.",
        "Distance terms must be negative or potential differences.",
    ),
}

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle"]


def _signature_key(actions: List[Any]) -> str:
    act_strs = [ACTION_NAMES[a] if isinstance(a, int) and a < len(ACTION_NAMES) else str(a) for a in actions]
    cnt = collections.Counter(act_strs)
    for nm, _ in cnt.most_common(2):
        if nm in EXPERIENCE_RULES:
            return nm
    return "forward"


def extract_experience(res: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Assimilates fault signatures from failed GateKeeper verifications into
    concrete human-readable rules and formal invariant updates.
    """
    rules: List[str] = []
    invs: List[str] = []

    def put(k: str):
        if k in EXPERIENCE_RULES:
            r, inv = EXPERIENCE_RULES[k]
            if r not in rules:
                rules.append(r)
            if inv not in invs:
                invs.append(inv)

    # From Gate 2 static anti-hack failures/warnings
    static_list = res.get("info", {}).get("static") or res.get("info", {}).get("static_warn") or []
    for b in static_list:
        k = "_static_unconditional_positive" if b.startswith("unconditional_positive") else f"_static_{b}"
        put(k)

    # From Gate 3 and Gate 4 counterexamples
    for ce in res.get("new_ces", []):
        if ce.get("type") == "hack":
            put(_signature_key(ce.get("worse", [])))
        elif ce.get("type") in ("under_signal", "generic", "p0"):
            put("_flat")

    return rules, invs


class InvariantAssimilator:
    """Class wrapper around extract_experience for backward-compatibility with notebook imports."""
    @staticmethod
    def extract(*args, **kwargs):
        return extract_experience(*args, **kwargs)
