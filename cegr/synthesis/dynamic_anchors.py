import json
from typing import Any, Dict, List, Optional

DEFAULT_DOORKEY_ANCHORS = {
    "version": 0,
    "milestones": ["key", "door", "goal"],
    "invariants": [
        "Reaching GOAL_POS must give strictly the highest total return of any trajectory.",
        "Every progress term must be one-time or potential-based phi(s2)-phi(s); no repeatable bonus.",
        "Idling (no-op / turning in place) must never earn positive reward.",
    ],
}

DEFAULT_METAWORLD_ANCHORS = {
    "version": 0,
    "milestones": ["reach_obj", "grasp_obj", "place_goal"],
    "invariants": [
        "Placing the object at the target goal must give strictly the highest cumulative return.",
        "Distance terms must be negative or potential differences phi(s2)-phi(s); never reward positive Euclidean distance.",
        "Opening/closing the gripper in mid-air must never yield positive reward.",
    ],
}

MILESTONE_TEXT = {
    "key": "pick up the key",
    "door": "open the door (needs the key)",
    "goal": "step on GOAL_POS",
    "reach_obj": "move hand to object",
    "grasp_obj": "grasp object firmly",
    "place_goal": "deliver object to goal location",
}

MAX_INVARIANTS = 8


class DynamicAnchorsManager:
    """
    Manages Dynamic Invariant Anchors specification.
    Upgrades formal invariants (v0 -> v1 -> ...) as the verifier discovers novel failure modes.
    """

    def __init__(self, domain: str = "doorkey", initial_anchors: Optional[Dict[str, Any]] = None):
        if initial_anchors:
            self.anchors = json.loads(json.dumps(initial_anchors))
        else:
            base = DEFAULT_DOORKEY_ANCHORS if domain == "doorkey" else DEFAULT_METAWORLD_ANCHORS
            self.anchors = json.loads(json.dumps(base))

    def update(self, new_invariants: List[str]) -> Dict[str, Any]:
        current_invs = list(self.anchors["invariants"])
        added = False
        for x in new_invariants:
            if x not in current_invs:
                current_invs.append(x)
                added = True
        self.anchors["invariants"] = current_invs[:MAX_INVARIANTS]
        if added:
            self.anchors["version"] += 1
        return self.anchors

    def render(self) -> str:
        milestones_str = " -> ".join(
            MILESTONE_TEXT.get(m, m) for m in self.anchors.get("milestones", [])
        )
        inv_lines = "".join(f"   - {x}\n" for x in self.anchors.get("invariants", []))
        return (
            f"SPECIFICATION ANCHORS (v{self.anchors['version']}):\n"
            f"  Milestones in order: {milestones_str}\n"
            f"  Invariants the reward MUST satisfy:\n{inv_lines}"
        )
