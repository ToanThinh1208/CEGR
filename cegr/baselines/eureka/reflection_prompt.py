from typing import Any, Dict, List
import numpy as np


def build_reward_reflection(
    task_success_curve: List[float],
    component_curves: Dict[str, List[float]],
    domain: str = "doorkey",
) -> str:
    """
    Constructs the exact Reward Reflection text specified by Eureka (Ma et al., 2024).
    Tracks values at 5 evenly spaced training checkpoints (20%, 40%, 60%, 80%, 100%).
    """
    def sample_5_checkpoints(seq: List[float]) -> List[float]:
        if not seq:
            return [0.0] * 5
        n = len(seq)
        idxs = [max(0, int(n * f) - 1) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
        return [float(seq[i]) for i in idxs]

    succ_5 = sample_5_checkpoints(task_success_curve)
    max_succ = max(succ_5) if succ_5 else 0.0
    final_succ = succ_5[-1] if succ_5 else 0.0

    lines = [
        "We trained an RL policy using your provided reward_components function.",
        "Below are the values of each reward component and task metrics tracked at every 20% of training:",
        f"- task_success_rate: [{', '.join(f'{v:.2f}' for v in succ_5)}] (Max: {max_succ:.2f}, Final: {final_succ:.2f})",
    ]

    for comp_name, vals in component_curves.items():
        c_5 = sample_5_checkpoints(vals)
        lines.append(f"- component '{comp_name}': [{', '.join(f'{v:.2f}' for v in c_5)}]")

    lines.extend([
        "Please carefully analyze the scale and progression of each reward component above,",
        "identify why the policy failed to maximize task_success_rate, and provide a new improved reward_components(s, a, s2) function."
    ])

    return "\n".join(lines)
