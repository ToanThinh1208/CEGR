from typing import Any, Dict, List, Optional
import numpy as np


def generate_card_feedback(
    syntax_error: Optional[str],
    tpe_violations: List[Dict[str, Any]],
    succ_returns: List[float],
    fail_returns: List[float],
    succ_comps: Dict[str, List[float]],
    fail_comps: Dict[str, List[float]],
    reflections_history: Optional[List[str]] = None,
) -> str:
    """
    Constructs the 3 structured feedback blocks mandated by the CARD framework (Sun et al., 2024):
      1. Process Feedback (execution status, type checks, degenerate constant components)
      2. Trajectory Feedback (mean & std across successful vs failed rollouts)
      3. Preference Feedback (explicit pairwise ranking inversions G(tau+) <= G(tau-))
    """
    blocks = []

    # 1. Process Feedback
    if syntax_error:
        blocks.append(
            f"=== PROCESS FEEDBACK ===\n"
            f"Execution Status: FAILED\n"
            f"Error details: {syntax_error}\n"
            f"Ensure `def reward_components(s, a, s2) -> dict` returns a dictionary of finite floats without raising exceptions."
        )
        return "\n\n".join(blocks)

    # Check for degenerate components (variance == 0 across rollouts)
    degenerate = []
    for k, vals in succ_comps.items():
        if len(vals) > 1 and float(np.std(vals)) < 1e-7 and abs(float(np.mean(vals))) < 1e-7:
            degenerate.append(k)

    degen_msg = (
        f"Warning: Component(s) {degenerate} remained identically 0.0 across all trajectories (degenerate component)."
        if degenerate else "Execution Status: PASSED. All components produce valid scalar numbers."
    )
    blocks.append(f"=== PROCESS FEEDBACK ===\n{degen_msg}")

    # 2. Trajectory Feedback
    m_succ = float(np.mean(succ_returns)) if succ_returns else 0.0
    s_succ = float(np.std(succ_returns)) if len(succ_returns) > 1 else 0.0
    m_fail = float(np.mean(fail_returns)) if fail_returns else 0.0
    s_fail = float(np.std(fail_returns)) if len(fail_returns) > 1 else 0.0

    traj_lines = [
        "=== TRAJECTORY FEEDBACK ===",
        f"Successful Trajectories (tau+): Mean Return = {m_succ:.3f} (std = {s_succ:.3f})",
        f"Failed/Suboptimal Trajectories (tau-): Mean Return = {m_fail:.3f} (std = {s_fail:.3f})",
        "Component breakdown on tau+ vs tau-:",
    ]
    for k in succ_comps.keys():
        v_s = float(np.mean(succ_comps.get(k, [0.0])))
        v_f = float(np.mean(fail_comps.get(k, [0.0])))
        traj_lines.append(f"  - Component '{k}': mean on tau+ = {v_s:+.3f} | mean on tau- = {v_f:+.3f}")

    blocks.append("\n".join(traj_lines))

    # 3. Preference Feedback
    pref_lines = [
        "=== PREFERENCE FEEDBACK (TPE Violations) ===",
        f"Total pairs evaluated: {len(succ_returns) if succ_returns else len(tpe_violations)}",
        f"Violated reference pairs: {len(tpe_violations)} pair(s) inverted or tied.",
    ]
    for viol in tpe_violations[:3]:
        tag = viol.get("pair", {}).get("tag", "reference_pair")
        rb = viol.get("Rb", 0.0)
        rw = viol.get("Rw", 0.0)
        pref_lines.append(f"  * [VIOLATION] Pair '{tag}': better trajectory return ({rb:+.3f}) <= worse trajectory return ({rw:+.3f}). Margin = {rb - rw:+.3f}")

    blocks.append("\n".join(pref_lines))

    # Optional Memory Reflection (CARD_REFL mode)
    if reflections_history:
        blocks.append(
            "=== TEXT REFLECTION MEMORY (Past Iteration Failures) ===\n" +
            "\n".join(f"- Iteration {idx+1}: {refl}" for idx, refl in enumerate(reflections_history[-3:]))
        )

    return "\n\n".join(blocks)
