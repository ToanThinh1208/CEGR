from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def classify_pair(
    Rb: float,
    Rw: float,
    eps_rel: float = 0.02,
    better_is_success: bool = True,
    thr: Optional[float] = None,
    **kwargs: Any,
) -> Optional[str]:
    """
    Classifies preference violation according to SYSTEM_LOGIC_SPEC:
      - 'hack': Worse trajectory has return >= better trajectory - 1e-9 (ranking inversion or near-tie).
      - 'under_signal': Better trajectory has higher return (Rb - Rw > 1e-9), but margin < thr.
      - None: Satisfies both ordering and margin requirement (Rb - Rw >= thr).
    """
    if "eps" in kwargs and kwargs["eps"] is not None:
        eps_rel = float(kwargs["eps"])

    if thr is None:
        if eps_rel > 0.1:  # Legacy absolute thr passed as eps
            thr = max(1e-4, float(eps_rel))
        else:
            thr = max(1e-4, float(eps_rel) * abs(float(Rb)))

    margin = float(Rb) - float(Rw)
    if margin <= 1e-9:
        return "hack"
    elif margin < thr:
        return "under_signal"
    return None


def pair_returns(env: Any, table: Any, ce: Dict[str, Any]) -> Tuple[float, float]:
    """Computes (Rb, Rw) returns for a pair on discrete DoorKey or continuous rollouts."""
    if table is not None and hasattr(table, "R") and "better" in ce and "worse" in ce:
        from cegr.envs.doorkey import traj_return
        b, w = ce["better"], ce["worse"]
        if isinstance(b, (list, tuple)) and (not b or isinstance(b[0], (int, np.integer))):
            rb = traj_return(env, table, b)
            rw = traj_return(env, table, w)
            return float(rb), float(rw)

    if callable(table) or (table is not None and not hasattr(table, "R")):
        reward_fn = table
        b_steps = ce.get("better_steps") or (ce.get("better") if isinstance(ce.get("better"), list) and len(ce["better"]) > 0 and hasattr(ce["better"][0], "s") else None)
        w_steps = ce.get("worse_steps") or (ce.get("worse") if isinstance(ce.get("worse"), list) and len(ce["worse"]) > 0 and hasattr(ce["worse"][0], "s") else None)
        if b_steps and w_steps:
            def eval_steps(steps):
                tot = 0.0
                for st in steps:
                    comps = reward_fn(st.s, st.a, st.s2)
                    if not isinstance(comps, dict):
                        comps = {"total": float(comps)}
                    tot += sum(float(v) for v in comps.values())
                return float(tot)
            try:
                rb = eval_steps(b_steps)
                rw = eval_steps(w_steps)
                return float(rb), float(rw)
            except Exception:
                pass

    rb = float(ce.get("Rb", ce.get("ref_return", 0.0)))
    rw = float(ce.get("Rw", ce.get("exploit_return", 0.0)))
    return rb, rw


def pair_margin(env: Any, table: Any, ce: Dict[str, Any]) -> float:
    """Computes margin Rb - Rw for a preference pair or counterexample."""
    rb, rw = pair_returns(env, table, ce)
    return float(rb - rw)


def pair_violated(
    margin: float,
    ce: Dict[str, Any],
    eps: float,
    classify: bool = True,
    mode: str = "margin",
    env: Optional[Any] = None,
    table: Optional[Any] = None,
) -> bool:
    """
    Returns True if trajectory pair violates order or required margin.
    - 'strict' mode (CARD/CARD_REFL): violated if margin <= 1e-9.
    - 'margin' mode (CEGR G3): violated if classify_pair is not None.
    """
    if mode == "strict":
        return margin <= 1e-9

    if env is not None and table is not None:
        rb, rw = pair_returns(env, table, ce)
        return classify_pair(rb, rw, eps_rel=0.02) is not None

    if ce.get("type") == "under_signal" and classify:
        return margin < eps
    return margin <= 1e-9


def signal_eps(ret_succ: float, eps_abs: float = 1e-3, eps_rel: float = 0.02) -> float:
    return max(eps_abs, eps_rel * abs(ret_succ))


class TPEEvaluator:
    """
    Shared Trajectory Preference Evaluation (TPE) algorithm for CARD, CARD_REFL, and CEGR (Gate G3).
    Evaluates discounted cumulative return across paired trajectories (tau+, tau-)
    and verifies strict order preservation: G(tau+) > G(tau-).
    """

    def __init__(self, gamma: float = 1.0, eps_rel: float = 0.02, eps_abs: float = 1e-3):
        self.gamma = gamma
        self.eps_rel = eps_rel
        self.eps_abs = eps_abs

    def compute_return(self, trajectory_steps: List[Any], reward_fn: Any) -> Tuple[float, Dict[str, float]]:
        """Computes discounted return and per-component breakdown for a trajectory."""
        tot = 0.0
        comp_totals = {}
        for t, step in enumerate(trajectory_steps):
            s, a, s2 = step.s, step.a, step.s2
            comps = reward_fn(s, a, s2)
            if not isinstance(comps, dict):
                comps = {"total": comps}
            step_r = sum(float(v) for v in comps.values())
            discount = self.gamma ** t
            tot += discount * step_r
            for k, v in comps.items():
                comp_totals[k] = comp_totals.get(k, 0.0) + discount * float(v)
        return float(tot), comp_totals

    def evaluate_pairs(
        self,
        pairs: List[Dict[str, Any]],
        reward_fn: Any,
        classify: bool = True,
        ref_succ_return: Optional[float] = None,
        mode: str = "strict",
    ) -> Dict[str, Any]:
        """
        Evaluates a set of trajectory pairs (tau+, tau-).
        Supports mode='strict' (CARD/CARD_REFL) and mode='margin' (CEGR G3).
        """
        eps = signal_eps(ref_succ_return or 1.0, self.eps_abs, self.eps_rel)
        violations = []
        all_margins = []

        for p in pairs:
            if reward_fn is not None and "better_steps" in p and "worse_steps" in p:
                Rb, _ = self.compute_return(p["better_steps"], reward_fn)
                Rw, _ = self.compute_return(p["worse_steps"], reward_fn)
            else:
                Rb = p.get("Rb", p.get("ref_return", 0.0))
                Rw = p.get("Rw", p.get("exploit_return", 0.0))

            margin = Rb - Rw
            all_margins.append(margin)

            if mode == "strict":
                is_viol = margin <= 1e-9
            else:
                is_viol = classify_pair(Rb, Rw, eps_rel=self.eps_rel) is not None

            if is_viol:
                violations.append({
                    "pair": p,
                    "margin": margin,
                    "type": p.get("type", "generic"),
                    "Rb": Rb,
                    "Rw": Rw,
                })

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "eps": eps,
            "num_violations": len(violations),
            "margins": all_margins,
        }


def build_p0_pairs(env: Any, seed: int = 1000, max_noisy: int = 20) -> List[Dict[str, Any]]:
    """
    Builds the unified reference set P_0 used across CARD, CARD_REFL, and CEGR G3.
    Consists of 4 milestone reference pairs:
      {(T3, T2), (T2, T1), (T1, T0), (T3, T0)}
    plus up to max_noisy (<= 20) noisy preference pairs generated with seed (stream 1000).
    """
    from cegr.envs.doorkey import DoorKeyEnv
    pairs = []
    if isinstance(env, DoorKeyEnv):
        from cegr.envs.doorkey import build_reference_trajs, build_expert_tables, sample_noisy, gt_score
        refs = build_reference_trajs(env)
        expert = build_expert_tables(env)
        base = [
            (refs["T3"], refs["T2"], "T3>T2"),
            (refs["T2"], refs["T1"], "T2>T1"),
            (refs["T1"], refs["T0"], "T1>T0"),
            (refs["T3"], refs["T0"], "T3>T0"),
        ]
        for b, w, tag in base:
            pairs.append({
                "better": b, "worse": w, "tag": tag, "type": "p0",
                "progress_pair": (int(tag[1]), int(tag[4])),
            })

        # Sample up to max_noisy noisy trajectories
        if max_noisy > 0:
            noisy_trajs = sample_noisy(env, expert, seed=seed, n=max_noisy)
            t3 = refs["T3"]
            s_t3 = gt_score(env, t3)
            for k, nt in enumerate(noisy_trajs):
                s_nt = gt_score(env, nt)
                if s_t3 > s_nt:
                    b, w, tag = t3, nt, f"T3>noisy_{k}"
                    prog = (3, min(2, int(s_nt)))
                else:
                    b, w, tag = nt, refs["T0"], f"noisy_{k}>T0"
                    prog = (max(1, int(s_nt)), 0)
                pairs.append({
                    "better": b, "worse": w, "tag": tag, "type": "p0_noisy",
                    "progress_pair": prog,
                })
    else:
        from cegr.envs.base_env import Step
        def rollout_actions(action_list):
            s, _ = env.reset(seed=seed)
            steps = []
            for t, a in enumerate(action_list):
                s2, r, done, trunc, _ = env.step(a)
                g = bool(s2.get("in_target", False))
                steps.append(Step(t=t, s=s, a=a, s2=s2, r=1.0 if g else 0.0, comps={"goal": 1.0 if g else 0.0}, goal=g))
                s = s2
                if g or done:
                    break
            return steps

        t3_actions = [np.array([0.0, 0.1, -0.05, -0.8], dtype=np.float32) for _ in range(50)]
        t2_actions = [np.array([0.0, 0.1, -0.05, 0.0], dtype=np.float32) for _ in range(50)]
        t1_actions = [np.array([0.0, 0.0, 0.0, 0.8 if i % 2 == 0 else -0.8], dtype=np.float32) for i in range(50)]
        t0_actions = [np.array([-0.5, -0.5, 0.2, 0.0], dtype=np.float32) for _ in range(50)]

        t3_steps = rollout_actions(t3_actions)
        t2_steps = rollout_actions(t2_actions)
        t1_steps = rollout_actions(t1_actions)
        t0_steps = rollout_actions(t0_actions)

        base = [
            (t3_steps, t2_steps, "T3>T2"),
            (t2_steps, t1_steps, "T2>T1"),
            (t1_steps, t0_steps, "T1>T0"),
            (t3_steps, t0_steps, "T3>T0"),
        ]
        for b, w, tag in base:
            pairs.append({
                "better": b, "worse": w, "better_steps": b, "worse_steps": w,
                "tag": tag, "type": "p0", "progress_pair": (int(tag[1]), int(tag[4])),
            })

        # Generate continuous noisy perturbations (up to max_noisy)
        rng = np.random.default_rng(seed)
        for k in range(min(max_noisy, 20)):
            noisy_acts = [a + rng.normal(0, 0.15, size=a.shape).astype(np.float32) for a in t3_actions]
            noisy_steps = rollout_actions(noisy_acts)
            pairs.append({
                "better": t3_steps, "worse": noisy_steps,
                "better_steps": t3_steps, "worse_steps": noisy_steps,
                "tag": f"T3>noisy_{k}", "type": "p0_noisy", "progress_pair": (3, 1),
            })
    return pairs

