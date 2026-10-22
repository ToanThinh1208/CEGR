import itertools
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from cegr.envs.doorkey import DoorKeyEnv, traj_return, traj_level, gt_score, DOOR_OPEN
from cegr.core.code_sandbox import RewardTable
from cegr.core.tpe_evaluator import classify_pair
from cegr.synthesis.ce_pool import make_ce

RHO_THR = 0.75
M_CE_PER_ROUND = 2


def spearman(x: List[float], y: List[float]) -> float:
    xs, ys = pd.Series(x).rank(), pd.Series(y).rank()
    if xs.std() == 0 or ys.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def stage_reward_std(env: DoorKeyEnv, table: RewardTable, actions: List[int]) -> List[float]:
    stds = {0: [], 1: [], 2: []}
    i = 0
    for a in actions:
        if env.terminal[i]:
            break
        s = env.states[i]
        st = 2 if s.door == DOOR_OPEN else (1 if s.has_key else 0)
        stds[st].append(float(table.R[i, a]))
        g = bool(env.goal_flag[i, a])
        i = int(env.nxt[i, a])
        if g:
            break
    return [float(np.std(v)) if len(v) >= 2 else float("inf") for v in (stds[0], stds[1], stds[2])]


def signal_check(
    env: DoorKeyEnv,
    kit: Any,
    table: RewardTable,
    rnd: int,
    classify: bool = True,
    eps_rel: float = 0.02,
    eps_abs: float = 1e-3,
) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Gate G4: Signal and progress correlation verification.
    Verifies:
      (a) Distinct positive margin across adjacent milestone stages (T3 > T2 > T1 > T0)
      (b) Positive reward variance across progress stages along goal-reaching trajectories
      (c) Spearman rank correlation rho >= 0.75 against ground-truth milestone score
    """
    ret_t3 = traj_return(env, table, kit.refs["T3"])
    eps = max(eps_abs, eps_rel * abs(ret_t3))
    R = kit.refs
    Rret = {k: traj_return(env, table, v) for k, v in R.items()}
    info: Dict[str, Any] = {"eps": eps}
    ces = []

    # (a) Milestone stage margins across all pairs (Ta > Tb with a > b)
    all_milestone_pairs = (
        ("T3", "T2"), ("T2", "T1"), ("T1", "T0"),
        ("T3", "T0"), ("T3", "T1"), ("T2", "T0"),
    )
    violated_milestones = []
    for hi, lo in all_milestone_pairs:
        rb, rw = Rret[hi], Rret[lo]
        typ = classify_pair(rb, rw, eps_rel=eps_rel)
        if typ is not None:
            violated_milestones.append(f"{hi}>{lo}:{typ}")
            ces.append(make_ce(R[hi], R[lo], typ if classify else "generic", rnd, f"milestone_pair:{hi}>{lo}"))

    # (b) Stage reward variance along success trajectory (does not produce CE, only feedback)
    sv = stage_reward_std(env, table, R["T3"])
    flat_stages = [k for k, v in enumerate(sv) if v <= 1e-9]

    # (c) Spearman rank correlation on independent signal stream (seed 2000)
    rs = [traj_return(env, table, t) for t in kit.signal_trajs]
    gs = [gt_score(env, t) for t in kit.signal_trajs]
    rho = spearman(rs, gs)

    info.update(
        flat_pairs=violated_milestones,
        flat_stages=flat_stages,
        spearman=rho,
    )

    # If rho < RHO_THR, check if any distinguishing pairs in signal_trajs are violated
    if rho < RHO_THR:
        cand = []
        for A, B in itertools.combinations(range(len(kit.signal_trajs)), 2):
            ta, tb = kit.signal_trajs[A], kit.signal_trajs[B]
            la, lb = traj_level(env, ta), traj_level(env, tb)
            if la == lb:
                continue
            b, w = (ta, tb) if la > lb else (tb, ta)
            rb, rw = traj_return(env, table, b), traj_return(env, table, w)
            typ = classify_pair(rb, rw, eps_rel=eps_rel)
            if typ is not None:
                cand.append((rb - rw, b, w, typ, max(la, lb), min(la, lb)))
        cand.sort(key=lambda x: x[0])
        for m, b, w, typ, l_hi, l_lo in cand[:M_CE_PER_ROUND]:
            ces.append(make_ce(b, w, typ if classify else "generic", rnd, f"low_rho_viol:L{l_hi}>L{l_lo}"))

    uniq = {c["key"]: c for c in ces}
    ok = (not violated_milestones) and (not flat_stages) and (rho >= RHO_THR)
    return ok, info, list(uniq.values())[:M_CE_PER_ROUND + 2]


class SignalChecker:
    """Class wrapper around signal_check for backward-compatibility with notebook imports."""
    @staticmethod
    def check(*args, **kwargs):
        return signal_check(*args, **kwargs)
