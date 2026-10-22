import collections
from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cegr.envs.base_env import Step, Trajectory
from cegr.envs.doorkey import DoorKeyEnv, LEFT, RIGHT, PICKUP, TOGGLE, FORWARD
from cegr.core.code_sandbox import RewardTable, RewardError, build_reward_table
from cegr.core.tpe_evaluator import classify_pair

EPS_HACK = 1e-6
EPS_MARGIN_REL = 0.01


@dataclass
class Analysis:
    ok: bool
    error: str = ""
    V_succ_30: float = float("nan")
    V_fail_30: float = float("nan")
    V_succ_60: float = float("nan")
    V_fail_60: float = float("nan")
    hacked_30: bool = False
    hacked_60: bool = False
    gap_rel: float = float("nan")
    tau_succ: Optional[Trajectory] = None
    tau_fail: Optional[Trajectory] = None
    names: List[str] = field(default_factory=list)
    subgoal_shaping: bool = False
    culprit: str = ""
    culprit_verified: bool = False
    signature: Tuple[str, ...] = ()
    table: Optional[RewardTable] = None

    @property
    def V_succ(self):
        return self.V_succ_30

    @property
    def V_fail(self):
        return self.V_fail_30

    @property
    def hacked(self):
        return self.hacked_30 or self.hacked_60

    @property
    def dp_aligned_both(self) -> bool:
        if not self.ok:
            return False
        m30 = self.V_succ_30 - self.V_fail_30
        m60 = self.V_succ_60 - self.V_fail_60
        thr30 = max(1e-4, EPS_MARGIN_REL * abs(self.V_succ_30))
        thr60 = max(1e-4, EPS_MARGIN_REL * abs(self.V_succ_60))
        return (m30 > thr30) and (m60 > thr60)


def exploit_signature(traj: Optional[Trajectory]) -> Tuple[str, ...]:
    if traj is None:
        return ()
    cnt = collections.Counter(traj.action_names())
    return tuple(sorted([n for n, _ in cnt.most_common(2)]))


def solve(env: DoorKeyEnv, table: RewardTable, T: int = 30, gamma: float = 1.0):
    """
    Exact Dynamic Programming solver computing:
      V_succ = max value reaching goal within horizon T
      V_fail = max value of policies that do NOT reach goal within horizon T
    """
    n = env.n
    R, nxt, gf = table.R, env.nxt, env.goal_flag
    fail = np.zeros(n)
    succ = np.full(n, -np.inf)
    pol_f = np.zeros((T, n), dtype=np.int64)
    pol_s = np.zeros((T, n), dtype=np.int64)

    for t in range(T - 1, -1, -1):
        cf = R + gamma * fail[nxt]
        cf = np.where(gf, -np.inf, cf)
        cs = R + gamma * np.where(gf, 0.0, succ[nxt])
        pol_f[t], pol_s[t] = cf.argmax(1), cs.argmax(1)
        fail, succ = cf.max(1), cs.max(1)

    V_fail, V_succ = float(fail[0]), float(succ[0])

    def extract(pol, kind):
        if kind == "succ" and not np.isfinite(V_succ):
            return None
        i, steps, ret = 0, [], 0.0
        for t in range(T):
            a = int(pol[t, i])
            j, g = int(env.nxt[i, a]), bool(env.goal_flag[i, a])
            comps = {
                nm: float(table.C[i, a, k])
                for k, nm in enumerate(table.names)
                if abs(table.C[i, a, k]) > 1e-12
            }
            steps.append(Step(t, env.states[i], a, env.states[j], float(R[i, a]), comps, g))
            ret += (gamma ** t) * R[i, a]
            i = j
            if g:
                break
        return Trajectory(steps, success=(kind == "succ"), ret=float(ret))

    return V_succ, V_fail, extract(pol_s, "succ"), extract(pol_f, "fail")


def find_ablation_culprit(env: DoorKeyEnv, table: RewardTable, T: int = 30) -> Tuple[str, bool]:
    """Leave-one-component-out ablation on table.C to mathematically pinpoint culprit component."""
    Vs0, Vf0, _, tau_f = solve(env, table, T=T)
    base_gap = Vf0 - Vs0
    best_comp, best_reduction, is_verified = "", -float("inf"), False

    for k, nm in enumerate(table.names):
        R_sub = table.R - table.C[:, :, k]
        sub_tab = RewardTable(R=R_sub, C=table.C, names=table.names)
        Vs, Vf, _, _ = solve(env, sub_tab, T=T)
        if not np.isfinite(Vs):
            continue
        reduction = base_gap - (Vf - Vs)
        if reduction > best_reduction:
            best_reduction = reduction
            best_comp = nm
            thr = max(1e-4, EPS_MARGIN_REL * abs(Vs))
            is_verified = bool((Vs - Vf) > thr)

    if not best_comp and tau_f is not None:
        tot = tau_f.comp_totals()
        if tot:
            best_comp = max(tot, key=lambda k: tot[k])
    return best_comp, is_verified


def check_subgoal_shaping(env: DoorKeyEnv, table: RewardTable, tau_succ: Optional[Trajectory]) -> bool:
    """Verifies that the reward provides legitimate guidance (milestone or potential-based phi(s2) - phi(s))."""
    if tau_succ is None or len(tau_succ.steps) < 2:
        return False
    non_goal_steps = [st for st in tau_succ.steps if not st.goal]
    if not non_goal_steps:
        return False
    vals = [st.r for st in non_goal_steps]

    if max(vals) - min(vals) <= 1e-5:
        return False

    turn_or_wait_vals = [st.r for st in non_goal_steps if st.a in (LEFT, RIGHT)]
    baseline_step = (
        float(np.median(turn_or_wait_vals))
        if turn_or_wait_vals
        else float(np.percentile(vals, 25))
    )

    milestone_gains = [
        st.r - baseline_step for st in non_goal_steps if st.a in (PICKUP, TOGGLE, FORWARD)
    ]
    return bool(max(milestone_gains, default=0.0) > 1e-4)


def analyze(env: DoorKeyEnv, code: str, cache_table: bool = True) -> Analysis:
    try:
        table = build_reward_table(env, code)
    except RewardError as e:
        return Analysis(ok=False, error=str(e))

    Vs30, Vf30, tau_s30, tau_f30 = solve(env, table, T=env.horizon)
    Vs60, Vf60, _, _ = solve(env, table, T=2 * env.horizon)

    hacked_30 = bool(Vf30 > Vs30 + EPS_HACK)
    hacked_60 = bool(Vf60 > Vs60 + EPS_HACK)
    gap = (Vf30 - Vs30) / (abs(Vs30) + 1e-9) if np.isfinite(Vs30) else float("inf")

    shaping = check_subgoal_shaping(env, table, tau_s30)
    culprit, c_ver = (
        find_ablation_culprit(env, table, T=env.horizon)
        if (hacked_30 or hacked_60)
        else ("", False)
    )

    return Analysis(
        ok=True,
        V_succ_30=Vs30,
        V_fail_30=Vf30,
        V_succ_60=Vs60,
        V_fail_60=Vf60,
        hacked_30=hacked_30,
        hacked_60=hacked_60,
        gap_rel=gap,
        tau_succ=tau_s30,
        tau_fail=tau_f30,
        names=table.names,
        subgoal_shaping=shaping,
        culprit=culprit,
        culprit_verified=c_ver,
        signature=exploit_signature(tau_f30),
        table=table if cache_table else None,
    )


def dp_attack(env: DoorKeyEnv, table: RewardTable, rnd: int, classify: bool = True) -> List[Dict[str, Any]]:
    """Generates counterexample pairs using DP optimal policy against non-goal policy."""
    from cegr.synthesis.ce_pool import make_ce
    ces = []
    for T in (env.horizon, 2 * env.horizon):
        Vs, Vf, ts, tf = solve(env, table, T=T)
        if not np.isfinite(Vs) or ts is None or tf is None:
            continue
        typ = classify_pair(Vs, Vf, eps_rel=0.02, better_is_success=True)
        if typ is not None:
            better = [st.a for st in ts.steps]
            worse = [st.a for st in tf.steps]
            ces.append(make_ce(
                better=better,
                worse=worse,
                typ=typ if classify else "generic",
                rnd=int(rnd),
                tag=f"dp_attack_T{T}",
            ))
    return ces
