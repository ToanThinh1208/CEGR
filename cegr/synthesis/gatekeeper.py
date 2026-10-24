from typing import Any, Dict, List, Optional, Tuple
import collections
import numpy as np

from cegr.envs.doorkey import DoorKeyEnv, traj_return, traj_level, replay, ACTION_NAMES
from cegr.core.code_sandbox import build_reward_table, RewardError, RewardTable, check_code, make_namespace, check_rollout_exec
from cegr.core.tpe_evaluator import pair_margin, pair_violated, classify_pair
from cegr.solvers.dp_solver import dp_attack
from cegr.solvers.rl_probe import run_probe
from cegr.synthesis.ce_pool import CEPool, make_ce
from cegr.synthesis.static_antihack import static_checks, STATIC_MSG
from cegr.synthesis.signal_checker import signal_check, RHO_THR

GATE_NAMES = {
    1: "exec",
    2: "anti-hack (static)",
    3: "TPE (P0 + CE pool) + DP attacker",
    4: "signal",
    5: "short RL probe",
}

SHORT_PROBE_EPISODES = 1000
SHORT_PROBE_SEEDS = 2
PROBE_TAIL_THR = 0.50
M_CE_PER_ROUND = 2


def _rle_segments(toks: List[str], max_period: int = 3, min_rep: int = 2):
    out, i = [], 0
    while i < len(toks):
        best = None
        for p in range(1, max_period + 1):
            pat = toks[i:i + p]
            if len(pat) < p:
                continue
            r = 1
            while toks[i + r * p:i + (r + 1) * p] == pat:
                r += 1
            if r >= min_rep and (best is None or r * p > best[0] * best[1]):
                best = (r, p, pat)
        if best:
            r, p, pat = best
            out.append((r, pat))
            i += r * p
        else:
            out.append((1, [toks[i]]))
            i += 1
    return out


def trace_text(table: Any, env: Any, actions: List[int], max_chars: int = 330) -> str:
    idxs, used, _ = replay(env, actions)
    toks = []
    for i, a in zip(idxs, used):
        c = ",".join(
            f"{nm}{table.C[i, a, k]:+.2f}"
            for k, nm in enumerate(table.names)
            if abs(table.C[i, a, k]) > 1e-12
        )
        toks.append(f"{ACTION_NAMES[a]}[{c}]" if c else ACTION_NAMES[a])
    segs = _rle_segments(toks)
    s = ", ".join(f"({','.join(p)})x{r}" if r > 1 else p[0] for r, p in segs)
    return s if len(s) <= max_chars else s[:max_chars] + " ..."


def describe_pair(table: Any, env: Any, ce: Dict[str, Any]) -> str:
    if isinstance(env, DoorKeyEnv) and hasattr(table, "R"):
        Rb = traj_return(env, table, ce["better"])
        Rw = traj_return(env, table, ce["worse"])
        kind_map = {
            "hack": "HACKING",
            "under_signal": "UNDER-SIGNAL",
            "p0": "REFERENCE",
            "generic": "RANKING",
        }
        kind = kind_map.get(ce["type"], "RANKING")
        return (
            f"  [{kind}] must score HIGHER (your return {Rb:+.3f}): {trace_text(table, env, ce['better'])}\n"
            f"            than (your return {Rw:+.3f}): {trace_text(table, env, ce['worse'])}\n"
        )
    else:
        # Continuous domain formatting
        rb = ce.get("Rb", ce.get("ref_return", 0.0))
        rw = ce.get("Rw", ce.get("exploit_return", 0.0))
        b_steps = ce.get("better_steps") or (ce.get("better") if isinstance(ce.get("better"), list) and len(ce["better"]) > 0 and hasattr(ce["better"][0], "s") else None)
        w_steps = ce.get("worse_steps") or (ce.get("worse") if isinstance(ce.get("worse"), list) and len(ce["worse"]) > 0 and hasattr(ce["worse"][0], "s") else None)
        if callable(table) and b_steps and w_steps:
            def eval_st(steps):
                tot = 0.0
                for st in steps:
                    c = table(st.s, st.a, st.s2)
                    tot += sum(float(v) for v in c.values()) if isinstance(c, dict) else float(c)
                return tot
            try:
                rb = eval_st(b_steps)
                rw = eval_st(w_steps)
            except Exception:
                pass
        tag = ce.get("tag", "continuous_pair")
        typ = ce.get("type", "hack").upper()
        return (
            f"  [{typ}] Reference must score HIGHER (your return {rb:+.3f}) than exploit (your return {rw:+.3f}) on pair '{tag}'.\n"
        )


class GateKeeper:
    """Multi-Tier GateKeeper Verifier for CEGR (Gates 1 through 5)."""

    def __init__(self, env: Any, kit: Any):
        self.env = env
        self.kit = kit
        self._table_cache: Dict[str, Tuple[Optional[RewardTable], str]] = {}
        self._cont_p0: Optional[List[Dict[str, Any]]] = None
        self._cont_expert_steps: Optional[List[Any]] = None

    def get_table(self, code: str) -> Tuple[Optional[Any], str]:
        if code not in self._table_cache:
            if isinstance(self.env, DoorKeyEnv):
                try:
                    self._table_cache[code] = (build_reward_table(self.env, code), "")
                except RewardError as e:
                    self._table_cache[code] = (None, str(e))
                except Exception as e:
                    self._table_cache[code] = (None, str(e))
            else:
                try:
                    check_code(code)
                    ns = make_namespace(self.env)
                    exec(compile(code, "<gatekeeper_cont>", "exec"), ns)
                    fn = ns["reward_components"]
                    cont_p0 = getattr(self.kit, "p0", None) or self._cont_p0 or []
                    check_rollout_exec(self.env, fn, cont_p0)
                    self._table_cache[code] = (fn, "")
                except RewardError as e:
                    self._table_cache[code] = (None, str(e))
                except Exception as e:
                    self._table_cache[code] = (None, str(e))
        return self._table_cache[code]

    def observe(self, table: Any, rnd: int, attacker_enabled: bool = True) -> Dict[str, Any]:
        if not isinstance(self.env, DoorKeyEnv):
            from cegr.solvers.trajectory_attacker import ContinuousTrajectoryAttacker
            from cegr.envs.base_env import Step
            attacker = ContinuousTrajectoryAttacker(self.env)
            ref_steps = getattr(self.kit, "expert_steps", None)
            if not ref_steps:
                if self._cont_expert_steps is None:
                    if hasattr(self.env, "get_milestone_trajectories"):
                        m_trajs = self.env.get_milestone_trajectories()
                        t0_steps, t1_steps, t2_steps, t3_steps = m_trajs[0], m_trajs[1], m_trajs[2], m_trajs[3]
                    else:
                        def rollout_actions(action_list):
                            s, _ = self.env.reset(seed=42)
                            steps = []
                            for t, a in enumerate(action_list):
                                s2, r, done, trunc, _ = self.env.step(a)
                                g = bool(s2.get("in_target", False))
                                steps.append(Step(t=t, s=s, a=a, s2=s2, r=1.0 if g else 0.0, comps={"goal": 1.0 if g else 0.0}, goal=g))
                                s = s2
                                if g or done:
                                    break
                            return steps

                        reach_acts = [np.array([0.0, 0.5, -0.7, 0.0], dtype=np.float32) for _ in range(10)]
                        grasp_acts = [np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32) for _ in range(10)]
                        pull_acts = [np.array([1.0, 0.0, 0.55, -1.0], dtype=np.float32) for _ in range(10)]
                        idle_grasp = [np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32) for _ in range(30)]

                        t0_actions = [np.array([-0.5, -0.5, 0.2, 0.0], dtype=np.float32) for _ in range(50)]
                        t1_actions = reach_acts + [np.zeros(4, dtype=np.float32) for _ in range(40)]
                        t2_actions = reach_acts + grasp_acts + idle_grasp
                        t3_actions = reach_acts + grasp_acts + pull_acts

                        t0_steps = rollout_actions(t0_actions)
                        t1_steps = rollout_actions(t1_actions)
                        t2_steps = rollout_actions(t2_actions)
                        t3_steps = rollout_actions(t3_actions)

                    self._cont_expert_steps = t3_steps
                    self._cont_p0 = [
                        make_ce(t3_steps, t2_steps, "p0", 0, "T3>T2", better_steps=t3_steps, worse_steps=t2_steps),
                        make_ce(t2_steps, t1_steps, "p0", 0, "T2>T1", better_steps=t2_steps, worse_steps=t1_steps),
                        make_ce(t1_steps, t0_steps, "p0", 0, "T1>T0", better_steps=t1_steps, worse_steps=t0_steps),
                        make_ce(t3_steps, t0_steps, "p0", 0, "T3>T0", better_steps=t3_steps, worse_steps=t0_steps),
                    ]
                ref_steps = self._cont_expert_steps

            p0_viol = []
            cont_p0 = getattr(self.kit, "p0", None) or self._cont_p0 or []
            from cegr.core.tpe_evaluator import pair_returns
            for p in cont_p0:
                rb, rw = pair_returns(self.env, table, p)
                typ = classify_pair(rb, rw, eps_rel=0.02)
                if typ is not None:
                    p0_viol.append((rb - rw, make_ce(p["better"], p["worse"], typ, rnd, "p0:" + p.get("tag", ""),
                                              better_steps=p.get("better_steps"), worse_steps=p.get("worse_steps"))))
            p0_viol.sort(key=lambda x: x[0])
            attack = attacker.attack(table, ref_steps, rnd, classify=True) if attacker_enabled else []
            return {"eps": 0.02, "p0_viol_all": [c for _, c in p0_viol], "attack": attack}

        ret_t3 = traj_return(self.env, table, self.kit.refs["T3"])
        eps = max(1e-3, 0.02 * abs(ret_t3))
        p0_viol = []
        for p in self.kit.p0:
            rb = traj_return(self.env, table, p["better"])
            rw = traj_return(self.env, table, p["worse"])
            typ = classify_pair(rb, rw, eps_rel=0.02)
            if typ is not None:
                p0_viol.append((rb - rw, make_ce(p["better"], p["worse"], typ, rnd, "p0:" + p.get("tag", ""))))
        p0_viol.sort(key=lambda x: x[0])
        attack = dp_attack(self.env, table, rnd, classify=True) if attacker_enabled else []
        return {"eps": eps, "p0_viol_all": [c for _, c in p0_viol], "attack": attack}


def run_gatekeeper(
    code: str,
    cfg: Dict[str, Any],
    pool: CEPool,
    rnd: int,
    env: Any,
    kit: Any,
    gk: GateKeeper,
    seed: int = 42,
) -> Dict[str, Any]:
    res: Dict[str, Any] = dict(
        passed=False,
        failed_gate=None,
        gates_passed=0,
        error="",
        table=None,
        new_ces=[],
        violated=[],
        info={},
        rl_runs=0,
        rl_episodes=0,
        n_viol=0,
        obs=None,
        probe_tail=0.0,
    )
    gates = cfg.get("gates", [1, 2, 3, 4, 5])

    # ---- Gate 1: Execution ----
    table, err = gk.get_table(code)
    if table is None:
        res.update(failed_gate=1, error=err)
        return res
    res["table"] = table
    if 1 in gates:
        res["gates_passed"] += 1

    obs = gk.observe(table, rnd, attacker_enabled=cfg.get("attacker", True))
    res["obs"] = obs
    eps = obs["eps"]

    def fail(g, new_ces=None, **info):
        res["failed_gate"] = g
        res["new_ces"] = (new_ces or [])[:M_CE_PER_ROUND + 2]
        res["info"].update(info)
        return res

    g2_mode = cfg.get("g2_mode", "merge")
    static_bad = []

    # Check Gate 2 (Static Anti-Hack)
    if 2 in gates:
        static_bad = static_checks(code, table, env)
        res["static_bad"] = static_bad
        if static_bad and g2_mode == "hard":
            res["gates_passed"] = 1
            return fail(2, new_ces=[], static=static_bad)
        if static_bad and g2_mode == "soft":
            res["info"]["static_warn"] = static_bad

    # Check Gate 3 (TPE & Attacker)
    if 3 in gates:
        viol = list(obs["p0_viol_all"])
        pool_viol = [
            c for c in (pool.items if cfg.get("pool", True) else [])
            if pair_violated(pair_margin(env, table, c), c, eps, cfg.get("classify", True), mode="margin", env=env, table=table)
        ]
        res["n_viol"] = len(viol) + len(pool_viol) + len(obs["attack"])
        g3_failed = bool(viol or pool_viol or (cfg.get("attacker", True) and obs["attack"]))

        if g3_failed:
            typed = [
                c if cfg.get("classify", True) else dict(c, type="generic", key=c["key"] + "g")
                for c in viol[:M_CE_PER_ROUND]
            ]
            atk = obs["attack"] if cfg.get("attacker", True) else []
            if not cfg.get("classify", True):
                atk = [dict(c, type="generic") for c in atk]
            # Merge Mode: if G2 also failed, gates_passed stops at 1; if G2 passed, gates_passed = 2
            res["gates_passed"] = 1 if (static_bad and g2_mode == "merge") else (2 if 2 in gates else 1)
            return fail(
                3,
                new_ces=atk + typed,
                tpe_violated=(pool_viol + typed)[:4],
                n_p0_viol=len(viol),
                n_pool_viol=len(pool_viol),
                attack=[c["tag"] for c in atk],
                static=(static_bad if g2_mode == "merge" else []),
            )

        # G3 passed! In Merge Mode, if G2 failed, stop at G2 without adding new CE:
        if static_bad and g2_mode == "merge":
            res["gates_passed"] = 1
            return fail(2, new_ces=[], static=static_bad)

        # Both G2 and G3 passed
        res["gates_passed"] = 3 if 2 in gates else 2
    elif 2 in gates:
        if static_bad and g2_mode != "soft":
            res["gates_passed"] = 1
            return fail(2, new_ces=[], static=static_bad)
        res["gates_passed"] = 2

    # Gate 4: Signal & Correlation
    if 4 in gates:
        if isinstance(env, DoorKeyEnv):
            ok, s_info, ces = signal_check(env, kit, table, rnd, classify=cfg.get("classify", True))
            if not ok:
                return fail(4, new_ces=ces, signal=s_info)
        res["gates_passed"] += 1

    # Gate 5: Short RL Probe
    if 5 in gates:
        if not isinstance(env, DoorKeyEnv):
            from cegr.solvers.continuous_rl_probe import ContinuousRLProbe
            probe = ContinuousRLProbe(env)
            pr = probe.probe(table, total_steps=20000, eval_episodes=5, seed=seed)
            res["rl_runs"] += 1
            res["rl_episodes"] += pr.get("total_steps", 20000)
            res["probe_tail"] = pr["tail"]
            if pr["tail"] < 0.40:
                w_steps = pr.get("failing_steps")
                b_steps = getattr(kit, "expert_steps", None)
                ces_g5 = []
                if w_steps and b_steps:
                    ces_g5.append(make_ce(b_steps, w_steps, "hack" if cfg.get("classify", True) else "generic", rnd, "probe_cont_fail", better_steps=b_steps, worse_steps=w_steps))
                return fail(
                    5,
                    new_ces=ces_g5,
                    probe={
                        "tail": pr["tail"],
                        "greedy_return": pr.get("mean_eval_return", 0.0),
                        "goal_return": 1.0,
                    },
                )
        else:
            ep = cfg.get("probe_episodes", SHORT_PROBE_EPISODES)
            sd = cfg.get("probe_seeds", SHORT_PROBE_SEEDS)
            pr = run_probe(env, table, ep, sd, seed_base=seed + 100, gamma=1.0)
            res["rl_runs"] += 1
            res["rl_episodes"] += ep * sd
            res["probe_tail"] = pr["tail"]
            if pr["tail"] < PROBE_TAIL_THR:
                fin = next((f for f in pr["finals"] if not f[1]), pr["finals"][0])
                acts = fin[0]
                lv = traj_level(env, acts)
                rg = traj_return(env, table, acts)
                r3 = traj_return(env, table, kit.refs["T3"])
                better = kit.refs[f"T{min(3, lv + 1)}"]
                typ = classify_pair(traj_return(env, table, better), rg, eps_rel=0.02) or "under_signal"
                ce = make_ce(better, acts, typ if cfg.get("classify", True) else "generic", rnd, f"probe:L{lv}")
                return fail(
                    5,
                    new_ces=[ce],
                    probe={
                        "tail": pr["tail"],
                        "level": lv,
                        "greedy_return": rg,
                        "goal_return": r3,
                        "actions": acts,
                    },
                )
        res["gates_passed"] += 1

    res["passed"] = True
    return res


def build_feedback(res: Dict[str, Any], cfg: Dict[str, Any], pool: CEPool, env: Any, kit: Any) -> str:
    if res["failed_gate"] == 1:
        return (
            f"SANDBOX EXECUTION ERROR in your previous code:\n`{res['error']}`\n"
            "Ensure `reward_components(s, a, s2)` returns a dict of finite numbers without errors."
        )

    table = res["table"]
    g = res["failed_gate"]
    info = res.get("info", {})
    head = f"VERIFIER REPORT: Candidate FAILED gate {g} ({GATE_NAMES.get(g, f'Gate {g}')}).\n"

    gate_msg = ""
    if g == 2:
        gate_msg = "Static anti-hack check FAILED:\n" + "".join(
            f"  - {STATIC_MSG.get(x.split(':')[0], x)} ({x})\n" for x in info.get("static", [])
        )
    elif g == 3:
        parts = []
        if info.get("n_p0_viol") or info.get("n_pool_viol"):
            parts.append(f"TPE ranking test FAILED on {info.get('n_p0_viol', 0)} reference pair(s) and {info.get('n_pool_viol', 0)} pool counterexample(s).")
        if info.get("attack"):
            parts.append(f"DP attacker found a superior non-goal policy ({', '.join(info['attack'])}).")
        gate_msg = " ".join(parts) + "\n"
    elif g == 4:
        s = info.get("signal", {})
        gate_msg = (
            f"Signal check FAILED: flat/narrow stages={s.get('flat_pairs')}, "
            f"rank correlation rho={s.get('spearman', 0.0):.2f} (need >= {RHO_THR}).\n"
        )
    elif g == 5:
        p = info.get("probe", {})
        gate_msg = (
            f"Short RL probe FAILED: tail success rate {p.get('tail', 0.0):.2f} (need >= {PROBE_TAIL_THR}). "
            f"Greedy return={p.get('greedy_return', 0.0):+.2f} vs goal return={p.get('goal_return', 0.0):+.2f}.\n"
        )

    # Counterexamples from pool & newly generated
    ce_lines = []
    if cfg.get("pool", True):
        shown = list(info.get("tpe_violated", [])) + list(res.get("new_ces", []))
        seen_keys = set()
        picked = []
        for c in shown[::-1]:
            if c["key"] in seen_keys:
                continue
            seen_keys.add(c["key"])
            picked.append(c)
            if len(picked) >= 3:
                break
        if picked:
            ce_lines.append(f"\nCOUNTEREXAMPLE POOL ({len(pool)} stored items). Must satisfy:\n")
            ce_lines.extend(describe_pair(table, env, c) for c in picked)

    return head + gate_msg + "".join(ce_lines)
