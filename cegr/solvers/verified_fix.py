import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cegr.envs.doorkey import DoorKeyEnv
from cegr.core.code_sandbox import ast_node_count, RewardTable
from cegr.solvers.dp_solver import Analysis, analyze, find_ablation_culprit, check_subgoal_shaping
from cegr.solvers.rl_probe import learnability_probe

culprit_ablation_oracle = find_ablation_culprit
subgoal_shaping_oracle = check_subgoal_shaping

SCORING_PROFILE = "relaxed"
KEYPOS_MODE = "agent_pos"
EPS_MARGIN_REL = 0.01
PROBE_SEEDS = 3
SEED = 3000

PROFILES = {
    "strict_v1": dict(
        probe_episodes=1200, tail_thr=0.66, probe_gamma=0.99,
        optimistic_init=0.0, alpha_decay=False, require_dense=True
    ),
    "relaxed": dict(
        probe_episodes=2500, tail_thr=0.50, probe_gamma=1.0,
        optimistic_init=1.0, alpha_decay=True, require_dense=True
    ),
}

_PROBE_CACHE: Dict[Tuple[str, Tuple], Any] = {}


def _table_key(table: RewardTable) -> str:
    return hashlib.md5(table.R.tobytes()).hexdigest()


def cached_learnability_probe(env: Any, table: RewardTable, **kw) -> List[Tuple[int, int]]:
    """learnability_probe có cache theo bảng reward (các bản sửa trùng nhau không train lại)."""
    key = (_table_key(table), tuple(sorted(kw.items())))
    if key not in _PROBE_CACHE:
        res = learnability_probe(env, table, **kw)
        if isinstance(res, tuple):
            res = res[0]  # take curve
        _PROBE_CACHE[key] = res
    return _PROBE_CACHE[key]


def evaluate_verified_fix(env: Any, orig_an: Optional[Analysis], orig_code: str,
                          new_code: str, run_probe: bool = True,
                          profile_name: Optional[str] = None,
                          profile_cfg: Optional[dict] = None,
                          new_an: Optional[Analysis] = None) -> dict:
    """
    Chấm điểm bản sửa theo profile cấu hình và báo cáo song song 2 tầng:
    - tier1_safe_aligned: loose_dp VÀ probe >= tail_thr (cho phép cả degenerate_sparse)
    - tier2_dense_vf    : loose_dp VÀ subgoal_shaping VÀ probe >= tail_thr
    Hỗ trợ cả môi trường Tabular DP (DoorKey) và Continuous Rollout (Meta-World).
    """
    p_name = profile_name or SCORING_PROFILE
    cfg = dict(profile_cfg or PROFILES[p_name])
    probe_episodes  = int(cfg.get("probe_episodes", 2500))
    tail_thr        = float(cfg.get("tail_thr", 0.50))
    probe_gamma     = float(cfg.get("probe_gamma", 1.0))
    optimistic_init = float(cfg.get("optimistic_init", 1.0))
    alpha_decay     = bool(cfg.get("alpha_decay", True))
    require_dense   = bool(cfg.get("require_dense", True))

    # --- Miền liên tục (Meta-World) ---
    if not isinstance(env, DoorKeyEnv):
        res = {
            "ok": True, "error": "",
            "dp_aligned_30": False, "dp_aligned_60": False, "loose_dp": False,
            "subgoal_shaping": True, "probe_success_rate": 0.0, "probe_aulc": 0.0,
            "tier1_safe_aligned": False,
            "tier2_dense_vf": False,
            "verified_fix": False,
            "category": "syntax_runtime_error",
            "culprit_targeted": False,
            "ast_diff": abs(ast_node_count(new_code) - ast_node_count(orig_code)),
            "margin_30": float("nan"), "margin_60": float("nan"),
            "analysis": None, "curve": [],
            "probe_params": {"domain": "metaworld_continuous"}
        }
        try:
            from cegr.core.code_sandbox import check_code, make_namespace
            check_code(new_code)
            ns = make_namespace(env)
            exec(compile(new_code, "<eval_cont>", "exec"), ns)
            fn = ns["reward_components"]
        except Exception as e:
            res.update(ok=False, error=str(e), category="syntax_runtime_error")
            return res

        from cegr.solvers.continuous_rl_probe import ContinuousRLProbe
        from cegr.solvers.trajectory_attacker import ContinuousTrajectoryAttacker
        from cegr.envs.base_env import Step

        # Run continuous RL probe (Oracle stream 3000)
        probe = ContinuousRLProbe(env)
        pr_res = probe.probe(fn, total_steps=25000, eval_episodes=5, seed=SEED)
        tail = float(pr_res.get("tail", 0.0))
        res["probe_success_rate"] = tail
        res["probe_aulc"] = float(np.mean(pr_res.get("curve", [tail])))

        # Run continuous trajectory attacker with extended budget
        attacker = ContinuousTrajectoryAttacker(env, cem_iterations=10, population_size=40)
        ref_steps = []
        s, _ = env.reset(seed=SEED)
        for t in range(50):
            a = np.array([0.0, 0.1, -0.05, -0.8], dtype=np.float32)
            s2, _, done, trunc, _ = env.step(a)
            g = bool(s2.get("in_target", False))
            ref_steps.append(Step(t=t, s=s, a=a, s2=s2, r=1.0 if g else 0.0, comps={"goal": 1.0 if g else 0.0}, goal=g))
            s = s2
            if g or done:
                break

        # Calculate expert return
        ref_return = 0.0
        for st in ref_steps:
            c = fn(st.s, st.a, st.s2)
            ref_return += sum(float(v) for v in c.values()) if isinstance(c, dict) else float(c)

        ces = attacker.attack(fn, ref_steps, rnd=0)
        r_exploit_max = max([float(ce.get("exploit_return", -np.inf)) for ce in ces], default=-np.inf)
        if r_exploit_max == -np.inf:
            margin_ok = True
        else:
            # Meta-World Oracle: R_expert - R_exploit_max > 0.01 * |R_expert| (Checklist 18)
            margin_ok = bool((ref_return - r_exploit_max) > EPS_MARGIN_REL * abs(ref_return))

        hacked = (not margin_ok) or (len(ces) > 0 and any(ce.get("type") == "hack" for ce in ces))
        res["dp_aligned_30"] = not hacked
        res["dp_aligned_60"] = not hacked
        res["loose_dp"] = not hacked
        res["margin_30"] = ref_return - r_exploit_max if r_exploit_max != -np.inf else 1.0
        res["margin_60"] = res["margin_30"]

        # Subgoal shaping check for Meta-World milestones
        subgoal_ok = True
        if hasattr(env, "get_milestone_trajectories"):
            m_trajs = env.get_milestone_trajectories()
            m_returns = []
            for m_t in m_trajs:
                r_m = 0.0
                for st in m_t:
                    c = fn(st.s, st.a, st.s2)
                    r_m += sum(float(v) for v in c.values()) if isinstance(c, dict) else float(c)
                m_returns.append(r_m)
            subgoal_ok = all(m_returns[i] < m_returns[i+1] for i in range(len(m_returns)-1))
        res["subgoal_shaping"] = subgoal_ok

        probe_passed = bool(tail >= tail_thr)
        res["tier1_safe_aligned"] = bool(res["loose_dp"] and probe_passed)
        res["tier2_dense_vf"] = bool(res["tier1_safe_aligned"] and subgoal_ok)
        res["verified_fix"] = bool(res["tier2_dense_vf"] if require_dense else res["tier1_safe_aligned"])
        res["category"] = "verified_fix" if res["verified_fix"] else ("unlearnable_shaping" if not probe_passed else "new_exploit_induced")
        return res

    # --- Miền rời rạc (DoorKey) ---
    if new_an is None:
        new_an = analyze(env, new_code, cache_table=True)

    probe_params_log = {
        "scoring_profile": p_name,
        "keypos_mode": KEYPOS_MODE,
        "probe_episodes": probe_episodes,
        "tail_thr": tail_thr,
        "probe_gamma": probe_gamma,
        "optimistic_init": optimistic_init,
        "alpha_decay": alpha_decay,
        "require_dense": require_dense,
        "probe_seeds": PROBE_SEEDS,
    }

    res = {
        "ok": new_an.ok, "error": new_an.error,
        "dp_aligned_30": False, "dp_aligned_60": False, "loose_dp": False,
        "subgoal_shaping": False, "probe_success_rate": 0.0, "probe_aulc": 0.0,
        "tier1_safe_aligned": False,
        "tier2_dense_vf": False,
        "verified_fix": False,
        "category": "syntax_runtime_error",
        "culprit_targeted": False,
        "ast_diff": abs(ast_node_count(new_code) - ast_node_count(orig_code)),
        "margin_30": float("nan"), "margin_60": float("nan"),
        "analysis": new_an, "curve": [],
        "probe_params": probe_params_log
    }
    if not new_an.ok:
        return res

    m30 = new_an.V_succ_30 - new_an.V_fail_30
    m60 = new_an.V_succ_60 - new_an.V_fail_60
    res["margin_30"], res["margin_60"] = m30, m60
    res["dp_aligned_30"] = bool(m30 > max(1e-4, EPS_MARGIN_REL * abs(new_an.V_succ_30)))
    res["dp_aligned_60"] = bool(m60 > max(1e-4, EPS_MARGIN_REL * abs(new_an.V_succ_60)))
    res["loose_dp"]      = res["dp_aligned_30"] and res["dp_aligned_60"]
    res["subgoal_shaping"] = new_an.subgoal_shaping

    if orig_an is not None:
        other_orig = [c for c in orig_an.names if c != orig_an.culprit]
        kept_others = sum(1 for c in other_orig if c in new_an.names)
        res["culprit_targeted"] = bool((kept_others >= max(1, len(other_orig) // 2)) and res["loose_dp"])
    else:
        res["culprit_targeted"] = False

    if not res["loose_dp"]:
        if orig_an is not None and new_an.signature == orig_an.signature:
            res["category"] = "same_exploit_persists"
        else:
            res["category"] = "new_exploit_induced"
        return res

    # 1c. Chạy Q-Learning Probe cho TẤT CẢ các mẫu đạt loose_dp (kể cả degenerate_sparse) để tính Tier 1
    probe_passed = True
    if run_probe and new_an.table is not None:
        curves = [
            cached_learnability_probe(      # [CARD-CEGIS] có cache theo bảng reward
                env, new_an.table,
                episodes=probe_episodes,
                gamma=probe_gamma,
                alpha=0.35,
                optimistic_init=optimistic_init,
                alpha_decay=alpha_decay,
                seed=SEED + s
            )
            for s in range(PROBE_SEEDS)
        ]
        mean_curve = [
            (curves[0][idx][0], float(np.mean([curves[s][idx][1] for s in range(PROBE_SEEDS)])))
            for idx in range(len(curves[0]))
        ]
        res["curve"] = mean_curve
        tail_rate = float(np.mean([pt[1] for pt in mean_curve[-3:]])) if len(mean_curve) >= 3 else float(np.mean([pt[1] for pt in mean_curve]))
        aulc      = float(np.mean([pt[1] for pt in mean_curve]))
        res["probe_success_rate"] = tail_rate
        res["probe_aulc"]         = aulc
        probe_passed = bool(tail_rate >= tail_thr)

    # Ghi nhận 2 tầng song song
    res["tier1_safe_aligned"] = bool(res["loose_dp"] and probe_passed)
    res["tier2_dense_vf"]     = bool(res["loose_dp"] and new_an.subgoal_shaping and probe_passed)

    # Phân loại category và verified_fix (ưu tiên kiểm tra subgoal_shaping trước để giữ nguyên nhãn degenerate_sparse)
    if not new_an.subgoal_shaping:
        res["category"]     = "degenerate_sparse"
        res["verified_fix"] = res["tier1_safe_aligned"] if not require_dense else False
    elif probe_passed:
        res["category"]     = "verified_fix"
        res["verified_fix"] = True
    else:
        res["category"]     = "unlearnable_shaping"
        res["verified_fix"] = False

    return res
