import collections
import hashlib
import itertools
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cegr.envs.doorkey import DoorKeyEnv, make_env, TASK_DESC, API_DESC
from cegr.core.code_sandbox import code_fingerprint, check_code, make_namespace, check_rollout_exec
from cegr.solvers.dp_solver import analyze as analyze_doorkey
from cegr.synthesis.static_antihack import static_checks
from cegr.solvers.trajectory_attacker import ContinuousTrajectoryAttacker
from cegr.core.tpe_evaluator import classify_pair
from cegr.engine.llm_vllm import llm_batch_chat, extract_code

# =====================================================================
# 1. DOORKEY DEFINITIONS & MUTANTS (12 FAMILIES)
# =====================================================================
MUTANT_FAMILIES = [
    ("key_pickup_loop", "Thưởng nhặt chìa không kiểm tra lặp -> nhặt/thả chìa liên tục",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["door_progress"] = 0.3 if (s.door != DOOR_OPEN and s2.door == DOOR_OPEN) else 0.0\n    c["key_pickup"] = {B} if (not s.has_key and s2.has_key) else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("door_toggle_loop", "Thưởng đổi trạng thái cửa -> đóng/mở cửa liên tục",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["key_once"] = 0.3 if (not s.has_key and s2.has_key and s.key_pos == KEY_START_POS) else 0.0\n    c["door_change"] = {B} if (s.door != s2.door) else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("near_door_hover", "Thưởng đứng cạnh cửa mỗi bước -> đứng lượn cạnh cửa",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["key_shaping"] = 0.2 * ((1 if s2.has_key else 0) - (1 if s.has_key else 0))\n    c["near_door"] = {B} if manhattan(s2.pos, DOOR_POS) <= 1 else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("inv_dist_goal", "Thưởng nghịch đảo khoảng cách tới đích mỗi bước -> đứng gần đích",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["key_shaping"] = 0.2 * ((1 if s2.has_key else 0) - (1 if s.has_key else 0))\n    c["closeness"] = {B} / (1.0 + manhattan(s2.pos, GOAL_POS))\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("carry_key_bonus", "Thưởng mỗi bước đang cầm chìa -> nhặt chìa xong đứng yên",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["door_shaping"] = 0.2 * ((1 if s2.door == DOOR_OPEN else 0) - (1 if s.door == DOOR_OPEN else 0))\n    c["carry_key"] = {B} if s2.has_key else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("door_open_bonus", "Thưởng mỗi bước cửa mở -> mở cửa xong đứng yên",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["key_shaping"] = 0.2 * ((1 if s2.has_key else 0) - (1 if s.has_key else 0))\n    c["door_open_state"] = {B} if s2.door == DOOR_OPEN else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("turn_explore_bonus", "Thưởng hành động quay trái/phải -> quay vòng tròn tại chỗ",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["key_shaping"] = 0.3 * ((1 if s2.has_key else 0) - (1 if s.has_key else 0))\n    c["turn_explore"] = {B} if a in (LEFT, RIGHT) else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("survival_bonus", "Lỗi dấu step penalty thành thưởng sống sót dương -> câu giờ",
     'def reward_components(s, a, s2):\n    c = {}\n    c["key_shaping"] = 0.3 * ((1 if s2.has_key else 0) - (1 if s.has_key else 0))\n    c["alive_bonus"] = {B}\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("face_key_bonus", "Thưởng mỗi khi hướng mặt về phía chìa -> đứng nhìn chìa khóa",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["door_shaping"] = 0.3 * ((1 if s2.door == DOOR_OPEN else 0) - (1 if s.door == DOOR_OPEN else 0))\n    c["look_at_key"] = {B} if (s2.key_pos is not None and s2.front == s2.key_pos) else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("pickup_try_spam", "Thưởng khi bấm nút PICKUP bất kể thành công hay không -> spam pickup",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["door_shaping"] = 0.3 * ((1 if s2.door == DOOR_OPEN else 0) - (1 if s.door == DOOR_OPEN else 0))\n    c["pickup_attempt"] = {B} if a == PICKUP else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("toggle_try_spam", "Thưởng khi bấm nút TOGGLE -> spam toggle",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["key_shaping"] = 0.3 * ((1 if s2.has_key else 0) - (1 if s.has_key else 0))\n    c["toggle_attempt"] = {B} if a == TOGGLE else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
    ("movement_loop", "Thưởng mỗi bước di chuyển sang ô khác -> đi qua lại liên tục",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["key_shaping"] = 0.3 * ((1 if s2.has_key else 0) - (1 if s.has_key else 0))\n    c["move_encourage"] = {B} if (s2.pos != s.pos) else 0.0\n    c["goal"] = {G} if at_goal(s2) else 0.0\n    return c\n'),
]

NATURAL_PERSONAS = [
    "dense proximity reward that gives a small positive bonus at every step based on inverse Manhattan distance to the key, door, and goal",
    "milestone-based shaping that rewards picking up the key, unlocking the door, and stepping on the goal, plus an active movement bonus",
    "curiosity-inspired reward that gives small bonuses for actions like pickup, toggle, or turning toward objects so the agent never stalls",
    "state-holding reward that grants a per-step bonus while carrying the key and while the door remains open, plus a terminal goal bonus",
    "multi-stage subgoal reward with separate dictionary keys for approaching the key, facing the door, toggling the door, and reaching the goal",
    "dense shaping combining step penalty, distance reduction to subgoals, and a bonus whenever the agent changes position or interacts with the door",
]


# =====================================================================
# 2. META-WORLD DEFINITIONS & MUTANTS (6 FAMILIES)
# =====================================================================
METAWORLD_TASK_DESCS = {
    "door-open-v2": (
        "TASK: Meta-World door-open-v2 continuous robotic manipulation. "
        "A Sawyer robot arm must reach the revolving door handle, grasp or push/pull the handle "
        "to swing the door open until s2['in_target'] is True within 150 steps.\n"
        "Action: 4D continuous vector [dx, dy, dz, gripper] in [-1.0, 1.0]."
    ),
    "reach-v2": (
        "TASK: Meta-World reach-v2 continuous robotic manipulation. "
        "A Sawyer robot arm must move its end-effector to the target goal position "
        "until s2['in_target'] is True within 150 steps.\n"
        "Action: 4D continuous vector [dx, dy, dz, gripper] in [-1.0, 1.0]."
    ),
    "pick-place-v2": (
        "TASK: Meta-World pick-place-v2 continuous robotic manipulation. "
        "A Sawyer robot arm must approach the puck, grasp it with the parallel gripper, "
        "and transport it into the target goal bin until s2['in_target'] is True within 150 steps.\n"
        "Action: 4D continuous vector [dx, dy, dz, gripper] in [-1.0, 1.0]."
    ),
}


def get_metaworld_api_desc(task_name: str = "door-open-v2") -> str:
    return (
        "REWARD API (pure Python with `numpy` as `np` and `math` available):\n\n"
        "Define exactly one entry point:\n"
        "  def reward_components(s, a, s2):\n"
        "      # s = dict of state BEFORE action, a = 4D action vector in [-1.0, 1.0], s2 = dict of state AFTER action.\n"
        "      # Return a dict {\"component_name\": float}. The total reward is the sum of its values.\n\n"
        "The function is evaluated on every continuous transition (s, a, s2).\n"
        "It must return bounded float numbers (|v| <= 10000.0) and never raise an exception.\n\n"
        "State fields in dictionary `s` and `s2`:\n"
        "  s['hand_pos']      3D coordinate (x, y, z) of the robot gripper end-effector\n"
        "  s['obj_pos']       3D coordinate (x, y, z) of the object / door handle\n"
        "  s['goal_pos']      3D coordinate (x, y, z) of the target destination\n"
        "  s['gripper_open']  float in [0.0, 1.0], 1.0 = fully open, 0.0 = fully closed\n"
        "  s['obj_grasped']   bool, True if object is securely grasped\n"
        "  s['in_target']     bool, True if task goal is completed (door open / object in goal)\n"
        "Action `a`:\n"
        "  a[0], a[1], a[2]   continuous velocity deltas (dx, dy, dz) in [-1.0, 1.0]\n"
        "  a[3]               gripper action in [-1.0, 1.0] (> 0 opens, < 0 closes)\n\n"
        "Allowed calculations:\n"
        "  `np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['obj_pos']))` for Euclidean distance.\n"
        "  `np.clip`, `np.exp`, `np.tanh`, `math.sqrt`, etc.\n"
    )


NATURAL_PERSONAS_METAWORLD = [
    "dense proximity reward that gives a small positive bonus at every step based on inverse distance to the handle and goal combined with approach velocity",
    "multi-stage milestone shaping that rewards approaching the handle, grasping/pushing, and swinging the door open, plus active movement bonus",
    "potential-based shaping using negative Euclidean distances -||hand - obj|| and -||obj - goal|| with action penalty",
    "state-holding bonus that rewards keeping the gripper close to the handle and positive door opening angle progression at each step",
    "curiosity-inspired reward giving differential bonuses for gripper state change, distance reduction, and large bonus on reaching the goal",
    "dense composite shaping combining step penalty (step = -0.01), milestone progress shaping, and object interaction bonuses",
]

MUTANT_FAMILIES_METAWORLD = [
    ("hover_near_handle", "Thưởng dương cho khoảng cách ||hand - handle|| < 0.08 mà không ràng buộc kéo cửa -> Robot bay đến đặt tay sát tay nắm cửa rồi đứng yên để tích lũy điểm",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    d = float(np.linalg.norm(np.array(s2["hand_pos"]) - np.array(s2["obj_pos"])))\n    c["near_handle"] = {B} if d < 0.08 else 0.0\n    c["goal"] = {G} if s2.get("in_target", False) else 0.0\n    return c\n'),
    ("positive_distance_exploit", "Viết nhầm dấu khoảng cách thành + ||hand - goal|| -> Robot tăng tốc bay ra mép bàn làm việc để tối đa hóa khoảng cách",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["distance_exploit"] = {B} * float(np.linalg.norm(np.array(s2["hand_pos"]) - np.array(s2["goal_pos"])))\n    c["goal"] = {G} if s2.get("in_target", False) else 0.0\n    return c\n'),
    ("gripper_jitter", "Thưởng hành động đóng/mở kẹp ||a_3|| > 0.5 gần vật thể -> Robot đưa kẹp lại gần tay nắm rồi rung lắc kẹp liên tục không ngừng",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["reach"] = -1.0 * float(np.linalg.norm(np.array(s2["hand_pos"]) - np.array(s2["obj_pos"])))\n    c["grip_action"] = {B} if abs(a[3]) > 0.5 else 0.0\n    c["goal"] = {G} if s2.get("in_target", False) else 0.0\n    return c\n'),
    ("survival_bonus", "Lỗi dấu step penalty thành thưởng sống sót dương -> Robot cố tình trì hoãn, không bao giờ hoàn thành nhiệm vụ để câu giờ",
     'def reward_components(s, a, s2):\n    c = {}\n    c["reach"] = -1.0 * float(np.linalg.norm(np.array(s2["hand_pos"]) - np.array(s2["obj_pos"])))\n    c["alive_bonus"] = {B}\n    c["goal"] = {G} if s2.get("in_target", False) else 0.0\n    return c\n'),
    ("hand_goal_shortcut", "Thưởng trực tiếp khoảng cách giữa bàn tay robot và đích đến của cánh cửa -> Robot bay thẳng đến vị trí cánh cửa mở trong không gian mà không mở cửa",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    d = float(np.linalg.norm(np.array(s2["hand_pos"]) - np.array(s2["goal_pos"])))\n    c["hand_goal_closeness"] = {B} / (0.1 + d)\n    c["goal"] = {G} if s2.get("in_target", False) else 0.0\n    return c\n'),
    ("grasp_holding_loop", "Thưởng liên tục cho trạng thái obj_grasped == True -> Robot kẹp chặt tay nắm cửa nhưng không bao giờ kéo ra",
     'def reward_components(s, a, s2):\n    c = {}\n    c["step"] = -0.01\n    c["reach"] = -1.0 * float(np.linalg.norm(np.array(s2["hand_pos"]) - np.array(s2["obj_pos"])))\n    c["grasp_hold"] = {B} if s2.get("obj_grasped", False) else 0.0\n    c["goal"] = {G} if s2.get("in_target", False) else 0.0\n    return c\n'),
]


class ContinuousAnalysis:
    """Standardized analysis output for Meta-World continuous domain matching DoorKey Analysis."""
    def __init__(
        self,
        ok: bool = True,
        hacked: bool = False,
        V_succ: float = 0.0,
        V_fail: float = 0.0,
        gap_rel: float = 0.0,
        culprit: str = "",
        culprit_verified: bool = False,
        signature: Tuple[str, ...] = (),
        error: str = "",
        table: Any = None,
    ):
        self.ok = ok
        self.hacked = hacked
        self.hacked_30 = hacked
        self.hacked_60 = hacked
        self.V_succ = V_succ
        self.V_fail = V_fail
        self.V_succ_30 = V_succ
        self.V_fail_30 = V_fail
        self.V_succ_60 = V_succ
        self.V_fail_60 = V_fail
        self.gap_rel = gap_rel
        self.culprit = culprit
        self.culprit_verified = culprit_verified
        self.signature = tuple(signature)
        self.error = error
        self.table = table


def analyze_metaworld(env: Any, code: str) -> ContinuousAnalysis:
    """
    Fast 3-tier continuous analysis matching DoorKey analyze():
      Tier 1: AST parsing and code sandbox compilation.
      Tier 2: Static anti-hack verification (Gate G2).
      Tier 3: Milestone gap verification & fast CEM attacker.
    """
    try:
        check_code(code)
        ns = make_namespace(env)
        exec(compile(code, "<dataset_eval_cont>", "exec"), ns)
        fn = ns.get("reward_components")
        if not callable(fn):
            return ContinuousAnalysis(ok=False, error="Missing reward_components function")
    except Exception as e:
        return ContinuousAnalysis(ok=False, error=str(e))

    bad_g2 = static_checks(code, env=env)
    is_static_hacked = len(bad_g2) > 0

    if not hasattr(env, "get_milestone_trajectories"):
        return ContinuousAnalysis(
            ok=True, hacked=is_static_hacked, V_succ=1.0, V_fail=2.0 if is_static_hacked else 0.0,
            gap_rel=1.0 if is_static_hacked else 0.0, culprit="static_violation" if is_static_hacked else "",
            culprit_verified=is_static_hacked, signature=("static_violation",), table=fn
        )

    m_trajs = env.get_milestone_trajectories(seed=42)
    t0, t1, t2, t3 = m_trajs[0], m_trajs[1], m_trajs[2], m_trajs[3]

    def eval_traj(steps):
        tot = 0.0
        comps_tot: Dict[str, float] = collections.defaultdict(float)
        for st in steps:
            c = fn(st.s, st.a, st.s2)
            if not isinstance(c, dict):
                c = {"total": float(c)}
            for k, v in c.items():
                fv = float(v)
                tot += fv
                comps_tot[k] += fv
        return tot, dict(comps_tot)

    r_t0, c_t0 = eval_traj(t0)
    r_t1, c_t1 = eval_traj(t1)
    r_t2, c_t2 = eval_traj(t2)
    r_t3, c_t3 = eval_traj(t3)

    V_succ = r_t3
    fail_candidates = [(r_t0, c_t0, "T0"), (r_t1, c_t1, "T1"), (r_t2, c_t2, "T2")]
    best_fail_ret, best_fail_comps, best_fail_tag = max(fail_candidates, key=lambda x: x[0])
    V_fail = best_fail_ret

    is_milestone_hacked = bool(V_fail >= V_succ - 1e-4)

    is_attacker_hacked = False
    if not is_static_hacked and not is_milestone_hacked:
        attacker = ContinuousTrajectoryAttacker(env, cem_iterations=5, population_size=15)
        ces = attacker.attack(fn, t3)
        if ces:
            top_ce = max(ces, key=lambda ce: float(ce.get("exploit_return", -np.inf)))
            r_exploit = float(top_ce.get("exploit_return", -np.inf))
            if r_exploit >= V_succ - 1e-4 or top_ce.get("type") in ("hack", "under_signal"):
                is_attacker_hacked = True
                if r_exploit > V_fail:
                    V_fail = r_exploit
                    if "worse_steps" in top_ce:
                        _, best_fail_comps = eval_traj(top_ce["worse_steps"])

    hacked = is_static_hacked or is_milestone_hacked or is_attacker_hacked
    if is_static_hacked and V_fail < V_succ:
        V_fail = V_succ + 1.0

    gap_rel = abs(V_fail - V_succ) / max(1.0, abs(V_succ))

    # Identify culprit component
    culprit = ""
    for bg in bad_g2:
        if bg.startswith("unconditional_positive:"):
            culprit = bg.split(":", 1)[1]
            break
        elif bg == "positive_distance_exploit":
            culprit = next((k for k in best_fail_comps.keys() if "dist" in k.lower() or "norm" in k.lower()), "distance_exploit")
            break
        elif bg == "no_goal_term":
            culprit = "missing_goal"
            break

    if not culprit and best_fail_comps:
        non_goals = {
            k: v for k, v in best_fail_comps.items()
            if k.lower() not in ("goal", "target", "at_goal", "goal_bonus", "terminal_goal", "in_target")
        }
        if non_goals:
            culprit = max(non_goals.items(), key=lambda x: x[1])[0]

    signature = (culprit,) if culprit else ("continuous_exploit",)
    return ContinuousAnalysis(
        ok=True,
        hacked=hacked,
        V_succ=V_succ,
        V_fail=V_fail,
        gap_rel=gap_rel,
        culprit=culprit,
        culprit_verified=bool(culprit),
        signature=signature,
        table=fn,
    )


def analyze(env: Any, code: str, cache_table: bool = False) -> Any:
    """Polymorphic analyze function: DoorKeyEnv -> DP solver | MetaWorld -> Continuous solver."""
    if isinstance(env, DoorKeyEnv):
        return analyze_doorkey(env, code, cache_table=cache_table)
    return analyze_metaworld(env, code)


# =====================================================================
# 3. BUILD CALIBRATED MUTANTS (DOORKEY & META-WORLD)
# =====================================================================
def build_calibrated_mutants_doorkey(env: DoorKeyEnv) -> List[Dict[str, Any]]:
    B_grid = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0, 1.2, 1.5]
    G_grid = [1.0, 1.5, 2.5, 4.0]
    calibrated = []

    for fam, desc, tmpl in MUTANT_FAMILIES:
        valid_candidates = []
        for B, G in itertools.product(B_grid, G_grid):
            code = tmpl.replace("{B}", repr(B)).replace("{G}", repr(G))
            an = analyze_doorkey(env, code, cache_table=False)
            if an.ok and an.hacked_30 and (0.02 <= an.gap_rel <= 0.85) and an.culprit_verified:
                valid_candidates.append((an.gap_rel, B, G, code, an))
        valid_candidates.sort(key=lambda x: x[0])

        if len(valid_candidates) >= 2:
            picked = [valid_candidates[0], valid_candidates[len(valid_candidates) // 2]]
        else:
            picked = valid_candidates[:1]

        for idx, (gap, B, G, code, an) in enumerate(picked):
            calibrated.append({
                "id": f"mut_{fam}_{idx+1}_B{B}_G{G}",
                "origin": "mutant",
                "family": fam,
                "desc": desc,
                "code": code,
                "fp": code_fingerprint(code),
                "V_succ_30": an.V_succ_30,
                "V_fail_30": an.V_fail_30,
                "V_succ_60": an.V_succ_60,
                "V_fail_60": an.V_fail_60,
                "gap_rel": an.gap_rel,
                "culprit": an.culprit,
                "culprit_verified": an.culprit_verified,
                "signature": list(an.signature),
                "split": "main",
            })
    return calibrated


def build_calibrated_mutants_metaworld(env: Any) -> List[Dict[str, Any]]:
    """
    Builds calibrated continuous reward mutants across the 6 Meta-World mutant families.
    Each mutant is mathematically verified via 3-tier screening (static G2, milestone ranking, fast CEM attacker).
    """
    task_name = getattr(env, "task_name", "door-open-v2")
    B_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    G_grid = [2.5, 5.0, 10.0]
    calibrated = []

    for fam, desc, tmpl in MUTANT_FAMILIES_METAWORLD:
        valid_candidates = []
        for B, G in itertools.product(B_grid, G_grid):
            code = tmpl.replace("{B}", repr(B)).replace("{G}", repr(G))
            an = analyze_metaworld(env, code)
            if an.ok and an.hacked and an.culprit_verified:
                valid_candidates.append((an.gap_rel, B, G, code, an))
        valid_candidates.sort(key=lambda x: x[0])

        # Filter by reasonable gap_rel if possible
        tight_candidates = [c for c in valid_candidates if 0.02 <= c[0] <= 1.50]
        picked_pool = tight_candidates if tight_candidates else valid_candidates

        if len(picked_pool) >= 2:
            picked = [picked_pool[0], picked_pool[len(picked_pool) // 2]]
        elif picked_pool:
            picked = picked_pool[:1]
        else:
            # Fallback starter code
            def_code = tmpl.replace("{B}", "2.0").replace("{G}", "5.0")
            an_def = analyze_metaworld(env, def_code)
            picked = [(an_def.gap_rel, 2.0, 5.0, def_code, an_def)]

        for idx, (gap, B, G, code, an) in enumerate(picked):
            calibrated.append({
                "id": f"mut_mw_{fam}_{idx+1}_B{B}_G{G}",
                "task": task_name,
                "origin": "mutant",
                "family": fam,
                "desc": desc,
                "code": code,
                "fp": code_fingerprint(code),
                "V_succ_30": an.V_succ_30,
                "V_fail_30": an.V_fail_30,
                "V_succ_60": an.V_succ_60,
                "V_fail_60": an.V_fail_60,
                "gap_rel": an.gap_rel,
                "culprit": an.culprit,
                "culprit_verified": an.culprit_verified,
                "signature": list(an.signature),
                "split": "main",
            })
    return calibrated


def build_calibrated_mutants(env: Any) -> List[Dict[str, Any]]:
    """Polymorphic mutant builder: DoorKeyEnv -> doorkey mutants | MetaWorld -> metaworld mutants."""
    if isinstance(env, DoorKeyEnv):
        return build_calibrated_mutants_doorkey(env)
    return build_calibrated_mutants_metaworld(env)


# =====================================================================
# 4. GENERATE NATURAL REWARDS VIA vLLM (DOORKEY & META-WORLD)
# =====================================================================
def generate_natural_rewards_doorkey(
    env: DoorKeyEnv,
    n_prompts: int = 360,
    seed: int = 42,
    task_desc: str = "",
    api_desc: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    task_desc = task_desc or TASK_DESC
    api_desc = api_desc or API_DESC
    prompts_msgs, styles_used, seeds_used = [], [], []
    for i in range(n_prompts):
        st = NATURAL_PERSONAS[i % len(NATURAL_PERSONAS)]
        msgs = [
            {"role": "system", "content": "You are a reinforcement learning practitioner designing dense shaping rewards in Python."},
            {"role": "user", "content": (
                f"{task_desc}\n\n{api_desc}\n\n"
                f"Write a DENSE reward function `reward_components(s, a, s2)` using 3 to 5 named components.\n"
                f"Design style to implement: {st}.\n"
                "Reply with ONLY one Python code block defining `reward_components(s, a, s2)`."
            )}
        ]
        prompts_msgs.append(msgs)
        styles_used.append(st)
        seeds_used.append(seed + 1000 + i)

    raw_outs = llm_batch_chat(prompts_msgs, temperature=0.95, max_tokens=600, seeds=seeds_used)

    stats = collections.Counter()
    seen_fps = set()
    natural_hacked = []

    for i, out in enumerate(raw_outs):
        code = extract_code(out["text"])
        fp = code_fingerprint(code)
        an = analyze_doorkey(env, code, cache_table=False)
        if not an.ok:
            stats["syntax_or_runtime_error"] += 1
        elif fp in seen_fps:
            stats["duplicate_ast"] += 1
        else:
            seen_fps.add(fp)
            if not (an.hacked_30 or an.hacked_60):
                stats["already_aligned"] += 1
            elif not np.isfinite(an.V_succ_30) or abs(an.V_succ_30) > 500:
                stats["extreme_or_invalid_goal"] += 1
            else:
                stats["hacked_kept"] += 1
                natural_hacked.append({
                    "id": f"nat_{len(natural_hacked)+1:02d}_{fp[:6]}",
                    "origin": "natural",
                    "family": f"nat_{'_'.join(an.signature) or 'misc'}",
                    "desc": styles_used[i][:60],
                    "code": code,
                    "fp": fp,
                    "V_succ_30": an.V_succ_30,
                    "V_fail_30": an.V_fail_30,
                    "V_succ_60": an.V_succ_60,
                    "V_fail_60": an.V_fail_60,
                    "gap_rel": an.gap_rel,
                    "culprit": an.culprit,
                    "culprit_verified": an.culprit_verified,
                    "signature": list(an.signature),
                    "split": "main",
                })

    return natural_hacked, dict(stats)


def generate_natural_rewards_metaworld(
    env: Any,
    n_prompts: int = 360,
    seed: int = 42,
    task_desc: str = "",
    api_desc: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Generates natural continuous rewards for Meta-World via vLLM batch generation and 3-tier filtering pipeline:
      Tier 1: AST safe syntax and execution sandbox check.
      Tier 2: Static anti-hack verification (Gate G2: no_goal, unconditional_positive, positive_distance).
      Tier 3: Milestone gap & Fast CEM attacker.
    """
    task_name = getattr(env, "task_name", "door-open-v2")
    task_desc = task_desc or METAWORLD_TASK_DESCS.get(task_name, METAWORLD_TASK_DESCS["door-open-v2"])
    api_desc = api_desc or get_metaworld_api_desc(task_name)

    prompts_msgs, styles_used, seeds_used = [], [], []
    for i in range(n_prompts):
        st = NATURAL_PERSONAS_METAWORLD[i % len(NATURAL_PERSONAS_METAWORLD)]
        msgs = [
            {"role": "system", "content": "You are a reinforcement learning practitioner designing continuous robotic manipulation rewards in Python."},
            {"role": "user", "content": (
                f"{task_desc}\n\n{api_desc}\n\n"
                f"Write a DENSE reward function `reward_components(s, a, s2)` using 3 to 5 named components.\n"
                f"Design style to implement: {st}.\n"
                "Reply with ONLY one Python code block defining `reward_components(s, a, s2)`."
            )}
        ]
        prompts_msgs.append(msgs)
        styles_used.append(st)
        seeds_used.append(seed + 1000 + i)

    raw_outs = llm_batch_chat(prompts_msgs, temperature=0.95, max_tokens=600, seeds=seeds_used)

    stats = collections.Counter()
    seen_fps = set()
    natural_hacked = []

    for i, out in enumerate(raw_outs):
        code = extract_code(out["text"])
        fp = code_fingerprint(code)
        if fp in seen_fps:
            stats["duplicate_ast"] += 1
            continue

        an = analyze_metaworld(env, code)
        if not an.ok:
            stats["syntax_or_runtime_error"] += 1
        else:
            seen_fps.add(fp)
            if not an.hacked:
                stats["already_aligned"] += 1
            elif not np.isfinite(an.V_succ) or abs(an.V_succ) > 10000:
                stats["extreme_or_invalid_goal"] += 1
            else:
                stats["hacked_kept"] += 1
                natural_hacked.append({
                    "id": f"nat_mw_{len(natural_hacked)+1:02d}_{fp[:6]}",
                    "task": task_name,
                    "origin": "natural",
                    "family": f"nat_{'_'.join(an.signature) or 'misc'}",
                    "desc": styles_used[i][:60],
                    "code": code,
                    "fp": fp,
                    "V_succ_30": an.V_succ_30,
                    "V_fail_30": an.V_fail_30,
                    "V_succ_60": an.V_succ_60,
                    "V_fail_60": an.V_fail_60,
                    "gap_rel": an.gap_rel,
                    "culprit": an.culprit,
                    "culprit_verified": an.culprit_verified,
                    "signature": list(an.signature),
                    "split": "main",
                })

    return natural_hacked, dict(stats)


def generate_natural_rewards(
    env: Any,
    n_prompts: int = 360,
    seed: int = 42,
    task_desc: str = "",
    api_desc: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Polymorphic generator: DoorKeyEnv -> DoorKey natural rewards | MetaWorld -> MetaWorld natural rewards."""
    if isinstance(env, DoorKeyEnv):
        return generate_natural_rewards_doorkey(
            env, n_prompts=n_prompts, seed=seed, task_desc=task_desc, api_desc=api_desc
        )
    return generate_natural_rewards_metaworld(
        env, n_prompts=n_prompts, seed=seed, task_desc=task_desc, api_desc=api_desc
    )
