import collections
import hashlib
import itertools
import json
import os
import random
from typing import Any, Dict, List, Tuple
import numpy as np

from cegr.envs.doorkey import DoorKeyEnv, make_env
from cegr.core.code_sandbox import code_fingerprint
from cegr.solvers.dp_solver import analyze
from cegr.engine.llm_vllm import llm_batch_chat, extract_code

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


def build_calibrated_mutants(env: DoorKeyEnv) -> List[Dict[str, Any]]:
    B_grid = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0, 1.2, 1.5]
    G_grid = [1.0, 1.5, 2.5, 4.0]
    calibrated = []

    for fam, desc, tmpl in MUTANT_FAMILIES:
        valid_candidates = []
        for B, G in itertools.product(B_grid, G_grid):
            code = tmpl.replace("{B}", repr(B)).replace("{G}", repr(G))
            an = analyze(env, code, cache_table=False)
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
            })
    return calibrated


def generate_natural_rewards(
    env: DoorKeyEnv,
    n_prompts: int = 360,
    seed: int = 42,
    task_desc: str = "",
    api_desc: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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
        an = analyze(env, code, cache_table=False)
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
                })

    return natural_hacked, dict(stats)


def build_and_freeze_manifest(out_file: str, env_name: str = "doorkey5", seed: int = 42):
    env = make_env(env_name)
    mut_list = build_calibrated_mutants(env)
    print(f"Generated {len(mut_list)} calibrated mutants across 12 families.")

    # In mock / CPU fallback, if vLLM unavailable, use existing starter mutants
    nat_list, gen_stats = generate_natural_rewards(env, n_prompts=30, seed=seed)
    print("Natural generation stats:", gen_stats)

    rng = random.Random(seed)
    rng.shuffle(mut_list)
    rng.shuffle(nat_list)

    n_p_mut = min(2, len(mut_list) // 4)
    n_p_nat = min(2, max(1, len(nat_list) // 5))
    for idx, r in enumerate(mut_list):
        r["split"] = "pilot" if idx < n_p_mut else "main"
    for idx, r in enumerate(nat_list):
        r["split"] = "pilot" if idx < n_p_nat else "main"

    all_rewards = [r for r in mut_list + nat_list if r["split"] == "main"]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_rewards, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_rewards)} rewards to {out_file}")


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_target = os.path.join(cur_dir, "manifest_planted_115.json")
    build_and_freeze_manifest(out_target)
