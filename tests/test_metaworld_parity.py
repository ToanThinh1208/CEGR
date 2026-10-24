import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ImportError:
    pytest = None
import numpy as np

from cegr.envs.metaworld_env import make_metaworld_env
from cegr.synthesis.dataset import (
    build_calibrated_mutants,
    generate_natural_rewards,
    MUTANT_FAMILIES_METAWORLD,
    NATURAL_PERSONAS_METAWORLD,
    METAWORLD_TASK_DESCS,
    get_metaworld_api_desc,
    analyze_metaworld,
)
from cegr.core.code_sandbox import check_code, make_namespace


def test_metaworld_task_descs_and_api_desc():
    for task_name in ["door-open-v2", "reach-v2", "pick-place-v2"]:
        assert task_name in METAWORLD_TASK_DESCS
        desc = METAWORLD_TASK_DESCS[task_name]
        assert "TASK: Meta-World" in desc
        assert "Action: 4D continuous vector" in desc

        api_desc = get_metaworld_api_desc(task_name)
        assert "reward_components(s, a, s2)" in api_desc
        assert "hand_pos" in api_desc
        assert "in_target" in api_desc


def test_mutant_families_metaworld_count():
    assert len(MUTANT_FAMILIES_METAWORLD) == 6
    expected_families = [
        "hover_near_handle",
        "positive_distance_exploit",
        "gripper_jitter",
        "survival_bonus",
        "hand_goal_shortcut",
        "grasp_holding_loop",
    ]
    actual_families = [fam for fam, _, _ in MUTANT_FAMILIES_METAWORLD]
    assert actual_families == expected_families


def test_build_calibrated_mutants_metaworld():
    env = make_metaworld_env("door-open-v2", horizon=150, use_mock=True)
    muts = build_calibrated_mutants(env)

    # Should contain mutants covering all 6 continuous families
    assert len(muts) >= 6
    families_present = set(m["family"] for m in muts)
    expected_families = set([
        "hover_near_handle",
        "positive_distance_exploit",
        "gripper_jitter",
        "survival_bonus",
        "hand_goal_shortcut",
        "grasp_holding_loop",
    ])
    assert expected_families.issubset(families_present)

    # Verify each mutant obeys interface and is mathematically verified as hacked
    for m in muts:
        assert m["origin"] == "mutant"
        assert m["split"] == "main"
        assert m["culprit_verified"] is True
        assert m["culprit"] != ""
        assert m["gap_rel"] >= 0.0

        # Interface validation
        check_code(m["code"])
        ns = make_namespace(env)
        exec(compile(m["code"], "<eval_test>", "exec"), ns)
        fn = ns["reward_components"]

        s, _ = env.reset(seed=42)
        a = np.zeros(4, dtype=np.float32)
        s2, _, _, _, _ = env.step(a)
        res = fn(s, a, s2)
        assert isinstance(res, dict)
        for k, v in res.items():
            assert isinstance(v, (int, float, np.floating))
            assert np.isfinite(float(v))
            assert abs(float(v)) <= 10000.0


def test_generate_natural_rewards_metaworld():
    env = make_metaworld_env("door-open-v2", horizon=150, use_mock=True)
    nats, stats = generate_natural_rewards(env, n_prompts=6, seed=42)

    assert isinstance(nats, list)
    assert isinstance(stats, dict)

    for r in nats:
        assert r["origin"] == "natural"
        assert r["task"] == "door-open-v2"
        assert "code" in r
        assert "fp" in r
        assert "culprit" in r

        check_code(r["code"])
        ns = make_namespace(env)
        exec(compile(r["code"], "<eval_nat>", "exec"), ns)
        fn = ns["reward_components"]

        s, _ = env.reset(seed=42)
        a = np.zeros(4, dtype=np.float32)
        s2, _, _, _, _ = env.step(a)
        res = fn(s, a, s2)
        assert isinstance(res, dict)


def test_analyze_metaworld_fast_continuous_screening():
    env = make_metaworld_env("door-open-v2", horizon=150, use_mock=True)

    # Static violation (positive distance)
    pos_dist_code = (
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c['step'] = -0.01\n"
        "    c['distance'] = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['goal_pos'])))\n"
        "    c['goal'] = 10.0 if s2.get('in_target', False) else 0.0\n"
        "    return c\n"
    )
    an_pos = analyze_metaworld(env, pos_dist_code)
    assert an_pos.ok is True
    assert an_pos.hacked is True
    assert an_pos.culprit == "distance"

    # Hovering near handle milestone violation
    hover_code = (
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c['step'] = -0.01\n"
        "    d = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['obj_pos'])))\n"
        "    c['near_handle'] = 2.0 if d < 0.08 else 0.0\n"
        "    c['goal'] = 5.0 if s2.get('in_target', False) else 0.0\n"
        "    return c\n"
    )
    an_hov = analyze_metaworld(env, hover_code)
    assert an_hov.ok is True
    assert an_hov.hacked is True
    assert an_hov.culprit == "near_handle"
    assert an_hov.V_fail >= an_hov.V_succ - 1e-4


if __name__ == "__main__":
    print("Running test_metaworld_task_descs_and_api_desc...")
    test_metaworld_task_descs_and_api_desc()
    print("Running test_mutant_families_metaworld_count...")
    test_mutant_families_metaworld_count()
    print("Running test_build_calibrated_mutants_metaworld...")
    test_build_calibrated_mutants_metaworld()
    print("Running test_generate_natural_rewards_metaworld...")
    test_generate_natural_rewards_metaworld()
    print("Running test_analyze_metaworld_fast_continuous_screening...")
    test_analyze_metaworld_fast_continuous_screening()
    print("ALL METAWORLD PARITY TESTS PASSED SUCCESSFULLY!")

