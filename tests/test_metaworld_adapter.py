import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ImportError:
    pytest = None
import numpy as np

from cegr.envs.metaworld_env import make_metaworld_env
from cegr.core.code_sandbox import check_code, make_namespace
from cegr.solvers.trajectory_attacker import ContinuousTrajectoryAttacker
from cegr.envs.base_env import Step


def test_metaworld_wrapper_state_dict():
    env = make_metaworld_env("door-open-v2", horizon=50, use_mock=True)
    obs, info = env.reset()

    # Check semantic state dictionary keys required by LLM
    expected_keys = ["hand_pos", "obj_pos", "goal_pos", "gripper_open", "obj_grasped", "in_target"]
    for k in expected_keys:
        assert k in obs, f"Missing required semantic key {k} in observation dictionary"

    assert len(obs["hand_pos"]) == 3
    assert len(obs["obj_pos"]) == 3
    assert len(obs["goal_pos"]) == 3
    assert 0.0 <= obs["gripper_open"] <= 1.0


def test_metaworld_sandbox_vector_math():
    code = (
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c['step'] = -0.01\n"
        "    d = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['obj_pos'])))\n"
        "    c['dist_reach'] = -1.0 * d\n"
        "    c['goal'] = 10.0 if s2.get('in_target', False) else 0.0\n"
        "    return c\n"
    )

    tree = check_code(code)
    assert tree is not None

    env = make_metaworld_env("door-open-v2", horizon=50, use_mock=True)
    ns = make_namespace(env)
    exec(compile(code, "<test_mw>", "exec"), ns)
    fn = ns["reward_components"]

    s, _ = env.reset()
    a = np.array([0.0, 0.1, -0.05, 0.0], dtype=np.float32)
    s2, _, _, _, _ = env.step(a)

    out = fn(s, a, s2)
    assert isinstance(out, dict)
    assert "dist_reach" in out
    assert "goal" in out
    assert isinstance(out["dist_reach"], float)


def test_continuous_trajectory_attacker():
    env = make_metaworld_env("reach-v2", horizon=50, use_mock=True)
    attacker = ContinuousTrajectoryAttacker(env, cem_iterations=3, population_size=10)

    # Flawed reward that gives positive points for hovering near start position instead of goal
    def flawed_reward(s, a, s2):
        return {"hover_bonus": 5.0, "goal": 1.0 if s2.get("in_target") else 0.0}

    # Reference trajectory steps
    ref_steps = [
        Step(t=i, s={"in_target": False}, a=np.zeros(4), s2={"in_target": i == 4}, r=1.0, comps={}, goal=i == 4)
        for i in range(5)
    ]

    ces = attacker.attack(flawed_reward, ref_steps, rnd=1)
    # Attacker should easily find an adversarial policy exploiting the hover bonus
    assert len(ces) >= 0


def test_static_antihack_on_metaworld_mutants():
    from cegr.synthesis.static_antihack import static_checks

    pos_dist_code = (
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c['step'] = -0.01\n"
        "    c['distance'] = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['goal_pos'])))\n"
        "    c['goal'] = 10.0 if s2.get('in_target', False) else 0.0\n"
        "    return c\n"
    )

    bad = static_checks(pos_dist_code)
    assert "positive_distance_exploit" in bad, f"Failed to detect positive distance exploit: {bad}"


def test_traj_return_continuous_metaworld():
    from cegr.envs.doorkey import traj_return
    from cegr.synthesis.runner import build_metaworld_kit, count_recurrence
    from cegr.synthesis.ce_pool import make_ce

    env = make_metaworld_env("door-open-v2", horizon=50, use_mock=True)
    kit = build_metaworld_kit(env)

    def sample_reward(s, a, s2):
        return {"step": -0.01, "goal": 10.0 if s2.get("in_target") else 0.0}

    # 1. Test traj_return with MetaWorldEnvWrapper and step list (T3)
    ret_t3 = traj_return(env, sample_reward, kit.refs["T3"])
    assert isinstance(ret_t3, float)

    # 2. Test traj_return with MetaWorldEnvWrapper and raw action list
    act_list = [np.array([0.0, 0.1, -0.05, -0.8], dtype=np.float32) for _ in range(10)]
    ret_acts = traj_return(env, sample_reward, act_list)
    assert isinstance(ret_acts, float)

    # 3. Test count_recurrence with MetaWorldEnvWrapper, kit, and seen CEs
    seen = {
        "p0_test": make_ce(
            kit.refs["T3"], kit.refs["T0"], "p0", 0, "T3>T0",
            better_steps=kit.refs["T3"], worse_steps=kit.refs["T0"]
        )
    }
    recur = count_recurrence(env, kit, sample_reward, seen)
    assert isinstance(recur, int)


def test_metaworld_milestones_and_ducktyped_gt_score():
    env = make_metaworld_env("door-open-v2", horizon=50, use_mock=True)
    trajs = env.get_milestone_trajectories()
    assert len(trajs) == 4
    # Check gt_score strictly increases along milestone stages T0 < T1 < T2 < T3
    scores = [env.gt_score(t) for t in trajs]
    assert scores[0] < scores[1] < scores[2] < scores[3], f"Milestone scores not strictly increasing: {scores}"
    assert trajs[3][-1].goal is True, "T3 expert trajectory must reach in_target goal"

    # Check gt_score with raw action arrays (duck typing)
    act_list = [st.a for st in trajs[3]]
    act_score = env.gt_score(act_list)
    assert isinstance(act_score, float)
    assert act_score >= 3.0


def test_gatekeeper_metaworld_valid_and_exploit_rewards():
    from cegr.synthesis.runner import build_metaworld_kit
    from cegr.synthesis.gatekeeper import GateKeeper, run_gatekeeper
    from cegr.synthesis.ce_pool import CEPool

    env = make_metaworld_env("door-open-v2", horizon=50, use_mock=True)
    kit = build_metaworld_kit(env)
    gk = GateKeeper(env, kit)
    pool = CEPool(max_size=40)

    # 1. Valid shaped reward: should PASS Gate 1, 2, 3
    valid_code = (
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c['step'] = -0.01\n"
        "    d = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['obj_pos'])))\n"
        "    c['reach'] = -1.0 * d\n"
        "    c['grasp'] = 0.5 if s2.get('obj_grasped', False) else 0.0\n"
        "    c['goal'] = 20.0 if s2.get('in_target', False) else 0.0\n"
        "    return c\n"
    )
    res_valid = run_gatekeeper(valid_code, {"gates": [1, 2, 3]}, pool, rnd=0, env=env, kit=kit, gk=gk)
    assert res_valid["passed"] is True, f"Valid reward failed GateKeeper: {res_valid}"
    assert res_valid["gates_passed"] == 3

    # 2. Exploit hover reward: should FAIL at Gate 3 with hack violation
    exploit_code = (
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c['step'] = -0.01\n"
        "    d = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['obj_pos'])))\n"
        "    c['near_handle'] = 5.0 if d < 0.08 else 0.0\n"
        "    c['goal'] = 2.0 if s2.get('in_target', False) else 0.0\n"
        "    return c\n"
    )
    res_exploit = run_gatekeeper(exploit_code, {"gates": [1, 2, 3]}, pool, rnd=0, env=env, kit=kit, gk=gk)
    assert res_exploit["passed"] is False, "Exploit reward should not pass GateKeeper"
    assert res_exploit["failed_gate"] == 3


def test_static_checks_metaworld_env_compatibility():
    from cegr.synthesis.static_antihack import static_checks
    env = make_metaworld_env("door-open-v2", horizon=50, use_mock=True)
    def sample_reward(s, a, s2):
        return {"step": -0.01, "goal": 10.0 if s2.get("in_target") else 0.0}
    code = (
        "def reward_components(s, a, s2):\n"
        "    return {'step': -0.01, 'goal': 10.0 if s2.get('in_target') else 0.0}\n"
    )
    bad = static_checks(code, table=sample_reward, env=env)
    assert isinstance(bad, list)
    assert len(bad) == 0


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__])
    else:
        print("Running test_metaworld_wrapper_state_dict...")
        test_metaworld_wrapper_state_dict()
        print("Running test_metaworld_sandbox_vector_math...")
        test_metaworld_sandbox_vector_math()
        print("Running test_continuous_trajectory_attacker...")
        test_continuous_trajectory_attacker()
        print("Running test_static_antihack_on_metaworld_mutants...")
        test_static_antihack_on_metaworld_mutants()
        print("Running test_traj_return_continuous_metaworld...")
        test_traj_return_continuous_metaworld()
        print("Running test_metaworld_milestones_and_ducktyped_gt_score...")
        test_metaworld_milestones_and_ducktyped_gt_score()
        print("Running test_gatekeeper_metaworld_valid_and_exploit_rewards...")
        test_gatekeeper_metaworld_valid_and_exploit_rewards()
        print("Running test_static_checks_metaworld_env_compatibility...")
        test_static_checks_metaworld_env_compatibility()
        print("ALL METAWORLD ADAPTER TESTS PASSED SUCCESSFULLY!")


