import pytest
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


if __name__ == "__main__":
    pytest.main([__file__])
