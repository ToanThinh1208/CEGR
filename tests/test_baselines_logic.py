import pytest

from cegr.envs.doorkey import make_env
from cegr.engine.llm_vllm import set_mock_llm
from cegr.baselines.eureka.runner import EurekaRunner
from cegr.baselines.card.runner import CardRunner


def mock_repair_code(messages):
    return (
        "```python\n"
        "def phi(st):\n"
        "    return 0.5 * (1.0 if st.has_key else 0.0) + 0.5 * (1.0 if st.door == DOOR_OPEN else 0.0)\n"
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c['step'] = -0.01\n"
        "    c['shaping'] = phi(s2) - phi(s)\n"
        "    c['goal'] = 1.0 if at_goal(s2) else 0.0\n"
        "    return c\n"
        "```"
    )


def test_eureka_execution_and_budget():
    set_mock_llm(mock_repair_code)
    env = make_env("doorkey5")
    config = {"budget": {"n_rounds": 4}}
    runner = EurekaRunner(env, config)

    reward_rec = {
        "id": "test_mutant_1",
        "code": (
            "def reward_components(s, a, s2):\n"
            "    c = {'step': -0.01, 'turn': 0.5 if a in (0, 1) else 0.0, 'goal': 1.0 if at_goal(s2) else 0.0}\n"
            "    return c\n"
        ),
    }

    res = runner.run_single_job(reward_rec, sample_idx=0)
    # Eureka must strictly execute 2 generations x 2 population = 4 LLM calls
    assert res["llm_calls"] == 4, f"Eureka made {res['llm_calls']} LLM calls instead of 4"
    assert "final_code" in res
    assert "vf" in res


def test_card_execution_and_budget():
    set_mock_llm(mock_repair_code)
    env = make_env("doorkey5")
    config = {"budget": {"n_rounds": 4}}
    runner = CardRunner(env, config, use_reflection_memory=False)

    reward_rec = {
        "id": "test_mutant_2",
        "code": (
            "def reward_components(s, a, s2):\n"
            "    c = {'step': -0.01, 'turn': 0.5 if a in (0, 1) else 0.0, 'goal': 1.0 if at_goal(s2) else 0.0}\n"
            "    return c\n"
        ),
    }

    res = runner.run_single_job(reward_rec, sample_idx=0)
    # CARD must not exceed max budget (4 calls)
    assert res["llm_calls"] <= 4, f"CARD exceeded budget: {res['llm_calls']} calls"
    assert "passed_gates" in res
    assert "final_code" in res


def test_card_metaworld_execution():
    from cegr.envs.metaworld_env import make_metaworld_env
    env = make_metaworld_env("door-open-v2", horizon=50, use_mock=True)

    def mock_mw_repair(messages):
        return (
            "```python\n"
            "def reward_components(s, a, s2):\n"
            "    c = {}\n"
            "    c['step'] = -0.01\n"
            "    d = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['obj_pos'])))\n"
            "    c['reach'] = -1.0 * d\n"
            "    c['goal'] = 10.0 if s2.get('in_target', False) else 0.0\n"
            "    return c\n"
            "```"
        )

    set_mock_llm(mock_mw_repair)
    config = {"budget": {"n_rounds": 2}}
    runner = CardRunner(env, config)
    rec = {
        "id": "test_mw_hover",
        "code": (
            "def reward_components(s, a, s2):\n"
            "    c = {'step': -0.01, 'near': 2.0, 'goal': 5.0 if s2.get('in_target') else 0.0}\n"
            "    return c\n"
        ),
    }
    res = runner.run_single_job(rec, sample_idx=0)
    assert "final_code" in res
    assert res["llm_calls"] >= 1


def test_cegr_metaworld_execution():
    from cegr.envs.metaworld_env import make_metaworld_env
    from cegr.synthesis.runner import CegrRunner
    env = make_metaworld_env("door-open-v2", horizon=50, use_mock=True)

    def mock_mw_repair(messages):
        return (
            "```python\n"
            "def reward_components(s, a, s2):\n"
            "    c = {}\n"
            "    c['step'] = -0.01\n"
            "    d = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['obj_pos'])))\n"
            "    c['reach'] = -1.0 * d\n"
            "    c['goal'] = 10.0 if s2.get('in_target', False) else 0.0\n"
            "    return c\n"
            "```"
        )

    set_mock_llm(mock_mw_repair)
    config = {"budget": {"n_rounds": 2}}
    runner = CegrRunner(env, config, method_name="CEGIS")
    rec = {
        "id": "test_mw_hover",
        "code": (
            "def reward_components(s, a, s2):\n"
            "    c = {'step': -0.01, 'near': 2.0, 'goal': 5.0 if s2.get('in_target') else 0.0}\n"
            "    return c\n"
        ),
    }
    res = runner.run_single_job(rec, sample_idx=0)
    assert "final_code" in res


if __name__ == "__main__":
    pytest.main([__file__])
