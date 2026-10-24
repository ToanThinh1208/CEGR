import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ImportError:
    pytest = None

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
    runner = CegrRunner(env, config, method_name="CEGR")
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


def test_card_prompts_reuse_common_prompt():
    from cegr.baselines.card import build_card_prompts, self_reflection_prompt, format_self_reflection_prompt

    feedback_sample = "=== PROCESS FEEDBACK ===\nExecution Status: PASSED.\n=== PREFERENCE FEEDBACK ===\nPair 'T3_T2' inverted."
    msgs = build_card_prompts(
        task_desc="Task description",
        api_desc="API description",
        current_code="def reward_components(s, a, s2):\n    return {'step': -0.01}",
        feedback_text=feedback_sample,
        round_idx=1,
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"

    user_content = msgs[1]["content"]
    # Verify core guidance from CARD common_prompt is present
    assert "I use the reward function you wrote to train the RL agent" in user_content
    assert "Please carefully analyze the above feedback step by step" in user_content
    assert "If the success rates are always near zero, then you must rewrite the entire reward function" in user_content
    assert "If some reward components' magnitude is significantly larger" in user_content
    assert "Do not invent any variable or attribute that is not given" in user_content
    assert "Pair 'T3_T2' inverted" in user_content


def test_eureka_prompts_match_eureka_spec():
    from cegr.baselines.eureka import (
        build_eureka_prompts,
        build_initial_eureka_prompts,
        INITIAL_SYSTEM,
        CODE_OUTPUT_TIP,
        CODE_FEEDBACK,
        POLICY_FEEDBACK,
    )

    # 1. Test build_eureka_prompts with reflection text
    refl_text = "- task_success_rate: [0.00, 0.00, 0.20, 0.40, 0.60] (Max: 0.60, Final: 0.60)"
    msgs = build_eureka_prompts(
        current_code="def reward_components(s, a, s2):\n    return {'step': -0.01}",
        reflection_text=refl_text,
        task_desc="Task description",
        api_desc="API description",
        generation=1,
    )
    assert len(msgs) == 2
    sys_content = msgs[0]["content"]
    user_content = msgs[1]["content"]

    # Verify Eureka system prompt and code output tip
    assert "You are a reward engineer" in sys_content
    assert "Some helpful tips for writing the reward function code" in sys_content

    # Verify Eureka training reflection & code feedback
    assert "We trained an RL policy" in user_content
    assert refl_text in user_content
    assert "Please carefully analyze the policy feedback" in user_content
    assert "If the success rates are always near zero" in user_content
    assert "If some reward components' magnitude is significantly larger" in user_content

    # 2. Test initial prompt builder (Generation 0)
    init_msgs = build_initial_eureka_prompts(
        task_desc="DoorKey-5x5 task",
        api_desc="DoorKey API",
    )
    assert len(init_msgs) == 2
    assert "You are a reward engineer" in init_msgs[0]["content"]
    assert "Write a reward function for the following task:" in init_msgs[1]["content"]


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__])
    else:
        print("Running test_eureka_execution_and_budget...")
        test_eureka_execution_and_budget()
        print("Running test_card_execution_and_budget...")
        test_card_execution_and_budget()
        print("Running test_card_metaworld_execution...")
        test_card_metaworld_execution()
        print("Running test_cegr_metaworld_execution...")
        test_cegr_metaworld_execution()
        print("Running test_card_prompts_reuse_common_prompt...")
        test_card_prompts_reuse_common_prompt()
        print("Running test_eureka_prompts_match_eureka_spec...")
        test_eureka_prompts_match_eureka_spec()
        print("ALL BASELINES LOGIC TESTS PASSED SUCCESSFULLY!")


