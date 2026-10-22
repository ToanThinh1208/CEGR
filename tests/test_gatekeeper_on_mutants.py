import json
import os
import pytest

from cegr.envs.doorkey import make_env
from cegr.core.code_sandbox import build_reward_table
from cegr.synthesis.ce_pool import CEPool
from cegr.synthesis.gatekeeper import GateKeeper, run_gatekeeper
from cegr.synthesis.runner import build_kit


def test_gatekeeper_on_mutants():
    env = make_env("doorkey5")
    kit = build_kit(env)
    gk = GateKeeper(env, kit)
    pool = CEPool(max_size=40)

    cfg = {
        "gates": [1, 2, 3, 4, 5],
        "pool": True,
        "attacker": True,
        "classify": True,
        "anchors": "dynamic",
        "experience": True,
        "g2_mode": "merge",
    }

    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "doorkey", "manifest_planted_115.json"
    )

    with open(manifest_path) as f:
        mutants = json.load(f)

    # All planted mutants must be caught by at least one gate in GateKeeper (Gate 2, 3, 4, or 5)
    for m in mutants[:5]:
        res = run_gatekeeper(m["code"], cfg, pool, rnd=1, env=env, kit=kit, gk=gk)
        assert not res["passed"], f"Mutant {m['id']} should have failed GateKeeper verification!"
        assert res["failed_gate"] in (2, 3, 4, 5), f"Unexpected failure gate: {res['failed_gate']}"


def test_gatekeeper_on_correct_code():
    env = make_env("doorkey5")
    kit = build_kit(env)
    gk = GateKeeper(env, kit)
    pool = CEPool(max_size=40)

    correct_code = (
        "def phi(st):\n"
        "    return 0.5 * (1.0 if st.has_key else 0.0) + 0.5 * (1.0 if st.door == DOOR_OPEN else 0.0)\n"
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c['step'] = -0.01\n"
        "    c['shaping'] = phi(s2) - phi(s)\n"
        "    c['goal'] = 1.0 if at_goal(s2) else 0.0\n"
        "    return c\n"
    )

    cfg = {
        "gates": [1, 2, 3, 4, 5],
        "pool": True,
        "attacker": True,
        "classify": True,
        "anchors": "dynamic",
        "g2_mode": "merge",
    }

    res = run_gatekeeper(correct_code, cfg, pool, rnd=1, env=env, kit=kit, gk=gk)
    assert res["passed"], f"Correct code failed verification at gate {res['failed_gate']}: {res['error']}"


def test_cegr_wave_batch_and_verified_fix():
    from cegr.synthesis.runner import CegrRunner
    from cegr.engine.llm_vllm import set_mock_llm

    def mock_correct(messages):
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

    set_mock_llm(mock_correct)
    env = make_env("doorkey5")
    config = {"budget": {"n_rounds": 2}}
    runner = CegrRunner(env, config, method_name="CEGR")

    test_records = [
        {
            "id": "mut_pickup_loop_quick_test",
            "origin": "mutant",
            "family": "key_pickup_loop",
            "code": (
                "def reward_components(s, a, s2):\n"
                "    c = {'step': -0.01, 'key_pickup': 0.3 if (not s.has_key and s2.has_key) else 0.0, 'goal': 1.0 if at_goal(s2) else 0.0}\n"
                "    return c\n"
            ),
        }
    ]

    results = runner.run_wave_batch(test_records, n_samples=1)
    assert len(results) == 1
    res = results[0]
    assert res["passed_gates"] is True
    assert res["vf"] is True
    assert res["tier1_safe"] is True
    assert res["loose_dp"] is True
    assert res["category"] == "verified_fix"


if __name__ == "__main__":
    pytest.main([__file__])
