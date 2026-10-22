from typing import Any, Dict, List
import pandas as pd

from cegr.core.base_runner import BaseRunner
from cegr.solvers.dp_solver import analyze
from cegr.solvers.verified_fix import evaluate_verified_fix


class OrigRunner(BaseRunner):
    """
    ORIG Baseline: Directly evaluates the initial un-repaired reward function R_0.
    Consumes 0 LLM calls, providing the empirical baseline for reward hacking severity.
    """

    def __init__(self, env: Any, config: Dict[str, Any]):
        super().__init__(config)
        self.env = env

    def run_single_job(self, reward_record: Dict[str, Any], sample_idx: int = 0) -> Dict[str, Any]:
        code = reward_record.get("code", "")
        from cegr.envs.doorkey import DoorKeyEnv
        an = analyze(self.env, code, cache_table=True) if isinstance(self.env, DoorKeyEnv) else None
        ev = evaluate_verified_fix(
            env=self.env,
            orig_an=an,
            orig_code=code,
            new_code=code,
            run_probe=True,
            new_an=an,
        )

        return {
            "job_id": f"orig__{reward_record['id']}__s{sample_idx}",
            "reward_id": reward_record["id"],
            "origin": reward_record.get("origin", "mutant"),
            "family": reward_record.get("family", "unknown"),
            "method": "ORIG",
            "sample_idx": sample_idx,
            "passed_gates": False,
            "rounds_to_pass": self.max_rounds + 1,
            "llm_calls": 0,
            "rl_runs": 1,
            "rl_episodes": 2500,
            "recurrence_count": 0,
            "vf": bool(ev["verified_fix"]),
            "tier1_safe": bool(ev["tier1_safe_aligned"]),
            "loose_dp": bool(ev["loose_dp"]),
            "category": ev["category"],
            "culprit_targeted": False,
            "rl_success": ev["probe_success_rate"],
            "hacked_dp": bool(an.hacked if an is not None else (not ev["loose_dp"])),
            "final_code": code,
        }

    def run_batch(self, reward_records: List[Dict[str, Any]], n_samples: int = 1) -> List[Dict[str, Any]]:
        results = []
        for r in reward_records:
            for s in range(n_samples):
                results.append(self.run_single_job(r, s))
        return results
