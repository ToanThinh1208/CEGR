from typing import Any, Dict, List, Optional
import numpy as np

from cegr.core.base_runner import BaseRunner
from cegr.core.code_sandbox import build_reward_table, RewardError
from cegr.envs.doorkey import DoorKeyEnv, build_reference_trajs, traj_return, TASK_DESC, API_DESC
from cegr.solvers.dp_solver import analyze
from cegr.solvers.verified_fix import evaluate_verified_fix
from cegr.engine.llm_vllm import llm_batch_chat, extract_code
from cegr.baselines.card.trajectory_buffer import TrajectoryBuffer
from cegr.baselines.card.feedback_generator import generate_card_feedback
from cegr.baselines.card.prompts import build_card_prompts


def traj_component_returns(env: Any, table: Any, actions: List[int]) -> Dict[str, float]:
    i = 0
    comp_tot = {k: 0.0 for k in table.names}
    for a in actions:
        if env.terminal[i]:
            break
        a = int(a)
        for ki, k in enumerate(table.names):
            comp_tot[k] += float(table.C[i, a, ki])
        g = bool(env.goal_flag[i, a])
        i = int(env.nxt[i, a])
        if g:
            break
    return comp_tot


class CardRunner(BaseRunner):
    """
    CARD & CARD_REFL Protocol Runner (Sun et al., 2024).
    Features:
      - Static Trajectory Preference Evaluation (TPE) on fixed reference pairs P_0
      - 3-part feedback: Process, Trajectory, and Preference Feedback
      - CARD (no history) vs. CARD_REFL (concatenated text memory)
      - Early stopping upon passing all P_0 preference tests
      - Up to N = 4 rounds (Strict budget compliance)
    """

    def __init__(
        self,
        env: Any,
        config: Dict[str, Any],
        use_reflection_memory: bool = False,
        task_desc: str = "",
        api_desc: str = "",
    ):
        super().__init__(config)
        self.env = env
        self.use_reflection_memory = use_reflection_memory
        self.task_desc = task_desc or TASK_DESC
        self.api_desc = api_desc or API_DESC
        self.is_continuous = not isinstance(env, DoorKeyEnv)
        self.buffer = self._init_buffer()

    def _init_buffer(self) -> TrajectoryBuffer:
        tb = TrajectoryBuffer()
        from cegr.core.tpe_evaluator import build_p0_pairs
        p0_pairs = build_p0_pairs(self.env, seed=1000, max_noisy=20)
        for p in p0_pairs:
            tb.add_pair(
                p["better"],
                p["worse"],
                tag=p["tag"],
                better_steps=p.get("better_steps"),
                worse_steps=p.get("worse_steps"),
            )
        return tb

    def _eval_tpe(self, code: str) -> Dict[str, Any]:
        """Evaluates reward code across trajectory buffer pairs."""
        if not self.is_continuous:
            try:
                table = build_reward_table(self.env, code)
            except Exception as e:
                return {
                    "ok": False,
                    "error": str(e),
                    "passed": False,
                    "violations": [],
                    "succ_returns": [],
                    "fail_returns": [],
                    "succ_comps": {},
                    "fail_comps": {},
                }

            violations = []
            succ_rets = []
            fail_rets = []
            succ_comps = {}
            fail_comps = {}

            for p in self.buffer.get_pairs():
                rb = traj_return(self.env, table, p["better"])
                rw = traj_return(self.env, table, p["worse"])
                cb = traj_component_returns(self.env, table, p["better"])
                cw = traj_component_returns(self.env, table, p["worse"])
                succ_rets.append(rb)
                fail_rets.append(rw)

                # Record true per-component averages
                for k in table.names:
                    succ_comps.setdefault(k, []).append(cb.get(k, 0.0))
                    fail_comps.setdefault(k, []).append(cw.get(k, 0.0))

                if rw >= rb - 1e-9:
                    violations.append({
                        "pair": p,
                        "Rb": rb,
                        "Rw": rw,
                        "margin": rb - rw,
                    })

            return {
                "ok": True,
                "error": "",
                "passed": len(violations) == 0,
                "violations": violations,
                "succ_returns": succ_rets,
                "fail_returns": fail_rets,
                "succ_comps": succ_comps,
                "fail_comps": fail_comps,
                "table": table,
            }
        else:
            try:
                from cegr.core.code_sandbox import make_namespace, check_code
                check_code(code)
                ns = make_namespace(self.env)
                exec(compile(code, "<card_cont>", "exec"), ns)
                fn = ns["reward_components"]
            except Exception as e:
                return {
                    "ok": False,
                    "error": str(e),
                    "passed": False,
                    "violations": [],
                    "succ_returns": [],
                    "fail_returns": [],
                    "succ_comps": {},
                    "fail_comps": {},
                }

            violations = []
            succ_rets, fail_rets = [], []
            succ_comps, fail_comps = {}, {}

            for p in self.buffer.get_pairs():
                def eval_steps(steps):
                    tot = 0.0
                    ctot = {}
                    for st in steps:
                        comps = fn(st.s, st.a, st.s2)
                        if not isinstance(comps, dict):
                            comps = {"total": float(comps)}
                        tot += sum(float(v) for v in comps.values())
                        for k, v in comps.items():
                            ctot[k] = ctot.get(k, 0.0) + float(v)
                    return tot, ctot

                rb, cb = eval_steps(p["better"])
                rw, cw = eval_steps(p["worse"])
                succ_rets.append(rb)
                fail_rets.append(rw)
                for k in set(list(cb.keys()) + list(cw.keys())):
                    succ_comps.setdefault(k, []).append(cb.get(k, 0.0))
                    fail_comps.setdefault(k, []).append(cw.get(k, 0.0))

                if rw >= rb - 1e-9:
                    violations.append({
                        "pair": p,
                        "Rb": rb,
                        "Rw": rw,
                        "margin": rb - rw,
                    })

            return {
                "ok": True,
                "error": "",
                "passed": len(violations) == 0,
                "violations": violations,
                "succ_returns": succ_rets,
                "fail_returns": fail_rets,
                "succ_comps": succ_comps,
                "fail_comps": fail_comps,
                "table": None,
            }

    def run_single_job(self, reward_record: Dict[str, Any], sample_idx: int = 0) -> Dict[str, Any]:
        current_code = reward_record.get("code", "")
        r0_code = current_code
        llm_calls = 0
        reflections_history: List[str] = []
        passed = False
        pass_round = None
        seen_violations: Dict[str, Any] = {}
        waves_log = []

        # Round 0: Evaluate initial R_0 without LLM calls
        init_tpe = self._eval_tpe(r0_code)
        latest_valid_code = r0_code if init_tpe["ok"] else None

        if init_tpe["passed"]:
            passed = True
            pass_round = 0
            final_code = r0_code
            waves_log.append({"round": 0, "passed": True, "violations": 0})
        else:
            for v in init_tpe["violations"]:
                tag = v.get("pair", {}).get("tag", "reference_pair")
                seen_violations[tag] = v.get("pair")
            feedback = generate_card_feedback(
                syntax_error=init_tpe["error"] if not init_tpe["ok"] else None,
                tpe_violations=init_tpe["violations"],
                succ_returns=init_tpe["succ_returns"],
                fail_returns=init_tpe["fail_returns"],
                succ_comps=init_tpe["succ_comps"],
                fail_comps=init_tpe["fail_comps"],
                reflections_history=None,
            )
            if self.use_reflection_memory:
                reflections_history.append(feedback)
            waves_log.append({
                "round": 0, "passed": False,
                "violations": len(init_tpe["violations"]),
                "error": init_tpe["error"]
            })

        for rnd in range(1, self.max_rounds + 1):
            if passed:
                break

            msgs = build_card_prompts(
                task_desc=self.task_desc,
                api_desc=self.api_desc,
                current_code=current_code,
                feedback_text=feedback,
                round_idx=rnd,
            )
            seed_val = 1000 + (sample_idx % 10) * 80 + rnd * 11
            out = llm_batch_chat([msgs], temperature=0.35, max_tokens=900, seeds=[seed_val])[0]
            llm_calls += 1
            cand_code = extract_code(out["text"])

            tpe_res = self._eval_tpe(cand_code)
            if tpe_res["ok"]:
                latest_valid_code = cand_code

            if tpe_res["passed"]:
                passed = True
                pass_round = rnd
                final_code = cand_code
                waves_log.append({"round": rnd, "passed": True, "violations": 0})
                break
            else:
                for v in tpe_res["violations"]:
                    tag = v.get("pair", {}).get("tag", "reference_pair")
                    seen_violations[tag] = v.get("pair")

                feedback = generate_card_feedback(
                    syntax_error=tpe_res["error"] if not tpe_res["ok"] else None,
                    tpe_violations=tpe_res["violations"],
                    succ_returns=tpe_res["succ_returns"],
                    fail_returns=tpe_res["fail_returns"],
                    succ_comps=tpe_res["succ_comps"],
                    fail_comps=tpe_res["fail_comps"],
                    reflections_history=reflections_history if self.use_reflection_memory else None,
                )
                if self.use_reflection_memory:
                    reflections_history.append(feedback)

                waves_log.append({
                    "round": rnd, "passed": False,
                    "violations": len(tpe_res["violations"]),
                    "error": tpe_res["error"]
                })
                current_code = cand_code

        if not passed:
            final_code = latest_valid_code if latest_valid_code is not None else r0_code

        # Post-hoc recurrence count replay on R_final using 'margin' mode
        recurrence_count = 0
        if seen_violations:
            final_tpe = self._eval_tpe(final_code)
            final_tab = final_tpe.get("table")
            from cegr.core.tpe_evaluator import pair_returns, classify_pair
            for p_tag, p in seen_violations.items():
                if p:
                    rb, rw = pair_returns(self.env, final_tab or final_tpe, p)
                    if classify_pair(rb, rw, eps_rel=0.02) is not None:
                        recurrence_count += 1

        # Independent Oracle evaluation
        orig_an = analyze(self.env, r0_code, cache_table=True) if isinstance(self.env, DoorKeyEnv) else None
        new_an = analyze(self.env, final_code, cache_table=True) if isinstance(self.env, DoorKeyEnv) else None
        ev = evaluate_verified_fix(
            env=self.env,
            orig_an=orig_an,
            orig_code=r0_code,
            new_code=final_code,
            run_probe=True,
            new_an=new_an,
        )

        method_name = "CARD_REFL" if self.use_reflection_memory else "CARD"

        return {
            "job_id": f"{method_name.lower()}__{reward_record['id']}__s{sample_idx}",
            "reward_id": reward_record["id"],
            "origin": reward_record.get("origin", "mutant"),
            "family": reward_record.get("family", "unknown"),
            "method": method_name,
            "sample_idx": sample_idx,
            "passed_gates": passed,
            "rounds_to_pass": pass_round if passed else self.max_rounds + 1,
            "llm_calls": llm_calls,
            "rl_runs": 1,
            "rl_episodes": 0,  # CARD does not train RL in the loop
            "recurrence_count": recurrence_count,
            "vf": bool(ev["verified_fix"]),
            "tier1_safe": bool(ev["tier1_safe_aligned"]),
            "loose_dp": bool(ev["loose_dp"]),
            "category": ev["category"],
            "culprit_targeted": bool(ev["culprit_targeted"]),
            "rl_success": ev["probe_success_rate"],
            "hacked_dp": bool(new_an.hacked if new_an else False),
            "final_code": final_code,
            "waves": waves_log,
        }

    def run_batch(self, reward_records: List[Dict[str, Any]], n_samples: int = 1) -> List[Dict[str, Any]]:
        results = []
        for r in reward_records:
            for s in range(n_samples):
                results.append(self.run_single_job(r, s))
        return results
