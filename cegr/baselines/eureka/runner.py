from typing import Any, Dict, List, Optional, Tuple
import gc
import numpy as np

from cegr.core.base_runner import BaseRunner
from cegr.core.code_sandbox import build_reward_table, RewardError
from cegr.envs.doorkey import DoorKeyEnv, TASK_DESC, API_DESC
from cegr.solvers.rl_probe import run_probe
from cegr.solvers.continuous_rl_probe import ContinuousRLProbe
from cegr.solvers.dp_solver import analyze
from cegr.solvers.verified_fix import evaluate_verified_fix
from cegr.engine.llm_vllm import llm_batch_chat, extract_code
from cegr.baselines.eureka.evolutionary_search import EvolutionarySearch, Candidate
from cegr.baselines.eureka.reflection_prompt import build_reward_reflection
from cegr.baselines.eureka.prompts import build_eureka_prompts


from cegr.core.tpe_evaluator import build_p0_pairs, count_recurrence


class EurekaRunner(BaseRunner):
    """
    EUREKA Protocol Runner (Ma et al., ICLR 2024).
    Features:
      - 2 Generations x 2 Candidates = 4 LLM calls (Strict N = 4 budget)
      - Full RL evaluation on every syntactically valid candidate
      - Reward Reflection based on 5 training checkpoints
      - Elitism selection
      - Independent of GateKeeper and Oracle Verified Fix during search
    """

    def __init__(self, env: Any, config: Dict[str, Any], task_desc: str = "", api_desc: str = ""):
        super().__init__(config)
        self.env = env
        self.task_desc = task_desc or TASK_DESC
        self.api_desc = api_desc or API_DESC
        self.is_continuous = not isinstance(env, DoorKeyEnv)
        self.pop_size = 2
        self.generations = 2
        self.p0_pairs = build_p0_pairs(env, seed=1000, max_noisy=20)

    def _collect_ces(self, code: str) -> List[Dict[str, Any]]:
        from cegr.core.tpe_evaluator import pair_returns, classify_pair
        from cegr.synthesis.ce_pool import make_ce
        ces = []
        if not self.is_continuous:
            try:
                table = build_reward_table(self.env, code)
                for p in self.p0_pairs:
                    rb, rw = pair_returns(self.env, table, p)
                    typ = classify_pair(rb, rw, eps_rel=0.02)
                    if typ is not None:
                        ces.append(make_ce(p["better"], p["worse"], typ, 0, p.get("tag", "")))
                from cegr.solvers.dp_solver import dp_attack
                ces.extend(dp_attack(self.env, table, rnd=0, classify=True))
            except Exception:
                pass
        else:
            try:
                from cegr.core.code_sandbox import make_namespace, check_code
                check_code(code)
                ns = make_namespace(self.env)
                exec(compile(code, "<eureka_ce>", "exec"), ns)
                fn = ns["reward_components"]
                for p in self.p0_pairs:
                    rb, rw = pair_returns(self.env, fn, p)
                    typ = classify_pair(rb, rw, eps_rel=0.02)
                    if typ is not None:
                        ces.append(make_ce(p["better"], p["worse"], typ, 0, p.get("tag", ""),
                                           better_steps=p.get("better_steps"), worse_steps=p.get("worse_steps")))
            except Exception:
                pass
        return ces

    def _eval_rl(self, code: str, seed: int) -> Tuple[bool, float, List[float], Dict[str, List[float]], int]:
        """Evaluates reward code using RL probe and tracks learning curve."""
        if not self.is_continuous:
            try:
                table = build_reward_table(self.env, code)
                pr = run_probe(
                    self.env,
                    table,
                    episodes=2500,
                    seeds=3,
                    seed_base=seed,
                    gamma=1.0,
                    optimistic_init=1.0,
                    alpha_decay=True,
                )
                comp_curves = pr.get("comp_curves", {})
                return True, pr["tail"], pr["curve"], comp_curves, 2500 * 3
            except Exception:
                return False, 0.0, [0.0] * 5, {}, 0
        else:
            try:
                from cegr.core.code_sandbox import make_namespace, check_code
                check_code(code)
                ns = make_namespace(self.env)
                exec(compile(code, "<eureka_cont>", "exec"), ns)
                reward_fn = ns["reward_components"]
                probe = ContinuousRLProbe(self.env)
                res = probe.probe(reward_fn, total_steps=25000, eval_episodes=5, seed=seed)
                return True, res["tail"], res.get("curve", [res["tail"]] * 5), res.get("comp_curves", {}), 25000
            except Exception:
                return False, 0.0, [0.0] * 5, {}, 0

    def run_single_job(self, reward_record: Dict[str, Any], sample_idx: int = 0) -> Dict[str, Any]:
        evo = EvolutionarySearch(pop_size=self.pop_size, max_generations=self.generations)
        r0_code = reward_record.get("code", "")
        llm_calls = 0
        rl_episodes = 0
        seen_ces: Dict[str, Any] = {}

        # Step 1: Initial R_0 RL evaluation (seed stream 1000)
        seed_base = 1000 + (sample_idx % 10) * 80
        ok_0, fitness_0, curve_0, comp_curves_0, eps_used = self._eval_rl(r0_code, seed=seed_base)
        rl_episodes += eps_used
        cand_0 = Candidate(code=r0_code, generation=0, fitness=fitness_0, rl_episodes=eps_used, valid=ok_0)
        evo.add_candidate(0, cand_0)
        for c in self._collect_ces(r0_code):
            seen_ces.setdefault(c["key"], c)

        reflection = build_reward_reflection(curve_0, comp_curves_0)
        current_parent_code = r0_code

        # Step 2: Generation 1 (K = 2 candidates, temperature=0.95)
        msgs_gen1 = []
        seeds_gen1 = []
        for k in range(self.pop_size):
            msgs = build_eureka_prompts(
                current_code=current_parent_code,
                reflection_text=reflection,
                task_desc=self.task_desc,
                api_desc=self.api_desc,
                generation=1,
            )
            msgs_gen1.append(msgs)
            seeds_gen1.append(seed_base + 10 + k * 4)

        outs_1 = llm_batch_chat(msgs_gen1, temperature=0.95, max_tokens=900, seeds=seeds_gen1)
        llm_calls += len(outs_1)

        for k, out in enumerate(outs_1):
            code_k = extract_code(out["text"])
            ok_k, fit_k, curve_k, comp_k, eps_k = self._eval_rl(code_k, seed=seed_base + 12 + k * 4)
            rl_episodes += eps_k
            refl_k = build_reward_reflection(curve_k, comp_k)
            cand_k = Candidate(code=code_k, generation=1, fitness=fit_k, reflection=refl_k, rl_episodes=eps_k, valid=ok_k)
            evo.add_candidate(1, cand_k)
            for c in self._collect_ces(code_k):
                seen_ces.setdefault(c["key"], c)

        # Parent selection for Generation 2: highest task_success_rate in Generation 1; fallback to R_0 if all fail/0
        gen1_cands = evo.generations.get(1, [])
        valid_gen1 = [c for c in gen1_cands if c.valid and c.fitness > 0]
        if valid_gen1:
            elite_gen1 = max(valid_gen1, key=lambda c: c.fitness)
        else:
            elite_gen1 = cand_0
        current_parent_code = elite_gen1.code
        reflection = elite_gen1.reflection or reflection

        # Step 3: Generation 2 (K = 2 candidates, temperature=0.35, strictly reaching N = 4)
        msgs_gen2 = []
        seeds_gen2 = []
        for k in range(self.pop_size):
            msgs = build_eureka_prompts(
                current_code=current_parent_code,
                reflection_text=reflection,
                task_desc=self.task_desc,
                api_desc=self.api_desc,
                generation=2,
            )
            msgs_gen2.append(msgs)
            seeds_gen2.append(seed_base + 30 + k * 4)

        outs_2 = llm_batch_chat(msgs_gen2, temperature=0.35, max_tokens=900, seeds=seeds_gen2)
        llm_calls += len(outs_2)

        for k, out in enumerate(outs_2):
            code_k = extract_code(out["text"])
            ok_k, fit_k, curve_k, comp_k, eps_k = self._eval_rl(code_k, seed=seed_base + 32 + k * 4)
            rl_episodes += eps_k
            refl_k = build_reward_reflection(curve_k, comp_k)
            cand_k = Candidate(code=code_k, generation=2, fitness=fit_k, reflection=refl_k, rl_episodes=eps_k, valid=ok_k)
            evo.add_candidate(2, cand_k)
            for c in self._collect_ces(code_k):
                seen_ces.setdefault(c["key"], c)

        # Final outcome: best candidate among generated candidates (Gen 1 and 2); tie-break by earlier candidate; fallback to R_0
        generated_cands = evo.generations.get(1, []) + evo.generations.get(2, [])
        valid_generated = [c for c in generated_cands if c.valid]
        if valid_generated:
            # max with stable sort keeps earlier candidate in case of tie
            best_overall = max(valid_generated, key=lambda c: c.fitness)
            final_code = best_overall.code
        else:
            best_overall = cand_0
            final_code = r0_code

        # Post-hoc Oracle evaluation (strictly independent)
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

        # Post-hoc recurrence count replay on R_final using 'margin' mode
        final_table = new_an.table if (new_an and new_an.table) else None
        if final_table is None and not self.is_continuous:
            try:
                final_table = build_reward_table(self.env, final_code)
            except Exception:
                final_table = None
        elif final_table is None and self.is_continuous:
            try:
                from cegr.core.code_sandbox import make_namespace, check_code
                check_code(final_code)
                ns = make_namespace(self.env)
                exec(compile(final_code, "<eval_final>", "exec"), ns)
                final_table = ns["reward_components"]
            except Exception:
                final_table = None

        recurrence_count = count_recurrence(self.env, final_table, seen_ces)

        gen_found = best_overall.generation if best_overall else 0
        rounds_to_pass = gen_found if ev["verified_fix"] else (self.generations + 1)

        return {
            "job_id": f"eureka__{reward_record['id']}__s{sample_idx}",
            "reward_id": reward_record["id"],
            "origin": reward_record.get("origin", "mutant"),
            "family": reward_record.get("family", "unknown"),
            "method": "EUREKA",
            "sample_idx": sample_idx,
            "passed_gates": bool(ev["verified_fix"]),
            "rounds_to_pass": rounds_to_pass,
            "llm_calls": llm_calls,
            "rl_runs": 1 + self.pop_size * self.generations,
            "rl_episodes": rl_episodes,
            "recurrence_count": recurrence_count,
            "vf": bool(ev["verified_fix"]),
            "tier1_safe": bool(ev["tier1_safe_aligned"]),
            "loose_dp": bool(ev["loose_dp"]),
            "category": ev["category"],
            "culprit_targeted": bool(ev["culprit_targeted"]),
            "rl_success": ev["probe_success_rate"],
            "hacked_dp": bool(new_an.hacked if new_an else False),
            "final_code": final_code,
            "waves": [{"wave": 1, "fitness": elite_gen1.fitness}, {"wave": 2, "fitness": best_overall.fitness}],
        }

    def run_batch(self, reward_records: List[Dict[str, Any]], n_samples: int = 1) -> List[Dict[str, Any]]:
        results = []
        total = len(reward_records) * n_samples
        count = 0
        for r in reward_records:
            for s in range(n_samples):
                count += 1
                res = self.run_single_job(r, s)
                results.append(res)
                gc.collect()
                if count % 10 == 0 or count == total:
                    print(f"[EUREKA] Tiến độ: {count}/{total} jobs ({count/total*100:.1f}%)")
        return results
