from typing import Any, Dict, List, Optional
import gc
import json
import numpy as np

from cegr.core.base_runner import BaseRunner
from cegr.envs.doorkey import (
    DoorKeyEnv, build_expert_tables, build_reference_trajs, sample_noisy,
    TASK_DESC, API_DESC
)
from cegr.engine.llm_vllm import llm_batch_chat, extract_code
from cegr.synthesis.ce_pool import CEPool, make_ce
from cegr.synthesis.dynamic_anchors import DynamicAnchorsManager
from cegr.synthesis.assimilator import extract_experience
from cegr.synthesis.gatekeeper import GateKeeper, run_gatekeeper, build_feedback
from cegr.solvers.dp_solver import analyze
from cegr.solvers.verified_fix import evaluate_verified_fix


class Kit:
    pass


def build_kit(env: DoorKeyEnv, ce_seed: int = 1000, signal_seed: int = 2000, eval_seed: int = 3000) -> Kit:
    k = Kit()
    k.expert = build_expert_tables(env)
    k.refs = build_reference_trajs(env)
    from cegr.core.tpe_evaluator import build_p0_pairs
    p0_pairs = build_p0_pairs(env, seed=ce_seed, max_noisy=20)
    k.p0 = [
        make_ce(
            p["better"], p["worse"], p["type"], 0, p["tag"],
            better_steps=p.get("better_steps"), worse_steps=p.get("worse_steps"),
        )
        for p in p0_pairs
    ]
    k.signal_trajs = list(k.refs.values()) + sample_noisy(env, k.expert, signal_seed, 20)
    k.eval_trajs = list(k.refs.values()) + sample_noisy(env, k.expert, eval_seed, 30)
    return k


def build_metaworld_kit(env: Any, ce_seed: int = 1000, signal_seed: int = 2000, eval_seed: int = 3000) -> Kit:
    k = Kit()
    from cegr.core.tpe_evaluator import build_p0_pairs
    p0_pairs = build_p0_pairs(env, seed=ce_seed, max_noisy=20)
    k.p0 = [
        make_ce(
            p["better"], p["worse"], p["type"], 0, p["tag"],
            better_steps=p.get("better_steps"), worse_steps=p.get("worse_steps"),
        )
        for p in p0_pairs
    ]
    t3 = [p["better_steps"] for p in p0_pairs if p["tag"] == "T3>T2"][0]
    t2 = [p["worse_steps"] for p in p0_pairs if p["tag"] == "T3>T2"][0]
    t1 = [p["worse_steps"] for p in p0_pairs if p["tag"] == "T2>T1"][0]
    t0 = [p["worse_steps"] for p in p0_pairs if p["tag"] == "T1>T0"][0]
    k.refs = {"T3": t3, "T2": t2, "T1": t1, "T0": t0}
    k.expert_steps = t3
    k.signal_trajs = [t3, t2, t1, t0]
    k.eval_trajs = [t3, t2, t1, t0]
    return k


def candidate_score(res: Dict[str, Any]) -> int:
    """Candidate score is based strictly on gate depth (gates_passed)."""
    return int(res.get("gates_passed", 0))


def count_recurrence(env: Any, kit: Any, table: Any, seen: Dict[str, Any]) -> int:
    """Calculates number of previously encountered counterexamples violated by candidate table."""
    if table is None or not seen:
        return 0
    from cegr.envs.doorkey import traj_return
    from cegr.core.tpe_evaluator import pair_margin, pair_violated
    ret_t3 = traj_return(env, table, kit.refs["T3"]) if kit and hasattr(kit, "refs") and "T3" in kit.refs else 1.0
    eps = max(1e-3, 0.02 * abs(ret_t3))
    return sum(
        1 for c in seen.values()
        if pair_violated(pair_margin(env, table, c), c, eps, True, mode="margin", env=env, table=table)
    )


REPAIR_INSTRUCTION = """Your job: produce a reward `reward_components(s, a, s2)` whose optimal policy reaches the goal within the horizon and that gives a learnable, dense signal.
Strict Requirements:
- Keep it a DENSE reward with meaningful sub-goal shaping (potential difference `phi(s2) - phi(s)` for getting the key, opening the door, or reducing distance). Do NOT return a sparse goal-only reward.
- Do NOT hard-code coordinates; use only the provided API constants and helpers.
- Reply with ONLY one Python code block defining `reward_components(s, a, s2)`."""


class CegrRunner(BaseRunner):
    """
    CEGR Runner: Coordinates CounterExample-Guided Reward Design with Multi-Tier GateKeeper.
    Supports both the proposed full CEGR protocol and all 6 ablation variants:
      - CEGR_noPool, CEGR_noAnchors, CEGR_noClass, CEGR_noG2, CEGR_noG4, CEGR_noG5
    """

    def __init__(
        self,
        env: Any,
        config: Dict[str, Any],
        method_name: str = "CEGR",
        variant_cfg: Optional[Dict[str, Any]] = None,
        task_desc: str = "",
        api_desc: str = "",
    ):
        super().__init__(config)
        self.env = env
        self.method_name = method_name
        default_task = TASK_DESC if isinstance(env, DoorKeyEnv) else ""
        default_api = API_DESC if isinstance(env, DoorKeyEnv) else ""
        self.task_desc = task_desc or config.get("task_desc", "") or default_task or TASK_DESC
        self.api_desc = api_desc or config.get("api_desc", "") or default_api or API_DESC
        self.kit = build_kit(env) if isinstance(env, DoorKeyEnv) else build_metaworld_kit(env)
        self.gk = GateKeeper(env, self.kit)

        # Method configuration (full CEGR or ablation)
        self.cfg = {
            "gates": [1, 2, 3, 4, 5],
            "pool": True,
            "attacker": True,
            "classify": True,
            "anchors": "dynamic",
            "experience": True,
            "feedback": "cegr",
            "cands": 1,
            "probe_episodes": 1000,
            "probe_seeds": 2,
        }
        # Pre-apply defaults for known ablation method names
        if self.method_name == "CEGR_noPool":
            self.cfg["pool"] = False
        elif self.method_name == "CEGR_noAnchors":
            self.cfg["anchors"] = "static"
        elif self.method_name == "CEGR_noClass":
            self.cfg["classify"] = False
        elif self.method_name == "CEGR_noG2":
            self.cfg["gates"] = [1, 3, 4, 5]
        elif self.method_name == "CEGR_noG4":
            self.cfg["gates"] = [1, 2, 3, 5]
        elif self.method_name == "CEGR_noG5":
            self.cfg["gates"] = [1, 2, 3, 4]
        elif self.method_name == "CEGR_noFeedback":
            self.cfg["feedback"] = "none"
            self.cfg["pool"] = False
            self.cfg["anchors"] = "none"
            self.cfg["experience"] = False

        if variant_cfg:
            self.cfg.update(variant_cfg)

        # Ensure noAnchors strictly keeps v0 without upgrading per SYSTEM_LOGIC_SPEC.md Sec 2.4F
        if self.method_name == "CEGR_noAnchors":
            self.cfg["anchors"] = "static"

    def _build_prompt(
        self,
        current_code: Optional[str],
        feedback: str,
        wave: int,
        anchors_mgr: DynamicAnchorsManager,
        experience: List[str],
        persona: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        system = "You are an expert in reinforcement learning reward design and formal verification."
        blocks = [b for b in [self.task_desc, self.api_desc] if b]
        if self.cfg.get("feedback") != "none":
            if self.cfg.get("anchors") != "none":
                blocks.append(anchors_mgr.render())
            if self.cfg.get("experience") and experience:
                blocks.append("LEARNED RULES FROM PAST FAILURES (apply them):\n" + "".join(f"  - {r}\n" for r in experience[-5:]))

        if not current_code:
            st = persona or "dense shaping combining step penalty, distance reduction to subgoals, and a bonus whenever the agent changes position or interacts with the door"
            blocks.append(
                f"Write a DENSE reward function `reward_components(s, a, s2)` using 3 to 5 named components.\n"
                f"Design style to implement: {st}.\nReply with ONLY one Python code block defining `reward_components(s, a, s2)`."
            )
        else:
            note = "" if wave <= 1 else f"[ROUND {wave}/{self.max_rounds} - the previous candidate still failed verification.]\n\n"
            if self.cfg.get("feedback") == "none":
                # noFeedback: resample under same budget without feedback or failure hints
                blocks.append(
                    f"{note}CURRENT REWARD CODE:\n```python\n{current_code.strip()}\n```\n\n"
                    f"Please improve this reward function to pass all verification requirements and achieve learnability.\n{REPAIR_INSTRUCTION}"
                )
            else:
                blocks.append(
                    f"{note}CURRENT REWARD CODE:\n```python\n{current_code.strip()}\n```\n\n"
                    f"VERIFIER FEEDBACK:\n{feedback[:4500]}\n{REPAIR_INSTRUCTION}"
                )
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(blocks)}]

    def run_single_job(self, reward_record: Dict[str, Any], sample_idx: int = 0) -> Dict[str, Any]:
        pool = CEPool(max_size=40)
        domain_str = "doorkey" if isinstance(self.env, DoorKeyEnv) else "metaworld"
        anchors_mgr = DynamicAnchorsManager(domain=domain_str)
        experience: List[str] = []
        current_code = reward_record.get("code", "")
        r0_code = current_code
        llm_calls = 0
        rl_episodes = 0
        passed = False
        pass_round = None
        feedback = ""
        waves_log = []

        # Round 0: Evaluate initial R_0 without consuming LLM calls
        seed_base = 1000 + (sample_idx % 10) * 80
        init_res = run_gatekeeper(current_code, self.cfg, pool, 0, self.env, self.kit, self.gk, seed=seed_base)
        rl_episodes += init_res["rl_episodes"]
        best_cand: Optional[Tuple[int, str, int]] = None
        if init_res["passed"]:
            passed = True
            pass_round = 0

        if not passed:
            feedback = build_feedback(init_res, self.cfg, pool, self.env, self.kit)

        seen_ce = {}
        total_recur = 0

        for rnd in range(1, self.max_rounds + 1):
            if passed:
                break

            # Prompt LLM for repair (seed stream 1000)
            msgs = self._build_prompt(current_code, feedback, rnd, anchors_mgr, experience)
            seed_val = seed_base + rnd * 13
            out = llm_batch_chat([msgs], temperature=0.35, max_tokens=900, seeds=[seed_val])[0]
            llm_calls += 1
            current_code = extract_code(out["text"])

            # Verify new candidate through GateKeeper
            res = run_gatekeeper(current_code, self.cfg, pool, rnd, self.env, self.kit, self.gk, seed=seed_val)
            rl_episodes += res["rl_episodes"]

            # Track best candidate by gate depth; tie-break by latest round
            g_depth = res.get("gates_passed", 0)
            if g_depth >= 1 and (best_cand is None or g_depth >= best_cand[0]):
                best_cand = (g_depth, current_code, rnd)

            recur = count_recurrence(self.env, self.kit, res.get("table"), seen_ce)
            total_recur += recur
            for c in res.get("new_ces", []):
                seen_ce.setdefault(c["key"], c)

            if self.cfg.get("pool") and not res["passed"]:
                for ce in res["new_ces"]:
                    pool.add(ce)

            if res["passed"]:
                passed = True
                pass_round = rnd
                waves_log.append({"wave": rnd, "passed": True, "failed_gate": None, "recur": recur})
                break

            # Failure handling: extract experience & update dynamic anchors
            if self.cfg.get("experience"):
                rules, invs = extract_experience(res)
                for r in rules:
                    if r not in experience:
                        experience.append(r)
                if self.cfg.get("anchors") == "dynamic":
                    anchors_mgr.update(invs)

            feedback = build_feedback(res, self.cfg, pool, self.env, self.kit)
            waves_log.append({
                "wave": rnd,
                "passed": False,
                "failed_gate": res["failed_gate"],
                "gates_passed": res.get("gates_passed", 0),
                "recur": recur,
                "pool_size": len(pool),
            })

        # Selection when not PASS: candidate passing G1 with highest gate depth; fallback to R_0
        if passed:
            final_code = current_code
        else:
            final_code = best_cand[1] if best_cand is not None else r0_code

        # Independent Oracle Verified Fix Evaluation
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

        final_table = new_an.table if (new_an and new_an.table) else self.gk.get_table(final_code)[0]
        post_hoc_recur = count_recurrence(self.env, self.kit, final_table, seen_ce)

        return {
            "job_id": f"{self.method_name.lower()}__{reward_record['id']}__s{sample_idx}",
            "reward_id": reward_record["id"],
            "origin": reward_record.get("origin", "mutant"),
            "family": reward_record.get("family", "unknown"),
            "method": self.method_name,
            "sample_idx": sample_idx,
            "passed_gates": passed,
            "rounds_to_pass": pass_round if passed else self.max_rounds + 1,
            "llm_calls": llm_calls,
            "architect_calls": 0,
            "recurrence_count": post_hoc_recur,
            "rl_runs": 1,
            "rl_episodes": rl_episodes,
            "vf": bool(ev["verified_fix"]),
            "tier1_safe": bool(ev["tier1_safe_aligned"]),
            "loose_dp": bool(ev["loose_dp"]),
            "category": ev["category"],
            "culprit_targeted": bool(ev["culprit_targeted"]),
            "rl_success": ev["probe_success_rate"],
            "hacked_dp": bool(new_an.hacked if new_an else False),
            "final_code": final_code,
            "pool_size": len(pool),
            "waves": waves_log,
        }

    def run_batch(self, reward_records: List[Dict[str, Any]], n_samples: int = 1) -> List[Dict[str, Any]]:
        """Sequential execution across reward records (convenient for local tests)."""
        results = []
        for r in reward_records:
            for s in range(n_samples):
                results.append(self.run_single_job(r, s))
                gc.collect()
        return results

    def run_wave_batch(
        self,
        reward_records: List[Dict[str, Any]],
        n_samples: int = 1,
        label: str = "main",
    ) -> List[Dict[str, Any]]:
        """
        Coordinates full Wave Batching search across all active jobs.
        In each wave w (1..max_rounds), aggregates prompts from all unfinished jobs into a single
        batched call to llm_batch_chat (vLLM), matching Section 3.4 & 4.5 of the experimental report.
        """
        jobs = []
        for r in reward_records:
            for s in range(n_samples):
                jobs.append({
                    "job_id": f"{self.method_name.lower()}__{r['id']}__s{s}",
                    "start": r,
                    "sample_idx": s,
                    "code": r.get("code"),
                    "orig_code": r.get("code", ""),
                    "pool": CEPool(max_size=self.cfg.get("pool_max", 40)),
                    "anchors_mgr": DynamicAnchorsManager(domain="doorkey" if isinstance(self.env, DoorKeyEnv) else "metaworld"),
                    "experience": [],
                    "seen": {},
                    "feedback": "",
                    "stall": 0,
                    "done": False,
                    "passed": False,
                    "pass_wave": None,
                    "pass_code": None,
                    "best": None,
                    "waves": [],
                    "llm_calls": 0,
                    "rl_runs": 0,
                    "rl_episodes": 0,
                })

        # Wave 0: Initial R_0 verification without LLM calls
        for jb in jobs:
            if jb["code"]:
                seed_r0 = 1000 + (jb["sample_idx"] % 10) * 80
                init_res = run_gatekeeper(
                    jb["code"], self.cfg, jb["pool"], 0, self.env, self.kit, self.gk, seed=seed_r0
                )
                jb["rl_runs"] += init_res["rl_runs"]
                jb["rl_episodes"] += init_res["rl_episodes"]
                g_depth = init_res.get("gates_passed", 0)
                if init_res["passed"]:
                    jb["done"] = True
                    jb["passed"] = True
                    jb["pass_wave"] = 0
                    jb["pass_code"] = jb["code"]
                    jb["waves"].append({"wave": 0, "passed": True, "failed_gate": None, "gates_passed": g_depth, "recur": 0})
                else:
                    jb["feedback"] = build_feedback(init_res, self.cfg, jb["pool"], self.env, self.kit)

        for w in range(1, self.max_rounds + 1):
            active = [jb for jb in jobs if not jb["done"]]
            if not active:
                break

            # Collect prompts for all active jobs (seed stream 1000)
            prompts_list = []
            seeds_list = []
            for jb in active:
                msgs = self._build_prompt(
                    jb["code"], jb["feedback"], w, jb["anchors_mgr"], jb["experience"],
                    persona=jb["start"].get("persona")
                )
                prompts_list.append(msgs)
                seeds_list.append(1000 + (jb["sample_idx"] % 10) * 80 + w * 13)

            outs = llm_batch_chat(
                prompts_list, temperature=self.config.get("llm", {}).get("temperature_repair", 0.35),
                max_tokens=self.config.get("llm", {}).get("max_tokens", 900), seeds=seeds_list
            )

            for jb, out in zip(active, outs):
                jb["llm_calls"] += 1
                cand_code = extract_code(out["text"])
                res = run_gatekeeper(
                    cand_code, self.cfg, jb["pool"], w, self.env, self.kit, self.gk, seed=seeds_list[active.index(jb)]
                )
                jb["rl_runs"] += res["rl_runs"]
                jb["rl_episodes"] += res["rl_episodes"]

                recur = count_recurrence(self.env, self.kit, res.get("table"), jb["seen"])
                if res.get("obs"):
                    for c in res["obs"].get("p0_viol_all", []) + res["obs"].get("attack", []):
                        jb["seen"].setdefault(c["key"], c)
                for c in res.get("new_ces", []):
                    jb["seen"].setdefault(c["key"], c)

                if self.cfg.get("pool") and not res["passed"]:
                    for c in res.get("new_ces", []):
                        jb["pool"].add(c)

                g_depth = res.get("gates_passed", 0)
                if g_depth >= 1:
                    if jb["best"] is None or g_depth >= jb["best"][1]:
                        jb["best"] = (cand_code, g_depth, w)

                jb["code"] = cand_code
                jb["waves"].append({
                    "wave": w,
                    "passed": res["passed"],
                    "failed_gate": res["failed_gate"],
                    "gates_passed": res["gates_passed"],
                    "recur": recur,
                    "pool_size": len(jb["pool"]),
                    "probe_tail": res.get("probe_tail", 0.0),
                })

                if res["passed"]:
                    jb["done"] = True
                    jb["passed"] = True
                    jb["pass_wave"] = w
                    jb["pass_code"] = cand_code
                    continue

                jb["feedback"] = build_feedback(res, self.cfg, jb["pool"], self.env, self.kit)
                if self.cfg.get("experience"):
                    rules, invs = extract_experience(res)
                    for r in rules:
                        if r not in jb["experience"]:
                            jb["experience"].append(r)
                    if self.cfg.get("anchors") == "dynamic":
                        jb["anchors_mgr"].update(invs)
            gc.collect()

        # Final evaluation
        final_results = []
        for jb in jobs:
            final_code = jb["pass_code"] if jb["passed"] else (jb["best"][0] if jb["best"] else jb["orig_code"])
            orig_an = analyze(self.env, jb["orig_code"], cache_table=True) if isinstance(self.env, DoorKeyEnv) else None
            new_an = analyze(self.env, final_code, cache_table=True) if isinstance(self.env, DoorKeyEnv) else None
            ev = evaluate_verified_fix(
                env=self.env,
                orig_an=orig_an,
                orig_code=jb["orig_code"],
                new_code=final_code,
                run_probe=True,
                new_an=new_an,
            )
            final_table = new_an.table if (new_an and new_an.table) else self.gk.get_table(final_code)[0]
            post_hoc_recur = count_recurrence(self.env, self.kit, final_table, jb["seen"])
            final_results.append({
                "job_id": jb["job_id"],
                "reward_id": jb["start"]["id"],
                "origin": jb["start"].get("origin", "mutant"),
                "family": jb["start"].get("family", "unknown"),
                "method": self.method_name,
                "sample_idx": jb["sample_idx"],
                "passed_gates": jb["passed"],
                "rounds_to_pass": jb["pass_wave"] if jb["passed"] else self.max_rounds + 1,
                "llm_calls": jb["llm_calls"],
                "architect_calls": 0,
                "recurrence_count": post_hoc_recur,
                "rl_runs": jb["rl_runs"],
                "rl_episodes": jb["rl_episodes"],
                "vf": bool(ev["verified_fix"]),
                "tier1_safe": bool(ev["tier1_safe_aligned"]),
                "loose_dp": bool(ev["loose_dp"]),
                "category": ev["category"],
                "culprit_targeted": bool(ev["culprit_targeted"]),
                "rl_success": ev["probe_success_rate"],
                "hacked_dp": bool(new_an.hacked if new_an else False),
                "final_code": final_code,
                "pool_size": len(jb["pool"]),
                "waves": jb["waves"],
            })
        return final_results
