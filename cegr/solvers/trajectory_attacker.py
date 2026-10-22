import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from cegr.envs.base_env import Trajectory, Step
from cegr.envs.metaworld_env import MetaWorldEnvWrapper
from cegr.core.tpe_evaluator import classify_pair


class ContinuousTrajectoryAttacker:
    """
    Adversarial Trajectory Attacker for Continuous Control Domains (Meta-World).
    Uses Cross-Entropy Method (CEM) and action perturbation to discover reward-hacking trajectories
    that exploit shaping loopholes (e.g., hovering near object, gripper jitter, escaping boundary)
    without achieving the task goal.
    """

    def __init__(
        self,
        env: MetaWorldEnvWrapper,
        cem_iterations: int = 10,
        population_size: int = 40,
        elite_frac: float = 0.15,
        action_noise_std: float = 0.08,
    ):
        self.env = env
        self.cem_iterations = cem_iterations
        self.population_size = population_size
        self.elite_frac = elite_frac
        self.action_noise_std = action_noise_std
        self.action_dim = env.action_dim
        self.horizon = min(env.horizon, 150)  # fast search horizon

    def rollout_policy_mean(
        self, action_seq: np.ndarray, reward_fn: Callable[[Dict[str, Any], np.ndarray, Dict[str, Any]], Dict[str, float]]
    ) -> Tuple[float, bool, List[Step]]:
        s, _ = self.env.reset()
        steps = []
        tot_reward = 0.0
        success = False

        for t in range(len(action_seq)):
            a = action_seq[t]
            s2, r_env, done, trunc, info = self.env.step(a)
            comps = reward_fn(s, a, s2)
            if not isinstance(comps, dict):
                comps = {"total": float(comps)}
            r_val = sum(float(v) for v in comps.values())
            tot_reward += r_val
            g = bool(s2.get("in_target", False))
            if g:
                success = True
            steps.append(Step(t=t, s=s, a=a, s2=s2, r=r_val, comps=comps, goal=g))
            s = s2
            if g or done:
                break
        return tot_reward, success, steps

    def attack(
        self,
        reward_fn: Callable[[Dict[str, Any], np.ndarray, Dict[str, Any]], Dict[str, float]],
        ref_expert_traj: List[Step],
        rnd: int = 0,
        classify: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Executes CEM search for an action sequence that maximizes candidate reward_fn
        while penalizing goal completion (finding purely exploitative policies).
        """
        H = min(len(ref_expert_traj), self.horizon)
        # Evaluate expert reference trajectory under candidate reward_fn
        ref_return = 0.0
        for st in ref_expert_traj:
            c = reward_fn(st.s, st.a, st.s2)
            ref_return += sum(c.values()) if isinstance(c, dict) else float(c)

        # CEM initialization
        mean = np.zeros((H, self.action_dim), dtype=np.float32)
        std = np.ones((H, self.action_dim), dtype=np.float32) * 0.4
        n_elite = max(2, int(self.population_size * self.elite_frac))

        best_exploit_return = -np.inf
        best_exploit_steps = None

        for _ in range(self.cem_iterations):
            samples = np.random.normal(mean, std, size=(self.population_size, H, self.action_dim))
            samples = np.clip(samples, -1.0, 1.0)
            scores = []
            rollout_records = []

            for pop_idx in range(self.population_size):
                act_seq = samples[pop_idx]
                r_cand, succ, steps = self.rollout_policy_mean(act_seq, reward_fn)
                # Attacker objective: maximize reward_fn, but heavily penalize actual goal completion
                score = r_cand - (1e4 if succ else 0.0)
                scores.append(score)
                rollout_records.append((r_cand, succ, steps))

            scores = np.array(scores)
            elite_idxs = np.argsort(scores)[-n_elite:]
            mean = np.mean(samples[elite_idxs], axis=0)
            std = np.std(samples[elite_idxs], axis=0) + self.action_noise_std

            # Check best elite
            top_idx = elite_idxs[-1]
            r_top, succ_top, steps_top = rollout_records[top_idx]
            if not succ_top and r_top > best_exploit_return:
                best_exploit_return = r_top
                best_exploit_steps = steps_top

        # If best non-goal policy scores higher or near expert return, generate counterexample
        ces = []
        if best_exploit_steps is not None:
            typ = classify_pair(ref_return, best_exploit_return, eps_rel=0.02, better_is_success=True)
            if typ is not None:
                better_acts = [tuple(st.a) if hasattr(st.a, "__iter__") else st.a for st in ref_expert_traj]
                worse_acts = [tuple(st.a) if hasattr(st.a, "__iter__") else st.a for st in best_exploit_steps]
                key_str = f"cont_atk_r{rnd}_{ref_return:.2f}_{best_exploit_return:.2f}"
                key = hashlib.md5(key_str.encode()).hexdigest()[:12]
                ces.append({
                    "key": key,
                    "better": better_acts,
                    "worse": worse_acts,
                    "better_steps": ref_expert_traj,
                    "worse_steps": best_exploit_steps,
                    "type": typ if classify else "generic",
                    "round": int(rnd),
                    "tag": f"continuous_cem_attack_{typ}",
                    "ref_return": ref_return,
                    "exploit_return": best_exploit_return,
                })

        return ces
