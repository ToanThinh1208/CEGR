from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from cegr.envs.metaworld_env import MetaWorldEnvWrapper


class ContinuousRLProbe:
    """
    Continuous RL Probe (Gate G5 and Post-Hoc Oracle Evaluation for Meta-World).
    Trains an agent on continuous state-action spaces to test policy learnability
    and detect unlearnable or degenerate continuous reward shapings.
    """

    def __init__(
        self,
        env: MetaWorldEnvWrapper,
        algorithm: str = "sac",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
    ):
        self.env = env
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.gamma = gamma

    def probe(
        self,
        reward_fn: Callable[[Dict[str, Any], np.ndarray, Dict[str, Any]], Dict[str, float]],
        total_steps: int = 10000,
        eval_episodes: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Runs short continuous policy optimization probe.
        Uses policy gradient / cross-entropy parameter search on linear policy weights for rapid, robust checking.
        """
        rng = np.random.default_rng(seed)
        obs_dim = 39
        act_dim = self.env.action_dim
        h_dim = 64

        # Parameterize 2-hidden-layer MLP policy (MLP 64-64)
        W1 = rng.normal(0, np.sqrt(2.0 / obs_dim), size=(h_dim, obs_dim)).astype(np.float32)
        b1 = np.zeros(h_dim, dtype=np.float32)
        W2 = rng.normal(0, np.sqrt(2.0 / h_dim), size=(h_dim, h_dim)).astype(np.float32)
        b2 = np.zeros(h_dim, dtype=np.float32)
        W3 = rng.normal(0, np.sqrt(2.0 / h_dim), size=(act_dim, h_dim)).astype(np.float32)
        b3 = np.zeros(act_dim, dtype=np.float32)

        def forward(x):
            h1 = np.tanh(W1 @ x + b1)
            h2 = np.tanh(W2 @ h1 + b2)
            out = np.tanh(W3 @ h2 + b3)
            return out, h1, h2

        def get_action(raw_obs, noise_scale=0.1):
            mean, _, _ = forward(raw_obs)
            if noise_scale > 0:
                mean = mean + rng.normal(0, noise_scale, size=act_dim)
            return np.clip(mean, -1.0, 1.0)

        step_count = 0
        curve = []
        succ_history = []
        comp_history: Dict[str, List[float]] = {}
        batch_size = 5

        # Policy optimization loop with batch advantage baseline
        while step_count < total_steps:
            batch_trajs = []
            for _ in range(batch_size):
                if step_count >= total_steps:
                    break
                s, _ = self.env.reset(seed=seed + step_count)
                ep_return = 0.0
                ep_steps = 0
                ep_states, ep_actions, ep_h1, ep_h2, ep_rewards = [], [], [], [], []
                ep_succ = False
                ep_comp: Dict[str, float] = {}

                for _ in range(min(self.env.horizon, 100)):
                    raw_obs = s.get("raw_obs", np.zeros(obs_dim, dtype=np.float32))
                    a_pred, h1, h2 = forward(raw_obs)
                    noise = rng.normal(0, 0.15, size=act_dim).astype(np.float32)
                    a = np.clip(a_pred + noise, -1.0, 1.0)
                    s2, _, done, trunc, _ = self.env.step(a)
                    comps = reward_fn(s, a, s2)
                    if not isinstance(comps, dict):
                        comps = {"total": float(comps)}
                    r = sum(float(v) for v in comps.values())

                    for k, v in comps.items():
                        ep_comp[k] = ep_comp.get(k, 0.0) + float(v)

                    ep_states.append(raw_obs)
                    ep_actions.append(a)
                    ep_h1.append(h1)
                    ep_h2.append(h2)
                    ep_rewards.append(r)
                    ep_return += r
                    ep_steps += 1
                    step_count += 1
                    s = s2
                    if s2.get("in_target", False):
                        ep_succ = True
                        break
                    if done or trunc:
                        break

                succ_history.append(1.0 if ep_succ else 0.0)
                for k, v in ep_comp.items():
                    comp_history.setdefault(k, []).append(v)
                curve.append(ep_return)

                batch_trajs.append({
                    "states": ep_states,
                    "actions": ep_actions,
                    "h1": ep_h1,
                    "h2": ep_h2,
                    "total_return": ep_return,
                })

            if not batch_trajs:
                break

            # Batch Advantage Normalization across trajectories in batch
            all_ep_returns = np.array([tr["total_return"] for tr in batch_trajs], dtype=np.float32)
            mean_ret = float(np.mean(all_ep_returns))
            std_ret = float(np.std(all_ep_returns)) + 1e-6

            g_W1, g_b1 = np.zeros_like(W1), np.zeros_like(b1)
            g_W2, g_b2 = np.zeros_like(W2), np.zeros_like(b2)
            g_W3, g_b3 = np.zeros_like(W3), np.zeros_like(b3)
            total_batch_steps = sum(len(tr["states"]) for tr in batch_trajs)

            for tr in batch_trajs:
                traj_adv = (tr["total_return"] - mean_ret) / std_ret
                for st_obs, act, h1, h2 in zip(tr["states"], tr["actions"], tr["h1"], tr["h2"]):
                    pred = np.tanh(W3 @ h2 + b3)
                    delta3 = (act - pred) * (1.0 - pred ** 2) * traj_adv
                    g_W3 += np.outer(delta3, h2)
                    g_b3 += delta3

                    delta2 = (W3.T @ delta3) * (1.0 - h2 ** 2)
                    g_W2 += np.outer(delta2, h1)
                    g_b2 += delta2

                    delta1 = (W2.T @ delta2) * (1.0 - h1 ** 2)
                    g_W1 += np.outer(delta1, st_obs)
                    g_b1 += delta1

            lr = self.learning_rate
            scale = max(1, total_batch_steps)
            W3 += lr * np.clip(g_W3 / scale, -1.0, 1.0)
            b3 += lr * np.clip(g_b3 / scale, -1.0, 1.0)
            W2 += lr * np.clip(g_W2 / scale, -1.0, 1.0)
            b2 += lr * np.clip(g_b2 / scale, -1.0, 1.0)
            W1 += lr * np.clip(g_W1 / scale, -1.0, 1.0)
            b1 += lr * np.clip(g_b1 / scale, -1.0, 1.0)

        # Evaluation rollouts
        from cegr.envs.base_env import Step
        eval_successes = []
        eval_returns = []
        failing_steps: Optional[List[Step]] = None

        for eval_idx in range(eval_episodes):
            s, _ = self.env.reset(seed=seed + 9999 + eval_idx)
            succ = False
            ret = 0.0
            rollout_steps = []
            for t in range(min(self.env.horizon, 150)):
                raw_obs = s.get("raw_obs", np.zeros(obs_dim, dtype=np.float32))
                a = get_action(raw_obs, noise_scale=0.0)
                s2, _, done, trunc, _ = self.env.step(a)
                comps = reward_fn(s, a, s2)
                r = sum(float(v) for v in comps.values()) if isinstance(comps, dict) else float(comps)
                ret += r
                g = bool(s2.get("in_target", False))
                rollout_steps.append(Step(t=t, s=s, a=a, s2=s2, r=r, comps=comps if isinstance(comps, dict) else {"total": r}, goal=g))
                if g:
                    succ = True
                    break
                s = s2
                if done or trunc:
                    break
            eval_successes.append(succ)
            eval_returns.append(ret)
            if not succ and failing_steps is None:
                failing_steps = rollout_steps

        tail_rate = float(np.mean(eval_successes)) if eval_successes else 0.0

        # Sample 5 checkpoint curves for Eureka reflection
        def to_5_points(seq):
            if not seq:
                return [0.0] * 5
            n = len(seq)
            return [float(np.mean(seq[max(0, int(n * f) - 5):int(n * f) + 1])) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]

        succ_curve = to_5_points(succ_history)
        comp_curves = {k: to_5_points(v) for k, v in comp_history.items()}

        return {
            "tail": tail_rate,
            "mean_eval_return": float(np.mean(eval_returns)) if eval_returns else 0.0,
            "success_rate": tail_rate,
            "curve": succ_curve,
            "comp_curves": comp_curves,
            "total_steps": step_count,
            "failing_steps": failing_steps,
        }
