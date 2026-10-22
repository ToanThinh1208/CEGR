from typing import Dict, List, Tuple, Any
import numpy as np

from cegr.envs.doorkey import DoorKeyEnv
from cegr.core.code_sandbox import RewardTable


def learnability_probe(
    env: DoorKeyEnv,
    table: RewardTable,
    episodes: int = 1200,
    gamma: float = 0.99,
    alpha: float = 0.35,
    optimistic_init: float = 1.0,
    alpha_decay: bool = True,
    eval_every: int = 100,
    seed: int = 0,
) -> Tuple[List[Tuple[int, int]], List[int], bool]:
    """
    Fast and deterministic Tabular Q-learning probe.
    Tests if candidate reward signal is learnable by an RL agent within standard horizon.
    Returns learning curve, final greedy action trajectory, and final greedy success status.
    """
    rng = np.random.default_rng(seed)
    scale = max(1.0, float(np.max(np.abs(table.R))))
    R_norm = table.R / scale

    Q = float(optimistic_init) * np.ones((env.n, 6), dtype=np.float64)
    curve = []

    comp_evals = []

    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.65 * episodes))
        alpha_t = max(0.05, alpha * (1.0 - ep / episodes)) if alpha_decay else alpha

        i = 0
        for _ in range(env.horizon):
            a = int(rng.integers(6)) if rng.random() < eps else int(Q[i].argmax())
            j, g = int(env.nxt[i, a]), bool(env.goal_flag[i, a])
            target = R_norm[i, a] + (0.0 if g else gamma * Q[j].max())
            Q[i, a] += alpha_t * (target - Q[i, a])
            i = j
            if g:
                break

        if (ep + 1) % eval_every == 0:
            i, ok = 0, False
            c_dict = {k: 0.0 for k in table.names}
            for _ in range(env.horizon):
                a = int(Q[i].argmax())
                for ki, k in enumerate(table.names):
                    c_dict[k] += float(table.C[i, a, ki])
                j, g = int(env.nxt[i, a]), bool(env.goal_flag[i, a])
                i = j
                if g:
                    ok = True
                    break
            curve.append((ep + 1, int(ok)))
            comp_evals.append(c_dict)

    # Final greedy policy evaluation
    i, ok = 0, False
    final_actions = []
    for _ in range(env.horizon):
        a = int(Q[i].argmax())
        final_actions.append(a)
        j, g = int(env.nxt[i, a]), bool(env.goal_flag[i, a])
        i = j
        if g:
            ok = True
            break

    return curve, final_actions, ok, comp_evals


def run_probe(
    env: DoorKeyEnv,
    table: RewardTable,
    episodes: int = 1000,
    seeds: int = 2,
    seed_base: int = 100,
    eval_every: int = 100,
    gamma: float = 1.0,
    alpha: float = 0.35,
    optimistic_init: float = 1.0,
    alpha_decay: bool = True,
) -> Dict[str, Any]:
    """Runs multiple seeds of learnability probe and calculates tail success rate."""
    all_finals = []
    all_curves = []
    all_comp_evals = []

    for s_idx in range(seeds):
        sd = seed_base + s_idx
        curve, final_acts, ok, comp_evals = learnability_probe(
            env=env,
            table=table,
            episodes=episodes,
            seed=sd,
            eval_every=eval_every,
            gamma=gamma,
            alpha=alpha,
            optimistic_init=optimistic_init,
            alpha_decay=alpha_decay,
        )
        all_finals.append((final_acts, ok))
        all_curves.append([pt[1] for pt in curve])
        all_comp_evals.append(comp_evals)

    # Tail success rate (average success across last 25% of probe evaluation points across all seeds)
    tail_evals = []
    for c in all_curves:
        n_tail = max(1, len(c) // 4)
        tail_evals.extend(c[-n_tail:])

    tail_rate = float(np.mean(tail_evals)) if tail_evals else 0.0

    comp_curves = {}
    if all_comp_evals and all_comp_evals[0]:
        n_evals = len(all_comp_evals[0])
        for k in table.names:
            comp_curves[k] = [
                float(np.mean([seed_runs[idx].get(k, 0.0) for seed_runs in all_comp_evals]))
                for idx in range(n_evals)
            ]

    return {
        "tail": tail_rate,
        "finals": all_finals,
        "curve": [float(np.mean(vals)) for vals in zip(*all_curves)] if all_curves else [],
        "comp_curves": comp_curves,
        "success_fraction": float(np.mean([f[1] for f in all_finals])),
    }


evaluate_qlearning_probe_v2 = run_probe
