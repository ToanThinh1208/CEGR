import os
import pickle
import numpy as np

from cegr.envs.doorkey import make_env, build_reference_trajs
from cegr.envs.metaworld_env import make_metaworld_env


def generate_doorkey_trajectories(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name in ["doorkey5", "doorkey6", "doorkey8"]:
        env = make_env(name)
        refs = build_reference_trajs(env)
        file_name = f"{name.replace('doorkey', 'doorkey_')}_p0.pkl" if "_" not in name else f"{name}_p0.pkl"
        target_path = os.path.join(out_dir, file_name)
        with open(target_path, "wb") as f:
            pickle.dump(refs, f)
        print(f"Generated {target_path}")


def generate_metaworld_trajectories(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tasks = ["reach-v2", "door-open-v2", "pick-place-v2"]
    for task in tasks:
        env = make_metaworld_env(task, horizon=150, use_mock=True)
        # Generate T3 (expert demo), T2 (suboptimal hold), T1 (jitter), T0 (flailing)
        # 1. T3: actions moving hand toward object and goal
        t3_actions = [np.array([0.0, 0.1, -0.05, -0.8], dtype=np.float32) for _ in range(50)]
        # 2. T2: moves to object and holds
        t2_actions = [np.array([0.0, 0.1, -0.05, 0.0], dtype=np.float32) for _ in range(50)]
        # 3. T1: jittering near start
        t1_actions = [np.array([0.0, 0.0, 0.0, 0.8 if i % 2 == 0 else -0.8], dtype=np.float32) for i in range(50)]
        # 4. T0: moving away
        t0_actions = [np.array([-0.5, -0.5, 0.2, 0.0], dtype=np.float32) for _ in range(50)]

        from cegr.envs.base_env import Step
        def rollout_to_steps(action_seq):
            s, _ = env.reset(seed=42)
            steps = []
            for t, a in enumerate(action_seq):
                s2, r, done, trunc, _ = env.step(a)
                g = bool(s2.get("in_target", False))
                steps.append(Step(t=t, s=s, a=a, s2=s2, r=1.0 if g else 0.0, comps={"goal": 1.0 if g else 0.0}, goal=g))
                s = s2
                if g or done:
                    break
            return steps

        trajs = {
            "T3": t3_actions, "T2": t2_actions, "T1": t1_actions, "T0": t0_actions,
            "T3_steps": rollout_to_steps(t3_actions),
            "T2_steps": rollout_to_steps(t2_actions),
            "T1_steps": rollout_to_steps(t1_actions),
            "T0_steps": rollout_to_steps(t0_actions),
        }
        target_name = f"{task.replace('-', '_')}_p0.pkl"
        target_path = os.path.join(out_dir, target_name)
        with open(target_path, "wb") as f:
            pickle.dump(trajs, f)
        print(f"Generated {target_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generate_doorkey_trajectories(os.path.join(base_dir, "doorkey", "reference_trajectories"))
    generate_metaworld_trajectories(os.path.join(base_dir, "metaworld", "reference_trajectories"))
