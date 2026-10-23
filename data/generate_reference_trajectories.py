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
        m_steps = env.get_milestone_trajectories(seed=42)
        t0_steps, t1_steps, t2_steps, t3_steps = m_steps[0], m_steps[1], m_steps[2], m_steps[3]

        trajs = {
            "T3": [st.a for st in t3_steps],
            "T2": [st.a for st in t2_steps],
            "T1": [st.a for st in t1_steps],
            "T0": [st.a for st in t0_steps],
            "T3_steps": t3_steps,
            "T2_steps": t2_steps,
            "T1_steps": t1_steps,
            "T0_steps": t0_steps,
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
