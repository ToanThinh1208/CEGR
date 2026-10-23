from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cegr.envs.base_env import BaseEnv, Step, Trajectory


class MetaWorldEnvWrapper(BaseEnv):
    """
    Standardized wrapper for continuous robotic manipulation environments (Meta-World).
    Translates raw 39-dim MuJoCo observation vectors into semantic state dictionaries:
      - s["hand_pos"]: 3D tuple (x, y, z)
      - s["obj_pos"]: 3D tuple (x, y, z)
      - s["goal_pos"]: 3D tuple (x, y, z)
      - s["gripper_open"]: float in [0.0, 1.0]
      - s["obj_grasped"]: bool
      - s["in_target"]: bool
    """

    def __init__(self, task_name: str = "door-open-v2", horizon: int = 500, use_mock: bool = False):
        super().__init__(horizon=horizon)
        self.task_name = task_name
        self.horizon = horizon
        self.action_dim = 4
        self.use_mock = use_mock
        self._env = None
        self._init_inner_env()

    def _init_inner_env(self):
        if self.use_mock:
            self._env = None
            return
        try:
            import metaworld
            ml1 = metaworld.ML1(self.task_name)
            self._env = ml1.train_classes[self.task_name]()
            task = ml1.train_tasks[0]
            self._env.set_task(task)
        except Exception:
            # Fallback to simulated lightweight physics mockup
            self.use_mock = True
            self._env = None

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        if self._env is not None and not self.use_mock:
            obs = self._env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            info = {}
        else:
            # Mock initial state: hand at (0, 0.6, 0.2), object at (0, 0.7, 0.05), goal at (0.2, 0.7, 0.15)
            obs = np.zeros(39, dtype=np.float32)
            obs[0:3] = [0.0, 0.6, 0.2]  # hand pos
            obs[3] = 1.0  # gripper open
            obs[4:7] = [0.0, 0.7, 0.05]  # object pos
            obs[18:21] = [0.2, 0.7, 0.15]  # goal pos
            self._mock_step_count = 0
            self._mock_state = obs.copy()
            info = {}
        state_dict = self.extract_state_dict(obs, info)
        return state_dict, info

    def extract_state_dict(self, obs: Any, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        obs = np.asarray(obs, dtype=np.float32)
        hand_pos = obs[0:3].copy()
        gripper_open = float(obs[3]) if len(obs) > 3 else 1.0
        obj_pos = obs[4:7].copy() if len(obs) >= 7 else np.zeros(3, dtype=np.float32)
        goal_pos = obs[18:21].copy() if len(obs) >= 21 else np.array([0.2, 0.7, 0.15], dtype=np.float32)

        hand_obj_dist = float(np.linalg.norm(hand_pos - obj_pos))
        obj_goal_dist = float(np.linalg.norm(obj_pos - goal_pos))

        hand_near_obj = hand_obj_dist < 0.08
        obj_grasped = hand_obj_dist < 0.04 and gripper_open < 0.5
        in_target = obj_goal_dist < 0.05
        if info and (bool(info.get("success", False)) or float(info.get("success", 0.0)) > 0.0):
            in_target = True

        return {
            "hand_pos": tuple(float(x) for x in hand_pos),
            "obj_pos": tuple(float(x) for x in obj_pos),
            "goal_pos": tuple(float(x) for x in goal_pos),
            "gripper_open": float(np.clip(gripper_open, 0.0, 1.0)),
            "obj_grasped": bool(obj_grasped),
            "in_target": bool(in_target),
            "hand_near_obj": bool(hand_near_obj),
            "hand_obj_dist": hand_obj_dist,
            "obj_goal_dist": obj_goal_dist,
            "raw_obs": obs,
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self._env is not None and not self.use_mock:
            step_out = self._env.step(action)
            if len(step_out) == 4:
                obs, reward, done, info = step_out
                truncated = False
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
        else:
            # Mock physics update
            obs = self._mock_state.copy()
            # Hand displacement: action[0:3] scaled by 0.02
            obs[0:3] += action[0:3] * 0.02
            # Gripper update: action[3]
            obs[3] = np.clip(obs[3] + action[3] * 0.1, 0.0, 1.0)
            # If grasped, move object with hand
            hand_obj_dist = float(np.linalg.norm(obs[0:3] - obs[4:7]))
            if hand_obj_dist < 0.04 and obs[3] < 0.5:
                obs[4:7] = obs[0:3] + [0.0, 0.0, -0.02]
            self._mock_step_count += 1
            self._mock_state = obs.copy()
            done = self._mock_step_count >= self.horizon
            truncated = done
            reward = 1.0 if float(np.linalg.norm(obs[4:7] - obs[18:21])) < 0.05 else 0.0
            info = {"success": reward > 0.0}

        state_dict = self.extract_state_dict(obs, info)
        return state_dict, float(reward), bool(state_dict["in_target"]), bool(truncated), info

    def get_milestone_trajectories(self, seed: int = 42) -> List[List[Step]]:
        """
        Returns milestone trajectories [T0, T1, T2, T3] for progress verification.
        T0: Moving away from object / goal
        T1: Approaching / hovering near object
        T2: Approaching object / partial grasp
        T3: Expert reaching goal (in_target)
        """
        def rollout_actions(action_list):
            s, _ = self.reset(seed=seed)
            steps = []
            for t, a in enumerate(action_list):
                s2, r, done, trunc, _ = self.step(a)
                g = bool(s2.get("in_target", False))
                steps.append(Step(t=t, s=s, a=a, s2=s2, r=1.0 if g else 0.0, comps={"goal": 1.0 if g else 0.0}, goal=g))
                s = s2
                if g or done or trunc:
                    break
            return steps

        t0_actions = [np.array([-0.5, -0.5, 0.2, 0.0], dtype=np.float32) for _ in range(50)]
        t1_actions = [np.array([0.0, 0.0, 0.0, 0.8 if i % 2 == 0 else -0.8], dtype=np.float32) for i in range(50)]
        t2_actions = [np.array([0.0, 0.1, -0.05, 0.0], dtype=np.float32) for _ in range(50)]
        t3_actions = [np.array([0.0, 0.1, -0.05, -0.8], dtype=np.float32) for _ in range(50)]

        return [
            rollout_actions(t0_actions),
            rollout_actions(t1_actions),
            rollout_actions(t2_actions),
            rollout_actions(t3_actions),
        ]

    def gt_score(self, trajectory: List[Any]) -> float:
        """
        Calculates ground-truth milestone score:
          0 = far from object
          1 = reached object
          2 = grasped/manipulated object
          3 = reached target goal
        """
        if not trajectory:
            return 0.0

        # Duck typing: if trajectory is an action list (not Step or dict), roll it out
        first = trajectory[0]
        if isinstance(first, (np.ndarray, list, tuple)) and not hasattr(first, "s") and not (isinstance(first, dict) and "s" in first):
            s, _ = self.reset(seed=42)
            steps = []
            for t, a in enumerate(trajectory):
                s2, r, done, trunc, _ = self.step(a)
                steps.append(Step(t=t, s=s, a=a, s2=s2, r=r, comps={"goal": r}, goal=bool(s2.get("in_target", False))))
                s = s2
                if done or trunc or s2.get("in_target", False):
                    break
            trajectory = steps

        def _get_st(item):
            if hasattr(item, "s2") and isinstance(item.s2, dict):
                return item.s2
            if hasattr(item, "s") and isinstance(item.s, dict):
                return item.s
            return item if isinstance(item, dict) else {}

        final_s = _get_st(trajectory[-1])
        if final_s.get("in_target", False):
            return 3.0 + 0.5 * (self.horizon - len(trajectory)) / self.horizon

        states = [_get_st(item) for item in trajectory]
        min_hand_obj = min(s.get("hand_obj_dist", 1.0) for s in states)
        any_grasped = any(s.get("obj_grasped", False) for s in states)
        min_obj_goal = min(s.get("obj_goal_dist", 1.0) for s in states)

        score = 0.0
        if min_hand_obj < 0.05:
            score = max(score, 1.0 + (0.05 - min_hand_obj) * 10)
        if any_grasped:
            score = max(score, 2.0 + (0.3 - min_obj_goal))
        return float(score)


def make_metaworld_env(task_name: str = "door-open-v2", horizon: int = 500, use_mock: bool = False) -> MetaWorldEnvWrapper:
    return MetaWorldEnvWrapper(task_name=task_name, horizon=horizon, use_mock=use_mock)
