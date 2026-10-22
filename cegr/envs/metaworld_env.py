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

        obj_grasped = hand_obj_dist < 0.04 and gripper_open < 0.5
        in_target = obj_goal_dist < 0.05

        return {
            "hand_pos": tuple(float(x) for x in hand_pos),
            "obj_pos": tuple(float(x) for x in obj_pos),
            "goal_pos": tuple(float(x) for x in goal_pos),
            "gripper_open": float(np.clip(gripper_open, 0.0, 1.0)),
            "obj_grasped": bool(obj_grasped),
            "in_target": bool(in_target),
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
