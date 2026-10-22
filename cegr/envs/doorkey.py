from collections import deque
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Any
import numpy as np

from cegr.envs.base_env import BaseEnv

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle"]
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = range(6)
DIR_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIR_NAMES = ["right", "down", "left", "up"]
DOOR_LOCKED, DOOR_CLOSED, DOOR_OPEN = 0, 1, 2
DOOR_NAMES = ["locked", "closed", "open"]

KEYPOS_MODE = "agent_pos"  # "agent_pos" | "none"

TASK_DESC = (
    "TASK: Minigrid DoorKey-5x5. The agent must pick up the key, unlock the door, "
    "and step on the green goal within the horizon.\nActions: left, right, forward, pickup, drop, toggle."
)


def get_api_desc(mode: Optional[str] = None) -> str:
    m = mode or KEYPOS_MODE
    if m == "agent_pos":
        key_pos_field = "  s.key_pos          tuple (x, y) of the key if it lies on the floor, and the agent's current position if the key is carried\n"
        key_pos_rule = ""
    else:
        key_pos_field = "  s.key_pos          tuple (x, y) if the key lies on the floor, otherwise None (key carried)\n"
        key_pos_rule = (
            "  - s.key_pos / s2.key_pos is None while the key is carried. Guard it before use, e.g.\n"
            "    `s2.key_pos is not None and manhattan(s2.pos, s2.key_pos) <= 1`. Never index it (key_pos[0]).\n"
        )

    return (
        "REWARD API (pure Python; NO imports; `math` is already available):\n\n"
        "Define exactly one entry point:\n"
        "  def reward_components(s, a, s2):\n"
        "      # s = State BEFORE the action, a = action id (int), s2 = State AFTER the action.\n"
        "      # Return a dict {\"component_name\": number}. The total reward is the sum of its values.\n\n"
        "The function is evaluated on EVERY reachable state and EVERY action (including no-op actions),\n"
        "so it must never raise an exception. It is stateless: it only sees (s, a, s2); there is no\n"
        "step counter, no history and no global variable.\n\n"
        "State fields (this is the COMPLETE list; nothing else exists):\n"
        "  s.x, s.y           ints, grid position (x grows to the right, y grows downward)\n"
        "  s.dir              int: 0=right, 1=down, 2=left, 3=up\n"
        "  s.has_key          bool, True while the key is carried\n"
        "  s.key_x, s.key_y   ints, key position on the floor; both are -1 while the key is carried\n"
        "  s.door             int: 0=locked, 1=closed, 2=open\n"
        "  s.pos              tuple (x, y)\n"
        "  s.front            tuple (x, y), the cell the agent is facing\n"
        f"{key_pos_field}\n"
        "Global constants (UPPERCASE; they are NOT attributes of s):\n"
        "  Actions: LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE   (compare with the argument `a`, e.g. `a == PICKUP`)\n"
        "  Door states: DOOR_LOCKED, DOOR_CLOSED, DOOR_OPEN      (compare with s.door / s2.door)\n"
        "  Positions as (x, y) tuples: GOAL_POS, DOOR_POS, KEY_START_POS\n"
        "Global helper functions (NOT methods of s):\n"
        "  manhattan(p, q)  -> int; p and q are (x, y) tuples, e.g. manhattan(s2.pos, GOAL_POS)\n"
        "  at_goal(state)   -> bool; takes a State, e.g. at_goal(s2)   (NOT a position)\n\n"
        "Dynamics: PICKUP grabs the key only if it lies in the faced cell. TOGGLE on the faced door opens\n"
        "it if it is closed, or if it is locked and the key is carried (toggling an open door closes it).\n"
        "The episode ends when the agent steps onto GOAL_POS.\n\n"
        "Rules that avoid common runtime errors:\n"
        f"{key_pos_rule}"
        "  - Never divide by a quantity that can be 0 (distances can be 0). Write 1.0 / (1.0 + d).\n"
        "  - Do not invent attributes (no s.door_pos, s.goal_pos, s.start_pos, s.inventory, s.step, ...),\n"
        "    and never call manhattan / at_goal as methods of s.\n"
        "  - Define any helper (e.g. a potential `phi(state)`) in the same code block, at top level, before use.\n"
        "  - Every component value must be a finite number (bool is fine) with |value| <= 1e4; keys are strings.\n"
        "  - Only these builtins exist: abs, min, max, sum, len, range, float, int, bool, round, enumerate,\n"
        "    zip, any, all, sorted, list, dict, tuple, set, isinstance, pow, divmod, map, filter, str, reversed.\n"
        "    Use max(...), not math.max. No eval/exec/open/getattr and no names starting with \"__\".\n\n"
        "Syntax illustration only (NOT a recommended design):\n"
        "def reward_components(s, a, s2):\n"
        "    c = {}\n"
        "    c[\"step\"] = -0.01\n"
        "    c[\"goal\"] = 1.0 if at_goal(s2) else 0.0\n"
        "    return c"
    )


API_DESC = get_api_desc(KEYPOS_MODE)


class State(NamedTuple):
    x: int
    y: int
    dir: int
    has_key: bool
    key_x: int
    key_y: int
    door: int

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def front(self):
        dx, dy = DIR_VECS[self.dir]
        return (self.x + dx, self.y + dy)

    @property
    def key_pos(self):
        if KEYPOS_MODE == "agent_pos":
            return (self.x, self.y) if self.has_key else (self.key_x, self.key_y)
        return None if self.has_key else (self.key_x, self.key_y)


class DoorKeyEnv(BaseEnv):
    def __init__(
        self,
        size: int = 5,
        wall_x: int = 2,
        door_pos: Tuple[int, int] = (2, 2),
        key_pos: Tuple[int, int] = (1, 1),
        start: Tuple[int, int] = (1, 3),
        start_dir: int = 0,
        goal: Tuple[int, int] = (3, 3),
        horizon: int = 30,
    ):
        super().__init__(horizon=horizon)
        self.size = size
        self.wall_x = wall_x
        self.door_pos = door_pos
        self.key_pos0 = key_pos
        self.start = start
        self.start_dir = start_dir
        self.goal = goal
        self.horizon = horizon
        self._current_state = self.initial_state()
        self._build_tables()

    def is_wall(self, x: int, y: int) -> bool:
        if x <= 0 or y <= 0 or x >= self.size - 1 or y >= self.size - 1:
            return True
        return x == self.wall_x and (x, y) != self.door_pos

    def initial_state(self) -> State:
        return State(
            self.start[0],
            self.start[1],
            self.start_dir,
            False,
            self.key_pos0[0],
            self.key_pos0[1],
            DOOR_LOCKED,
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[State, Dict[str, Any]]:
        self._current_state = self.initial_state()
        return self._current_state, {}

    def extract_state_dict(self, obs: Any, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(obs, State):
            return {
                "x": obs.x,
                "y": obs.y,
                "dir": obs.dir,
                "has_key": obs.has_key,
                "key_x": obs.key_x,
                "key_y": obs.key_y,
                "door": obs.door,
                "pos": obs.pos,
                "front": obs.front,
                "key_pos": obs.key_pos,
            }
        return dict(obs)

    def _blocked(self, s: State, cell: Tuple[int, int]) -> bool:
        if self.is_wall(*cell):
            return True
        if cell == self.door_pos and s.door != DOOR_OPEN:
            return True
        if (not s.has_key) and cell == (s.key_x, s.key_y):
            return True
        return False

    def step_state(self, s: State, a: int) -> Tuple[State, bool]:
        fx, fy = s.front
        if a == LEFT:
            return s._replace(dir=(s.dir - 1) % 4), False
        if a == RIGHT:
            return s._replace(dir=(s.dir + 1) % 4), False
        if a == FORWARD:
            if self._blocked(s, (fx, fy)):
                return s, False
            s2 = s._replace(x=fx, y=fy)
            return s2, (fx, fy) == self.goal
        if a == PICKUP:
            if (not s.has_key) and (fx, fy) == (s.key_x, s.key_y):
                return s._replace(has_key=True, key_x=-1, key_y=-1), False
            return s, False
        if a == DROP:
            free = (not self.is_wall(fx, fy)) and (fx, fy) != self.door_pos and (fx, fy) != self.goal
            if s.has_key and free:
                return s._replace(has_key=False, key_x=fx, key_y=fy), False
            return s, False
        if a == TOGGLE:
            if (fx, fy) == self.door_pos:
                if s.door == DOOR_LOCKED and s.has_key:
                    return s._replace(door=DOOR_OPEN), False
                if s.door == DOOR_CLOSED:
                    return s._replace(door=DOOR_OPEN), False
                if s.door == DOOR_OPEN:
                    return s._replace(door=DOOR_CLOSED), False
            return s, False
        raise ValueError(f"Unknown action {a}")

    def step(self, action: int) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        s2, goal_reached = self.step_state(self._current_state, action)
        self._current_state = s2
        reward = 1.0 if goal_reached else 0.0
        return s2, reward, goal_reached, False, {"goal": goal_reached}

    def _build_tables(self):
        s0 = self.initial_state()
        idx = {s0: 0}
        states = [s0]
        nxt, goal, terminal = [], [], []
        i = 0
        while i < len(states):
            s = states[i]
            if s.pos == self.goal:
                nxt.append([i] * 6)
                goal.append([False] * 6)
                terminal.append(True)
                i += 1
                continue
            row, grow = [], []
            for a in range(6):
                s2, g = self.step_state(s, a)
                if s2 not in idx:
                    idx[s2] = len(states)
                    states.append(s2)
                row.append(idx[s2])
                grow.append(g)
            nxt.append(row)
            goal.append(grow)
            terminal.append(False)
            i += 1
        self.states = states
        self.idx = idx
        self.n = len(states)
        self.nxt = np.array(nxt, dtype=np.int64)
        self.goal_flag = np.array(goal, dtype=bool)
        self.terminal = np.array(terminal, dtype=bool)

    def gt_score(self, actions: List[int]) -> float:
        return gt_score(self, actions)


def replay(env: DoorKeyEnv, actions: List[int]) -> Tuple[List[int], List[int], bool]:
    i, idxs, used, reached = 0, [0], [], False
    for a in actions:
        if env.terminal[i]:
            break
        a = int(a)
        g = bool(env.goal_flag[i, a])
        used.append(a)
        i = int(env.nxt[i, a])
        idxs.append(i)
        if g:
            reached = True
            break
    return idxs, used, reached


def traj_return(env: DoorKeyEnv, table: Any, actions: List[int]) -> float:
    i, tot = 0, 0.0
    for a in actions:
        if env.terminal[i]:
            break
        a = int(a)
        tot += float(table.R[i, a])
        g = bool(env.goal_flag[i, a])
        i = int(env.nxt[i, a])
        if g:
            break
    return tot


def traj_level(env: Any, actions: Any) -> int:
    """True milestone progress: 0=start, 1=has key/near object, 2=door opened/grasped, 3=goal reached."""
    if not actions:
        return 0
    first = actions[0]
    # Duck typing: support List[Step] and List[dict]
    if hasattr(first, "goal") or (isinstance(first, dict) and ("goal" in first or "s2" in first)):
        lv = 0
        for st in actions:
            g = getattr(st, "goal", None) if hasattr(st, "goal") else st.get("goal", False)
            s2 = getattr(st, "s2", None) if hasattr(st, "s2") else st.get("s2", {})
            if g or (isinstance(s2, dict) and s2.get("in_target", False)):
                return 3
            if isinstance(s2, dict):
                if s2.get("door_open") or s2.get("obj_grasped"):
                    lv = max(lv, 2)
                elif s2.get("has_key") or s2.get("hand_near_obj"):
                    lv = max(lv, 1)
        return lv

    # Default discrete replay on DoorKey
    idxs, used, reached = replay(env, actions)
    if reached:
        return 3
    lv = 0
    for i in idxs:
        s = env.states[i]
        if s.door == DOOR_OPEN:
            lv = max(lv, 2)
        elif s.has_key:
            lv = max(lv, 1)
    return lv


def gt_score(env: Any, actions: Any) -> float:
    lv = traj_level(env, actions)
    if not actions:
        return float(lv)
    first = actions[0]
    H = getattr(env, "horizon", 30)
    if hasattr(first, "goal") or (isinstance(first, dict) and ("goal" in first or "s2" in first)):
        return float(lv) + (0.5 * (H - len(actions)) / H if lv == 3 else 0.0)
    _, used, _ = replay(env, actions)
    return float(lv) + (0.5 * (H - len(used)) / H if lv == 3 else 0.0)


def _reverse_bfs(env: DoorKeyEnv, pred) -> np.ndarray:
    rev = [[] for _ in range(env.n)]
    for i in range(env.n):
        if env.terminal[i]:
            continue
        for a in range(6):
            j = int(env.nxt[i, a])
            if j != i:
                rev[j].append((i, a))
    dist = np.full(env.n, 10**6, dtype=np.int64)
    q = deque()
    for i, s in enumerate(env.states):
        if pred(s):
            dist[i] = 0
            q.append(i)
    while q:
        j = q.popleft()
        for i, a in rev[j]:
            if dist[i] > dist[j] + 1:
                dist[i] = dist[j] + 1
                q.append(i)
    best = np.zeros(env.n, dtype=np.int64)
    for i in range(env.n):
        if env.terminal[i]:
            continue
        best[i] = int(np.argmin([dist[int(env.nxt[i, a])] for a in range(6)]))
    return best


def build_expert_tables(env: DoorKeyEnv) -> Dict[str, np.ndarray]:
    return {
        "key": _reverse_bfs(env, lambda s: s.has_key),
        "door": _reverse_bfs(env, lambda s: s.door == DOOR_OPEN),
        "goal": _reverse_bfs(env, lambda s: s.pos == env.goal),
    }


def bfs_actions(env: DoorKeyEnv, pred) -> List[int]:
    q = deque([0])
    prev = {0: None}
    while q:
        i = q.popleft()
        if pred(env.states[i]):
            path = []
            while prev[i] is not None:
                i, a = prev[i]
                path.append(a)
            return path[::-1]
        if env.terminal[i]:
            continue
        for a in range(6):
            j = int(env.nxt[i, a])
            if j not in prev:
                prev[j] = (i, a)
                q.append(j)
    raise RuntimeError("BFS could not reach target")


def _idle_action(env: DoorKeyEnv, i: int) -> int:
    for a in (PICKUP, DROP, TOGGLE):
        if int(env.nxt[i, a]) == i:
            return a
    return LEFT


def build_reference_trajs(env: DoorKeyEnv) -> Dict[str, List[int]]:
    H = env.horizon

    def pad_idle(acts):
        idxs, used, _ = replay(env, acts)
        i = idxs[-1]
        acts = list(used)
        while len(acts) < H:
            a = _idle_action(env, i)
            acts.append(a)
            i = int(env.nxt[i, a])
        return acts

    return {
        "T3": bfs_actions(env, lambda s: s.pos == env.goal),
        "T2": pad_idle(bfs_actions(env, lambda s: s.door == DOOR_OPEN)),
        "T1": pad_idle(bfs_actions(env, lambda s: s.has_key)),
        "T0": pad_idle([]),
    }


def noisy_expert_actions(
    env: DoorKeyEnv, expert: Dict[str, np.ndarray], rng: np.random.Generator, p_follow: float
) -> List[int]:
    i, acts = 0, []
    for _ in range(env.horizon):
        s = env.states[i]
        tgt = "goal" if s.door == DOOR_OPEN else ("door" if s.has_key else "key")
        a = int(expert[tgt][i]) if rng.random() < p_follow else int(rng.integers(6))
        acts.append(a)
        g = bool(env.goal_flag[i, a])
        i = int(env.nxt[i, a])
        if g:
            break
    return acts


def sample_noisy(
    env: DoorKeyEnv, expert: Dict[str, np.ndarray], seed: int, n: int
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    return [
        noisy_expert_actions(env, expert, rng, (0.3, 0.6, 0.9)[k % 3])
        for k in range(n)
    ]


ENV_CONFIGS = {
    "doorkey5": dict(
        size=5,
        wall_x=2,
        door_pos=(2, 2),
        key_pos=(1, 1),
        start=(1, 3),
        start_dir=0,
        goal=(3, 3),
        horizon=30,
    ),
    "doorkey6": dict(
        size=6,
        wall_x=3,
        door_pos=(3, 2),
        key_pos=(1, 1),
        start=(1, 4),
        start_dir=0,
        goal=(4, 4),
        horizon=40,
    ),
    "doorkey8": dict(
        size=8,
        wall_x=4,
        door_pos=(4, 3),
        key_pos=(1, 1),
        start=(1, 6),
        start_dir=0,
        goal=(6, 6),
        horizon=60,
    ),
}


def make_env(name: str) -> DoorKeyEnv:
    if name not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment name: {name}. Available: {list(ENV_CONFIGS.keys())}")
    return DoorKeyEnv(**ENV_CONFIGS[name])
