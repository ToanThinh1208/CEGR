import ast
import builtins
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Dict, List, Optional
import numpy as np


class RewardError(Exception):
    pass


@dataclass
class RewardTable:
    R: np.ndarray  # [n, 6]
    C: np.ndarray  # [n, 6, K]
    names: List[str]


_SAFE_NAMES = [
    "abs", "min", "max", "sum", "len", "range", "float", "int", "bool", "round",
    "enumerate", "zip", "any", "all", "sorted", "list", "dict", "tuple", "set",
    "isinstance", "pow", "divmod", "map", "filter", "str", "reversed"
]
def _safe_import(name, *args, **kwargs):
    if name in ("math", "numpy", "typing"):
        return builtins.__import__(name, *args, **kwargs)
    raise RewardError(f"import_forbidden:{name}")

SAFE_BUILTINS = {n: getattr(builtins, n) for n in _SAFE_NAMES}
SAFE_BUILTINS["__import__"] = _safe_import
_FORBIDDEN = {
    "eval", "exec", "open", "compile", "globals", "locals", "getattr",
    "setattr", "delattr", "input", "breakpoint", "vars", "dir", "exit", "quit"
}

# Safe numpy sub-module functions for continuous vector calculations
class _SafeNumpy:
    ndarray = np.ndarray
    linalg = type("linalg", (), {"norm": staticmethod(np.linalg.norm)})
    clip = staticmethod(np.clip)
    exp = staticmethod(np.exp)
    tanh = staticmethod(np.tanh)
    abs = staticmethod(np.abs)
    sum = staticmethod(np.sum)
    mean = staticmethod(np.mean)
    min = staticmethod(np.min)
    max = staticmethod(np.max)
    asarray = staticmethod(np.asarray)
    array = staticmethod(np.array)
    dot = staticmethod(np.dot)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    sqrt = staticmethod(np.sqrt)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)
    square = staticmethod(np.square)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    atan2 = staticmethod(np.arctan2)
    arctan2 = staticmethod(np.arctan2)
    float32 = np.float32
    float64 = np.float64
    inf = np.inf
    pi = np.pi

SAFE_NP = _SafeNumpy()


def check_code(code: str) -> ast.AST:
    """
    Parses and checks candidate reward code for syntax and safety.
    Allows pure Python, math, typing, and safe vector numpy calls. Blocks file I/O, eval, dunders.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RewardError(f"SyntaxError: {e}")

    has_fn = False
    for node in ast.walk(tree):
        # Disallow raw import statements except numpy/math/typing
        if isinstance(node, ast.Import):
            for alias in getattr(node, "names", []):
                if alias.name not in ("math", "numpy", "typing"):
                    raise RewardError(f"import_forbidden:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = getattr(node, "module", "") or ""
            if mod not in ("math", "numpy", "typing"):
                raise RewardError(f"import_forbidden:{mod}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RewardError("dunder_forbidden")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN:
            raise RewardError(f"forbidden_name:{node.id}")
        if isinstance(node, ast.FunctionDef) and node.name == "reward_components":
            has_fn = True

    if not has_fn:
        raise RewardError("missing_reward_components")
    return tree


def code_fingerprint(code: str) -> str:
    try:
        return hashlib.md5(ast.dump(ast.parse(code)).encode()).hexdigest()
    except Exception:
        return hashlib.md5(code.encode()).hexdigest()


def ast_node_count(code: str) -> int:
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except Exception:
        return 0


def make_namespace(env: Any) -> Dict[str, Any]:
    """Builds a sandbox namespace containing safe builtins, math, numpy, typing, and environment helpers."""
    import typing
    ns = dict(
        __builtins__=SAFE_BUILTINS,
        math=math,
        np=SAFE_NP,
        numpy=SAFE_NP,
        typing=typing,
        Dict=typing.Dict,
        Tuple=typing.Tuple,
        List=typing.List,
        Optional=typing.Optional,
        Any=typing.Any,
    )

    # If DoorKey environment
    if hasattr(env, "goal") and hasattr(env, "door_pos"):
        gx, gy = env.goal
        def manhattan(p, q):
            return abs(p[0] - q[0]) + abs(p[1] - q[1])

        def at_goal(st):
            pos = getattr(st, "pos", st)
            return (pos[0], pos[1]) == (gx, gy)

        from cegr.envs.doorkey import (
            LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE,
            DOOR_LOCKED, DOOR_CLOSED, DOOR_OPEN
        )

        ns.update(
            LEFT=LEFT, RIGHT=RIGHT, FORWARD=FORWARD, PICKUP=PICKUP, DROP=DROP, TOGGLE=TOGGLE,
            DOOR_LOCKED=DOOR_LOCKED, DOOR_CLOSED=DOOR_CLOSED, DOOR_OPEN=DOOR_OPEN,
            GOAL_POS=env.goal, DOOR_POS=env.door_pos, KEY_START_POS=env.key_pos0,
            manhattan=manhattan, at_goal=at_goal,
        )

    # Helper for Meta-World / Continuous control
    def l2_norm(p, q):
        return float(np.linalg.norm(np.array(p) - np.array(q)))

    ns["l2_norm"] = l2_norm
    return ns


def build_reward_table(env: Any, code: str) -> RewardTable:
    """Precomputes tabular transition reward table for DP solver on DoorKey."""
    check_code(code)
    ns = make_namespace(env)
    try:
        exec(compile(code, "<reward>", "exec"), ns)
    except Exception as e:
        raise RewardError(f"DefError:{type(e).__name__}:{e}")

    f = ns["reward_components"]
    entries, names = {}, []
    try:
        for i, s in enumerate(env.states):
            if env.terminal[i]:
                continue
            for a in range(6):
                s2 = env.states[env.nxt[i, a]]
                out = f(s, a, s2)
                if not isinstance(out, dict):
                    out = {"total": out}
                clean = {}
                for k, v in out.items():
                    if isinstance(v, bool):
                        v = float(v)
                    v = float(v)
                    if not math.isfinite(v):
                        raise RewardError("non_finite_reward")
                    if abs(v) > 1e4:
                        raise RewardError("extreme_scale")
                    k = str(k)
                    if k not in names:
                        names.append(k)
                    clean[k] = v
                entries[(i, a)] = clean
    except RewardError:
        raise
    except Exception as e:
        raise RewardError(f"RuntimeError:{type(e).__name__}:{e}")

    if not names:
        names = ["total"]
    K = len(names)
    C = np.zeros((env.n, 6, K), dtype=np.float64)
    for (i, a), d in entries.items():
        for k, v in d.items():
            C[i, a, names.index(k)] = v
    return RewardTable(R=C.sum(-1), C=C, names=names)


def check_rollout_exec(env: Any, reward_fn: Any, p0_trajs: List[Any]) -> None:
    """
    Executes candidate reward_fn on all transition steps in P_0 reference rollouts.
    Ensures return value is a dict of finite scalar floats bounded within [-1e4, 1e4].
    Raises RewardError on execution failures, NaN/Inf, or extreme scale.
    """
    if not p0_trajs:
        return
    for item in p0_trajs:
        steps = item if isinstance(item, list) else item.get("better_steps", item.get("better", []))
        if not isinstance(steps, list):
            continue
        for st in steps:
            if not hasattr(st, "s") or not hasattr(st, "s2"):
                continue
            try:
                out = reward_fn(st.s, st.a, st.s2)
            except Exception as e:
                raise RewardError(f"RuntimeError:{type(e).__name__}:{e}")
            if not isinstance(out, dict):
                out = {"total": out}
            for k, v in out.items():
                if isinstance(v, bool):
                    v = float(v)
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    raise RewardError(f"invalid_component_type:{type(v).__name__}")
                if not math.isfinite(v):
                    raise RewardError("non_finite_reward")
                if abs(v) > 1e4:
                    raise RewardError("extreme_scale")
