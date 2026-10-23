import ast
from typing import Any, Dict, List, Optional
import numpy as np

STATIC_MSG = {
    "no_goal_term": "the reward never references the goal (at_goal / GOAL_POS / in_target).",
    "positive_reward_on_noop": "some no-op action (state unchanged) receives POSITIVE reward, so idling can be farmed forever.",
    "shaping_exceeds_goal": "a non-goal transition can pay as much as or more than reaching the goal in one step; the goal must be the largest single reward.",
    "positive_distance_exploit": "the reward gives positive reward proportional to distance (+norm), encouraging the agent to diverge away to infinity.",
}


def ast_unconditional_positive_consts(code: str) -> List[str]:
    """Scans AST for top-level assignments of positive constants (unconditional survival bonuses)."""
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out

    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "reward_components"), None)
    if fn is None:
        return out

    for stmt in fn.body:
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, (int, float))
            and not isinstance(stmt.value.value, bool)
            and stmt.value.value > 0
        ):
            for t in stmt.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant):
                    out.append(str(t.slice.value))
    return out


def ast_check_positive_distance(code: str) -> bool:
    """Detects continuous control anti-pattern: rewarding positive distance instead of negative distance."""
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    norm_vars = set()

    def is_norm_call(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Call) and getattr(expr.func, "id", "") == "float" and expr.args:
            expr = expr.args[0]
        if isinstance(expr, ast.Call):
            fn_name = getattr(expr.func, "attr", getattr(expr.func, "id", ""))
            return fn_name in ("norm", "l2_norm")
        return False

    # First pass: find variables assigned directly from norm/l2_norm without negation
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            val = node.value
            if is_norm_call(val):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        norm_vars.add(t.id)

    # Second pass: check assignments to reward dictionary
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_is_dict_subscript = any(
                isinstance(t, ast.Subscript) for t in node.targets
            )
            if not target_is_dict_subscript:
                continue

            val = node.value
            # Unwrap float(...) call if present
            if isinstance(val, ast.Call) and getattr(val.func, "id", "") == "float" and val.args:
                val = val.args[0]

            # Direct call to norm or l2_norm without negative factor
            if is_norm_call(val):
                return True

            # Direct variable reference to norm result without negation
            if isinstance(val, ast.Name) and val.id in norm_vars:
                return True

            # Positive scalar multiplier: e.g. 1.0 * norm(...) or 1.0 * dist_var
            if isinstance(val, ast.BinOp) and isinstance(val.op, ast.Mult):
                if isinstance(val.left, ast.Constant) and isinstance(val.left.value, (int, float)) and val.left.value > 0:
                    r_val = val.right
                    if is_norm_call(r_val):
                        return True
                    if isinstance(r_val, ast.Name) and r_val.id in norm_vars:
                        return True

    patterns = ["+ np.linalg.norm", "+ l2_norm", "+ np.linalg.norm(hand", "+ np.linalg.norm(obj"]
    return any(p in code for p in patterns)


def static_checks(code: str, table: Optional[Any] = None, env: Optional[Any] = None) -> List[str]:
    """
    Executes Gate G2 static anti-hack verification.
    Returns list of violation flags if candidate violates mathematical or AST design rules.
    """
    bad = []

    # 1. Goal reference check
    goal_keywords = ("at_goal", "GOAL_POS", "in_target", "goal_pos", "goal_reached")
    if not any(k in code for k in goal_keywords):
        bad.append("no_goal_term")

    # 2. Unconditional positive constant check
    pos_consts = ast_unconditional_positive_consts(code)
    bad += [f"unconditional_positive:{nm}" for nm in pos_consts]

    # 3. Continuous domain positive distance check (Meta-World only)
    from cegr.envs.doorkey import DoorKeyEnv
    if env is not None and not isinstance(env, DoorKeyEnv):
        if ast_check_positive_distance(code):
            bad.append("positive_distance_exploit")

    # 4. Tabular environment checks (if table and env available)
    if (
        table is not None
        and hasattr(table, "R")
        and env is not None
        and isinstance(env, DoorKeyEnv)
        and getattr(env, "nxt", None) is not None
        and getattr(env, "terminal", None) is not None
        and getattr(env, "n", None) is not None
    ):
        selfloop = (env.nxt == np.arange(env.n)[:, None]) & (~env.terminal[:, None])
        if np.any(table.R[selfloop] > 1e-9):
            bad.append("positive_reward_on_noop")

        if hasattr(env, "goal_flag") and env.goal_flag is not None and env.goal_flag.any():
            gmax = float(table.R[env.goal_flag].max())
            nongoal = table.R[(~env.goal_flag) & (~env.terminal[:, None])]
            if nongoal.size and float(nongoal.max()) >= gmax - 1e-9:
                bad.append("shaping_exceeds_goal")

    return bad
