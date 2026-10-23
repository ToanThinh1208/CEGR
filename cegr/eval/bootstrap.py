from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def bootstrap_ci(
    vals: Any,
    n_boot: int = 5000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Computes empirical mean and bootstrap confidence interval at the specified CI level (default 95%).
    Calculates percentiles dynamically from the given CI without hardcoded bounds.
    """
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[indices].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lo_pct = alpha * 100.0
    hi_pct = (1.0 - alpha) * 100.0
    lo, hi = np.percentile(means, [lo_pct, hi_pct])
    return float(vals.mean()), float(lo), float(hi)


def per_reward(
    df: pd.DataFrame,
    col: str,
    cluster_col: Optional[str] = None,
    method_col: str = "method",
) -> pd.DataFrame:
    """
    Cluster by reward/task ID first, taking the mean across seeds prior to bootstrap.
    Dynamically resolves clustering column ('reward_id', 'id', 'task_id', 'env_id')
    without hardcoding a single fixed column schema.
    """
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()

    if cluster_col is None:
        for cand in ["reward_id", "id", "task_id", "env_id"]:
            if cand in df.columns:
                cluster_col = cand
                break
        if cluster_col is None:
            non_numeric = [c for c in df.columns if c not in [col, method_col]]
            cluster_col = non_numeric[0] if non_numeric else df.index.name or "index"

    if cluster_col not in df.columns or method_col not in df.columns:
        return pd.DataFrame()

    return df.groupby([cluster_col, method_col])[col].mean().unstack(method_col)


def paired_test(
    piv: pd.DataFrame,
    a: str,
    b: str,
    n_perm: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
    as_percent: bool = True,
) -> Dict[str, Any]:
    """
    Performs Paired Permutation Test between method 'a' and baseline 'b'.
    Dynamically computes delta, bootstrap CI, and exact two-tailed permutation p-value
    from the empirical sample differences.
    """
    if piv is None or piv.empty or a not in piv or b not in piv:
        return dict(pair=f"{a} - {b}", delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_val=np.nan)

    diff = (piv[a] - piv[b]).dropna().values
    if len(diff) == 0:
        return dict(pair=f"{a} - {b}", delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_val=np.nan)

    scale = 100.0 if as_percent else 1.0
    m, lo, hi = bootstrap_ci(diff, seed=seed, ci=ci)

    rng = np.random.default_rng(seed)
    perm = (diff[None, :] * rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))).mean(axis=1)
    p_val = float((np.abs(perm) >= abs(m)).mean())

    return {
        "pair": f"{a} - {b}",
        "delta": float(m * scale),
        "ci_lo": float(lo * scale),
        "ci_hi": float(hi * scale),
        "p_val": float(p_val),
    }

