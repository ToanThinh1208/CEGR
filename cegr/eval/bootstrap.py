from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def bootstrap_ci(vals: Any, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes mean and 95% bootstrap confidence interval (2.5% to 97.5%)."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def per_reward(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cluster by reward_id first, taking the mean across LLM seeds prior to bootstrap."""
    return df.groupby(["reward_id", "method"])[col].mean().unstack("method")


def paired_test(piv: pd.DataFrame, a: str, b: str, n_perm: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """
    Performs Paired Permutation Test between method 'a' and baseline 'b'.
    Returns delta (percentage points), 95% CI, and exact p-value.
    """
    if a not in piv or b not in piv:
        return dict(pair=f"{a} - {b}", delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_val=np.nan)
    diff = (piv[a] - piv[b]).dropna().values
    if len(diff) == 0:
        return dict(pair=f"{a} - {b}", delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_val=np.nan)

    m, lo, hi = bootstrap_ci(diff, seed=seed)
    rng = np.random.default_rng(seed)
    perm = (diff[None, :] * rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))).mean(axis=1)
    p_val = float((np.abs(perm) >= abs(m)).mean())

    return {
        "pair": f"{a} - {b}",
        "delta": m * 100,
        "ci_lo": lo * 100,
        "ci_hi": hi * 100,
        "p_val": p_val,
    }
