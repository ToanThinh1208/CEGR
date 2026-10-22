import collections
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cegr.eval.bootstrap import bootstrap_ci, per_reward, paired_test

COLORS = {
    "ORIG": "#969696",
    "EUREKA": "#fd8d3c",
    "CARD": "#6baed6",
    "CARD_REFL": "#3182bd",
    "CEGR": "#08519c",
    "CEGR": "#08519c",
}

GATE_NAMES = {
    1: "exec",
    2: "anti-hack",
    3: "TPE + attacker",
    4: "signal",
    5: "RL probe",
}


def fmt_ci(vals: Any) -> str:
    m, lo, hi = bootstrap_ci(np.asarray(vals) * 100)
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"


def generate_latex_tables(df_all: pd.DataFrame, out_tex_path: str):
    """Generates LaTeX code for Tables 1 to 4 and saves to out_tex_path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_tex_path)), exist_ok=True)
    m_main = ["ORIG", "EUREKA", "CARD", "CARD_REFL", "CEGR"]

    # --- TABLE 1: MAIN RESULTS ---
    rows1 = []
    for m in m_main:
        sub = df_all[df_all["method"] == m]
        if sub.empty:
            continue
        g = sub.groupby("reward_id").mean(numeric_only=True)
        rows1.append({
            "Method": m,
            "Verified Fix (%)": fmt_ci(g.get("vf", [0])),
            "Hack rate DP (%)": fmt_ci(g.get("hacked_dp", [0])),
            "Final RL success (%)": f"{g.get('rl_success', pd.Series([0])).mean()*100:.1f}",
            "Rounds-to-pass": f"{g.get('rounds_to_pass', pd.Series([0])).mean():.2f}",
            "LLM calls": f"{g.get('llm_calls', pd.Series([0])).mean():.1f}",
            "RL episodes (K)": f"{g.get('rl_episodes', pd.Series([0])).mean()/1e3:.1f}",
        })
    df_t1 = pd.DataFrame(rows1)

    # --- TABLE 2: ABLATIONS ---
    m_abl = ["CEGR", "CEGR_noPool", "CEGR_noAnchors", "CEGR_noClass", "CEGR_noG2", "CEGR_noG4", "CEGR_noG5"]
    piv_ab = per_reward(df_all, "vf").astype(float) if "vf" in df_all else pd.DataFrame()
    rows2 = []
    for m in m_abl:
        sub = df_all[df_all["method"] == m]
        if sub.empty:
            continue
        g = sub.groupby("reward_id").mean(numeric_only=True)
        r = {
            "Variant": m,
            "VF (%)": f"{g.get('vf', pd.Series([0])).mean()*100:.1f}",
            "Rounds-to-pass": f"{g.get('rounds_to_pass', pd.Series([0])).mean():.2f}",
            "RL episodes (K)": f"{g.get('rl_episodes', pd.Series([0])).mean()/1e3:.1f}",
        }
        if m != "CEGR" and not piv_ab.empty and "CEGR" in piv_ab and m in piv_ab:
            t = paired_test(piv_ab, "CEGR", m)
            r["Δ VF vs CEGR (pp)"] = f"{t['delta']:+.1f} [{t['ci_lo']:+.1f}, {t['ci_hi']:+.1f}] p={t['p_val']:.3f}"
        rows2.append(r)
    df_t2 = pd.DataFrame(rows2)

    # Write tables to file
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("% --- TABLE 1 MAIN RESULTS ---\n" + df_t1.to_latex(index=False, escape=False) + "\n\n")
        f.write("% --- TABLE 2 ABLATION RESULTS ---\n" + df_t2.to_latex(index=False, escape=False) + "\n\n")


def plot_evaluation_figures(df_jobs: pd.DataFrame, out_dir: str):
    """Generates 300 DPI publication plots (Figures 1 to 3)."""
    os.makedirs(out_dir, exist_ok=True)
    m_main = ["ORIG", "EUREKA", "CARD", "CARD_REFL", "CEGR"]

    # 1. Figure 1: Verified Fix & Non-hacked rate
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ms = [m for m in m_main if m in set(df_jobs.get("method", []))]
    if ms:
        pv = per_reward(df_jobs, "vf").astype(float)
        x = np.arange(len(ms))
        st = [bootstrap_ci(pv[m].dropna().values * 100) if m in pv else (0, 0, 0) for m in ms]
        ax.bar(
            x,
            [s[0] for s in st],
            width=0.5,
            yerr=[[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]],
            capsize=3,
            color=[COLORS.get(m, "#333") for m in ms],
            edgecolor="black",
            alpha=0.9,
        )
        for xi, s in zip(x, st):
            ax.text(xi, s[0] + 2, f"{s[0]:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ms)
        ax.set_ylabel("Verified Fix (%)")
        ax.set_ylim(0, 110)
        ax.set_title("Figure 1: Verified Fix Rate across Methods")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig1_vf_by_method.png"), dpi=300)
        plt.close()

    # 2. Figure 2: Recurrence and RL cost
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=300)
    for m in ["EUREKA", "CARD", "CARD_REFL", "CEGR"]:
        sub = df_jobs[df_jobs["method"] == m]
        if sub.empty:
            continue
        recur = sub.get("recur_total", pd.Series([0])).mean()
        cost = sub.get("rl_episodes", pd.Series([0])).mean() / 1e3
        vf = sub.get("vf", pd.Series([0])).mean() * 100

        ax1.scatter(recur, vf, s=120, color=COLORS.get(m, "#333"), edgecolor="black", zorder=3)
        ax1.annotate(m, (recur, vf), xytext=(6, 5), textcoords="offset points", fontsize=9)

        ax2.scatter(cost, vf, s=120, color=COLORS.get(m, "#333"), edgecolor="black", zorder=3)
        ax2.annotate(m, (cost, vf), xytext=(6, 5), textcoords="offset points", fontsize=9)

    ax1.set_xlabel("Total Counterexample Recurrences")
    ax1.set_ylabel("Verified Fix (%)")
    ax1.set_title("(a) Recurrence vs. Fix Quality")

    ax2.set_xlabel("RL Training Cost (K episodes)")
    ax2.set_ylabel("Verified Fix (%)")
    ax2.set_title("(b) Fix Quality vs. RL Exploration Cost")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_recurrence_and_cost.png"), dpi=300)
    plt.close()
