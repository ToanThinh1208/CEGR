import collections
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd

from cegr.eval.bootstrap import bootstrap_ci, per_reward, paired_test

# Unified color palette for standard methods matching publication standards
COLORS = {
    "ORIG": "#969696",
    "EUREKA": "#fd8d3c",
    "CARD": "#6baed6",
    "CARD_REFL": "#3182bd",
    "CEGR": "#08519c",
}

# Dynamic color cycle fallback for novel methods or ablation variants
PALETTE_FALLBACK = [
    "#08519c", "#3182bd", "#6baed6", "#9ecae1",
    "#fd8d3c", "#fdae6b", "#74c476", "#31a354",
    "#756bb1", "#969696", "#cb181d", "#bcbddc",
]


def get_method_color(method: str, idx: int = 0) -> str:
    """Returns color mapped to method name, or dynamic palette color if novel method."""
    if method in COLORS:
        return COLORS[method]
    return PALETTE_FALLBACK[idx % len(PALETTE_FALLBACK)]


# Gate names for GateKeeper (Gates 1 to 5)
GATE_NAMES = {
    1: "exec",
    2: "anti-hack",
    3: "TPE",
    4: "signal",
    5: "short RL pro",
}

# Categories and colors for outcome breakdown (Figure 3a)
CATEGORY_COLORS = {
    "Verified Fix": "#1b9e77",
    "Degenerate / sparse": "#a1d99b",
    "DP đúng nhưng khó học": "#fed976",
    "Exploit cũ còn": "#fd8d3c",
    "Exploit mới": "#e41a1c",
    "Lỗi cú pháp/sandbox": "#999999",
}

CATEGORY_ORDER = [
    "Verified Fix",
    "Degenerate / sparse",
    "DP đúng nhưng khó học",
    "Exploit cũ còn",
    "Exploit mới",
    "Lỗi cú pháp/sandbox",
]

RAW_CATEGORY_MAP = {
    "verified_fix": "Verified Fix",
    "degenerate_sparse": "Degenerate / sparse",
    "unlearnable_shaping": "DP đúng nhưng khó học",
    "same_exploit_persists": "Exploit cũ còn",
    "new_exploit_induced": "Exploit mới",
    "syntax_runtime_error": "Lỗi cú pháp/sandbox",
}


def fmt_ci(vals: Any, scale: float = 100.0) -> str:
    """Formats bootstrap 95% CI string 'mean [lo, hi]' purely from empirical sample values."""
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "0.0 [0.0, 0.0]"
    m, lo, hi = bootstrap_ci(arr * scale)
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes method names in dataframe without altering original data."""
    if df is None or df.empty:
        return pd.DataFrame()
    df_norm = df.copy()
    if "method" in df_norm.columns:
        df_norm["method"] = df_norm["method"].apply(
            lambda m: "CEGR" if m == "CEGIS" else ("CEGR" + m[5:] if str(m).startswith("CEGIS_") else str(m))
        )
    return df_norm


def _save_and_display_figure(fig: Any, filepaths: List[str], show: bool = True):
    """
    Saves figure to specified output paths (300 DPI publication quality)
    and renders it immediately in the notebook cell (Cell 9) and console.
    """
    for fp in filepaths:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(fp)), exist_ok=True)
            fig.savefig(fp, dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"Lưu file biểu đồ thất bại tại {fp}: {e}")

    if show:
        displayed = False
        try:
            from IPython.display import display
            display(fig)
            displayed = True
        except Exception:
            pass

        if not displayed:
            try:
                plt.show()
            except Exception:
                pass
    plt.close(fig)


def _display_and_print_tables(tables: Dict[str, pd.DataFrame]):
    """
    Displays styled tables in Jupyter notebook cell output (Cell 9) and prints formatted
    tables in standard console/terminal logs.
    """
    # 1. Print clean tables in standard terminal / text stream
    for title, df in tables.items():
        if df is None or df.empty:
            continue
        print("\n" + "=" * 80)
        print(f"📊 {title}")
        print("=" * 80)
        print(df.to_string(index=False))

    # 2. Rich HTML display in Jupyter Notebook (Cell 9)
    try:
        from IPython.display import display, HTML
        for title, df in tables.items():
            if df is None or df.empty:
                continue
            html_title = f"<h3 style='color:#08519c; margin-top:22px; margin-bottom:8px; font-weight:bold;'>📊 {title}</h3>"
            display(HTML(html_title))
            display(df)
    except Exception:
        pass


def generate_latex_tables(df_all: pd.DataFrame, out_tex_path: str, show: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Generates LaTeX code for Tables 1 to 4 purely from empirical data in df_all,
    saves to out_tex_path, and renders formatted tables directly in the Jupyter Notebook Cell (Cell 9).
    No hardcoded values or synthetic baselines are used.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_tex_path)), exist_ok=True)
    df_norm = _normalize_df(df_all)

    # Discover evaluated methods dynamically from data
    canonical_order = ["ORIG", "EUREKA", "CARD", "CARD_REFL", "CEGR"]
    methods_in_df = df_norm["method"].dropna().unique().tolist() if ("method" in df_norm.columns and not df_norm.empty) else []
    m_main = [m for m in canonical_order if m in methods_in_df] + [m for m in methods_in_df if m not in canonical_order and not str(m).startswith("CEGR_")]
    if not m_main and methods_in_df:
        m_main = methods_in_df

    # --- TABLE 1: MAIN RESULTS ---
    rows1 = []
    for m in m_main:
        sub = df_norm[df_norm["method"] == m] if not df_norm.empty else pd.DataFrame()
        if sub.empty:
            continue
        cluster_col = "reward_id" if "reward_id" in sub.columns else ("id" if "id" in sub.columns else None)
        g = sub.groupby(cluster_col).mean(numeric_only=True) if cluster_col else sub.mean(numeric_only=True).to_frame().T
        rows1.append({
            "Method": m,
            "Verified Fix (%)": fmt_ci(g.get("vf", [0])),
            "Hack rate DP (%)": fmt_ci(g.get("hacked_dp", [0])),
            "Final RL success (%)": f"{g.get('rl_success', pd.Series([0.0])).mean() * 100:.1f}",
            "Rounds-to-pass": f"{g.get('rounds_to_pass', pd.Series([0.0])).mean():.2f}",
            "LLM calls": f"{g.get('llm_calls', pd.Series([0.0])).mean():.1f}",
            "RL episodes (K)": f"{g.get('rl_episodes', pd.Series([0.0])).mean() / 1e3:.1f}",
        })
    df_t1 = pd.DataFrame(rows1)

    # --- TABLE 2: ABLATIONS ---
    known_abls = ["CEGR", "CEGR_noPool", "CEGR_noAnchors", "CEGR_noClass", "CEGR_noG2", "CEGR_noG4", "CEGR_noG5"]
    abl_in_df = [m for m in methods_in_df if str(m).startswith("CEGR_")]
    m_abl = ([m for m in ["CEGR"] if m in methods_in_df]) + [m for m in known_abls if m in abl_in_df] + [m for m in abl_in_df if m not in known_abls]

    piv_ab = per_reward(df_norm, "vf").astype(float) if ("vf" in df_norm.columns and not df_norm.empty) else pd.DataFrame()
    rows2 = []
    for m in m_abl:
        sub = df_norm[df_norm["method"] == m] if not df_norm.empty else pd.DataFrame()
        if sub.empty:
            continue
        cluster_col = "reward_id" if "reward_id" in sub.columns else ("id" if "id" in sub.columns else None)
        g = sub.groupby(cluster_col).mean(numeric_only=True) if cluster_col else sub.mean(numeric_only=True).to_frame().T
        r = {
            "Variant": m,
            "VF (%)": f"{g.get('vf', pd.Series([0.0])).mean() * 100:.1f}",
            "Rounds-to-pass": f"{g.get('rounds_to_pass', pd.Series([0.0])).mean():.2f}",
            "RL episodes (K)": f"{g.get('rl_episodes', pd.Series([0.0])).mean() / 1e3:.1f}",
        }
        if m != "CEGR" and not piv_ab.empty and "CEGR" in piv_ab and m in piv_ab:
            t = paired_test(piv_ab, "CEGR", m)
            if not np.isnan(t.get("delta", np.nan)):
                r["Δ VF vs CEGR (pp)"] = f"{t['delta']:+.1f} [{t['ci_lo']:+.1f}, {t['ci_hi']:+.1f}] p={t['p_val']:.3f}"
            else:
                r["Δ VF vs CEGR (pp)"] = "N/A"
        rows2.append(r)
    df_t2 = pd.DataFrame(rows2)

    # --- TABLE 3: OUTCOME CLASSIFICATION BREAKDOWN (%) ---
    rows3 = []
    for m in m_main:
        sub = df_norm[df_norm["method"] == m] if not df_norm.empty else pd.DataFrame()
        if sub.empty or "category" not in sub.columns:
            continue
        total_n = len(sub)
        mapped_cats = sub["category"].map(lambda c: RAW_CATEGORY_MAP.get(c, c))
        cat_counts = mapped_cats.value_counts()
        row = {"Method": m}
        for cat in CATEGORY_ORDER:
            pct = (cat_counts.get(cat, 0) / total_n) * 100.0 if total_n > 0 else 0.0
            row[cat] = f"{pct:.1f}%"
        rows3.append(row)
    df_t3 = pd.DataFrame(rows3)

    # --- TABLE 4: GATE REJECTIONS BREAKDOWN (%) ---
    rows4 = []
    gates = [1, 2, 3, 4, 5]
    for m in m_main:
        sub = df_norm[df_norm["method"] == m] if not df_norm.empty else pd.DataFrame()
        if sub.empty or "waves" not in sub.columns:
            continue
        gate_counts = collections.Counter()
        tot_rejected = 0
        for _, r in sub.iterrows():
            w_list = r.get("waves")
            if isinstance(w_list, list):
                for w in w_list:
                    if isinstance(w, dict) and not w.get("passed", True):
                        fg = w.get("failed_gate")
                        if fg is not None:
                            try:
                                gate_counts[int(fg)] += 1
                                tot_rejected += 1
                            except (ValueError, TypeError):
                                pass
        if tot_rejected > 0:
            rows4.append({
                "Method": m,
                "G1 exec (%)": f"{(gate_counts[1] / tot_rejected) * 100.0:.1f}%",
                "G2 anti-hack (%)": f"{(gate_counts[2] / tot_rejected) * 100.0:.1f}%",
                "G3 TPE (%)": f"{(gate_counts[3] / tot_rejected) * 100.0:.1f}%",
                "G4 signal (%)": f"{(gate_counts[4] / tot_rejected) * 100.0:.1f}%",
                "G5 short RL probe (%)": f"{(gate_counts[5] / tot_rejected) * 100.0:.1f}%",
                "Total Rejected": tot_rejected,
            })
        elif len(sub) > 0:
            rows4.append({
                "Method": m,
                "G1 exec (%)": "0.0%",
                "G2 anti-hack (%)": "0.0%",
                "G3 TPE (%)": "0.0%",
                "G4 signal (%)": "0.0%",
                "G5 short RL probe (%)": "0.0%",
                "Total Rejected": 0,
            })
    df_t4 = pd.DataFrame(rows4)

    # Write tables to LaTeX file
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("% =========================================================\n")
        f.write("% CEGR BENCHMARK EVALUATION TABLES (EMPIRICALLY GENERATED)\n")
        f.write("% =========================================================\n\n")
        if not df_t1.empty:
            f.write("% --- TABLE 1: MAIN RESULTS ---\n" + df_t1.to_latex(index=False, escape=False) + "\n\n")
        if not df_t2.empty:
            f.write("% --- TABLE 2: ABLATION RESULTS ---\n" + df_t2.to_latex(index=False, escape=False) + "\n\n")
        if not df_t3.empty:
            f.write("% --- TABLE 3: OUTCOME CLASSIFICATION BREAKDOWN ---\n" + df_t3.to_latex(index=False, escape=False) + "\n\n")
        if not df_t4.empty:
            f.write("% --- TABLE 4: GATEKEEPER REJECTION RATES ---\n" + df_t4.to_latex(index=False, escape=False) + "\n\n")

    # Render and print in Jupyter notebook / console (Cell 9)
    if show:
        tables_to_display = {
            "Bảng 1: Kết quả đối sánh chính (Main Benchmark Results)": df_t1,
            "Bảng 2: Kết quả các biến thể Ablation (Ablation Analysis)": df_t2,
            "Bảng 3: Phân loại kết quả cuối (Outcome Classification Breakdown %)": df_t3,
            "Bảng 4: Tỉ lệ candidate bị loại tại các cổng kiểm định GateKeeper (%)": df_t4,
        }
        _display_and_print_tables(tables_to_display)

    return {"table1": df_t1, "table2": df_t2, "table3": df_t3, "table4": df_t4}


def plot_figure1_vf(df_jobs: pd.DataFrame, out_dir: str, show: bool = True):
    """
    Figure 1: Verified Fix Rate across Methods (with bootstrap 95% CI).
    Dynamically computed from actual experimental data.
    """
    if df_jobs is None or df_jobs.empty or "method" not in df_jobs.columns:
        print("Không có dữ liệu thực nghiệm để vẽ Figure 1.")
        return

    df_norm = _normalize_df(df_jobs)
    canonical_order = ["ORIG", "EUREKA", "CARD", "CARD_REFL", "CEGR"]
    all_methods = df_norm["method"].dropna().unique().tolist()
    ms = [m for m in canonical_order if m in all_methods] + [m for m in all_methods if m not in canonical_order and not str(m).startswith("CEGR_")]
    if not ms:
        ms = all_methods
    if not ms:
        return

    pv = per_reward(df_norm, "vf").astype(float) if "vf" in df_norm.columns else pd.DataFrame()
    st = []
    for m in ms:
        if not pv.empty and m in pv and len(pv[m].dropna()) > 0:
            st.append(bootstrap_ci(pv[m].dropna().values * 100.0))
        elif "vf" in df_norm.columns:
            sub = df_norm[df_norm["method"] == m]
            vals = sub["vf"].dropna().values * 100.0
            st.append(bootstrap_ci(vals) if len(vals) > 0 else (0.0, 0.0, 0.0))
        else:
            st.append((0.0, 0.0, 0.0))

    fig, ax = plt.subplots(figsize=(max(5.5, len(ms) * 1.3), 4.2), dpi=300)
    x = np.arange(len(ms))
    y_means = [s[0] for s in st]
    y_err_lo = [max(0.0, s[0] - s[1]) for s in st]
    y_err_hi = [max(0.0, s[2] - s[0]) for s in st]
    colors = [get_method_color(m, i) for i, m in enumerate(ms)]

    ax.bar(
        x,
        y_means,
        width=0.48,
        yerr=[y_err_lo, y_err_hi],
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.9,
    )
    for xi, s in zip(x, st):
        ax.text(xi, s[0] + 1.5, f"{s[0]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ms, fontsize=9, rotation=15 if len(ms) > 5 else 0)
    ax.set_ylabel("Verified Fix (%)", fontsize=9.5)
    max_top = max([s[2] for s in st], default=10.0)
    ax.set_ylim(0, max(105.0, max_top * 1.15))
    ax.set_title("Figure 1: Verified Fix Rate across Methods", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="-", color="#ececec", linewidth=0.8)

    out_file = os.path.join(out_dir, "fig1_vf_by_method.png")
    _save_and_display_figure(fig, [out_file], show=show)


def plot_figure2_recurrence_and_cost(df_jobs: pd.DataFrame, out_dir: str, show: bool = True):
    """
    Figure 2 (Image 1):
      (a) Lỗi tái phát: CE pool vs trí nhớ văn bản vs không nhớ
      (b) Chất lượng theo chi phí RL (cùng ngân sách gọi LLM)
    Dynamically computed from actual experimental data.
    """
    if df_jobs is None or df_jobs.empty or "method" not in df_jobs.columns:
        print("Không có dữ liệu thực nghiệm để vẽ Figure 2.")
        return

    df_norm = _normalize_df(df_jobs)
    canonical_methods = ["EUREKA", "CARD", "CARD_REFL", "CEGR"]
    all_methods = df_norm["method"].dropna().unique().tolist()
    methods_a = [m for m in canonical_methods if m in all_methods] + [m for m in all_methods if m not in canonical_methods and m != "ORIG" and not str(m).startswith("CEGR_")]
    if not methods_a:
        methods_a = [m for m in all_methods if m != "ORIG"]
    if not methods_a:
        methods_a = all_methods

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=300)

    # -------------------------------------------------------------
    # (a) Lỗi tái phát theo Wave (Cumulative Recurrences)
    # -------------------------------------------------------------
    max_wave = 0
    for _, r in df_norm.iterrows():
        w_list = r.get("waves")
        if isinstance(w_list, list):
            for w in w_list:
                if isinstance(w, dict):
                    w_idx = w.get("wave", w.get("round", None))
                    if isinstance(w_idx, (int, float)):
                        max_wave = max(max_wave, int(w_idx))
    max_wave = max(max_wave, 4)
    waves = np.arange(0, max_wave + 1, dtype=float)

    markers = ["o", "s", "^", "D", "v", "p", "*"]
    max_recur_val = 0.0

    for i, m in enumerate(methods_a):
        sub = df_norm[df_norm["method"] == m]
        cum_lists = []
        if not sub.empty and "waves" in sub.columns:
            for _, r in sub.iterrows():
                w_list = r.get("waves")
                if isinstance(w_list, list) and len(w_list) > 0:
                    cums = []
                    curr_tot = 0.0
                    for w in range(max_wave + 1):
                        matches = [item for item in w_list if isinstance(item, dict) and (item.get("wave") == w or item.get("round") == w)]
                        if matches:
                            rec_step = float(matches[0].get("recur", matches[0].get("violations", 0.0)))
                            curr_tot += rec_step
                        cums.append(curr_tot)
                    cum_lists.append(cums)
                elif "recurrence_count" in r and pd.notna(r["recurrence_count"]):
                    rec_final = float(r["recurrence_count"])
                    cum_lists.append([0.0] + [rec_final] * max_wave)

        if cum_lists:
            y_vals = np.mean(cum_lists, axis=0).tolist()
        elif not sub.empty and "recurrence_count" in sub.columns:
            mean_rec = float(sub["recurrence_count"].dropna().mean()) if len(sub["recurrence_count"].dropna()) > 0 else 0.0
            y_vals = [0.0] + [mean_rec] * max_wave
        else:
            y_vals = [0.0] * (max_wave + 1)

        max_recur_val = max(max_recur_val, max(y_vals) if y_vals else 0.0)
        marker = markers[i % len(markers)]
        color = get_method_color(m, i)
        ax1.plot(
            waves,
            y_vals,
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.9,
            label=m,
            zorder=4,
        )

    ax1.set_xlim(-0.2, max_wave + 0.2)
    ax1.set_ylim(-0.5, max(max_recur_val * 1.15, 10.0))
    ax1.set_xticks(waves)
    ax1.set_xlabel("Wave (0 = reward gốc R0)", fontsize=9.5)
    ax1.set_ylabel("Số lỗi cũ tái phát (tích lũy, TB/job)", fontsize=9.5)
    ax1.set_title("(a) Lỗi tái phát: CE pool vs trí nhớ văn bản vs không nhớ", fontsize=10.5)
    ax1.grid(True, linestyle="-", color="#ececec", linewidth=0.8)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=8)

    # -------------------------------------------------------------
    # (b) Chất lượng theo chi phí RL (cùng ngân sách gọi LLM)
    # -------------------------------------------------------------
    scatter_pts = {}
    for m in methods_a:
        sub = df_norm[df_norm["method"] == m]
        if not sub.empty:
            vf_val = float(sub["vf"].dropna().mean() * 100.0) if ("vf" in sub and len(sub["vf"].dropna()) > 0) else 0.0
            cost_val = float(sub["rl_episodes"].dropna().mean() / 1e3) if ("rl_episodes" in sub and len(sub["rl_episodes"].dropna()) > 0) else 0.0
        else:
            cost_val, vf_val = 0.0, 0.0
        scatter_pts[m] = (cost_val, vf_val)

    max_cost = max([pt[0] for pt in scatter_pts.values()], default=1.0)
    max_vf = max([pt[1] for pt in scatter_pts.values()], default=10.0)

    for i, m in enumerate(methods_a):
        x_c, y_v = scatter_pts[m]
        color = get_method_color(m, i)
        ax2.scatter(
            x_c,
            y_v,
            s=125,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
        ax2.annotate(
            m,
            (x_c, y_v),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
            color="black",
        )

    ax2.set_xlim(-0.05 * max(max_cost, 1.0), max(max_cost * 1.15, 1.0))
    ax2.set_ylim(-0.5, max(max_vf * 1.2, 10.0))
    ax2.set_xlabel("Chi phí RL trung bình / job (nghìn episode Q-learning)", fontsize=9.5)
    ax2.set_ylabel("Verified Fix (%)", fontsize=9.5)
    ax2.set_title("(b) Chất lượng theo chi phí RL (cùng ngân sách gọi LLM)", fontsize=10.5)
    ax2.grid(True, linestyle="-", color="#ececec", linewidth=0.8)

    out_file1 = os.path.join(out_dir, "fig2_recurrence_and_cost.png")
    out_file2 = os.path.join(out_dir, "fig_image1_recurrence_cost.png")
    _save_and_display_figure(fig, [out_file1, out_file2], show=show)


def plot_figure3_classification_and_gates(df_jobs: pd.DataFrame, out_dir: str, show: bool = True):
    """
    Figure 3 (Image 2):
      (a) Phân loại kết quả (planted)
      (b) Cổng nào loại candidate (rẻ → đắt)
    Dynamically computed from actual experimental data.
    """
    if df_jobs is None or df_jobs.empty or "method" not in df_jobs.columns:
        print("Không có dữ liệu thực nghiệm để vẽ Figure 3.")
        return

    df_norm = _normalize_df(df_jobs)
    canonical_order = ["ORIG", "EUREKA", "CARD", "CARD_REFL", "CEGR"]
    all_methods = df_norm["method"].dropna().unique().tolist()
    methods_cat = [m for m in canonical_order if m in all_methods] + [m for m in all_methods if m not in canonical_order and not str(m).startswith("CEGR_")]
    if not methods_cat:
        methods_cat = all_methods
    if not methods_cat:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.2), dpi=300)

    # -------------------------------------------------------------
    # (a) Phân loại kết quả (planted) - 100% Stacked Bar Chart
    # -------------------------------------------------------------
    cat_pcts = {cat: [] for cat in CATEGORY_ORDER}

    for m in methods_cat:
        sub = df_norm[df_norm["method"] == m]
        if not sub.empty and "category" in sub.columns:
            total_n = len(sub)
            mapped_cats = sub["category"].map(lambda c: RAW_CATEGORY_MAP.get(c, c))
            cat_counts = mapped_cats.value_counts()
            for cat in CATEGORY_ORDER:
                pct = (cat_counts.get(cat, 0) / total_n) * 100.0 if total_n > 0 else 0.0
                cat_pcts[cat].append(pct)
        else:
            for cat in CATEGORY_ORDER:
                cat_pcts[cat].append(0.0)

    x_indices = np.arange(len(methods_cat))
    bar_width = 0.48
    bottom_vals = np.zeros(len(methods_cat))

    for cat in CATEGORY_ORDER:
        vals = np.array(cat_pcts[cat])
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        ax1.bar(
            x_indices,
            vals,
            bottom=bottom_vals,
            width=bar_width,
            label=cat,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        for i, val in enumerate(vals):
            if val >= 5.0:
                y_pos = bottom_vals[i] + val / 2.0
                ax1.text(
                    x_indices[i],
                    y_pos,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    fontweight="bold",
                    color="black",
                )
        bottom_vals += vals

    ax1.set_ylim(0, 105)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(methods_cat, rotation=15 if len(methods_cat) > 4 else 0, ha="right" if len(methods_cat) > 4 else "center", fontsize=9.5)
    ax1.set_ylabel("% kết quả cuối", fontsize=9.5)
    ax1.set_title("(a) Phân loại kết quả", fontsize=10.5)
    ax1.grid(axis="y", linestyle="-", color="#ececec", linewidth=0.8)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, edgecolor="#d0d0d0", fontsize=7.5)

    # -------------------------------------------------------------
    # (b) Cổng nào loại candidate (rẻ → đắt) - Grouped Bar Chart
    # -------------------------------------------------------------
    gates = [1, 2, 3, 4, 5]
    gate_labels = [
        "G1\nexec",
        "G2\nanti-hack",
        "G3\nTPE",
        "G4\nsignal",
        "G5\nshort RL pro",
    ]

    gate_methods = [m for m in ["CEGR", "CARD", "EUREKA"] if m in all_methods]
    if not gate_methods:
        gate_methods = [m for m in all_methods if m != "ORIG"]
    if not gate_methods:
        gate_methods = all_methods

    gate_rejection_data = {}
    for m in gate_methods:
        sub = df_norm[df_norm["method"] == m]
        gate_counts = collections.Counter()
        tot_rejected = 0
        if not sub.empty and "waves" in sub.columns:
            for _, r in sub.iterrows():
                w_list = r.get("waves")
                if isinstance(w_list, list):
                    for w in w_list:
                        if isinstance(w, dict) and not w.get("passed", True):
                            fg = w.get("failed_gate")
                            if fg is not None:
                                try:
                                    gate_counts[int(fg)] += 1
                                    tot_rejected += 1
                                except (ValueError, TypeError):
                                    pass
        if tot_rejected > 0:
            g_rates = [(gate_counts[g] / tot_rejected) * 100.0 for g in gates]
        else:
            g_rates = [0.0] * len(gates)
        gate_rejection_data[m] = g_rates

    x_g = np.arange(len(gates))
    n_gm = max(len(gate_methods), 1)
    sub_bar_width = min(0.8 / n_gm, 0.28)
    offsets = np.linspace(-sub_bar_width * (n_gm - 1) / 2.0, sub_bar_width * (n_gm - 1) / 2.0, n_gm) if n_gm > 1 else [0.0]

    for m, offset, i in zip(gate_methods, offsets, range(len(gate_methods))):
        vals = gate_rejection_data[m]
        color = get_method_color(m, i)
        ax2.bar(
            x_g + offset,
            vals,
            width=sub_bar_width,
            label=m,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )

    ax2.set_ylim(0, 105)
    ax2.set_xticks(x_g)
    ax2.set_xticklabels(gate_labels, fontsize=8.5)
    ax2.set_ylabel("% candidate bị loại tại cổng", fontsize=9.5)
    ax2.set_title("(b) Cổng nào loại candidate (rẻ → đắt)", fontsize=10.5)
    ax2.grid(axis="y", linestyle="-", color="#ececec", linewidth=0.8)
    ax2.legend(loc="upper left", frameon=True, edgecolor="#d0d0d0", fontsize=8)

    out_file1 = os.path.join(out_dir, "fig3_breakdown_and_gates.png")
    out_file2 = os.path.join(out_dir, "fig_image2_classification_gates.png")
    _save_and_display_figure(fig, [out_file1, out_file2], show=show)


def plot_evaluation_figures(df_jobs: pd.DataFrame, out_dir: str, show: bool = True):
    """
    Generates 300 DPI publication plots (Figures 1, 2, and 3),
    saves them to out_dir, and immediately renders them inline in the notebook cell (Cell 9).
    All data is computed dynamically from df_jobs without hardcoding.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Figure 1: Verified Fix across Methods
    plot_figure1_vf(df_jobs, out_dir, show=show)

    # 2. Figure 2 (Image 1): Recurrence by Wave & Quality vs. RL Exploration Cost
    plot_figure2_recurrence_and_cost(df_jobs, out_dir, show=show)

    # 3. Figure 3 (Image 2): Outcome Classification & Gatekeeper Rejection Rates
    plot_figure3_classification_and_gates(df_jobs, out_dir, show=show)
