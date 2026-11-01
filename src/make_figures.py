"""
make_figures.py

PURPOSE
-------
Turns the numeric tables train_pipeline.py and evaluate.py already produced
into the actual figures and formatted tables for the paper. This file does
NOT compute anything new -- no model fitting, no bootstrap, no SHAP
computation (the one exception, predictor-level SHAP aggregation, only
re-aggregates ALREADY-COMPUTED raw SHAP arrays -- see Figure 4 below).

Generates:
  Figure 1: CV ablation -- ROC-AUC per model, baseline vs lifestyle-augmented,
            tuned primary phenotype B, with CV std as error bars
  Figure 2: Held-out test paired comparison -- forest plot of the
            lifestyle-vs-baseline ROC-AUC delta + 95% CI per model
  Figure 3: Cross-model SHAP stability heatmap (top-10 feature Jaccard
            overlap), lifestyle-augmented arm and baseline arm
  Figure 4: Top-10 PREDICTOR-LEVEL SHAP importance for the best model --
            one-hot-encoded columns (e.g. RIDRETH3_1.0, RIDRETH3_3.0, ...)
            are aggregated back to their source predictor (RIDRETH3) before
            ranking, so the chart reports on conceptual variables, not
            encoded columns. See aggregate_shap_to_predictor_level().
  Figure 5: Phenotype A vs. B held-out test ROC-AUC comparison -- two
            side-by-side panels, same structure as Figure 1 but for the
            fixed 10% test set, one panel per phenotype definition.
  Table 1:  Cohort flow + phenotype breakdown, formatted as a LaTeX-ready
            table (paper/table1_cohort.tex)

USAGE
-----
    python src/make_figures.py

OUTPUT
------
    outputs/figures/fig1_cv_ablation.pdf
    outputs/figures/fig2_test_paired_comparison.pdf
    outputs/figures/fig3_shap_stability_heatmap_{baseline,lifestyle_augmented}.pdf
    outputs/figures/fig4_top_features_predictor_level_<model>.pdf
    outputs/figures/fig5_phenotype_comparison.pdf
    paper/table1_cohort.tex

All figures are generated from the tables and SHAP arrays written by
evaluate.py; rerunning this script after evaluate.py regenerates them.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")
SHAP_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "shap_values")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "bagged_decision_tree": "Bagged Decision Tree",
    "xgboost": "XGBoost",
}
MODEL_ORDER = list(MODEL_LABELS.keys())

# ATTENTION: consistent color scheme across all figures so "baseline" and
# "lifestyle_augmented" always mean the same color to a reader flipping
# between figures in the paper.
ARM_COLORS = {"baseline": "#4C72B0", "lifestyle_augmented": "#DD8452"}
ARM_LABELS = {"baseline": "Baseline", "lifestyle_augmented": "Lifestyle-augmented"}

# Original categorical predictor names -- one-hot columns for these get
# aggregated back together in Figure 4. Mirrors preprocessing.py's
# CATEGORICAL_FEATURES exactly (hardcoded here rather than imported so this
# file has no dependency on preprocessing.py's heavier import chain).
CATEGORICAL_BASES = {"RIDRETH3", "DMDEDUC2", "drinking_status", "smoking_status"}


def _setup_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "axes.titlesize": 11,
        "axes.titlepad": 12,  # ATTENTION: extra title padding, applied
                                # globally -- part of the title-overlap fix,
                                # combined with generous figsize and always
                                # calling tight_layout() below.
    })


def _finalize_and_save(fig, out_path: str):
    """
    ATTENTION -- TITLE-SAFETY: single choke point every figure in this file
    goes through before saving. Combines tight_layout() (auto-fits margins
    to actual content) with bbox_inches='tight' on save (trims any residual
    whitespace/overflow) -- if a title needed manual wrapping it should
    have been shortened or the figure widened BEFORE calling this, not
    patched here -- see each figure's own width choice and title text
    length, chosen deliberately short enough to fit.
    """
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 1: CV ABLATION
# ---------------------------------------------------------------------------

def make_fig1_cv_ablation():
    df = pd.read_csv(os.path.join(TABLES_DIR, "cv_results.csv"))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    for i, arm in enumerate(["baseline", "lifestyle_augmented"]):
        sub = df[df["feature_set"] == arm].set_index("model").loc[MODEL_ORDER]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset, sub["roc_auc_mean"], width,
            yerr=sub["roc_auc_std"], capsize=3,
            label=ARM_LABELS[arm], color=ARM_COLORS[arm],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC (5-fold CV mean \u00b1 SD)")
    ax.set_ylim(0.65, 0.85)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Tuned CV performance: baseline vs. lifestyle-augmented")

    _finalize_and_save(fig, os.path.join(FIGURES_DIR, "fig1_cv_ablation.pdf"))


# ---------------------------------------------------------------------------
# FIGURE 2: HELD-OUT TEST PAIRED COMPARISON (FOREST PLOT)
# ---------------------------------------------------------------------------

def make_fig2_paired_comparison():
    df = pd.read_csv(os.path.join(TABLES_DIR, "paired_comparison.csv"))
    df = df.set_index("model").loc[MODEL_ORDER].reset_index()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(df))

    err_lo = df["roc_auc_delta"] - df["roc_auc_delta_ci_lo"]
    err_hi = df["roc_auc_delta_ci_hi"] - df["roc_auc_delta"]

    colors = ["#2A9D8F" if sig else "#888888" for sig in df["roc_auc_significant"]]
    ax.errorbar(
        df["roc_auc_delta"], y, xerr=[err_lo, err_hi],
        fmt="o", capsize=3, color="black", ecolor="black", markersize=0,
    )
    ax.scatter(df["roc_auc_delta"], y, color=colors, zorder=3, s=60)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[m] for m in df["model"]])
    ax.set_xlabel("\u0394 ROC-AUC (lifestyle-augmented \u2212 baseline), held-out test")
    ax.set_title("Paired bootstrap comparison, tuned models (95% CI)")
    ax.invert_yaxis()
    xmax = max(abs(err_lo.max()), abs(err_hi.max())) + abs(df["roc_auc_delta"]).max()
    ax.set_xlim(-xmax * 1.3, xmax * 1.3)

    _finalize_and_save(fig, os.path.join(FIGURES_DIR, "fig2_test_paired_comparison.pdf"))
    print("  Note: gray markers = paired 95% CI includes zero.")


# ---------------------------------------------------------------------------
# FIGURE 3: SHAP STABILITY HEATMAP
# ---------------------------------------------------------------------------

def make_fig3_shap_stability_heatmap(arm: str = "lifestyle_augmented", path_suffix: str = ""):
    filename = f"shap_stability{path_suffix}.csv"
    df = pd.read_csv(os.path.join(TABLES_DIR, filename))
    df = df[df["feature_set"] == arm]

    matrix = pd.DataFrame(1.0, index=MODEL_ORDER, columns=MODEL_ORDER)
    for _, row in df.iterrows():
        matrix.loc[row["model_a"], row["model_b"]] = row["jaccard"]
        matrix.loc[row["model_b"], row["model_a"]] = row["jaccard"]

    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    im = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=1)

    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(MODEL_ORDER)):
        for j in range(len(MODEL_ORDER)):
            val = matrix.values[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Top-10 feature Jaccard overlap")
    # ATTENTION -- TITLE-SAFETY: kept deliberately short. "Lifestyle-augmented"
    # abbreviated to "Lifestyle" here specifically because the "(Phenotype A)"
    # suffix pushed the full-length version past the figure edge and got
    # truncated -- confirmed by actually rendering this exact variant before
    # this fix. Widened figsize too, but the real fix is a title that fits
    # even in the longest (phenotype-A) case, not relying on width alone.
    arm_short = "Lifestyle" if arm == "lifestyle_augmented" else "Baseline"
    label_suffix = " (Phenotype A)" if path_suffix else ""
    ax.set_title(f"SHAP stability: {arm_short}{label_suffix}", fontsize=10)

    out_name = f"fig3_shap_stability_heatmap_{arm}{path_suffix}.pdf"
    _finalize_and_save(fig, os.path.join(FIGURES_DIR, out_name))


# ---------------------------------------------------------------------------
# FIGURE 4: PREDICTOR-LEVEL SHAP IMPORTANCE (one-hot columns aggregated)
# ---------------------------------------------------------------------------

def aggregate_shap_to_predictor_level(shap_values: np.ndarray, feature_names: list) -> tuple:
    """
    Aggregate one-hot-encoded SHAP columns back to their source predictor.

    ATTENTION -- WHY THIS EXISTS: reporting SHAP
    importance at the encoded-column level means "race/ethnicity" never
    appears as a single ranked item -- it's fragmented across
    RIDRETH3_1.0, RIDRETH3_3.0, etc., each individually less important than
    the concept as a whole. This under-represents categorical predictors
    relative to numeric ones in any top-N ranking.

    ATTENTION -- WHY WE SUM RAW SIGNED VALUES, NOT THE ALREADY-AVERAGED
    mean_abs_shap PER COLUMN: for a one-hot group, exactly one column is
    "active" per sample. Summing the RAW (signed) SHAP values across the
    group for each sample gives that sample's total contribution from the
    original categorical variable; taking the absolute value of THAT sum,
    then averaging across samples, is the mathematically correct
    aggregation. Simply adding up each column's already-computed
    mean(|shap|) would overcount whenever different categories' effects
    partially offset across samples. This is why this function takes the
    RAW per-sample arrays (loaded from outputs/shap_values/*.npy) rather
    than working from the summarized shap_importance.csv.

    Returns: (aggregated_shap_matrix [n_samples x n_aggregated_features],
              aggregated_feature_names)
    """
    groups = {}
    passthrough_idx = []
    passthrough_names = []

    for i, name in enumerate(feature_names):
        matched_base = None
        for base in CATEGORICAL_BASES:
            if name == base or name.startswith(base + "_"):
                matched_base = base
                break
        if matched_base:
            groups.setdefault(matched_base, []).append(i)
        else:
            passthrough_idx.append(i)
            passthrough_names.append(name)

    agg_columns = []
    agg_names = []
    for base, idxs in groups.items():
        agg_columns.append(shap_values[:, idxs].sum(axis=1))
        agg_names.append(base)
    for i, name in zip(passthrough_idx, passthrough_names):
        agg_columns.append(shap_values[:, i])
        agg_names.append(name)

    return np.column_stack(agg_columns), agg_names


def _reconstruct_feature_names(feature_list: list) -> list:
    """
    ATTENTION: Reconstructs the post-one-hot-encoding column names WITHOUT
    importing preprocessing.py's full dependency chain -- duplicates just
    enough of build_column_transformer's logic (fit a OneHotEncoder on the
    known categorical columns) to recover the same names in the same
    order. If preprocessing.py's feature lists or encoding logic ever
    change, this must be updated to match -- it is deliberately NOT a
    shared import specifically to keep make_figures.py lightweight and
    independently runnable, at the cost of needing to stay manually in
    sync. If this ever feels fragile, importing directly from
    preprocessing.py is the more robust (if heavier) alternative.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import build_column_transformer, get_feature_matrix, load_and_split

    train_df, _ = load_and_split()
    ct = build_column_transformer(feature_list)
    X_sample = get_feature_matrix(train_df, feature_list)
    ct.fit(X_sample)
    return ct.get_feature_names_out().tolist()


def make_fig4_top_features_predictor_level(arm: str = "lifestyle_augmented", top_n: int = 10):
    """
    ATTENTION: "Best model" is picked by CV ROC-AUC on this arm -- a
    defensible, pre-specifiable choice, not whichever model's chart looks
    nicest after the fact.
    """
    from preprocessing import MODELING_BASELINE_FEATURES, MODELING_LIFESTYLE_FEATURES

    cv_df = pd.read_csv(os.path.join(TABLES_DIR, "cv_results.csv"))
    arm_cv = cv_df[cv_df["feature_set"] == arm].sort_values("roc_auc_mean", ascending=False)
    best_model = arm_cv.iloc[0]["model"]
    print(f"  Best model on {arm} by CV ROC-AUC: {best_model} "
          f"({arm_cv.iloc[0]['roc_auc_mean']:.3f})")

    feature_list = (MODELING_BASELINE_FEATURES if arm == "baseline"
                     else MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES)
    feature_names = _reconstruct_feature_names(feature_list)

    npy_path = os.path.join(SHAP_DIR, f"{arm}__{best_model}.npy")
    shap_values = np.load(npy_path)

    agg_matrix, agg_names = aggregate_shap_to_predictor_level(shap_values, feature_names)
    mean_abs_shap = np.abs(agg_matrix).mean(axis=0)

    order = np.argsort(mean_abs_shap)[-top_n:]
    top_names = [agg_names[i] for i in order]
    top_values = mean_abs_shap[order]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(top_names, top_values, color=ARM_COLORS[arm])
    ax.set_xlabel("Mean |SHAP value| (one-hot columns aggregated to source predictor)")
    ax.set_title(f"Top {top_n} predictors: {MODEL_LABELS[best_model]} ({ARM_LABELS[arm]})",
                 fontsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    out_name = f"fig4_top_features_predictor_level_{best_model}.pdf"
    _finalize_and_save(fig, os.path.join(FIGURES_DIR, out_name))


# ---------------------------------------------------------------------------
# FIGURE 5: PHENOTYPE A vs B HELD-OUT TEST COMPARISON
# ---------------------------------------------------------------------------

def make_fig5_phenotype_comparison():
    """
    Two side-by-side panels (Phenotype B primary, Phenotype A sensitivity),
    each showing held-out test ROC-AUC with 95% CI, baseline vs
    lifestyle-augmented, per model -- directly visualizes both (a) the
    phenotype A vs B predictability gap and (b) that the "no detectable
    lifestyle benefit" pattern replicates across both phenotypes.
    """
    df_b = pd.read_csv(os.path.join(TABLES_DIR, "test_metrics.csv"))
    df_a = pd.read_csv(os.path.join(TABLES_DIR, "test_metrics_phenotype_a.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    for ax, df, title in [(axes[0], df_b, "Phenotype B (primary)"),
                           (axes[1], df_a, "Phenotype A (sensitivity)")]:
        for i, arm in enumerate(["baseline", "lifestyle_augmented"]):
            sub = df[df["feature_set"] == arm].set_index("model").loc[MODEL_ORDER]
            err_lo = sub["roc_auc"] - sub["roc_auc_ci_lo"]
            err_hi = sub["roc_auc_ci_hi"] - sub["roc_auc"]
            offset = (i - 0.5) * width
            ax.bar(
                x + offset, sub["roc_auc"], width,
                yerr=[err_lo, err_hi], capsize=3,
                label=ARM_LABELS[arm], color=ARM_COLORS[arm],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.6, 0.9)

    axes[0].set_ylabel("ROC-AUC (held-out test, 95% CI)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Held-out predictive performance by phenotype definition", fontsize=11, y=1.02)

    _finalize_and_save(fig, os.path.join(FIGURES_DIR, "fig5_phenotype_comparison.pdf"))


# ---------------------------------------------------------------------------
# TABLE 1: COHORT FLOW + PHENOTYPE BREAKDOWN -> LATEX
# ---------------------------------------------------------------------------

def _parse_cohort_flow_md() -> list:
    path = os.path.join(TABLES_DIR, "cohort_flow.md")
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(r"\|\s*(.+?)\s*\|\s*(\d+)\s*\|", line)
            if m:
                rows.append((m.group(1), int(m.group(2))))
    return rows


def _parse_phenotype_breakdown_md() -> dict:
    path = os.path.join(TABLES_DIR, "phenotype_breakdown.md")
    with open(path) as f:
        text = f.read()

    result = {}
    for label, key in [
        ("A - measured BP only", "A"),
        ("B - measured OR medication (PRIMARY)", "B"),
        ("C - B OR self-reported diagnosis", "C"),
    ]:
        section = re.search(
            rf"## {re.escape(label)}\n(.*?)(?=\n##|\Z)", text, re.DOTALL
        )
        if not section:
            continue
        block = section.group(1)
        n_elig = int(re.search(r"eligible N.*?:\s*(\d+)", block).group(1))
        n_pos = int(re.search(r"Positive:\s*(\d+)", block).group(1))
        prevalence = float(re.search(r"Prevalence:\s*([\d.]+)", block).group(1))
        result[key] = {"n_eligible": n_elig, "n_positive": n_pos, "prevalence": prevalence}
    return result


def make_table1_cohort():
    flow_rows = _parse_cohort_flow_md()
    phenotype = _parse_phenotype_breakdown_md()

    os.makedirs(PAPER_DIR, exist_ok=True)
    out_path = os.path.join(PAPER_DIR, "table1_cohort.tex")

    with open(out_path, "w") as f:
        f.write("% Auto-generated by make_figures.py -- do not edit by hand.\n")
        f.write("\\begin{table}[htbp]\n\\floatconts\n  {tab:cohort}\n")
        f.write("  {\\caption{Cohort construction and hypertension phenotype prevalence}}\n")
        f.write("  {\\begin{tabular}{lr}\n  \\toprule\n")
        f.write("  \\bfseries Cohort step & \\bfseries N \\\\\n  \\midrule\n")
        for step, n in flow_rows:
            f.write(f"  {step} & {n:,} \\\\\n")
        f.write("  \\midrule\n")
        for key, nice_name in [("A", "Phenotype A (measured only)"),
                                 ("B", "Phenotype B (measured or meds, primary)"),
                                 ("C", "Phenotype C (B or self-report)")]:
            if key in phenotype:
                p = phenotype[key]
                f.write(f"  {nice_name}: N positive (\\%) & "
                        f"{p['n_positive']:,} ({p['prevalence']:.1f}\\%) \\\\\n")
        f.write("  \\bottomrule\n  \\end{tabular}}\n\\end{table}\n")

    print(f"[wrote] {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    _setup_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("--- Figure 1: CV ablation (tuned primary) ---")
    make_fig1_cv_ablation()

    print("\n--- Figure 2: Held-out test paired comparison (tuned primary) ---")
    make_fig2_paired_comparison()

    print("\n--- Figure 3: SHAP stability heatmaps ---")
    make_fig3_shap_stability_heatmap("lifestyle_augmented")
    make_fig3_shap_stability_heatmap("baseline")
    if os.path.exists(os.path.join(TABLES_DIR, "shap_stability_phenotype_a.csv")):
        make_fig3_shap_stability_heatmap("lifestyle_augmented", path_suffix="_phenotype_a")
        make_fig3_shap_stability_heatmap("baseline", path_suffix="_phenotype_a")

    print("\n--- Figure 4: Predictor-level top features, best model ---")
    make_fig4_top_features_predictor_level("lifestyle_augmented")

    if os.path.exists(os.path.join(TABLES_DIR, "test_metrics_phenotype_a.csv")):
        print("\n--- Figure 5: Phenotype A vs B comparison ---")
        make_fig5_phenotype_comparison()
    else:
        print("\n--- Figure 5 skipped: test_metrics_phenotype_a.csv not found ---")

    print("\n--- Table 1: Cohort + phenotype (LaTeX) ---")
    make_table1_cohort()

    print("\nDone. Figures in outputs/figures/, table in paper/table1_cohort.tex")


if __name__ == "__main__":
    main()
