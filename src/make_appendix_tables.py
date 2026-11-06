"""
make_appendix_tables.py

PURPOSE
-------
Format the supplementary-analysis result tables in outputs/tables/ as
LaTeX table snippets for the paper's appendices. Nothing is recomputed:
every number is read from a CSV written by an earlier script and only
rounded here, so the paper tables cannot drift from the saved outputs.

Source scripts for the CSVs read here:
    shap_agreement_controls.py   -> shap_agreement_controls*.csv,
                                    shap_domain_importance*.csv
    calibration_analysis.py      -> calibration_metrics.csv,
                                    calibration_paired_deltas.csv
    robustness_refits.py         -> untreated_results.csv, ablation_results.csv,
                                    train_seed_*.csv, repeated_splits_*.csv
    train_pipeline.py --tune     -> tuning_results.csv
    train_pipeline.py (untuned)  -> outputs/untuned_primary/cv_results.csv
    feature_redundancy.py        -> redundancy_*.csv

USAGE
-----
    python src/make_appendix_tables.py

OUTPUT
------
    paper/appendix_tables/*.tex   (one file per table)
"""

import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR = os.path.join(ROOT, "outputs", "tables")
OUT_DIR = os.path.join(ROOT, "paper", "appendix_tables")

MODELS = ["logistic_regression", "random_forest", "gradient_boosting",
          "bagged_decision_tree", "xgboost"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "bagged_decision_tree": "Bagged Decision Tree",
    "xgboost": "XGBoost",
}
DOMAIN_LABELS = {
    "demographic_ses": "Demographic/SES",
    "anthropometric": "Anthropometric",
    "sleep": "Sleep",
    "smoking": "Smoking",
    "alcohol": "Alcohol",
    "activity": "Activity",
}
FEATURE_LABELS = {
    "total_vigorous_min_wk": "Vigorous activity (min/wk)",
    "total_moderate_min_wk": "Moderate activity (min/wk)",
    "sedentary_min_day": "Sedentary time (min/day)",
    "drinking_status_current": "Drinking status: current",
    "drinking_status_former": "Drinking status: former",
    "drinking_status_never": "Drinking status: never",
    "alq130_avg_drinks_filled": "Average drinks/day",
    "smoking_status_current": "Smoking status: current",
    "smoking_status_former": "Smoking status: former",
    "smoking_status_never": "Smoking status: never",
    "smoked_recently": "Recent tobacco use",
    "SLD012": "Weekday sleep duration",
    "SLD013": "Weekend sleep duration",
    "trouble_sleeping": "Trouble sleeping",
    "SLQ120": "Daytime sleepiness",
}


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------------

def num(x: float, dp: int = 3, sign: bool = False) -> str:
    """Round to dp places with a LaTeX minus sign and optional plus sign.
    A value that rounds to zero is printed unsigned (e.g. 0.000)."""
    s = f"{abs(x):.{dp}f}"
    if float(s) == 0:
        return s
    if x < 0:
        return f"$-${s}"
    return f"$+${s}" if sign else s


def bound(x: float, dp: int = 3) -> str:
    """CI bound: if a nonzero bound rounds to zero, add places (up to 5)
    so its sign stays visible, e.g. +0.00001 rather than 0.000."""
    while x != 0 and float(f"{abs(x):.{dp}f}") == 0 and dp < 5:
        dp += 1
    return num(x, dp, sign=True)


def ci(lo: float, hi: float, dp: int = 3) -> str:
    return f"[{bound(lo, dp)}, {bound(hi, dp)}]"


def write(name: str, body: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body.strip() + "\n")
    print(f"wrote {os.path.relpath(path, ROOT)}")


def table(env: str, label: str, caption: str, size: str, spec: str,
          header: str, rows: list, colsep: str = "") -> str:
    return "\n".join([
        f"\\begin{{{env}}}[htbp]",
        "\\floatconts",
        f"  {{{label}}}",
        f"  {{\\caption{{{caption}}}}}",
        f"  {{{size}"
        + (f"\\setlength{{\\tabcolsep}}{{{colsep}}}" if colsep else "")
        + f"\\begin{{tabular}}{{{spec}}}",
        "  \\toprule",
        f"  {header} \\\\",
        "  \\midrule",
        *[f"  {r}" for r in rows],
        "  \\bottomrule",
        "  \\end{tabular}}",
        f"\\end{{{env}}}",
    ])


# ---------------------------------------------------------------------------
# SHAP CONTROLS
# ---------------------------------------------------------------------------

def shap_robustness() -> str:
    settings = [("Primary", ""), ("Phenotype A", "_phenotype_a"),
                ("No SMOTE", "_no_smote")]
    metrics = [("spearman", "Spearman $\\rho$"), ("top10_jaccard", "Top-10 Jaccard")]
    rows = []
    for i, (label, suf) in enumerate(settings):
        df = pd.read_csv(os.path.join(TABLES_DIR, f"shap_agreement_controls{suf}.csv"))
        for j, (metric, mlabel) in enumerate(metrics):
            full = df[(df.view == "full") & (df.metric == metric)].iloc[0]
            sh = df[(df.view == "shared19") & (df.metric == metric)].iloc[0]
            rows.append(" & ".join([
                label if j == 0 else "", mlabel.replace("Top-10 ", ""),
                num(full.baseline), num(full.lifestyle_augmented),
                f"{num(full.delta, sign=True)} {ci(full.delta_ci_lo, full.delta_ci_hi)}",
                num(sh.lifestyle_augmented),
                f"{num(sh.delta, sign=True)} {ci(sh.delta_ci_lo, sh.delta_ci_hi)}",
            ]) + " \\\\")
        if i < len(settings) - 1:
            rows.append("\\midrule")
    header = ("\\bfseries Setting & \\bfseries Metric & \\bfseries Base. & "
              "\\bfseries Life., all & \\bfseries $\\Delta$ all [95\\% CI] & "
              "\\bfseries Life., shared & \\bfseries $\\Delta$ shared [95\\% CI]")
    caption = ("Cross-model SHAP agreement by setting (mean pairwise over five models; "
               "Base.\\ = baseline arm, Life.\\ = lifestyle-augmented arm). "
               "``All'' uses each arm's full encoded column set (19 vs.\\ 36); "
               "``shared'' restricts the lifestyle arm to the 19 baseline columns, "
               "so the baseline value is unchanged. Jaccard is top-10 overlap. "
               "$\\Delta$ is lifestyle minus baseline; "
               "95\\% CIs are paired bootstraps over the 300 explained participants "
               "and do not include model-refit variability. Chance-level top-10 "
               "Jaccard is 0.365 for 19 columns and 0.167 for 36.")
    return table("table*", "tab:shap_robustness", caption, "\\scriptsize",
                 "llccccc", header, rows, "4pt")


def shap_domain() -> str:
    df = pd.read_csv(os.path.join(TABLES_DIR, "shap_domain_importance.csv"))
    share = df.groupby("domain").net_share.mean().sort_values(ascending=False)
    rank = df.pivot(index="domain", columns="model", values="rank")
    rows = []
    for d in share.index:
        ranks = " & ".join(str(int(rank.loc[d, m])) for m in MODELS)
        rows.append(f"{DOMAIN_LABELS[d]} & {100 * share[d]:.1f}\\% & {ranks} \\\\")
    ctl = pd.read_csv(os.path.join(TABLES_DIR, "shap_agreement_controls.csv"))
    dom = ctl[ctl.view == "domain"].iloc[0]
    header = ("\\bfseries Domain & \\bfseries Share & \\bfseries LR & "
              "\\bfseries RF & \\bfseries GB & \\bfseries BDT & \\bfseries XGB")
    caption = ("Domain-level SHAP importance, lifestyle-augmented arm, primary "
               "phenotype. Per-sample signed SHAP values are summed within each "
               "domain before absolute values are averaged over samples (the net "
               "grouped contribution, as in Figure~\\ref{fig:top_features}); opposing "
               "attributions within a domain can cancel. ``Mean share'' averages each "
               "domain's share of net grouped SHAP magnitude across the five models; "
               "remaining columns give each model's domain rank. Mean pairwise "
               f"Spearman agreement over the six domains is {num(dom.lifestyle_augmented)} "
               f"{ci(dom.delta_ci_lo, dom.delta_ci_hi)}.")
    return table("table", "tab:shap_domain", caption, "\\scriptsize",
                 "@{}lcccccc@{}", header, rows, "3pt")


# ---------------------------------------------------------------------------
# CALIBRATION AND DECISION CURVES
# ---------------------------------------------------------------------------

def calibration() -> str:
    met = pd.read_csv(os.path.join(TABLES_DIR, "calibration_metrics.csv"))
    dl = pd.read_csv(os.path.join(TABLES_DIR, "calibration_paired_deltas.csv"))
    rows = []
    for m in MODELS:
        b = met[(met.feature_set == "baseline") & (met.model == m)].iloc[0]
        l = met[(met.feature_set == "lifestyle_augmented") & (met.model == m)].iloc[0]
        d = dl[(dl.model == m) & dl.metric.str.startswith("brier")].iloc[0]
        rows.append(" & ".join([
            MODEL_LABELS[m],
            f"{num(b.brier)} / {num(l.brier)}",
            f"{num(b.cal_slope, 2)} / {num(l.cal_slope, 2)}",
            f"{num(b.cal_in_the_large, 2, True)} / {num(l.cal_in_the_large, 2, True)}",
            f"{num(b.mean_pred)} / {num(l.mean_pred)}",
            f"{num(d.delta_lifestyle_minus_baseline, 4, True)} "
            f"{ci(d.ci_lo, d.ci_hi, 4)}",
        ]) + " \\\\")
    obs = met.observed_rate.iloc[0]
    header = ("\\bfseries Model & \\bfseries Brier & \\bfseries Cal.\\ slope & "
              "\\bfseries CITL & \\bfseries Mean pred. & "
              "\\bfseries $\\Delta$ Brier [95\\% CI]")
    caption = ("Calibration on the held-out test set, frozen models (baseline / "
               "lifestyle-augmented). Calibration-in-the-large is the intercept with "
               "the slope fixed at 1; calibration slope is estimated separately with a "
               "free intercept. Both are point estimates. Observed prevalence is "
               f"{num(obs)}. $\\Delta$ Brier is lifestyle minus baseline with a paired "
               "bootstrap 95\\% CI (lower Brier is better).")
    return table("table*", "tab:calibration", caption, "\\footnotesize",
                 "lccccc", header, rows, "4pt")


def net_benefit() -> str:
    dl = pd.read_csv(os.path.join(TABLES_DIR, "calibration_paired_deltas.csv"))
    nb = dl[dl.metric.str.startswith("net_benefit")].copy()
    nb["t"] = nb.metric.str.split("@").str[1]
    thresholds = sorted(nb.t.unique(), key=float)
    rows = []
    for m in MODELS:
        cells = []
        for t in thresholds:
            r = nb[(nb.model == m) & (nb.t == t)].iloc[0]
            cell = (f"{num(r.delta_lifestyle_minus_baseline, sign=True)} "
                    f"{ci(r.ci_lo, r.ci_hi)}")
            cells.append(f"\\textbf{{{cell}}}" if not r.ci_includes_zero else cell)
        rows.append(f"{MODEL_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    header = "\\bfseries Model & " + " & ".join(
        f"\\bfseries $p_t = {float(t):.1f}$" for t in thresholds)
    caption = ("Exploratory decision-curve analysis: paired lifestyle-minus-baseline "
               "differences in net benefit at four risk thresholds $p_t$, held-out "
               "test set, with paired bootstrap 95\\% CIs. These 20 comparisons are "
               "correlated and unadjusted; the single interval excluding zero is in "
               "bold. Thresholds are illustrative, not clinically validated decision "
               "points.")
    return table("table*", "tab:net_benefit", caption, "\\scriptsize",
                 "lcccc", header, rows, "3pt")


# ---------------------------------------------------------------------------
# UNTREATED SUBGROUP
# ---------------------------------------------------------------------------

def untreated() -> str:
    df = pd.read_csv(os.path.join(TABLES_DIR, "untreated_results.csv")).set_index("model")
    rows = []
    for m in MODELS:
        r = df.loc[m]
        delta = f"{num(r.delta, 4, True)} {ci(r.ci_lo, r.ci_hi, 4)}"
        if not r.ci_includes_zero:
            delta = f"\\textbf{{{delta}}}"
        rows.append(" & ".join([
            MODEL_LABELS[m],
            f"{num(r.auc_baseline)} {r.auc_baseline_ci}",
            f"{num(r.auc_lifestyle)} {r.auc_lifestyle_ci}",
            delta,
        ]) + " \\\\")
    r0 = df.iloc[0]
    n_dev = f"{int(r0.n_dev):,}".replace(",", "{,}")
    header = ("\\bfseries Model & \\bfseries Baseline AUC [95\\% CI] & "
              "\\bfseries Lifestyle AUC [95\\% CI] & "
              "\\bfseries $\\Delta$ [95\\% CI]")
    caption = ("Exploratory untreated-adult subgroup: participants definitely not "
               "taking antihypertensive medication, measured-BP Phenotype A outcome "
               f"(development $n = {n_dev}$, prevalence {100 * r0.prev_dev:.1f}\\%; test $n = {int(r0.n_test)}$, "
               f"prevalence {100 * r0.prev_test:.1f}\\%). Hyperparameters are the frozen "
               "primary-analysis values, not retuned. $\\Delta$ is lifestyle minus "
               "baseline with a paired bootstrap 95\\% CI; intervals excluding zero are "
               "in bold. The five comparisons are not adjusted for multiplicity.")
    return table("table*", "tab:untreated", caption, "\\footnotesize",
                 "lccc", header, rows, "4pt")


# ---------------------------------------------------------------------------
# LIFESTYLE--BASELINE REDUNDANCY
# ---------------------------------------------------------------------------

def redundancy() -> str:
    res = pd.read_csv(os.path.join(TABLES_DIR, "redundancy_residual_association.csv"))
    pred = pd.read_csv(os.path.join(TABLES_DIR, "redundancy_baseline_predicts_lifestyle.csv"))
    pred = pred.set_index("lifestyle_feature")
    res = res.reindex(res.rho_with_outcome.abs().sort_values(ascending=False).index)
    rows = []
    for _, r in res.iterrows():
        p = pred.loc[r.feature]
        metric = "$R^2$" if p.metric == "r2" else "AUC"
        rows.append(" & ".join([
            FEATURE_LABELS[r.feature],
            num(r.rho_with_outcome, 2, True),
            num(r.rho_with_residual, 2, True),
            f"{metric} {num(p.cv_mean, 2)} $\\pm$ {num(p.cv_sd, 2)}",
        ]) + " \\\\")
    header = ("\\bfseries Lifestyle variable & \\bfseries $\\rho$ with outcome & "
              "\\bfseries $\\rho$ with residual & \\bfseries Predicted from baseline")
    caption = ("Lifestyle--baseline redundancy diagnostics, development set only. "
               "``$\\rho$ with outcome'' is the Spearman correlation with Phenotype B; "
               "``$\\rho$ with residual'' is the correlation with the outcome minus the "
               "out-of-fold probability from a baseline-only model. ``Predicted from "
               "baseline'' is the 5-fold cross-validated performance (mean $\\pm$ SD) "
               "of predicting the lifestyle variable from all baseline predictors "
               "($R^2$ for continuous, ROC-AUC for binary indicators). Each statistic "
               "uses participants with non-missing values. These are descriptive and "
               "do not establish conditional independence or causation.")
    return table("table*", "tab:redundancy", caption, "\\footnotesize",
                 "lccc", header, rows, "4pt")


# ---------------------------------------------------------------------------
# FEATURE-SUBSET ABLATIONS
# ---------------------------------------------------------------------------

SUBSET_LABELS = [
    ("age_only", "Age only"),
    ("age_bmi", "Age + BMI"),
    ("demographic_ses_only", "Demographic/SES only"),
    ("anthropometric_only", "Anthropometric only"),
    ("baseline_full", "Full baseline"),
    ("lifestyle_only", "Lifestyle only"),
    ("baseline_plus_lifestyle", "Baseline + lifestyle"),
]


def ablation() -> str:
    df = pd.read_csv(os.path.join(TABLES_DIR, "ablation_results.csv"))
    rows = []
    for key, label in SUBSET_LABELS:
        sub = df[df.feature_subset == key].set_index("model")
        n = int(sub.n_predictors.iloc[0])
        aucs = " & ".join(num(sub.loc[m, "auc"]) for m in MODELS)
        lo = num(sub.delta_vs_baseline.min(), sign=True)
        hi = num(sub.delta_vs_baseline.max(), sign=True)
        rng = "---" if key == "baseline_full" else f"{lo} to {hi}"
        rows.append(f"{label} & {n} & {aucs} & {rng} \\\\")
    header = ("\\bfseries Predictors & \\bfseries $k$ & \\bfseries LR & \\bfseries RF & "
              "\\bfseries GB & \\bfseries BDT & \\bfseries XGB & "
              "\\bfseries $\\Delta$ vs.\\ full baseline")
    caption = ("Feature-subset ablations: held-out ROC-AUC on the original test set "
               "by predictor subset ($k$ = number of source predictors) and model. "
               "Subsets reuse the frozen hyperparameters of the corresponding arm "
               "(baseline arm for baseline subsets, lifestyle-augmented arm for "
               "subsets including lifestyle) and were not retuned, so they may be "
               "slightly disadvantaged. The last column gives the range across the "
               "five models of the paired difference from the full baseline.")
    return table("table*", "tab:ablation", caption, "\\footnotesize",
                 "lccccccc", header, rows, "4pt")


# ---------------------------------------------------------------------------
# TRAINING-SEED AND REPEATED-SPLIT VARIABILITY
# ---------------------------------------------------------------------------

def variability() -> str:
    parts = [("Training seeds, original split", "train_seed_summary.csv"),
             ("Repeated random 90/10 splits (exploratory)", "repeated_splits_summary.csv")]
    rows = []
    for i, (title, fname) in enumerate(parts):
        df = pd.read_csv(os.path.join(TABLES_DIR, fname)).set_index("model")
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\itshape {title}}} \\\\")
        for m in MODELS:
            r = df.loc[m]
            rows.append(" & ".join([
                f"\\quad {MODEL_LABELS[m]}",
                f"{num(r.mean_auc_baseline)} / {num(r.mean_auc_lifestyle)}",
                num(r.mean_delta, 4, True), num(r.sd_delta, 4),
                f"{num(r.min_delta, 4, True)} to {num(r.max_delta, 4, True)}",
                f"{int(r.n_positive)}/{int(r.n_seeds)}",
                f"{int(r.n_ci_excludes_zero)}/{int(r.n_seeds)}",
            ]) + " \\\\")
        if i == 0:
            rows.append("\\midrule")
    header = ("\\bfseries Model & \\bfseries Mean AUC (base / life) & "
              "\\bfseries Mean $\\Delta$ & \\bfseries SD & \\bfseries Range & "
              "\\bfseries $\\Delta > 0$ & \\bfseries CI excl.\\ 0")
    caption = ("Variability of the lifestyle-minus-baseline held-out ROC-AUC "
               "difference over ten seeds. Training seeds keep the original "
               "development/test split and vary only the model and SMOTE random "
               "seeds. Repeated splits redraw the stratified 90/10 partition with "
               "ten seeds (including the original); because hyperparameters were "
               "tuned on the original development set, part of each new test set "
               "contributed to tuning, so these results describe split sensitivity "
               "conditional on that tuning, not independent validation. Each "
               "comparison has its own paired bootstrap 95\\% CI; the last column "
               "counts intervals excluding zero.")
    return table("table*", "tab:variability", caption, "\\footnotesize",
                 "@{}lcccccc@{}", header, rows, "4pt")


# ---------------------------------------------------------------------------
# HYPERPARAMETER SEARCH LOG SUMMARY
# ---------------------------------------------------------------------------

UNTUNED_CV = os.path.join(ROOT, "outputs", "untuned_primary", "cv_results.csv")


def tuning_summary() -> str:
    tr = pd.read_csv(os.path.join(TABLES_DIR, "tuning_results.csv"))
    untuned = pd.read_csv(UNTUNED_CV) if os.path.exists(UNTUNED_CV) else None
    rows = []
    for arm, arm_label in [("baseline", "Baseline"),
                           ("lifestyle_augmented", "Lifestyle-augmented")]:
        for j, m in enumerate(MODELS):
            g = tr[(tr.feature_set == arm) & (tr.model == m)]
            g = g.sort_values("mean_test_roc_auc", ascending=False)
            best, second = g.mean_test_roc_auc.iloc[0], g.mean_test_roc_auc.iloc[1]
            within = int((g.mean_test_roc_auc >= best - 0.002).sum())
            default = "---"
            if untuned is not None:
                u = untuned[(untuned.feature_set == arm) & (untuned.model == m)]
                default = num(u.roc_auc_mean.iloc[0])
            rows.append(" & ".join([
                arm_label if j == 0 else "", MODEL_LABELS[m], str(len(g)),
                default, num(best), num(best - second, 4),
                f"{within}/{len(g)}", num(g.std_test_roc_auc.iloc[0], 3),
            ]) + " \\\\")
        if arm == "baseline":
            rows.append("\\midrule")
    header = ("\\bfseries Arm & \\bfseries Model & \\bfseries Cand. & "
              "\\bfseries Default & \\bfseries Best & \\bfseries Gap to 2nd & "
              "\\bfseries Within 0.002 & \\bfseries SD of best")
    caption = ("Hyperparameter search summary, 5-fold development-set CV ROC-AUC. "
               "``Default'' is library-default CV ROC-AUC before tuning; ``Best'' is "
               "the selected configuration; ``Gap to 2nd'' is its margin over the "
               "next-best candidate; ``Within 0.002'' counts candidates within 0.002 "
               "of the best; ``SD of best'' is the CV fold standard deviation of the "
               "selected configuration. Logistic Regression used an exhaustive "
               "five-value grid; tree models used 25 randomized candidates.")
    return table("table*", "tab:tuning_summary", caption, "\\footnotesize",
                 "@{}llcccccc@{}", header, rows, "4pt")


def main() -> None:
    write("shap_robustness", shap_robustness())
    write("shap_domain", shap_domain())
    write("calibration", calibration())
    write("net_benefit", net_benefit())
    write("untreated", untreated())
    write("redundancy", redundancy())
    write("ablation", ablation())
    write("variability", variability())
    write("tuning_summary", tuning_summary())


if __name__ == "__main__":
    main()
