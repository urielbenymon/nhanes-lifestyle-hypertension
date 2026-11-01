"""
calibration_analysis.py

PURPOSE
-------
Supplementary analysis: equal ROC-AUC does
not imply equal calibration or equal clinical usefulness. This script
evaluates the SAME 10 saved pipelines evaluate.py uses, on the SAME held-out
test set, along three additional axes:

  1. Calibration: Brier score, calibration-in-the-large (intercept with the
     model's logit as an offset; 0 = perfect), and calibration slope (1 =
     perfect; <1 = predictions too extreme). Reliability diagram per model.
  2. Clinical utility: decision-curve analysis (net benefit across risk
     thresholds; Vickers & Elkin 2006), with treat-all / treat-none refs.
  3. Paired lifestyle-minus-baseline deltas with paired bootstrap 95% CIs
     (reusing evaluate.py's paired_bootstrap_delta) for Brier score and for
     net benefit at a few fixed thresholds.

No refitting, no hyperparameter changes: frozen models, test set only, the
same way evaluate.py computes ROC-AUC.

NOTE ON SMOTE: the pipelines oversample during training, which shifts
predicted probabilities toward 50/50. Some miscalibration in BOTH arms is
therefore expected; the question of interest is whether the arms DIFFER.

USAGE
-----
    python src/calibration_analysis.py
    python src/calibration_analysis.py --target-column hypertension_a --run-label phenotype_a

OUTPUT
------
    outputs/tables/calibration_metrics.csv
    outputs/tables/calibration_paired_deltas.csv
    outputs/tables/decision_curve.csv
    outputs/figures/fig_calibration.pdf
    outputs/figures/fig_decision_curve.pdf
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

from preprocessing import load_and_split, get_feature_matrix, TARGET_COLUMN
from evaluate import (
    load_all_models, paired_bootstrap_delta, _table_path,
    FEATURE_SET_ARMS, MODEL_NAMES,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")

DCA_THRESHOLDS = np.round(np.arange(0.05, 0.81, 0.01), 2)
NB_REPORT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
N_CAL_BINS = 10
EPS = 1e-6

PRETTY = {
    "logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting", "bagged_decision_tree": "Bagged Decision Tree",
    "xgboost": "XGBoost",
}


# ---------------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------------

def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope(y, p) -> float:
    lr = LogisticRegression(C=np.inf, max_iter=1000)  # unpenalized
    lr.fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0])


def calibration_in_the_large(y, p, n_iter: int = 50) -> float:
    """Intercept a in logit(P(y=1)) = a + logit(p), slope fixed at 1 (Newton)."""
    off = _logit(p)
    a = 0.0
    for _ in range(n_iter):
        mu = 1 / (1 + np.exp(-(a + off)))
        grad = np.sum(y - mu)
        hess = np.sum(mu * (1 - mu))
        step = grad / hess
        a += step
        if abs(step) < 1e-10:
            break
    return float(a)


def reliability_curve(y, p, n_bins: int = N_CAL_BINS):
    """Quantile bins: mean predicted vs observed rate per bin."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    pred, obs = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            pred.append(p[m].mean())
            obs.append(y[m].mean())
    return np.array(pred), np.array(obs)


# ---------------------------------------------------------------------------
# DECISION CURVE
# ---------------------------------------------------------------------------

def net_benefit(y, p, t: float) -> float:
    n = len(y)
    pos = p >= t
    tp = np.sum(pos & (y == 1))
    fp = np.sum(pos & (y == 0))
    return tp / n - fp / n * (t / (1 - t))


def treat_all_nb(y, t: float) -> float:
    prev = y.mean()
    return prev - (1 - prev) * (t / (1 - t))


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def get_test_predictions(models: dict, test_df: pd.DataFrame) -> dict:
    preds = {}
    for (arm, model_name), pipe in models.items():
        X = get_feature_matrix(test_df, FEATURE_SET_ARMS[arm])
        preds[(arm, model_name)] = pipe.predict_proba(X)[:, 1]
    return preds


def calibration_table(y, preds: dict) -> pd.DataFrame:
    rows = []
    for (arm, model_name), p in preds.items():
        rows.append({
            "feature_set": arm, "model": model_name,
            "brier": brier_score_loss(y, p),
            "cal_in_the_large": calibration_in_the_large(y, p),
            "cal_slope": calibration_slope(y, p),
            "mean_pred": p.mean(), "observed_rate": y.mean(),
        })
    return pd.DataFrame(rows)


def paired_deltas(y, preds: dict) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_NAMES:
        pa = preds[("baseline", model_name)]
        pb = preds[("lifestyle_augmented", model_name)]
        d, lo, hi = paired_bootstrap_delta(y, pa, pb, brier_score_loss)
        rows.append({"model": model_name, "metric": "brier (lower=better)",
                     "delta_lifestyle_minus_baseline": d, "ci_lo": lo, "ci_hi": hi})
        for t in NB_REPORT_THRESHOLDS:
            fn = lambda yy, pp, t=t: net_benefit(yy, pp, t)
            d, lo, hi = paired_bootstrap_delta(y, pa, pb, fn)
            rows.append({"model": model_name, "metric": f"net_benefit@{t:.2f}",
                         "delta_lifestyle_minus_baseline": d, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out["ci_includes_zero"] = (out.ci_lo <= 0) & (out.ci_hi >= 0)
    return out


def decision_curve_table(y, preds: dict) -> pd.DataFrame:
    rows = []
    for t in DCA_THRESHOLDS:
        rows.append({"threshold": t, "feature_set": "reference", "model": "treat_all",
                     "net_benefit": treat_all_nb(y, t)})
        rows.append({"threshold": t, "feature_set": "reference", "model": "treat_none",
                     "net_benefit": 0.0})
        for (arm, model_name), p in preds.items():
            rows.append({"threshold": t, "feature_set": arm, "model": model_name,
                         "net_benefit": net_benefit(y, p, t)})
    return pd.DataFrame(rows)


def plot_calibration(y, preds: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(3.0 * len(MODEL_NAMES), 3.2), sharey=True)
    for ax, model_name in zip(axes, MODEL_NAMES):
        ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--")
        for arm, color, marker in [("baseline", "#4C72B0", "o"), ("lifestyle_augmented", "#DD8452", "s")]:
            px, oy = reliability_curve(y, preds[(arm, model_name)])
            ax.plot(px, oy, marker=marker, ms=3.5, lw=1.2, color=color,
                    label="Baseline" if arm == "baseline" else "Lifestyle-augmented")
        ax.set_title(PRETTY[model_name], fontsize=9)
        ax.set_xlabel("Mean predicted risk", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Observed rate", fontsize=8)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_decision_curve(dca: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(3.0 * len(MODEL_NAMES), 3.2), sharey=True)
    ta = dca[dca.model == "treat_all"]
    for ax, model_name in zip(axes, MODEL_NAMES):
        ax.plot(ta.threshold, ta.net_benefit, color="0.6", lw=1, label="Treat all")
        ax.axhline(0, color="0.3", lw=0.8, label="Treat none")
        for arm, color in [("baseline", "#4C72B0"), ("lifestyle_augmented", "#DD8452")]:
            d = dca[(dca.model == model_name) & (dca.feature_set == arm)]
            ax.plot(d.threshold, d.net_benefit, color=color, lw=1.3,
                    label="Baseline" if arm == "baseline" else "Lifestyle-augmented")
        ax.set_title(PRETTY[model_name], fontsize=9)
        ax.set_xlabel("Risk threshold", fontsize=8)
        ax.set_ylim(-0.05, max(0.6, ta.net_benefit.max() + 0.02))
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Net benefit", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(models: dict, test_df: pd.DataFrame, target_column: str, run_label: str):
    y = test_df[target_column].astype(int).values
    preds = get_test_predictions(models, test_df)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cal = calibration_table(y, preds)
    cal.to_csv(_table_path(run_label, "calibration_metrics.csv"), index=False)
    print("--- Calibration (held-out test) ---")
    print(cal.round(4).to_string(index=False))

    deltas = paired_deltas(y, preds)
    deltas.to_csv(_table_path(run_label, "calibration_paired_deltas.csv"), index=False)
    print("\n--- Paired deltas, lifestyle-augmented minus baseline (95% CI) ---")
    print(deltas.round(4).to_string(index=False))

    dca = decision_curve_table(y, preds)
    dca.to_csv(_table_path(run_label, "decision_curve.csv"), index=False)

    suffix = "" if run_label == "primary" else f"_{run_label}"
    plot_calibration(y, preds, os.path.join(FIGURES_DIR, f"fig_calibration{suffix}.pdf"))
    plot_decision_curve(dca, os.path.join(FIGURES_DIR, f"fig_decision_curve{suffix}.pdf"))
    print("\n[wrote] calibration tables + figures")
    return cal, deltas, dca


def main():
    parser = argparse.ArgumentParser(description="Calibration + decision-curve analysis.")
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--run-label", default="primary")
    args = parser.parse_args()

    _, test_df = load_and_split()
    models = load_all_models(run_label=args.run_label)
    run(models, test_df, args.target_column, args.run_label)
    print("\nDone.")


if __name__ == "__main__":
    main()
