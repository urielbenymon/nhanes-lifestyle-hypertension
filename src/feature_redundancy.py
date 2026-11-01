"""
feature_redundancy.py

PURPOSE
-------
Supplementary analysis: how much of the
lifestyle block is already captured by the demographic/SES + anthropometric
baseline? This supports (or fails to support) the Discussion's interpretation
that the baseline "already captures much of the discriminative signal these
lifestyle measures would otherwise contribute."

Two complementary views, both computed on the 90% DEVELOPMENT split only
(the held-out test set is not touched):

  1. Pairwise Spearman correlations between every lifestyle predictor and
     every baseline predictor (rank-based, so it handles skewed minutes/
     drinks and ordinal scales). Categorical predictors are expanded to
     one-hot indicators first, matching the post-encoding feature space.

  2. Multivariate redundancy: for each lifestyle predictor, how well can the
     FULL baseline set predict it? 5-fold CV R^2 (numeric) or ROC-AUC
     (binary indicators) from a HistGradientBoosting model. Pairwise
     correlations understate redundancy when a lifestyle feature is
     explained by a combination of baseline variables (e.g. age + sex).

  3. Outcome association context: Spearman correlation of each predictor
     with the primary outcome, so the lifestyle-outcome signal can be
     compared with the baseline-outcome signal.

  4. Residual association: correlation of each lifestyle predictor with the
     part of the outcome the baseline does NOT explain (y minus out-of-fold
     baseline predicted probability). A large raw outcome correlation that
     shrinks toward zero here means the lifestyle signal is mostly shared
     with (e.g. confounded by) the baseline.

USAGE
-----
    python src/feature_redundancy.py

OUTPUT
------
    outputs/tables/redundancy_pairwise_spearman.csv
    outputs/tables/redundancy_baseline_predicts_lifestyle.csv
    outputs/tables/redundancy_outcome_association.csv
    outputs/tables/redundancy_residual_association.csv
    outputs/figures/fig_redundancy_heatmap.pdf
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold

from preprocessing import (
    load_and_split,
    MODELING_BASELINE_FEATURES,
    MODELING_LIFESTYLE_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
)

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

READABLE = {
    "RIDAGEYR": "Age", "is_female": "Female sex", "INDFMPIR": "Income-to-poverty",
    "BMXWT": "Weight", "BMXHT": "Height", "BMXBMI": "BMI", "BMXWAIST": "Waist",
    "total_vigorous_min_wk": "Vigorous min/wk", "total_moderate_min_wk": "Moderate min/wk",
    "sedentary_min_day": "Sedentary min/day", "alq130_avg_drinks_filled": "Drinks/day",
    "smoked_recently": "Smoked recently", "SLD012": "Sleep hrs (weekday)",
    "SLD013": "Sleep hrs (weekend)", "trouble_sleeping": "Trouble sleeping",
    "SLQ120": "Daytime sleepiness",
}


def _encode(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One-hot encode categoricals (NaN kept as NaN across indicators)."""
    out = []
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            d = pd.get_dummies(df[c], prefix=c, dtype=float)
            d[df[c].isna()] = np.nan
            out.append(d)
        else:
            out.append(df[[c]].astype(float))
    return pd.concat(out, axis=1)


def _label(col: str) -> str:
    for raw, nice in READABLE.items():
        if col == raw:
            return nice
    return col


def pairwise_spearman(base: pd.DataFrame, life: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lc in life.columns:
        for bc in base.columns:
            m = life[lc].notna() & base[bc].notna()
            if m.sum() < 30 or life.loc[m, lc].nunique() < 2 or base.loc[m, bc].nunique() < 2:
                continue
            rho, p = spearmanr(life.loc[m, lc], base.loc[m, bc])
            rows.append({"lifestyle_feature": lc, "baseline_feature": bc,
                         "n": int(m.sum()), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def baseline_predicts_lifestyle(base: pd.DataFrame, life: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lc in life.columns:
        y = life[lc]
        m = y.notna()
        X, yv = base[m].values, y[m].values
        is_binary = set(np.unique(yv)) <= {0.0, 1.0}
        if is_binary:
            if min((yv == 1).sum(), (yv == 0).sum()) < 25:
                continue
            model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
            cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X, yv.astype(int), cv=cv, scoring="roc_auc")
            metric = "roc_auc"
        else:
            model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
            cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X, yv, cv=cv, scoring="r2")
            metric = "r2"
        rows.append({"lifestyle_feature": lc, "metric": metric, "n": int(m.sum()),
                     "cv_mean": scores.mean(), "cv_sd": scores.std()})
        print(f"  baseline -> {lc:32s} {metric}={scores.mean():.3f} (sd {scores.std():.3f})")
    return pd.DataFrame(rows)


def outcome_association(base: pd.DataFrame, life: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for block, df in [("baseline", base), ("lifestyle", life)]:
        for c in df.columns:
            m = df[c].notna()
            if df.loc[m, c].nunique() < 2:
                continue
            rho, p = spearmanr(df.loc[m, c], y[m])
            rows.append({"block": block, "feature": c, "spearman_rho_with_outcome": rho,
                         "abs_rho": abs(rho), "p_value": p})
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def residual_association(base: pd.DataFrame, life: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    p_oof = cross_val_predict(model, base.values, y.values, cv=cv, method="predict_proba")[:, 1]
    resid = pd.Series(y.values - p_oof, index=y.index)
    rows = []
    for c in life.columns:
        m = life[c].notna()
        if life.loc[m, c].nunique() < 2:
            continue
        raw, _ = spearmanr(life.loc[m, c], y[m])
        res, p = spearmanr(life.loc[m, c], resid[m])
        rows.append({"feature": c, "rho_with_outcome": raw, "rho_with_residual": res,
                     "residual_p_value": p,
                     "share_remaining": abs(res) / abs(raw) if raw != 0 else np.nan})
    return pd.DataFrame(rows).sort_values("rho_with_outcome", key=abs, ascending=False)


def heatmap(pair_df: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = pair_df.pivot(index="lifestyle_feature", columns="baseline_feature", values="spearman_rho")
    keep_cols = [c for c in piv.columns if not c.startswith(("RIDRETH3_", "DMDEDUC2_"))]
    piv = piv[keep_cols]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [_label(c) for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)), [_label(c) for c in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.3 else "black")
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("Lifestyle vs. baseline predictors (development set)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    train_df, _ = load_and_split()   # development split only; test set unused
    base = _encode(train_df, MODELING_BASELINE_FEATURES)
    life = _encode(train_df, MODELING_LIFESTYLE_FEATURES)
    y = train_df[TARGET_COLUMN].astype(int)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("--- Pairwise Spearman (lifestyle x baseline) ---")
    pair_df = pairwise_spearman(base, life)
    pair_df.to_csv(os.path.join(TABLES_DIR, "redundancy_pairwise_spearman.csv"), index=False)
    top = (pair_df.assign(abs_rho=pair_df.spearman_rho.abs())
           .sort_values("abs_rho", ascending=False)
           .groupby("lifestyle_feature").head(1))
    print(top[["lifestyle_feature", "baseline_feature", "spearman_rho"]].to_string(index=False))

    print("\n--- Baseline set predicting each lifestyle feature (5-fold CV) ---")
    pred_df = baseline_predicts_lifestyle(base, life)
    pred_df.to_csv(os.path.join(TABLES_DIR, "redundancy_baseline_predicts_lifestyle.csv"), index=False)

    print("\n--- Outcome association ---")
    assoc_df = outcome_association(base, life, y)
    assoc_df.to_csv(os.path.join(TABLES_DIR, "redundancy_outcome_association.csv"), index=False)
    print(assoc_df.head(15).to_string(index=False))

    print("\n--- Lifestyle association with outcome residual after baseline ---")
    resid_df = residual_association(base, life, y)
    resid_df.to_csv(os.path.join(TABLES_DIR, "redundancy_residual_association.csv"), index=False)
    print(resid_df.to_string(index=False))

    heatmap(pair_df, os.path.join(FIGURES_DIR, "fig_redundancy_heatmap.pdf"))
    print("\nDone.")


if __name__ == "__main__":
    main()
