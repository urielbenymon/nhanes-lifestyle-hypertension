"""
smote_sensitivity.py

PURPOSE
-------
A controlled sensitivity analysis answering ONE question: given phenotype B's
near-balanced class distribution (53.7%/46.3%), does removing SMOTE from the
pipeline materially change results?

This is deliberately a CLEAN intervention:
    same participants, split, features, preprocessing, model hyperparameters,
    random seeds, and evaluation procedure -- only SMOTE present vs. absent.

WHAT THIS SCRIPT DOES NOT DO (all deliberate)
---------------------------------------------
- Does NOT re-tune. It loads the 10 ALREADY-FROZEN hyperparameter
  configurations from best_hyperparameters.csv (5 models x 2 arms) and uses
  them unchanged for both the SMOTE and no-SMOTE conditions. Re-tuning the
  no-SMOTE variant would change two things at once (resampling AND model
  configuration), making the comparison impossible to interpret cleanly.
- Does NOT run phenotype A. Phenotype B only --
  doubling this into 10 more phenotype-A runs isn't justified unless this
  primary sensitivity turns up something surprising.
- Does NOT recompute SHAP (see shap_refit_controls.py --part nosmote for that). This question is about predictive discrimination,
  not interpretability -- SHAP is expensive (the BaggingClassifier fallback
  alone took ~15-20 min per arm in earlier runs) and answers a different
  question than the one this script exists to answer.
- Does NOT use the held-out test set to pick a "better" pipeline. The
  primary analysis (WITH SMOTE) was already prespecified and run. This
  script tests robustness of those conclusions to SMOTE's removal -- it
  does not retroactively select whichever variant scores higher on test.
  Both conditions' test results are reported side by side, plainly, with
  no cherry-picking.

WHAT "MATERIALLY DIFFERENT" WOULD MEAN
------------------------------------------
- Differences ~negligible -> the cleanest result: conclusions are insensitive
  to synthetic oversampling despite the near-balanced primary outcome.
- No-SMOTE moderately better -> a real methodological finding, and grounds
  to reconsider SMOTE's place in the primary analysis / report as a
  limitation.
- SMOTE moderately better -> empirical justification for keeping it despite
  modest imbalance.
- Differences vary by model -> SMOTE's effect is model-dependent, not
  uniformly beneficial or harmful -- also a legitimate, reportable finding.
None of these outcomes is "the right answer" to hope for -- report whichever
one actually happens.

USAGE
-----
    python src/smote_sensitivity.py

OUTPUT
------
    outputs/tables/smote_sensitivity_cv_results.csv       -- CV mean/std,
        SMOTE and no-SMOTE, both arms, all 5 models (20 rows)
    outputs/tables/smote_sensitivity_test_metrics.csv     -- held-out test
        point estimate + 95% bootstrap CI, both conditions (20 rows)
    outputs/tables/smote_sensitivity_paired_comparison.csv -- paired
        bootstrap delta (SMOTE minus no-SMOTE) on the held-out test set,
        per model per arm (10 rows) -- this is the actual answer to the
        sensitivity question

"""

import os
import ast
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocessing import (
    load_and_split,
    get_feature_matrix,
    build_column_transformer,
    MODELING_BASELINE_FEATURES,
    MODELING_LIFESTYLE_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
)
from evaluate import bootstrap_metric_ci, paired_bootstrap_delta

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")
N_FOLDS = 5

FEATURE_SET_ARMS = {
    "baseline": MODELING_BASELINE_FEATURES,
    "lifestyle_augmented": MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES,
}
MODEL_NAMES = [
    "logistic_regression", "random_forest", "gradient_boosting",
    "bagged_decision_tree", "xgboost",
]


def load_frozen_hyperparameters() -> dict:
    """
    ATTENTION: Intentionally a self-contained copy of train_pipeline.py's
    function of the same name, NOT an import from it -- train_pipeline.py
    imports xgboost/imblearn at module level, and duplicating this ~10-line
    function here keeps this script's own import chain minimal and lets it
    fail fast with a clear file-not-found message if best_hyperparameters.csv
    is missing, rather than crashing on an unrelated import first.
    """
    path = os.path.join(TABLES_DIR, "best_hyperparameters.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. This sensitivity analysis requires the frozen "
            f"tuned hyperparameters from the primary --tune run -- run "
            f"`python src/train_pipeline.py --tune` first."
        )
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        arm, model = row["feature_set"], row["model"]
        result.setdefault(arm, {})[model] = ast.literal_eval(row["best_parameters"])
    return result


def build_tuned_model(model_name: str, params: dict):
    """
    Construct one classifier with its frozen tuned hyperparameters applied.
    Mirrors train_pipeline.py's build_models() + set_params() logic exactly,
    for a single model rather than all 5 at once.
    """
    defaults = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "bagged_decision_tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
    }
    model = defaults[model_name]
    if params:
        model.set_params(**params)
    return model


def build_pipeline_variant(feature_list: list, model_name: str, model, use_smote: bool) -> ImbPipeline:
    """
    Same structure as train_pipeline.py's build_pipeline(), with SMOTE
    included or omitted based on use_smote. Everything else -- encoding,
    imputation, scaling-for-LR-only -- is IDENTICAL between the two
    conditions, so SMOTE's presence/absence is the only thing that varies.
    """
    column_transformer = build_column_transformer(feature_list)
    steps = [
        ("preprocess", column_transformer),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if model_name == "logistic_regression":
        steps.append(("scale", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("classifier", model))
    return ImbPipeline(steps)  # ImbPipeline works fine with zero samplers too


# ---------------------------------------------------------------------------
# CV FOR ONE (arm, model, condition)
# ---------------------------------------------------------------------------

def run_cv(train_df: pd.DataFrame, arm_name: str, feature_list: list,
           model_name: str, frozen_params: dict, use_smote: bool) -> dict:
    X = get_feature_matrix(train_df, feature_list)
    y = train_df[TARGET_COLUMN].astype(int)

    model = build_tuned_model(model_name, frozen_params)
    pipeline = build_pipeline_variant(feature_list, model_name, model, use_smote)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["roc_auc", "average_precision"],
        n_jobs=1,
    )
    elapsed = time.time() - t0

    return {
        "feature_set": arm_name, "model": model_name,
        "smote": use_smote,
        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
        "pr_auc_mean": scores["test_average_precision"].mean(),
        "pr_auc_std": scores["test_average_precision"].std(),
        "fit_time_sec": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# FIT ON FULL TRAIN, SCORE ON HELD-OUT TEST
# ---------------------------------------------------------------------------

def fit_and_score_test(train_df: pd.DataFrame, test_df: pd.DataFrame, arm_name: str,
                        feature_list: list, model_name: str, frozen_params: dict,
                        use_smote: bool) -> tuple:
    """
    Returns (metrics_row, y_true, proba) -- proba is kept so the paired
    bootstrap comparison can be computed afterward from these exact
    predictions, without needing to refit anything a second time.
    """
    X_train = get_feature_matrix(train_df, feature_list)
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = get_feature_matrix(test_df, feature_list)
    y_test = test_df[TARGET_COLUMN].astype(int).values

    model = build_tuned_model(model_name, frozen_params)
    pipeline = build_pipeline_variant(feature_list, model_name, model, use_smote)
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    auc_point, auc_lo, auc_hi = bootstrap_metric_ci(y_test, proba, roc_auc_score)
    pr_point, pr_lo, pr_hi = bootstrap_metric_ci(y_test, proba, average_precision_score)

    row = {
        "feature_set": arm_name, "model": model_name, "smote": use_smote,
        "roc_auc": auc_point, "roc_auc_ci_lo": auc_lo, "roc_auc_ci_hi": auc_hi,
        "pr_auc": pr_point, "pr_auc_ci_lo": pr_lo, "pr_auc_ci_hi": pr_hi,
    }
    return row, y_test, proba


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("--- Loading frozen tuned hyperparameters (primary phenotype B) ---")
    frozen = load_frozen_hyperparameters()

    print("--- Loading train/test split (same fixed split as the primary pipeline) ---")
    train_df, test_df = load_and_split()

    cv_rows = []
    test_rows = []
    proba_cache = {}  # (arm, model, smote_bool) -> (y_true, proba), for the paired comparison

    print("\n--- Cross-validation, SMOTE vs. no-SMOTE, all 10 frozen configurations ---")
    for arm_name, feature_list in FEATURE_SET_ARMS.items():
        for model_name in MODEL_NAMES:
            params = frozen[arm_name][model_name]
            for use_smote in [True, False]:
                row = run_cv(train_df, arm_name, feature_list, model_name, params, use_smote)
                cv_rows.append(row)
                label = "SMOTE   " if use_smote else "no-SMOTE"
                print(f"  [{arm_name}] {model_name:22s} {label}: "
                      f"ROC-AUC={row['roc_auc_mean']:.4f}+/-{row['roc_auc_std']:.4f} "
                      f"({row['fit_time_sec']}s)")

    cv_df = pd.DataFrame(cv_rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    cv_path = os.path.join(TABLES_DIR, "smote_sensitivity_cv_results.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"\n[wrote] {cv_path}")

    print("\n--- Held-out test: SMOTE vs. no-SMOTE, all 10 frozen configurations ---")
    for arm_name, feature_list in FEATURE_SET_ARMS.items():
        for model_name in MODEL_NAMES:
            params = frozen[arm_name][model_name]
            for use_smote in [True, False]:
                row, y_true, proba = fit_and_score_test(
                    train_df, test_df, arm_name, feature_list, model_name, params, use_smote
                )
                test_rows.append(row)
                proba_cache[(arm_name, model_name, use_smote)] = (y_true, proba)
                label = "SMOTE   " if use_smote else "no-SMOTE"
                print(f"  [{arm_name}] {model_name:22s} {label}: "
                      f"ROC-AUC={row['roc_auc']:.3f} [{row['roc_auc_ci_lo']:.3f}, {row['roc_auc_ci_hi']:.3f}]")

    test_df_out = pd.DataFrame(test_rows)
    test_path = os.path.join(TABLES_DIR, "smote_sensitivity_test_metrics.csv")
    test_df_out.to_csv(test_path, index=False)
    print(f"\n[wrote] {test_path}")

    print("\n--- Paired bootstrap: SMOTE minus no-SMOTE, on the held-out test set ---")
    paired_rows = []
    for arm_name in FEATURE_SET_ARMS:
        for model_name in MODEL_NAMES:
            y_true_smote, proba_smote = proba_cache[(arm_name, model_name, True)]
            y_true_nosmote, proba_nosmote = proba_cache[(arm_name, model_name, False)]
            # ATTENTION: both conditions were fit/scored on the identical
            # test set, so y_true is the same array either way -- using
            # y_true_smote for both arguments below is correct, not a bug.
            assert (y_true_smote == y_true_nosmote).all()

            auc_delta, auc_lo, auc_hi = paired_bootstrap_delta(
                y_true_smote, proba_nosmote, proba_smote, roc_auc_score
            )
            pr_delta, pr_lo, pr_hi = paired_bootstrap_delta(
                y_true_smote, proba_nosmote, proba_smote, average_precision_score
            )
            significant = not (auc_lo <= 0 <= auc_hi)
            paired_rows.append({
                "feature_set": arm_name, "model": model_name,
                "roc_auc_delta_smote_minus_nosmote": auc_delta,
                "roc_auc_delta_ci_lo": auc_lo, "roc_auc_delta_ci_hi": auc_hi,
                "pr_auc_delta_smote_minus_nosmote": pr_delta,
                "pr_auc_delta_ci_lo": pr_lo, "pr_auc_delta_ci_hi": pr_hi,
                "roc_auc_significant": significant,
            })
            sig_label = "SIGNIFICANT" if significant else "not significant"
            print(f"  [{arm_name}] {model_name:22s} delta(SMOTE-noSMOTE)={auc_delta:+.4f} "
                  f"[{auc_lo:+.4f}, {auc_hi:+.4f}] -- {sig_label}")

    paired_df = pd.DataFrame(paired_rows)
    paired_path = os.path.join(TABLES_DIR, "smote_sensitivity_paired_comparison.csv")
    paired_df.to_csv(paired_path, index=False)
    print(f"\n[wrote] {paired_path}")

    print("\n--- Summary ---")
    n_sig = paired_df["roc_auc_significant"].sum()
    print(f"{n_sig} of {len(paired_df)} (arm, model) combinations show a significant "
          f"SMOTE effect on held-out ROC-AUC.")
    if n_sig == 0:
        print("Interpretation: conclusions appear insensitive to SMOTE's presence/absence "
              "despite the near-balanced primary phenotype -- report this plainly, it is "
              "the cleanest possible outcome for this sensitivity check.")
    else:
        print("Interpretation: report which specific (arm, model) combinations were "
              "affected and in which direction -- do not generalize to 'SMOTE matters' "
              "or 'SMOTE doesn't matter' if the effect is model-dependent.")

    print("\nDone.")


if __name__ == "__main__":
    main()
