"""
evaluate.py

PURPOSE
-------
Final held-out test-set evaluation of the 10 pipelines train_pipeline.py
already fit and saved. Three things happen here, and only here -- this is
the first script that touches the test set at all:

  1. Point estimates + bootstrap confidence intervals for ROC-AUC and PR-AUC,
     per model per feature-set arm, on the held-out test set.
  2. PAIRED bootstrap comparison of baseline vs. lifestyle-augmented, per
     model, on the SAME test set -- this is the actual statistical test
     behind the paper's central claim (does lifestyle information help),
     not just eyeballing whether one mean is bigger than another.
  3. SHAP values per model, and a cross-model interpretability-stability
     comparison (do RF/GB/XGBoost/Bagged/LR agree on which features matter
     most?) -- this is the Area 2 interpretability contribution.

WHY PAIRED BOOTSTRAP, NOT TWO SEPARATE CIs
---------------------------------------------
If you bootstrap baseline ROC-AUC and lifestyle-augmented ROC-AUC
separately and just eyeball whether the CIs overlap, you're throwing away
the fact that both models are evaluated on the SAME test participants --
their errors are correlated. The correct comparison resamples the SAME
test-set indices for both models in each bootstrap iteration and looks at
the distribution of (lifestyle_score - baseline_score) directly. That
delta's own confidence interval is what tells you whether the improvement
is real, not an artifact of resampling noise being counted twice.

WHAT THIS FILE DOES NOT DO
-----------------------------
No refitting. Every prediction here comes from loading train_pipeline.py's
saved outputs/models/*.joblib pipelines and calling .predict_proba() on the
untouched test set -- if you find yourself wanting to change how a model is
trained, that change belongs in train_pipeline.py, not here.

USAGE
-----
    python src/evaluate.py

OUTPUT
------
    outputs/tables/test_metrics.csv        -- point estimate + 95% CI per model/arm
    outputs/tables/paired_comparison.csv   -- lifestyle-vs-baseline delta + CI per model
    outputs/tables/shap_importance.csv     -- mean |SHAP value| per feature per model/arm
    outputs/tables/shap_stability.csv      -- pairwise cross-model top-feature agreement
    outputs/shap_values/{arm}__{model}.npy -- raw SHAP arrays, for make_figures.py

NOTE ON SHAP OUTPUT SHAPES
--------------------------
shap.TreeExplainer returns different array shapes depending on model type
(see _normalize_shap_output); all outputs are normalized to a single
(n_samples, n_features) positive-class array before aggregation.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocessing import (
    load_and_split,
    get_feature_matrix,
    MODELING_BASELINE_FEATURES,
    MODELING_LIFESTYLE_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
)

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
SHAP_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "shap_values")

N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
SHAP_TEST_SAMPLE_SIZE = 300   # ATTENTION: SHAP on the full test set can be slow
                               # for non-tree models (KernelExplainer fallback);
                               # subsampling is a deliberate, disclosed tradeoff,
                               # not silent. Report actual runtime if this feels
                               # too slow/fast, adjust it here.
SHAP_BACKGROUND_SIZE = 100    # background sample for LinearExplainer/fallback

FEATURE_SET_ARMS = {
    "baseline": MODELING_BASELINE_FEATURES,
    "lifestyle_augmented": MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES,
}
MODEL_NAMES = [
    "logistic_regression", "random_forest", "gradient_boosting",
    "bagged_decision_tree", "xgboost",
]


# ---------------------------------------------------------------------------
# RUN-LABEL-AWARE PATH HELPERS -- mirrors train_pipeline.py's naming scheme
# exactly, so evaluate.py can find the right saved models for a given run.
# ---------------------------------------------------------------------------

def _model_path(run_label: str, arm_name: str, model_name: str) -> str:
    if run_label == "primary":
        return os.path.join(MODELS_DIR, f"{arm_name}__{model_name}.joblib")
    return os.path.join(MODELS_DIR, f"{run_label}__{arm_name}__{model_name}.joblib")


def _table_path(run_label: str, base_name: str) -> str:
    if run_label == "primary":
        return os.path.join(TABLES_DIR, base_name)
    stem, ext = os.path.splitext(base_name)
    return os.path.join(TABLES_DIR, f"{stem}_{run_label}{ext}")


def _shap_path(run_label: str, arm_name: str, model_name: str) -> str:
    if run_label == "primary":
        return os.path.join(SHAP_DIR, f"{arm_name}__{model_name}.npy")
    return os.path.join(SHAP_DIR, f"{run_label}__{arm_name}__{model_name}.npy")


# ---------------------------------------------------------------------------
# LOAD SAVED MODELS
# ---------------------------------------------------------------------------

def load_all_models(run_label: str = "primary") -> dict:
    """
    Load all 10 fitted pipelines saved by train_pipeline.py for a given run.

    ATTENTION: Returns {(arm_name, model_name): pipeline}. Loading (not
    refitting) is the whole point of this file's design -- the pipeline
    loaded here is bit-for-bit what train_pipeline.py cross-validated.
    run_label must match whatever --run-label train_pipeline.py was called
    with, or this will raise FileNotFoundError pointing at the expected path
    -- that's deliberate, a mismatched run_label should fail loud, not
    silently load the wrong run's models.
    """
    models = {}
    for arm_name in FEATURE_SET_ARMS:
        for model_name in MODEL_NAMES:
            path = _model_path(run_label, arm_name, model_name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Run train_pipeline.py first "
                    f"(with --run-label {run_label} if not the primary run)."
                )
            models[(arm_name, model_name)] = joblib.load(path)
    return models



# ---------------------------------------------------------------------------
# BOOTSTRAP STATISTICS
# ---------------------------------------------------------------------------

def bootstrap_metric_ci(
    y_true: np.ndarray, y_proba: np.ndarray, metric_fn,
    n_boot: int = N_BOOTSTRAP, ci: float = CI_LEVEL, random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Point estimate + percentile bootstrap CI for one metric on one model's
    predictions.

    ATTENTION: Resamples ROW INDICES with replacement (not the metric
    itself) -- the standard nonparametric bootstrap for AUC-type metrics,
    which don't have a simple closed-form variance formula.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point = metric_fn(y_true, y_proba)

    boot_scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        # ATTENTION: a bootstrap resample can occasionally be single-class
        # (all positives or all negatives) by chance, especially for small
        # test sets -- roc_auc_score/average_precision_score raise on that.
        # Skip and retry rather than crash the whole loop.
        if len(np.unique(y_true[idx])) < 2:
            boot_scores[b] = np.nan
            continue
        boot_scores[b] = metric_fn(y_true[idx], y_proba[idx])

    boot_scores = boot_scores[~np.isnan(boot_scores)]
    alpha = 1 - ci
    lo = np.percentile(boot_scores, 100 * alpha / 2)
    hi = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return point, lo, hi


def paired_bootstrap_delta(
    y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, metric_fn,
    n_boot: int = N_BOOTSTRAP, ci: float = CI_LEVEL, random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Paired bootstrap CI for (metric(b) - metric(a)), e.g. lifestyle-augmented
    minus baseline, on the SAME resampled test indices each iteration.

    ATTENTION: This is the statistic that actually answers "is the lifestyle
    improvement real." If the returned CI excludes 0, the improvement is
    significant at the `ci` level for that model. If it includes 0, the
    honest conclusion is "not distinguishable from no improvement for this
    model on this test set" -- report that plainly if it happens, don't
    round up to "trending toward significance" language.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point = metric_fn(y_true, proba_b) - metric_fn(y_true, proba_a)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            deltas[i] = np.nan
            continue
        deltas[i] = metric_fn(y_true[idx], proba_b[idx]) - metric_fn(y_true[idx], proba_a[idx])

    deltas = deltas[~np.isnan(deltas)]
    alpha = 1 - ci
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    return point, lo, hi


# ---------------------------------------------------------------------------
# TEST-SET METRICS
# ---------------------------------------------------------------------------

def evaluate_test_metrics(models: dict, test_df: pd.DataFrame,
                           target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    Point estimate + bootstrap CI for ROC-AUC and PR-AUC, per model per arm,
    on the held-out test set.
    """
    y_true = test_df[target_column].astype(int).values
    rows = []

    for (arm_name, model_name), pipeline in models.items():
        X_test = get_feature_matrix(test_df, FEATURE_SET_ARMS[arm_name])
        proba = pipeline.predict_proba(X_test)[:, 1]

        auc_point, auc_lo, auc_hi = bootstrap_metric_ci(y_true, proba, roc_auc_score)
        pr_point, pr_lo, pr_hi = bootstrap_metric_ci(y_true, proba, average_precision_score)

        rows.append({
            "feature_set": arm_name, "model": model_name,
            "roc_auc": auc_point, "roc_auc_ci_lo": auc_lo, "roc_auc_ci_hi": auc_hi,
            "pr_auc": pr_point, "pr_auc_ci_lo": pr_lo, "pr_auc_ci_hi": pr_hi,
        })
        print(f"  [{arm_name}] {model_name}: "
              f"ROC-AUC={auc_point:.3f} [{auc_lo:.3f}, {auc_hi:.3f}], "
              f"PR-AUC={pr_point:.3f} [{pr_lo:.3f}, {pr_hi:.3f}]")

    return pd.DataFrame(rows)


def evaluate_paired_comparisons(models: dict, test_df: pd.DataFrame,
                                 target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    For each model, paired bootstrap comparison of lifestyle-augmented vs.
    baseline ROC-AUC/PR-AUC on the SAME test set.
    """
    y_true = test_df[target_column].astype(int).values
    rows = []

    for model_name in MODEL_NAMES:
        X_baseline = get_feature_matrix(test_df, FEATURE_SET_ARMS["baseline"])
        X_lifestyle = get_feature_matrix(test_df, FEATURE_SET_ARMS["lifestyle_augmented"])

        proba_baseline = models[("baseline", model_name)].predict_proba(X_baseline)[:, 1]
        proba_lifestyle = models[("lifestyle_augmented", model_name)].predict_proba(X_lifestyle)[:, 1]

        auc_delta, auc_lo, auc_hi = paired_bootstrap_delta(
            y_true, proba_baseline, proba_lifestyle, roc_auc_score
        )
        pr_delta, pr_lo, pr_hi = paired_bootstrap_delta(
            y_true, proba_baseline, proba_lifestyle, average_precision_score
        )

        significant = not (auc_lo <= 0 <= auc_hi)
        rows.append({
            "model": model_name,
            "roc_auc_delta": auc_delta, "roc_auc_delta_ci_lo": auc_lo, "roc_auc_delta_ci_hi": auc_hi,
            "pr_auc_delta": pr_delta, "pr_auc_delta_ci_lo": pr_lo, "pr_auc_delta_ci_hi": pr_hi,
            "roc_auc_significant": significant,
        })
        sig_label = "SIGNIFICANT" if significant else "not significant"
        print(f"  {model_name}: ROC-AUC delta={auc_delta:+.4f} "
              f"[{auc_lo:+.4f}, {auc_hi:+.4f}] -- {sig_label}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SHAP -- feature transformation helper
# ---------------------------------------------------------------------------

def transform_for_shap(pipeline, X: pd.DataFrame) -> tuple:
    """
    Manually apply a pipeline's preprocessing steps (one-hot + impute +
    scale-if-present) to X, SKIPPING the SMOTE step, and return the
    transformed array plus output feature names.

    ATTENTION: Do NOT use pipeline[:-1].transform(X) here. imblearn's SMOTE
    step implements fit_resample, not transform -- slicing the pipeline and
    calling .transform() on a sub-pipeline that still includes the smote
    step is not a supported operation and may error or behave unexpectedly.
    Walking named_steps explicitly and skipping 'smote' by name is the safe
    approach, and it's also semantically correct: SMOTE should never be
    applied to data you're about to explain/predict on regardless.
    """
    ct = pipeline.named_steps["preprocess"]
    imputer = pipeline.named_steps["impute"]

    X_transformed = ct.transform(X)
    feature_names = ct.get_feature_names_out().tolist()
    X_transformed = imputer.transform(X_transformed)

    if "scale" in pipeline.named_steps:
        X_transformed = pipeline.named_steps["scale"].transform(X_transformed)

    return X_transformed, feature_names


# ---------------------------------------------------------------------------
# SHAP -- explainer selection + computation
# ---------------------------------------------------------------------------

def compute_shap_values(pipeline, model_name: str, X_background: np.ndarray, X_explain: np.ndarray):
    """
    Compute SHAP values for one fitted model.

    Tree-based
    models (random_forest, gradient_boosting, bagged_decision_tree, xgboost)
    use shap.TreeExplainer, which is fast and exact for tree ensembles.
    logistic_regression uses shap.LinearExplainer. BaggingClassifier is NOT
    guaranteed to be supported by TreeExplainer (it wraps arbitrary base
    estimators, and shap's tree-model support list is version-dependent) --
    if TreeExplainer raises for bagged_decision_tree specifically, that's
    expected and the except branch below falls back to the slower
    model-agnostic shap.Explainer with a background sample. If that
    fallback triggers, it is noticeably slower (it is the dominant SHAP
    runtime cost in this project), which is expected.
    """
    import shap  # imported here, not at module top, so the rest of this
                  # file's testable logic doesn't hard-fail in environments
                  # without shap installed

    classifier = pipeline.named_steps["classifier"]

    if model_name == "logistic_regression":
        explainer = shap.LinearExplainer(classifier, X_background)
        shap_values = explainer.shap_values(X_explain)
        shap_values = _normalize_shap_output(shap_values)
    else:
        try:
            explainer = shap.TreeExplainer(classifier)
            raw = explainer.shap_values(X_explain)
            shap_values = _normalize_shap_output(raw)
        except Exception as e:
            print(f"    [!] TreeExplainer failed for {model_name} ({e}); "
                  f"falling back to model-agnostic Explainer (slower)")
            explainer = shap.Explainer(classifier.predict_proba, X_background)
            shap_values = explainer(X_explain)[:, :, 1].values

    return shap_values


def _normalize_shap_output(raw):
    """
    Normalize any of the SHAP output shapes actually observed across
    shap/model-type combinations into a single (n_samples, n_features)
    array for the positive class.

    ATTENTION -- REAL BUG THIS FIXES: confirmed on real project data that
    shap.TreeExplainer.shap_values() returns DIFFERENT shapes depending on
    model type, even within the SAME shap version: RandomForestClassifier
    returned a 3D array (n_samples, n_features, n_classes), while XGBoost
    and GradientBoostingClassifier returned a plain 2D array already. The
    original isinstance(raw, list) check only handled an older
    list-of-per-class-arrays convention and silently let the 3D case
    through unnormalized -- every downstream "mean_abs_shap" value for
    random_forest became a length-2 array instead of a scalar, which
    crashed sort_values() with an ambiguous-truth-value error deep inside
    compute_shap_stability(). This function handles all three shapes
    (list, 2D, 3D) explicitly rather than assuming one.
    """
    if isinstance(raw, list):
        return raw[1] if len(raw) > 1 else raw[0]
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) -- take the positive class,
        # last index along the class axis (binary: index 1 == index -1).
        return arr[:, :, -1]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def run_shap_analysis(models: dict, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       run_label: str = "primary") -> tuple:
    """
    Compute SHAP values for all 10 models, save raw arrays, and build the
    mean-|SHAP|-per-feature importance table used for the cross-model
    stability comparison.
    """
    os.makedirs(SHAP_DIR, exist_ok=True)
    importance_rows = []
    shap_by_arm_model = {}

    for arm_name, feature_list in FEATURE_SET_ARMS.items():
        X_train_sample = get_feature_matrix(
            train_df.sample(n=min(SHAP_BACKGROUND_SIZE, len(train_df)), random_state=RANDOM_STATE),
            feature_list,
        )
        X_test_sample = get_feature_matrix(
            test_df.sample(n=min(SHAP_TEST_SAMPLE_SIZE, len(test_df)), random_state=RANDOM_STATE),
            feature_list,
        )

        for model_name in MODEL_NAMES:
            pipeline = models[(arm_name, model_name)]
            print(f"  Computing SHAP: [{arm_name}] {model_name}...")

            X_bg_transformed, feature_names = transform_for_shap(pipeline, X_train_sample)
            X_explain_transformed, _ = transform_for_shap(pipeline, X_test_sample)

            shap_values = compute_shap_values(
                pipeline, model_name, X_bg_transformed, X_explain_transformed
            )

            out_path = _shap_path(run_label, arm_name, model_name)
            np.save(out_path, shap_values)
            shap_by_arm_model[(arm_name, model_name)] = (shap_values, feature_names)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            for feat, val in zip(feature_names, mean_abs_shap):
                importance_rows.append({
                    "feature_set": arm_name, "model": model_name,
                    "feature": feat, "mean_abs_shap": val,
                })

    importance_df = pd.DataFrame(importance_rows)
    return importance_df, shap_by_arm_model


def compute_shap_stability(importance_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Cross-model interpretability stability: for each feature-set arm, how
    much do the models agree on which features matter most?

    ATTENTION: Uses top-N Jaccard overlap (simple, interpretable: "7 of the
    top 10 features are shared between RF and XGBoost") rather than a rank
    correlation -- more directly quotable in a 4-page paper, and doesn't
    require the two models to rank ALL features similarly, just agree on
    the most important ones, which is the actual interpretability claim
    being made. See compute_shap_rank_correlation() below for the
    complementary full-ranking metric.
    """
    rows = []
    for arm_name in FEATURE_SET_ARMS:
        arm_df = importance_df[importance_df["feature_set"] == arm_name]
        top_features = {}
        for model_name in MODEL_NAMES:
            model_df = arm_df[arm_df["model"] == model_name].sort_values(
                "mean_abs_shap", ascending=False
            )
            top_features[model_name] = set(model_df["feature"].head(top_n))

        for i, model_a in enumerate(MODEL_NAMES):
            for model_b in MODEL_NAMES[i + 1:]:
                overlap = top_features[model_a] & top_features[model_b]
                jaccard = len(overlap) / len(top_features[model_a] | top_features[model_b])
                rows.append({
                    "feature_set": arm_name, "model_a": model_a, "model_b": model_b,
                    "top_n": top_n, "n_shared": len(overlap), "jaccard": jaccard,
                })
    return pd.DataFrame(rows)


def compute_shap_rank_correlation(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-model interpretability agreement via Spearman rank correlation
    over the FULL feature-importance ranking, per arm, per model pair.

    ATTENTION -- WHY THIS IS A DIFFERENT QUESTION FROM JACCARD, NOT A
    DUPLICATE: Jaccard (compute_shap_stability, above) asks "do two models
    select the same features into their top 10" -- a set-membership
    question that ignores order and ignores everything outside the top 10.
    Spearman rho asks "do two models rank ALL features similarly" -- e.g.
    two models could have identical top-10 sets (Jaccard=1.0) but disagree
    sharply on which of those 10 is most vs. least important, and Spearman
    would catch that where Jaccard cannot. Reporting both gives a more
    complete stability picture: "these models agree on WHAT matters
    (Jaccard) and/or HOW MUCH it matters relative to everything else
    (Spearman)" are genuinely separate claims.

    Computed directly from the already-saved shap_importance.csv -- no
    SHAP recomputation, no model refitting. Uses scipy.stats.spearmanr on
    each pair's mean_abs_shap values, aligned by feature name (via a pivot
    to guarantee identical feature ordering between the two models being
    compared, since dict/groupby iteration order isn't guaranteed to match
    otherwise).
    """
    from scipy.stats import spearmanr

    rows = []
    for arm_name in FEATURE_SET_ARMS:
        arm_df = importance_df[importance_df["feature_set"] == arm_name]
        # Pivot to (feature x model) so every model's importance values are
        # aligned to the SAME feature order before correlating -- critical,
        # since spearmanr just correlates two same-length arrays positionally
        # and has no concept of "feature name" itself.
        pivot = arm_df.pivot(index="feature", columns="model", values="mean_abs_shap")

        for i, model_a in enumerate(MODEL_NAMES):
            for model_b in MODEL_NAMES[i + 1:]:
                rho, p_value = spearmanr(pivot[model_a], pivot[model_b])
                rows.append({
                    "feature_set": arm_name, "model_a": model_a, "model_b": model_b,
                    "n_features": len(pivot), "spearman_rho": rho, "p_value": p_value,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Held-out test evaluation + SHAP interpretability."
    )
    parser.add_argument(
        "--target-column", default=TARGET_COLUMN,
        help=f"Which phenotype column to use as the label (default: {TARGET_COLUMN}). "
             f"Must match whatever train_pipeline.py was run with."
    )
    parser.add_argument(
        "--run-label", default="primary",
        help="Must match the --run-label train_pipeline.py was called with -- "
             "this is how evaluate.py finds the right saved models. 'primary' "
             "(default) reads/writes the SAME unprefixed filenames as before."
    )
    args = parser.parse_args()

    print(f"--- Run label: {args.run_label} | Target column: {args.target_column} ---")
    print("--- Loading train/test split (same split train_pipeline.py used) ---")
    train_df, test_df = load_and_split()

    print("\n--- Loading saved models ---")
    models = load_all_models(run_label=args.run_label)
    print(f"Loaded {len(models)} pipelines")

    print("\n--- Held-out test metrics (with bootstrap 95% CIs) ---")
    metrics_df = evaluate_test_metrics(models, test_df, target_column=args.target_column)
    os.makedirs(TABLES_DIR, exist_ok=True)
    metrics_path = _table_path(args.run_label, "test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[wrote] {metrics_path}")

    print("\n--- Paired comparison: lifestyle-augmented vs baseline ---")
    paired_df = evaluate_paired_comparisons(models, test_df, target_column=args.target_column)
    paired_path = _table_path(args.run_label, "paired_comparison.csv")
    paired_df.to_csv(paired_path, index=False)
    print(f"[wrote] {paired_path}")

    print("\n--- SHAP analysis ---")
    try:
        importance_df, _ = run_shap_analysis(models, train_df, test_df, run_label=args.run_label)
        importance_path = _table_path(args.run_label, "shap_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"[wrote] {importance_path}")

        stability_df = compute_shap_stability(importance_df)
        stability_path = _table_path(args.run_label, "shap_stability.csv")
        stability_df.to_csv(stability_path, index=False)
        print(f"[wrote] {stability_path}")

        rank_corr_df = compute_shap_rank_correlation(importance_df)
        rank_corr_path = _table_path(args.run_label, "shap_rank_correlation.csv")
        rank_corr_df.to_csv(rank_corr_path, index=False)
        print(f"[wrote] {rank_corr_path}")

        print("\n--- Cross-model top-10 feature agreement (lifestyle-augmented arm) ---")
        print(stability_df[stability_df["feature_set"] == "lifestyle_augmented"]
              [["model_a", "model_b", "n_shared", "jaccard"]].to_string(index=False))
        print("\n--- Cross-model full-ranking Spearman agreement (lifestyle-augmented arm) ---")
        print(rank_corr_df[rank_corr_df["feature_set"] == "lifestyle_augmented"]
              [["model_a", "model_b", "spearman_rho", "p_value"]].to_string(index=False))
    except Exception as e:
        print(f"\n[!] SHAP analysis failed: {e}")
        print("    Metrics and paired-comparison results above are still valid and saved --")
        print("    this failure only affects the interpretability analysis.")
        raise

    print("\nDone.")


if __name__ == "__main__":
    main()
