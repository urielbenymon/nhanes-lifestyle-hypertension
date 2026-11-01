"""
train_pipeline.py

PURPOSE
-------
The leakage-safe cross-validation engine and the central ablation experiment:
does lifestyle information improve hypertension prediction beyond a
demographic/anthropometric baseline, and how does that hold across 5 model
families?

Runs EVERY model x EVERY feature-set arm combination (5 models x 2 arms = 10
pipelines), each evaluated with 5-fold stratified CV, where imputation,
one-hot encoding, and SMOTE are ALL fit fresh inside each fold via an
imblearn Pipeline -- never on the full training set before CV starts. This
is the single most important methodological property this project claims
over the baseline paper it extends (which used a single 90/10 split with no
CV at all).

After CV, also fits each of the 10 pipelines on the FULL training set (still
using the same imputer/encoder/SMOTE structure, just fit once on all of
train rather than per-fold) and saves them to outputs/models/ -- these are
what evaluate.py loads for held-out test-set metrics and SHAP.

WHEN --tune IS USED: the saved model is instead the result of CV-based
hyperparameter SELECTION (via GridSearchCV/RandomizedSearchCV over the
fixed 90% development set) followed by an automatic refit on that same 90%
-- a different, more precise claim than "fit once after ordinary CV." See
tune_primary_models() below for the exact procedure and why it is
deliberately NOT nested CV.

WHY imblearn.pipeline.Pipeline, NOT sklearn.pipeline.Pipeline
----------------------------------------------------------------
SMOTE changes the number of rows (it generates synthetic minority-class
samples). sklearn's own Pipeline doesn't support a step that resamples --
only imblearn's Pipeline does, and critically, it only applies the resampler
during .fit(), never during .transform()/.predict() -- meaning validation
and test folds are NEVER synthetically oversampled, only the training
portion of each fold is. This is exactly the fold-safe behavior required;
using plain SMOTE fit once on the whole training set before CV (as many
NHANES-ML tutorials do) would leak synthetic near-duplicates of validation
rows into the training data and inflate CV scores artificially.

USAGE
-----
    python src/train_pipeline.py

OUTPUT
------
    outputs/tables/cv_results.csv       -- mean/std ROC-AUC and PR-AUC per
                                            model per feature-set arm
    outputs/models/{arm}__{model}.joblib -- each pipeline, fit on full train,
                                            ready for evaluate.py to load

NOTE ON REPRODUCIBILITY
-----------------------
Runtime (and, across library versions, exact CV numbers) may differ
slightly by machine; see README.md for the versions used in the paper.
"""

import os
import time
import argparse
import ast
import numpy as np
import pandas as pd
import joblib

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

from preprocessing import (
    load_and_split,
    get_feature_matrix,
    build_column_transformer,
    MODELING_BASELINE_FEATURES,
    MODELING_LIFESTYLE_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")

N_FOLDS = 5

# ATTENTION -- REPRODUCIBILITY / MULTI-RUN NAMING: `run_label` distinguishes
# output files between the primary analysis (phenotype B) and sensitivity
# analyses (e.g. phenotype A). "primary" writes to the SAME unprefixed
# filenames this script has always used (cv_results.csv,
# outputs/models/{arm}__{model}.joblib) so nothing changes for the default
# invocation and nothing you already have gets silently overwritten by a
# differently-labeled run. Any other run_label gets that label prepended to
# every output filename instead.
def _cv_results_path(run_label: str) -> str:
    if run_label == "primary":
        return os.path.join(TABLES_DIR, "cv_results.csv")
    return os.path.join(TABLES_DIR, f"cv_results_{run_label}.csv")


def _model_path(run_label: str, arm_name: str, model_name: str) -> str:
    if run_label == "primary":
        return os.path.join(MODELS_DIR, f"{arm_name}__{model_name}.joblib")
    return os.path.join(MODELS_DIR, f"{run_label}__{arm_name}__{model_name}.joblib")


# ATTENTION: This IS the central ablation. "baseline" and "lifestyle" here
# are the two feature-set arms the whole paper's research question rests on
# -- everything downstream (evaluate.py's incremental-value comparison)
# depends on these two lists being exactly right.
FEATURE_SET_ARMS = {
    "baseline": MODELING_BASELINE_FEATURES,
    "lifestyle_augmented": MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES,
}

# ATTENTION -- TUNING ARCHITECTURE. Tuning
# is NOT "wrap each model in a search INSIDE the existing per-fold CV
# structure" -- that WOULD be nested CV (search inside every outer fold),
# which was deliberately not used. The actual (pragmatic) architecture, implemented in
# tune_primary_models() below, has exactly ONE CV layer:
#   fixed 90% development set -> GridSearchCV/RandomizedSearchCV (5-fold
#   CV over that 90%, full pipeline searched, SMOTE/imputer/encoder
#   refit per search-fold) -> best params selected -> automatic refit on
#   all of the 90% -> saved -> scored once on the fixed 10% test partition
#   (excluded from fitting and hyperparameter selection).
# run_cv_for_arm() below (the untuned default-hyperparameter path) is
# UNCHANGED and still used when --tune is not passed.
# ATTENTION -- PHENOTYPE-A SENSITIVITY, HYPERPARAMETER INVARIANT: phenotype A
# is refit with the SAME hyperparameters frozen for primary phenotype B
# (per model, per arm), loaded from best_hyperparameters.csv -- never
# retuned and never left on library defaults. The invariant is "everything
# about the model is identical between A and B except the label column";
# anything else would confound the phenotype comparison with a
# configuration difference.
def _strip_classifier_prefix(params: dict) -> dict:
    """
    Convert pipeline-prefixed search params (e.g. {'classifier__C': 1.0,
    'classifier__estimator__max_depth': 5}) into params settable directly
    via classifier.set_params() (e.g. {'C': 1.0, 'estimator__max_depth': 5}).

    ATTENTION: The nested 'estimator__' prefix for BaggingClassifier's base
    tree is preserved as-is after stripping only the leading 'classifier__'
    -- sklearn's set_params() natively understands 'estimator__max_depth'
    as "set max_depth on my estimator sub-object," so no further parsing
    is needed for that case.
    """
    return {k.replace("classifier__", "", 1): v for k, v in params.items()}


def build_models(tuned_params: dict = None) -> dict:
    """
    ATTENTION: Returns FRESH, unfit model instances -- called once per
    pipeline construction (see build_pipeline()) so no fitted state is ever
    accidentally reused across arms/folds.

    tuned_params, if provided, is {model_name: {param: value, ...}} using
    ALREADY-STRIPPED parameter names (see _strip_classifier_prefix above) --
    applied via .set_params() on top of the default-constructed instance.
    When None (default), behavior is UNCHANGED from before tuning existed:
    every model uses library defaults / the lightly-set values below.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "bagged_decision_tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }
    if tuned_params:
        for model_name, params in tuned_params.items():
            if model_name in models and params:
                models[model_name].set_params(**params)
    return models


# ---------------------------------------------------------------------------
# PIPELINE CONSTRUCTION
# ---------------------------------------------------------------------------

def build_pipeline(feature_list: list, model_name: str, model) -> ImbPipeline:
    """
    Assemble one fold-fittable pipeline: one-hot encode -> impute -> (scale
    if LR) -> SMOTE -> classifier.

    ATTENTION -- ORDER MATTERS:
      1. ColumnTransformer (from preprocessing.py's FACTORY -- a fresh
         instance every call) one-hot encodes categoricals and passes
         numeric columns through with NaNs intact.
      2. SimpleImputer(median) fills any remaining numeric NaNs. Only
         numeric columns can still have NaN at this point -- the one-hot
         columns (including DMDEDUC2_nan, per the earlier decision to keep
         missingness-as-a-category rather than impute it) are already
         complete 0/1 indicators with nothing to impute.
      3. StandardScaler ONLY for logistic_regression -- tree-based models
         (RF/GB/BaggedDT/XGBoost) are scale-invariant and don't need it;
         adding it for them would be harmless but is deliberately omitted
         to keep each model's pipeline minimal and easy to justify in the
         methods section.
      4. SMOTE -- imblearn Pipeline guarantees this only resamples the
         TRAINING portion of whatever data .fit() sees; it is a no-op
         during .transform()/.predict(). This is what makes SMOTE
         fold-safe here.
      5. The classifier itself.
    """
    column_transformer = build_column_transformer(feature_list)

    steps = [
        ("preprocess", column_transformer),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if model_name == "logistic_regression":
        steps.append(("scale", StandardScaler()))
    steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("classifier", model))

    return ImbPipeline(steps)


# ---------------------------------------------------------------------------
# HYPERPARAMETER TUNING (pragmatic, single-CV-layer -- see module docstring)
# ---------------------------------------------------------------------------

# ATTENTION: All keys are pipeline-prefixed ("classifier__...") since the
# FULL pipeline (encoder + imputer + [scaler] + SMOTE + classifier) is what
# gets searched, not the bare classifier -- this is what keeps every
# candidate's preprocessing/SMOTE fold-safe during the search (each search
# fold refits the ENTIRE pipeline, encoder and SMOTE included, on just that
# fold's training portion). No 'smote__' keys appear anywhere here,
# deliberately -- SMOTE's own parameters are not part of this search.
TUNING_PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
}
TUNING_PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5, 10],
        "classifier__max_features": ["sqrt", "log2", 0.5],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__min_samples_leaf": [1, 5, 10, 20],
    },
    "bagged_decision_tree": {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__estimator__max_depth": [None, 5, 10, 20],
        "classifier__estimator__min_samples_leaf": [1, 2, 5, 10],
        "classifier__max_samples": [0.5, 0.7, 1.0],
    },
    "xgboost": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__min_child_weight": [1, 3, 5],
    },
}
N_RANDOM_ITER = 25  # candidates for RandomizedSearchCV models (RF/GB/Bagged/XGBoost)


def tune_primary_models(train_df: pd.DataFrame) -> dict:
    """
    Pragmatic hyperparameter tuning for all 5 models x 2 feature arms (10
    independent searches), on the fixed 90% development set (train_df) --
    ALWAYS uses TARGET_COLUMN (primary phenotype B). This function should
    never be called for a phenotype-A run; phenotype A inherits these
    frozen results instead (see main()'s --run-label handling).

    ATTENTION -- WHY BASELINE AND LIFESTYLE ARMS ARE TUNED INDEPENDENTLY,
    NOT SHARED: the two arms have different feature counts (9 vs 20) and
    different dimensionality -- e.g. random_forest's max_features fraction
    or XGBoost's colsample_bytree interact meaningfully with how many
    columns exist. Forcing both arms to share one model's hyperparameters
    would bias the comparison toward whichever arm's optimal settings
    happened to be picked. 10 independent parameter sets, one per
    (arm, model) combination.

    ATTENTION -- ONE CV LAYER, NOT NESTED: each search's internal 5-fold CV
    IS the only cross-validation happening here. There is no outer
    cross_validate() wrapping this -- do not add one. GridSearchCV/
    RandomizedSearchCV(refit=...) automatically refits the best pipeline on
    the ENTIRE 90% development set after the search completes; that refit
    pipeline (search.best_estimator_) is the final artifact saved to disk,
    not a separate fit_and_save_final_models() call.

    ATTENTION -- REPORTED CV SCORE IS A DEVELOPMENT/SELECTION ESTIMATE, NOT
    AN UNBIASED GENERALIZATION ESTIMATE: because hyperparameters were
    chosen using this same CV score, best_score_ is mildly optimistic
    relative to genuinely out-of-sample performance. The fixed 10% held-out
    test partition (excluded from fitting and hyperparameter selection,
    scored in evaluate.py, never touched here) is the unbiased estimate.
    State this explicitly in the manuscript -- do not describe the
    post-tuning CV number as if it were a fresh, unbiased CV result. Also
    do not describe the test partition as a never-before-viewed prospective
    holdout -- earlier default-model results already inspected it once;
    what matters is that fitting and hyperparameter SELECTION never used it.

    Returns: {arm_name: {model_name: stripped_best_params}} -- ready to
    pass directly into build_models(tuned_params=...) for either the
    primary phenotype-B final fit or a later phenotype-A refit.

    Also writes:
      outputs/tables/best_hyperparameters.csv -- one row per (arm, model)
      outputs/tables/tuning_results.csv -- the COMPLETE cv_results_
        DataFrame from every search, concatenated across all (arm, model)
        combinations with feature_set/model columns added -- includes
        every candidate's per-split scores, fit/score times, and
        parameters, not just the hand-picked summary columns. Disk cost is
        negligible; this is the actual full audit trail for the appendix.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_params_by_arm = {}
    best_rows = []
    full_results_dfs = []

    for arm_name, feature_list in FEATURE_SET_ARMS.items():
        X = get_feature_matrix(train_df, feature_list)
        y = train_df[TARGET_COLUMN].astype(int)
        best_params_by_arm[arm_name] = {}

        for model_name, base_model in build_models().items():
            pipeline = build_pipeline(feature_list, model_name, base_model)

            if model_name in TUNING_PARAM_GRIDS:
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=TUNING_PARAM_GRIDS[model_name],
                    scoring=["roc_auc", "average_precision"],
                    refit="roc_auc",
                    cv=cv,
                )
                search_type, n_candidates = "GridSearchCV", len(
                    TUNING_PARAM_GRIDS[model_name]["classifier__C"]
                )
            else:
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=TUNING_PARAM_DISTRIBUTIONS[model_name],
                    n_iter=N_RANDOM_ITER,
                    scoring=["roc_auc", "average_precision"],
                    refit="roc_auc",
                    cv=cv,
                    random_state=RANDOM_STATE,
                )
                search_type, n_candidates = "RandomizedSearchCV", N_RANDOM_ITER

            print(f"  Tuning [{arm_name}] {model_name} ({search_type}, "
                  f"{n_candidates} candidates x {N_FOLDS} folds)...")
            t0 = time.time()
            search.fit(X, y)
            elapsed = time.time() - t0

            stripped = _strip_classifier_prefix(search.best_params_)
            best_params_by_arm[arm_name][model_name] = stripped

            best_idx = search.best_index_
            cvres = search.cv_results_
            best_rows.append({
                "feature_set": arm_name, "model": model_name,
                "best_cv_roc_auc": cvres["mean_test_roc_auc"][best_idx],
                "best_cv_roc_auc_std": cvres["std_test_roc_auc"][best_idx],
                "best_cv_pr_auc": cvres["mean_test_average_precision"][best_idx],
                "best_cv_pr_auc_std": cvres["std_test_average_precision"][best_idx],
                "best_parameters": str(stripped),
                "search_type": search_type,
                "n_candidates": n_candidates,
                "n_folds": N_FOLDS,
                "random_state": RANDOM_STATE,
                "fit_time_sec": round(elapsed, 1),
            })
            print(f"    best ROC-AUC={cvres['mean_test_roc_auc'][best_idx]:.4f} "
                  f"params={stripped} ({elapsed:.1f}s)")

            # ATTENTION: dump the COMPLETE cv_results_
            # DataFrame, not a hand-picked subset of fields -- this preserves
            # per-split scores, fit/score times, and every parameter column
            # sklearn already computed, at negligible disk cost. Tag each
            # search's rows with feature_set/model before concatenating so
            # the combined file stays queryable across all 10 searches.
            search_df = pd.DataFrame(search.cv_results_)
            search_df.insert(0, "model", model_name)
            search_df.insert(0, "feature_set", arm_name)
            full_results_dfs.append(search_df)

            # ATTENTION: search.best_estimator_ is ALREADY refit on the full
            # 90% development set (GridSearchCV/RandomizedSearchCV default
            # refit behavior) -- save it directly. Do NOT call
            # fit_and_save_final_models() for the tuned path; that would be
            # a redundant, separate refit outside the search's own.
            out_path = _model_path("primary", arm_name, model_name)
            joblib.dump(search.best_estimator_, out_path)
            print(f"    [saved] {out_path}")

    pd.DataFrame(best_rows).to_csv(
        os.path.join(TABLES_DIR, "best_hyperparameters.csv"), index=False
    )
    pd.concat(full_results_dfs, ignore_index=True).to_csv(
        os.path.join(TABLES_DIR, "tuning_results.csv"), index=False
    )
    print(f"\n[wrote] {os.path.join(TABLES_DIR, 'best_hyperparameters.csv')}")
    print(f"[wrote] {os.path.join(TABLES_DIR, 'tuning_results.csv')}")

    return best_params_by_arm


# ---------------------------------------------------------------------------
# CROSS-VALIDATION
# ---------------------------------------------------------------------------

def run_cv_for_arm(train_df: pd.DataFrame, arm_name: str, feature_list: list,
                    target_column: str = TARGET_COLUMN, tuned_params: dict = None) -> list:
    """
    Run 5-fold stratified CV for all 5 models on one feature-set arm.

    ATTENTION: A FRESH build_models() call happens here (not passed in from
    outside) so every arm gets genuinely independent, never-before-fit model
    instances -- avoids any risk of a model object carrying fitted state
    from the baseline arm into the lifestyle-augmented arm's evaluation.

    tuned_params, if provided, is {model_name: stripped_params} for THIS
    arm specifically (e.g. best_params_by_arm["baseline"] from
    tune_primary_models()) -- used for the phenotype-A sensitivity run,
    which reports a CV score under the frozen primary-B hyperparameters
    rather than re-tuning. When None (default/untuned path), unchanged.
    """
    X = get_feature_matrix(train_df, feature_list)
    y = train_df[target_column].astype(int)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for model_name, model in build_models(tuned_params=tuned_params).items():
        pipeline = build_pipeline(feature_list, model_name, model)

        t0 = time.time()
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["roc_auc", "average_precision"],
            n_jobs=1,  # ATTENTION: kept serial deliberately -- some models
                       # (RF/XGBoost) already parallelize internally via
                       # n_jobs=-1; parallelizing cross_validate ON TOP of
                       # that oversubscribes CPU cores and can be slower,
                       # not faster, plus makes error messages harder to read.
        )
        elapsed = time.time() - t0

        results.append({
            "feature_set": arm_name,
            "model": model_name,
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "pr_auc_mean": scores["test_average_precision"].mean(),
            "pr_auc_std": scores["test_average_precision"].std(),
            "fit_time_sec": round(elapsed, 1),
        })
        print(f"  [{arm_name}] {model_name}: "
              f"ROC-AUC={scores['test_roc_auc'].mean():.3f}±{scores['test_roc_auc'].std():.3f}, "
              f"PR-AUC={scores['test_average_precision'].mean():.3f}±{scores['test_average_precision'].std():.3f} "
              f"({elapsed:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# FINAL FIT ON FULL TRAINING SET (for evaluate.py to consume)
# ---------------------------------------------------------------------------

def load_frozen_hyperparameters() -> dict:
    """
    Load best_hyperparameters.csv (written by tune_primary_models()) and
    reconstruct the {arm_name: {model_name: stripped_params}} structure
    build_models(tuned_params=...) expects.

    ATTENTION: Used by the phenotype-A path to inherit primary-B's frozen
    tuned settings -- per the hyperparameter invariant documented above
    build_models(), phenotype A must NOT be independently tuned. Returns
    None (with a printed warning, not a silent empty dict) if the file
    doesn't exist yet -- e.g. if you're running phenotype-A as a quick
    smoke test before tuning has been done for primary B. Callers should
    treat None as "fall back to library defaults for this run," not crash.
    """
    path = os.path.join(TABLES_DIR, "best_hyperparameters.csv")
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- tuning hasn't been run for primary B yet. "
              f"Falling back to library defaults for this run.")
        return None

    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        arm, model = row["feature_set"], row["model"]
        result.setdefault(arm, {})[model] = ast.literal_eval(row["best_parameters"])
    print(f"  Loaded frozen tuned hyperparameters from {path}")
    return result


def fit_and_save_final_models(train_df: pd.DataFrame, target_column: str = TARGET_COLUMN,
                               run_label: str = "primary", tuned_params: dict = None) -> None:
    """
    Fit each of the 10 (model x arm) pipelines on the FULL training set
    (no CV split here -- CV already happened in run_cv_for_arm; this is a
    separate, final fit) and save each to outputs/models/.

    ATTENTION: This is the SAME pipeline SPECIFICATION (same imputer, same
    encoder, same SMOTE settings, same model hyperparameters) as each CV
    fold used -- but this is a fresh fit on the full training set, not one
    of the 5 fold-fitted estimators from run_cv_for_arm. evaluate.py should
    LOAD this saved pipeline rather than refitting, so the model evaluate.py
    scores on the held-out test set is a known, fixed artifact matching the
    configuration whose CV performance is in cv_results.csv -- not a
    silently different refit with different random initialization, just not
    literally the same fitted weights as any individual CV fold.

    tuned_params, if provided, is {arm_name: {model_name: stripped_params}}
    -- used for the phenotype-A path to fit with primary-B's frozen tuned
    settings (via load_frozen_hyperparameters()) rather than defaults.

    ATTENTION: For the PRIMARY tuned path, this function is NOT used --
    tune_primary_models() already saves search.best_estimator_ directly
    (already refit on the full 90%), since calling this function afterward
    would be a redundant second refit with a different SMOTE random draw.
    This function remains for: (a) the untuned default path, and (b) the
    phenotype-A path, which needs a fresh fit under y=hypertension_a using
    frozen hyperparameters that came from a DIFFERENT (primary-B) search.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    for arm_name, feature_list in FEATURE_SET_ARMS.items():
        X = get_feature_matrix(train_df, feature_list)
        y = train_df[target_column].astype(int)

        arm_tuned_params = tuned_params.get(arm_name) if tuned_params else None
        for model_name, model in build_models(tuned_params=arm_tuned_params).items():
            pipeline = build_pipeline(feature_list, model_name, model)
            pipeline.fit(X, y)

            out_path = _model_path(run_label, arm_name, model_name)
            joblib.dump(pipeline, out_path)
            print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe CV + ablation training pipeline."
    )
    parser.add_argument(
        "--target-column", default=TARGET_COLUMN,
        help=f"Which phenotype column to use as the label (default: {TARGET_COLUMN}, "
             f"i.e. phenotype B / primary). Pass hypertension_a for the phenotype-A "
             f"sensitivity analysis."
    )
    parser.add_argument(
        "--run-label", default="primary",
        help="Prefix for output filenames. 'primary' (default) writes the SAME "
             "unprefixed filenames this script has always used -- do not change "
             "this for the main phenotype-B run. Use e.g. 'phenotype_a' for the "
             "sensitivity analysis so its outputs never collide with or overwrite "
             "the primary results."
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run pragmatic hyperparameter tuning (GridSearchCV/RandomizedSearchCV, "
             "one CV layer, on the fixed 90%% development set) instead of the "
             "default-hyperparameter path. ONLY valid with --run-label primary -- "
             "phenotype-A runs automatically inherit primary B's frozen tuned "
             "settings (if available) regardless of this flag; see "
             "load_frozen_hyperparameters()."
    )
    args = parser.parse_args()

    if args.tune and args.run_label != "primary":
        raise ValueError(
            "--tune is only valid with --run-label primary. Phenotype-A and other "
            "sensitivity runs must inherit primary B's frozen hyperparameters, not "
            "be independently tuned -- see the hyperparameter invariant documented "
            "above build_models(). Run without --tune for a non-primary run_label; "
            "it will automatically use load_frozen_hyperparameters() if available."
        )
    if args.tune and args.target_column != TARGET_COLUMN:
        raise ValueError(
            f"--tune is only valid for the primary target column ({TARGET_COLUMN}). "
            f"tune_primary_models() always tunes on {TARGET_COLUMN} internally "
            f"regardless of --target-column -- running with a different "
            f"--target-column here would print a misleading header (claiming "
            f"'{args.target_column}' while silently tuning on {TARGET_COLUMN}) and "
            f"save phenotype-B models under the 'primary' filenames. Phenotype-A "
            f"sensitivity must inherit the hyperparameters selected under the "
            f"primary phenotype, not trigger a separate (and mislabeled) tuning run."
        )

    print(f"--- Run label: {args.run_label} | Target column: {args.target_column} "
          f"| Tune: {args.tune} ---")
    print("--- Loading train/test split ---")
    train_df, test_df = load_and_split()

    if args.tune:
        # PRIMARY TUNED PATH: one CV layer (the search itself). Reports CV
        # results directly from each search's cv_results_ (see
        # tune_primary_models' docstring for why no separate cross_validate
        # call happens here) and saves search.best_estimator_ directly --
        # already refit on the full 90% by the search's own refit behavior.
        print("\n--- Tuning (GridSearchCV/RandomizedSearchCV, one CV layer, "
              "5 models x 2 arms) ---")
        best_params_by_arm = tune_primary_models(train_df)

        best_hp_df = pd.read_csv(os.path.join(TABLES_DIR, "best_hyperparameters.csv"))
        results_df = best_hp_df.rename(columns={
            "best_cv_roc_auc": "roc_auc_mean", "best_cv_roc_auc_std": "roc_auc_std",
            "best_cv_pr_auc": "pr_auc_mean", "best_cv_pr_auc_std": "pr_auc_std",
        })[["feature_set", "model", "roc_auc_mean", "roc_auc_std",
            "pr_auc_mean", "pr_auc_std", "fit_time_sec"]]
        results_df.to_csv(_cv_results_path(args.run_label), index=False)
        print(f"\n[wrote] {_cv_results_path(args.run_label)} (from tuning search results -- "
              f"NOTE: these are development/model-selection CV estimates, not a fresh "
              f"unbiased CV run -- see tune_primary_models() docstring)")

    else:
        # DEFAULT / UNTUNED PATH, or PHENOTYPE-A WITH INHERITED FROZEN PARAMS.
        # For any non-primary run_label, automatically try to inherit
        # primary B's frozen tuned settings -- falls back to library
        # defaults with a printed warning if tuning hasn't been run yet
        # (e.g. an early phenotype-A smoke test before tuning existed).
        tuned_params_by_arm = None
        if args.run_label != "primary":
            tuned_params_by_arm = load_frozen_hyperparameters()

        print("\n--- Running cross-validation (5-fold, 5 models x 2 feature arms) ---")
        all_results = []
        for arm_name, feature_list in FEATURE_SET_ARMS.items():
            print(f"\nArm: {arm_name} ({len(feature_list)} features)")
            arm_tuned = tuned_params_by_arm.get(arm_name) if tuned_params_by_arm else None
            all_results.extend(
                run_cv_for_arm(train_df, arm_name, feature_list,
                                target_column=args.target_column, tuned_params=arm_tuned)
            )

        results_df = pd.DataFrame(all_results)
        os.makedirs(TABLES_DIR, exist_ok=True)
        out_path = _cv_results_path(args.run_label)
        results_df.to_csv(out_path, index=False)
        print(f"\n[wrote] {out_path}")

        print("\n--- CV results summary ---")
        print(results_df.to_string(index=False))

        print("\n--- Incremental value of lifestyle features (ROC-AUC) ---")
        pivot = results_df.pivot(index="model", columns="feature_set", values="roc_auc_mean")
        pivot["delta"] = pivot["lifestyle_augmented"] - pivot["baseline"]
        print(pivot.to_string())

        print("\n--- Fitting final models on full training set ---")
        fit_and_save_final_models(train_df, target_column=args.target_column,
                                   run_label=args.run_label, tuned_params=tuned_params_by_arm)

    print("\nDone. Test set (held out from fitting and hyperparameter selection) has",
          len(test_df), "rows -- evaluate.py will use it next.")


if __name__ == "__main__":
    main()
