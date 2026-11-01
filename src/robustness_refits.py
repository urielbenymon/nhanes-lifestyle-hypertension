"""
robustness_refits.py

PURPOSE
-------
Supplementary analyses that REFIT models with
the FROZEN tuned hyperparameters from best_hyperparameters.csv. No tuning,
no SHAP -- discrimination only. The primary saved models are never
overwritten (nothing is written to outputs/models/).

  --part trainseeds  CLEAN check of training randomness. ORIGINAL seed-42
                    dev/test split held fixed (test set never used for any
                    decision here); only the model and SMOTE random_state
                    vary (default 10 seeds). Refit both arms x 5 models,
                    paired lifestyle-minus-baseline delta per seed.

  --part splits     EXPLORATORY split sensitivity, CONDITIONAL ON SEED-42
                    TUNING. Repeated random 90/10 splits; refit with the
                    frozen hyperparameters. NOT independent validation:
                    hyperparameters were tuned on the seed-42 development
                    set, so for other seeds part of each new test set
                    informed tuning. Report it labeled as such. (Per-split
                    retuning would need tune_primary_models(), which
                    OVERWRITES best_hyperparameters.csv -- do not call it.)

  --part ablations  On the ORIGINAL seed-42 split: nested predictor subsets
                    (age only; age + BMI; demographic/SES only;
                    anthropometric only; baseline; lifestyle only; baseline +
                    lifestyle). Held-out ROC-AUC with bootstrap CI and paired
                    delta vs the full baseline arm. Subsets of the baseline
                    reuse the BASELINE arm's frozen hyperparameters; lifestyle-
                    only and baseline+lifestyle reuse the LIFESTYLE arm's.
                    Not retuned per subset -- disclose this.

  --part untreated  Untreated-subgroup SENSITIVITY analysis (not a
                    medication-effect estimate). ORIGINAL seed-42 split,
                    restricted in both dev and test to participants with a
                    DEFINITE "not currently on BP medication" from the
                    questionnaire skip pattern:
                        BPQ020 == 2 (never told high BP), or
                        BPQ040A == 2 (told high BP, not told to take meds), or
                        BPQ050A == 2 (told to take meds, not currently taking).
                    Participants whose status is unknown (missing BPQ020, or
                    BPQ020 == 1 with missing BPQ040A; 14 people in the
                    cohort) are EXCLUDED rather than assumed untreated.
                    Outcome = measured-BP Phenotype A. Refit both arms.

USAGE
-----
    python src/robustness_refits.py --part splits --n-seeds 1   # seed-42 reproduction check FIRST
    python src/robustness_refits.py --part ablations
    python src/robustness_refits.py --part untreated
    python src/robustness_refits.py --part trainseeds
    python src/robustness_refits.py --part splits               # exploratory, 10 seeds

OUTPUT
------
    outputs/tables/train_seed_deltas.csv
    outputs/tables/train_seed_summary.csv
    outputs/tables/repeated_splits_deltas.csv
    outputs/tables/repeated_splits_summary.csv
    outputs/tables/ablation_results.csv
    outputs/tables/untreated_results.csv
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocessing import (
    load_and_split, get_feature_matrix,
    MODELING_BASELINE_FEATURES, MODELING_LIFESTYLE_FEATURES, TARGET_COLUMN,
)
from evaluate import bootstrap_metric_ci, paired_bootstrap_delta, MODEL_NAMES
from smote_sensitivity import (
    load_frozen_hyperparameters, build_tuned_model, build_pipeline_variant,
)

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")
DEFAULT_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # 42 first: should reproduce Table 9
TRAIN_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_BOOT_SPLITS = 500                                # per-seed paired CI (speed)

ARMS = {
    "baseline": MODELING_BASELINE_FEATURES,
    "lifestyle_augmented": MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES,
}
DEMOGRAPHIC_SES = ["RIDAGEYR", "is_female", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]
ANTHROPOMETRIC = ["BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]
ABLATIONS = {
    # name: (feature list, which arm's frozen hyperparameters to reuse)
    "age_only": (["RIDAGEYR"], "baseline"),
    "age_bmi": (["RIDAGEYR", "BMXBMI"], "baseline"),
    "demographic_ses_only": (DEMOGRAPHIC_SES, "baseline"),
    "anthropometric_only": (ANTHROPOMETRIC, "baseline"),
    "baseline_full": (MODELING_BASELINE_FEATURES, "baseline"),
    "lifestyle_only": (MODELING_LIFESTYLE_FEATURES, "lifestyle_augmented"),
    "baseline_plus_lifestyle": (MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES,
                                "lifestyle_augmented"),
}


def _reseed(pipe, model_name, seed):
    """Change only the stochastic seeds (classifier + SMOTE), nothing else."""
    pipe.set_params(smote__random_state=seed)
    if model_name == "bagged_decision_tree":
        pipe.set_params(classifier__random_state=seed, classifier__estimator__random_state=seed)
    else:
        pipe.set_params(classifier__random_state=seed)
    return pipe


def fit_predict(train_df, test_df, features, model_name, params, target, model_seed=None):
    model = build_tuned_model(model_name, params)
    pipe = build_pipeline_variant(features, model_name, model, use_smote=True)
    if model_seed is not None:
        _reseed(pipe, model_name, model_seed)
    pipe.fit(get_feature_matrix(train_df, features), train_df[target].astype(int))
    return pipe.predict_proba(get_feature_matrix(test_df, features))[:, 1]


# ---------------------------------------------------------------------------
# PART 1: REPEATED SPLITS
# ---------------------------------------------------------------------------

def _summarize(deltas):
    return deltas.groupby("model").agg(
        n_seeds=("delta", "size"), mean_delta=("delta", "mean"), sd_delta=("delta", "std"),
        min_delta=("delta", "min"), max_delta=("delta", "max"),
        n_positive=("delta", lambda s: int((s > 0).sum())),
        n_ci_excludes_zero=("ci_includes_zero", lambda s: int((~s).sum())),
        mean_auc_baseline=("auc_baseline", "mean"), mean_auc_lifestyle=("auc_lifestyle", "mean"),
    ).reindex(MODEL_NAMES).reset_index()


def run_train_seeds(frozen, seeds):
    train_df, test_df = load_and_split()      # ORIGINAL split, fixed
    y = test_df[TARGET_COLUMN].astype(int).values
    rows = []
    for seed in seeds:
        t0 = time.time()
        for m in MODEL_NAMES:
            p = {arm: fit_predict(train_df, test_df, feats, m, frozen[arm][m], TARGET_COLUMN, model_seed=seed)
                 for arm, feats in ARMS.items()}
            d, lo, hi = paired_bootstrap_delta(y, p["baseline"], p["lifestyle_augmented"],
                                               roc_auc_score, n_boot=N_BOOT_SPLITS)
            rows.append({"train_seed": seed, "model": m,
                         "auc_baseline": roc_auc_score(y, p["baseline"]),
                         "auc_lifestyle": roc_auc_score(y, p["lifestyle_augmented"]),
                         "delta": d, "ci_lo": lo, "ci_hi": hi, "ci_includes_zero": lo <= 0 <= hi})
        print(f"  train seed {seed}: done in {time.time() - t0:.0f}s")
    deltas = pd.DataFrame(rows)
    return deltas, _summarize(deltas)


def run_splits(frozen, seeds):
    rows = []
    for seed in seeds:
        t0 = time.time()
        train_df, test_df = load_and_split(random_state=seed)
        y = test_df[TARGET_COLUMN].astype(int).values
        for m in MODEL_NAMES:
            p = {arm: fit_predict(train_df, test_df, feats, m, frozen[arm][m], TARGET_COLUMN)
                 for arm, feats in ARMS.items()}
            d, lo, hi = paired_bootstrap_delta(y, p["baseline"], p["lifestyle_augmented"],
                                               roc_auc_score, n_boot=N_BOOT_SPLITS)
            rows.append({"seed": seed, "model": m,
                         "auc_baseline": roc_auc_score(y, p["baseline"]),
                         "auc_lifestyle": roc_auc_score(y, p["lifestyle_augmented"]),
                         "delta": d, "ci_lo": lo, "ci_hi": hi,
                         "ci_includes_zero": lo <= 0 <= hi})
        print(f"  seed {seed}: done in {time.time() - t0:.0f}s")
    deltas = pd.DataFrame(rows)
    return deltas, _summarize(deltas)


# ---------------------------------------------------------------------------
# PART 2: ABLATIONS
# ---------------------------------------------------------------------------

def run_ablations(frozen):
    train_df, test_df = load_and_split()
    y = test_df[TARGET_COLUMN].astype(int).values
    rows = []
    for m in MODEL_NAMES:
        preds = {}
        for name, (feats, arm) in ABLATIONS.items():
            preds[name] = fit_predict(train_df, test_df, feats, m, frozen[arm][m], TARGET_COLUMN)
        for name, (feats, arm) in ABLATIONS.items():
            auc, lo, hi = bootstrap_metric_ci(y, preds[name], roc_auc_score)
            d, dlo, dhi = paired_bootstrap_delta(y, preds["baseline_full"], preds[name], roc_auc_score)
            rows.append({"model": m, "feature_subset": name, "n_predictors": len(feats),
                         "hyperparams_from": arm, "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi,
                         "delta_vs_baseline": d, "delta_ci_lo": dlo, "delta_ci_hi": dhi})
        print(f"  ablations: {m} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART 3: UNTREATED ADULTS
# ---------------------------------------------------------------------------

def run_untreated(frozen):
    train_df, test_df = load_and_split()
    target = "hypertension_a"
    def keep(d):
        definitely_untreated = (d["BPQ020"] == 2) | (d["BPQ040A"] == 2) | (d["BPQ050A"] == 2)
        return d[definitely_untreated].copy()
    tr, te = keep(train_df), keep(test_df)
    y = te[target].astype(int).values
    print(f"  untreated: dev n={len(tr)} (prev {tr[target].mean():.3f}), "
          f"test n={len(te)} (prev {te[target].mean():.3f})")
    rows = []
    for m in MODEL_NAMES:
        p = {arm: fit_predict(tr, te, feats, m, frozen[arm][m], target) for arm, feats in ARMS.items()}
        auc_b = bootstrap_metric_ci(y, p["baseline"], roc_auc_score)
        auc_l = bootstrap_metric_ci(y, p["lifestyle_augmented"], roc_auc_score)
        d, lo, hi = paired_bootstrap_delta(y, p["baseline"], p["lifestyle_augmented"], roc_auc_score)
        rows.append({"model": m, "n_dev": len(tr), "n_test": len(te),
                     "prev_dev": tr[target].mean(), "prev_test": te[target].mean(),
                     "auc_baseline": auc_b[0], "auc_baseline_ci": f"[{auc_b[1]:.3f}, {auc_b[2]:.3f}]",
                     "auc_lifestyle": auc_l[0], "auc_lifestyle_ci": f"[{auc_l[1]:.3f}, {auc_l[2]:.3f}]",
                     "delta": d, "ci_lo": lo, "ci_hi": hi, "ci_includes_zero": lo <= 0 <= hi})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "trainseeds", "splits", "ablations", "untreated"], default="all")
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()

    frozen = load_frozen_hyperparameters()
    os.makedirs(TABLES_DIR, exist_ok=True)
    pd.set_option("display.width", 200)

    if args.part in ("all", "trainseeds"):
        print("\n=== Training-seed variability (original split fixed) ===")
        deltas, summary = run_train_seeds(frozen, TRAIN_SEEDS[:args.n_seeds])
        deltas.to_csv(os.path.join(TABLES_DIR, "train_seed_deltas.csv"), index=False)
        summary.to_csv(os.path.join(TABLES_DIR, "train_seed_summary.csv"), index=False)
        print(summary.round(4).to_string(index=False))

    if args.part in ("all", "splits"):
        print("\n=== EXPLORATORY repeated splits (frozen seed-42 hyperparameters; not independent validation) ===")
        deltas, summary = run_splits(frozen, DEFAULT_SEEDS[:args.n_seeds])
        deltas.to_csv(os.path.join(TABLES_DIR, "repeated_splits_deltas.csv"), index=False)
        summary.to_csv(os.path.join(TABLES_DIR, "repeated_splits_summary.csv"), index=False)
        print(summary.round(4).to_string(index=False))
        if 42 in deltas.seed.values:
            print("\n  Seed-42 check -- compare to paper Table 8/9:")
            print(deltas[deltas.seed == 42][["model", "auc_baseline", "auc_lifestyle", "delta"]].round(4).to_string(index=False))

    if args.part in ("all", "ablations"):
        print("\n=== PART 2: predictor-subset ablations (seed-42 split) ===")
        abl = run_ablations(frozen)
        abl.to_csv(os.path.join(TABLES_DIR, "ablation_results.csv"), index=False)
        print(abl.pivot(index="feature_subset", columns="model", values="auc")
              .reindex(list(ABLATIONS)).round(3).to_string())

    if args.part in ("all", "untreated"):
        print("\n=== PART 3: untreated adults, Phenotype A (seed-42 split) ===")
        un = run_untreated(frozen)
        un.to_csv(os.path.join(TABLES_DIR, "untreated_results.csv"), index=False)
        print(un.drop(columns=["n_dev", "n_test", "prev_dev", "prev_test"]).round(4).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
