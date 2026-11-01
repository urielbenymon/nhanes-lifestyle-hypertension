"""
shap_refit_controls.py

PURPOSE
-------
Supplementary analyses that need SHAP recomputed on
REFIT models (frozen hyperparameters, no tuning). Slower than the other
supplementary scripts because of the Bagged Decision Tree permutation explainer.

  --part shuffled   Shuffled-lifestyle control. Every lifestyle column is
                    independently permuted (within dev and within test), so
                    the lifestyle-augmented arm keeps its 36 encoded columns
                    and marginal distributions but the lifestyle columns
                    carry no information about anyone. Refit the 5 lifestyle-
                    arm models (lifestyle-arm frozen hyperparameters), compute
                    SHAP on the SAME 300 test participants evaluate.py used,
                    and compare cross-model agreement against (a) the saved
                    baseline arm and (b) the saved real lifestyle arm.
                    If shuffled agreement drops as much as the real arm does,
                    that is CONSISTENT WITH (not proof of) a drop driven by
                    adding columns. One shuffle = one realization; use
                    --n-shuffles for more. Permuting columns independently
                    can create implausible combinations; exploratory only.

  --part nosmote    Refit both arms without SMOTE (frozen hyperparameters;
                    same as smote_sensitivity.py) and recompute SHAP, saved
                    under run label "no_smote" so shap_agreement_controls.py
                    can analyze it:  python src/shap_agreement_controls.py --run-label no_smote

The primary saved models and SHAP arrays are never overwritten.

USAGE
-----
    python src/shap_refit_controls.py                    # both parts
    python src/shap_refit_controls.py --part shuffled [--n-shuffles 3]
    python src/shap_refit_controls.py --part nosmote

OUTPUT
------
    outputs/shap_values/shuffled_lifestyle_s{seed}__lifestyle_augmented__{model}.npy
    outputs/tables/shap_importance_shuffled_lifestyle_s{seed}.csv
    outputs/tables/shap_shuffled_control_summary.csv
    outputs/shap_values/no_smote__{arm}__{model}.npy
    outputs/tables/shap_importance_no_smote.csv
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

from preprocessing import (
    load_and_split, get_feature_matrix,
    MODELING_BASELINE_FEATURES, MODELING_LIFESTYLE_FEATURES, TARGET_COLUMN, RANDOM_STATE,
)
from evaluate import (
    transform_for_shap, compute_shap_values, run_shap_analysis,
    SHAP_TEST_SAMPLE_SIZE, SHAP_BACKGROUND_SIZE, SHAP_DIR, TABLES_DIR, MODEL_NAMES,
)
from smote_sensitivity import (
    load_frozen_hyperparameters, build_tuned_model, build_pipeline_variant,
)
import shap_agreement_controls as sac

ARMS = {
    "baseline": MODELING_BASELINE_FEATURES,
    "lifestyle_augmented": MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES,
}
SHUFFLE_SEED = 2026


def fit(train_df, features, model_name, params, use_smote):
    model = build_tuned_model(model_name, params)
    pipe = build_pipeline_variant(features, model_name, model, use_smote=use_smote)
    return pipe.fit(get_feature_matrix(train_df, features), train_df[TARGET_COLUMN].astype(int))


def shuffle_lifestyle(df: pd.DataFrame, rng) -> pd.DataFrame:
    out = df.copy()
    for c in MODELING_LIFESTYLE_FEATURES:
        out[c] = rng.permutation(out[c].values)
    return out


# ---------------------------------------------------------------------------
# PART: SHUFFLED-LIFESTYLE CONTROL
# ---------------------------------------------------------------------------

def run_shuffled(frozen, shuffle_seed=SHUFFLE_SEED):
    os.makedirs(SHAP_DIR, exist_ok=True)
    tag = f"shuffled_lifestyle_s{shuffle_seed}"
    train_df, test_df = load_and_split()
    rng = np.random.RandomState(shuffle_seed)
    tr_s, te_s = shuffle_lifestyle(train_df, rng), shuffle_lifestyle(test_df, rng)
    feats = ARMS["lifestyle_augmented"]

    # Same participants evaluate.py explained/used as background (row order preserved
    # because shuffling changes values, not the index).
    bg = get_feature_matrix(tr_s.sample(n=min(SHAP_BACKGROUND_SIZE, len(tr_s)), random_state=RANDOM_STATE), feats)
    ex = get_feature_matrix(te_s.sample(n=min(SHAP_TEST_SAMPLE_SIZE, len(te_s)), random_state=RANDOM_STATE), feats)

    rows = []
    for m in MODEL_NAMES:
        t0 = time.time()
        pipe = fit(tr_s, feats, m, frozen["lifestyle_augmented"][m], use_smote=True)
        Xb, names = transform_for_shap(pipe, bg)
        Xe, _ = transform_for_shap(pipe, ex)
        sv = compute_shap_values(pipe, m, Xb, Xe)
        np.save(os.path.join(SHAP_DIR, f"{tag}__lifestyle_augmented__{m}.npy"), sv)
        for f, val in zip(names, np.abs(sv).mean(axis=0)):
            rows.append({"feature_set": "lifestyle_augmented", "model": m, "feature": f, "mean_abs_shap": float(val)})
        print(f"  shuffled (seed {shuffle_seed}): {m} done in {time.time() - t0:.0f}s")
    imp = pd.DataFrame(rows)
    imp.to_csv(os.path.join(TABLES_DIR, f"shap_importance_{tag}.csv"), index=False)

    # Compare: saved baseline arm, saved real lifestyle arm, shuffled lifestyle arm.
    real = sac.load_shap("primary")
    shuf = {}
    for m in MODEL_NAMES:
        arr = np.load(os.path.join(SHAP_DIR, f"{tag}__lifestyle_augmented__{m}.npy"))
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        shuf[("lifestyle_augmented", m)] = (arr, imp[imp.model == m].feature.tolist())
    base_cols = real[("baseline", MODEL_NAMES[0])][1]

    def summarize(label, dct, arm):
        full = {m: np.abs(dct[(arm, m)][0]).mean(0) for m in MODEL_NAMES}
        names = dct[(arm, MODEL_NAMES[0])][1]
        idx = [names.index(c) for c in base_cols]
        shared = {m: full[m][idx] for m in MODEL_NAMES}
        a_full = sac.mean_agreement(full, sac.TOP_N)
        a_sh = sac.mean_agreement(shared, sac.TOP_N)
        lifestyle_share = np.mean([1 - full[m][idx].sum() / full[m].sum() for m in MODEL_NAMES])
        return {"condition": label, "n_columns": len(names),
                "jaccard_full": a_full["jaccard"], "spearman_full": a_full["spearman"],
                "jaccard_shared19": a_sh["jaccard"], "spearman_shared19": a_sh["spearman"],
                "share_of_shap_on_added_columns": lifestyle_share}

    summary = pd.DataFrame([
        summarize("baseline (saved)", real, "baseline"),
        summarize("lifestyle_augmented (saved, real lifestyle)", real, "lifestyle_augmented"),
        summarize(f"lifestyle_augmented (SHUFFLED lifestyle, seed {shuffle_seed})", shuf, "lifestyle_augmented"),
    ])
    return summary


# ---------------------------------------------------------------------------
# PART: NO-SMOTE SHAP
# ---------------------------------------------------------------------------

def run_nosmote(frozen):
    train_df, test_df = load_and_split()
    models = {}
    for arm, feats in ARMS.items():
        for m in MODEL_NAMES:
            models[(arm, m)] = fit(train_df, feats, m, frozen[arm][m], use_smote=False)
    print("  no-SMOTE models fit; computing SHAP (slow step: bagged_decision_tree)...")
    imp, _ = run_shap_analysis(models, train_df, test_df, run_label="no_smote")
    imp.to_csv(os.path.join(TABLES_DIR, "shap_importance_no_smote.csv"), index=False)
    print("  [wrote] shap_importance_no_smote.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "shuffled", "nosmote"], default="all")
    ap.add_argument("--n-shuffles", type=int, default=1,
                    help="Number of independent shuffles (each ~ one lifestyle-arm SHAP run).")
    args = ap.parse_args()
    frozen = load_frozen_hyperparameters()
    pd.set_option("display.width", 200)

    if args.part in ("all", "shuffled"):
        print("\n=== Shuffled-lifestyle control ===")
        parts = []
        for k in range(args.n_shuffles):
            sm = run_shuffled(frozen, SHUFFLE_SEED + k)
            parts.append(sm if k == 0 else sm.iloc[[2]])
            print(sm.round(3).to_string(index=False))
        summary = pd.concat(parts, ignore_index=True)
        summary.to_csv(os.path.join(TABLES_DIR, "shap_shuffled_control_summary.csv"), index=False)
        print("\n--- Shuffled-control summary (one row per shuffle) ---")
        print(summary.round(3).to_string(index=False))

    if args.part in ("all", "nosmote"):
        print("\n=== No-SMOTE SHAP ===")
        run_nosmote(frozen)
        print("\n--- Agreement controls on no-SMOTE SHAP ---")
        import sys
        sys.argv = [sys.argv[0], "--run-label", "no_smote"]
        sac.main()

    print("\nDone.")


if __name__ == "__main__":
    main()
