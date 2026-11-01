"""
shap_agreement_controls.py

PURPOSE
-------
Supplementary analysis: is the lower cross-
model SHAP agreement in the lifestyle-augmented arm (Table 2) just a product
of ranking more columns (36 vs 19 encoded features)?

Uses ONLY the SHAP arrays evaluate.py already saved (outputs/shap_values/
*.npy) -- no models are loaded, no explainer is re-run. Both arms explain
the SAME 300 test participants (test_df.sample(300, random_state=42)), so
row i of every array is the same person, which makes paired resampling valid.

Analyses (each reported for baseline arm vs lifestyle-augmented arm):

  1. full        -- the paper's Table 2 (all encoded columns: 19 vs 36).
  2. shared19    -- lifestyle arm restricted to the 19 baseline columns, so
                    both arms rank the SAME 19 items. If agreement on these
                    19 still drops, the lifestyle features changed how
                    models distribute credit among baseline features; if
                    it does not, the Table 2 drop is about list length /
                    the added columns themselves.
  3. predictor   -- one-hot columns summed back to source predictors per
                    sample (as in Figure 2): 9 vs 20 predictors. Still
                    unequal-length lists -- descriptive only.
  4. domain      -- per-sample SHAP summed within domain (demographic/SES,
                    anthropometric, activity, alcohol, smoking, sleep);
                    agreement on the ordering of the 6 domains (lifestyle
                    arm only -- the baseline arm has just 2 domains).
  5. chance      -- expected top-10 Jaccard for two random rankings of
                    19 vs 36 items (Spearman's chance level is 0 for both).

AGGREGATION DEFINITION (views 3-4): SIGNED per-sample SHAP values are
summed within a predictor/domain, THEN the absolute value is averaged over
samples -- the same definition as the paper's Figure 2 / Appendix D. This
is the NET grouped contribution; opposing within-group attributions can
cancel. It is not the sum of per-column mean |SHAP|. The "share" column in
shap_domain_importance.csv uses this net definition.

The 300 explained participants are identical across all arrays BY
CONSTRUCTION: evaluate.py draws test_df.sample(300, random_state=42) from
the same seed-42 test split for every arm and model; the arrays do not
store participant IDs, so this is not re-checked here.

Uncertainty: paired bootstrap over the 300 explained participants
(resample rows, recompute mean |SHAP| per model, recompute mean pairwise
agreement in both arms), giving a 95% CI on the lifestyle-minus-baseline
difference in mean agreement. This captures sampling variability of the
explained set only -- not model-refit variability.

USAGE
-----
    python src/shap_agreement_controls.py                       # primary
    python src/shap_agreement_controls.py --run-label phenotype_a

OUTPUT
------
    outputs/tables/shap_agreement_controls.csv          (+ _phenotype_a)
    outputs/tables/shap_domain_importance.csv           (+ _phenotype_a)
"""

import os
import argparse
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
SHAP_DIR = os.path.join(ROOT, "outputs", "shap_values")
TABLES_DIR = os.path.join(ROOT, "outputs", "tables")

MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting",
               "bagged_decision_tree", "xgboost"]
ARMS = ["baseline", "lifestyle_augmented"]
TOP_N = 10
N_BOOT = 1000
SEED = 42

BASELINE_PREDICTORS = ["RIDAGEYR", "is_female", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
                       "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]
DOMAINS = {
    "demographic_ses": ["RIDAGEYR", "is_female", "RIDRETH3", "DMDEDUC2", "INDFMPIR"],
    "anthropometric": ["BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"],
    "activity": ["total_vigorous_min_wk", "total_moderate_min_wk", "sedentary_min_day"],
    "alcohol": ["drinking_status", "alq130_avg_drinks_filled"],
    "smoking": ["smoking_status", "smoked_recently"],
    "sleep": ["SLD012", "SLD013", "trouble_sleeping", "SLQ120"],
}
CATEGORICAL_PREFIXES = ["RIDRETH3", "DMDEDUC2", "drinking_status", "smoking_status"]


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def _paths(run_label):
    pre = "" if run_label == "primary" else f"{run_label}__"
    suf = "" if run_label == "primary" else f"_{run_label}"
    imp = os.path.join(TABLES_DIR, f"shap_importance{suf}.csv")
    return pre, suf, imp


def load_shap(run_label: str) -> dict:
    """{(arm, model): (array n_samples x n_features, feature_names)}"""
    pre, _, imp_path = _paths(run_label)
    imp = pd.read_csv(imp_path)
    out = {}
    for arm in ARMS:
        for m in MODEL_NAMES:
            arr = np.load(os.path.join(SHAP_DIR, f"{pre}{arm}__{m}.npy"))
            if arr.ndim == 3:   # (n, features, classes) -> positive class
                arr = arr[:, :, -1]
            names = imp[(imp.feature_set == arm) & (imp.model == m)].feature.tolist()
            assert arr.shape[1] == len(names), (arm, m, arr.shape, len(names))
            # sanity check: saved importances must match the array
            saved = imp[(imp.feature_set == arm) & (imp.model == m)].mean_abs_shap.values
            assert np.allclose(np.abs(arr).mean(0), saved, rtol=1e-4, atol=1e-8), (arm, m)
            out[(arm, m)] = (arr, names)
    return out


def source_of(col: str) -> str:
    for p in CATEGORICAL_PREFIXES:
        if col.startswith(p + "_"):
            return p
    return col


def aggregate(arr, names, groups: dict) -> tuple:
    """Sum per-sample SHAP within groups -> (n x G array, group names)."""
    cols = []
    gnames = []
    for g, members in groups.items():
        idx = [i for i, c in enumerate(names) if source_of(c) in members]
        if idx:
            cols.append(arr[:, idx].sum(axis=1))
            gnames.append(g)
    return np.column_stack(cols), gnames


def views(shap: dict) -> dict:
    """
    Build the per-sample matrices for each analysis view.
    Returns {view: {(arm, model): (matrix, names)}}.
    """
    v = {"full": {}, "shared19": {}, "predictor": {}, "domain": {}}
    base_cols = shap[("baseline", MODEL_NAMES[0])][1]
    for (arm, m), (arr, names) in shap.items():
        v["full"][(arm, m)] = (arr, names)
        idx = [names.index(c) for c in base_cols]
        v["shared19"][(arm, m)] = (arr[:, idx], base_cols)
        preds = sorted({source_of(c) for c in names}, key=lambda s: [source_of(c) for c in names].index(s))
        v["predictor"][(arm, m)] = aggregate(arr, names, {p: [p] for p in preds})
        if arm == "lifestyle_augmented":
            v["domain"][(arm, m)] = aggregate(arr, names, DOMAINS)
    return v


# ---------------------------------------------------------------------------
# AGREEMENT
# ---------------------------------------------------------------------------

def jaccard_top(a, b, k):
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(ta & tb) / len(ta | tb)


def mean_agreement(imps: dict, k: int) -> dict:
    """imps: {model: mean|SHAP| vector}. Mean pairwise Jaccard@k and Spearman."""
    js, rs = [], []
    for a, b in combinations(MODEL_NAMES, 2):
        if k is not None:
            js.append(jaccard_top(imps[a], imps[b], k))
        rs.append(spearmanr(imps[a], imps[b])[0])
    return {"jaccard": np.mean(js) if js else np.nan, "spearman": np.mean(rs)}


def importances(view_dict, arm, rows=None):
    out = {}
    for m in MODEL_NAMES:
        arr = view_dict[(arm, m)][0]
        if rows is not None:
            arr = arr[rows]
        out[m] = np.abs(arr).mean(axis=0)
    return out


def chance_jaccard(n_items: int, k: int) -> float:
    """Expected Jaccard of two independent random top-k sets from n items."""
    exp = 0.0
    for s in range(0, k + 1):
        p = comb(k, s) * comb(n_items - k, k - s) / comb(n_items, k)
        exp += p * s / (2 * k - s)
    return exp


def analyze(v: dict, n_rows: int) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    boot_rows = [rng.randint(0, n_rows, n_rows) for _ in range(N_BOOT)]
    results = []
    for view in ["full", "shared19", "predictor"]:
        n_items = {arm: len(v[view][(arm, MODEL_NAMES[0])][1]) for arm in ARMS}
        k = TOP_N if view != "predictor" else 5
        point = {arm: mean_agreement(importances(v[view], arm), k) for arm in ARMS}
        boots = {"jaccard": [], "spearman": []}
        for rows in boot_rows:
            b = {arm: mean_agreement(importances(v[view], arm, rows), k) for arm in ARMS}
            for met in boots:
                boots[met].append(b["lifestyle_augmented"][met] - b["baseline"][met])
        for met in ["jaccard", "spearman"]:
            d = np.array(boots[met])
            results.append({
                "view": view, "metric": f"top{k}_jaccard" if met == "jaccard" else "spearman",
                "n_items_baseline": n_items["baseline"],
                "n_items_lifestyle": n_items["lifestyle_augmented"],
                "baseline": point["baseline"][met],
                "lifestyle_augmented": point["lifestyle_augmented"][met],
                "delta": point["lifestyle_augmented"][met] - point["baseline"][met],
                "delta_ci_lo": np.percentile(d, 2.5), "delta_ci_hi": np.percentile(d, 97.5),
                "chance_baseline": chance_jaccard(n_items["baseline"], k) if met == "jaccard" else 0.0,
                "chance_lifestyle": chance_jaccard(n_items["lifestyle_augmented"], k) if met == "jaccard" else 0.0,
            })
    # domain view: lifestyle arm only, Spearman over 6 domains
    imps = importances(v["domain"], "lifestyle_augmented")
    dom = mean_agreement(imps, None)
    bs = []
    for rows in boot_rows:
        bs.append(mean_agreement(importances(v["domain"], "lifestyle_augmented", rows), None)["spearman"])
    results.append({
        "view": "domain", "metric": "spearman", "n_items_baseline": np.nan,
        "n_items_lifestyle": len(DOMAINS), "baseline": np.nan,
        "lifestyle_augmented": dom["spearman"], "delta": np.nan,
        "delta_ci_lo": np.percentile(bs, 2.5), "delta_ci_hi": np.percentile(bs, 97.5),
        "chance_baseline": np.nan, "chance_lifestyle": 0.0,
    })
    return pd.DataFrame(results)


def domain_table(v: dict) -> pd.DataFrame:
    rows = []
    for m in MODEL_NAMES:
        arr, names = v["domain"][("lifestyle_augmented", m)]
        imp = np.abs(arr).mean(axis=0)
        share = imp / imp.sum()
        order = np.argsort(-imp)
        for rank, i in enumerate(order, 1):
            rows.append({"model": m, "domain": names[i], "net_mean_abs_shap": imp[i],
                         "net_share": share[i], "rank": rank})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-label", default="primary")
    args = ap.parse_args()
    _, suf, _ = _paths(args.run_label)

    shap = load_shap(args.run_label)
    n_rows = shap[("baseline", MODEL_NAMES[0])][0].shape[0]
    v = views(shap)

    res = analyze(v, n_rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    res.to_csv(os.path.join(TABLES_DIR, f"shap_agreement_controls{suf}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"--- SHAP agreement controls [{args.run_label}] (n explained = {n_rows}) ---")
    print(res.round(3).to_string(index=False))

    dt = domain_table(v)
    dt.to_csv(os.path.join(TABLES_DIR, f"shap_domain_importance{suf}.csv"), index=False)
    print("\n--- Domain-level importance (lifestyle arm) ---")
    print(dt.pivot(index="domain", columns="model", values="rank").to_string())
    print(dt.groupby("domain").net_share.mean().sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
