"""
preprocessing.py

PURPOSE
-------
Defines the modeling feature sets and transformation LOGIC for this project.
Deliberately does NOT fit anything on the full dataset -- per the
build_dataset.py / preprocessing.py / train_pipeline.py responsibility
split:

  build_dataset.py   -> deterministic participant-level data construction
  preprocessing.py   -> feature-set definitions + UNFIT transformer factory
  train_pipeline.py  -> imputation, scaling, SMOTE, model fitting -- all
                         fold-fitted inside cross-validation

This file's two jobs:
  1. Decide, explicitly and documented, which columns the models actually
     receive (see "RAW VS DERIVED LIFESTYLE FEATURES" below -- this is a
     real modeling decision, not a mechanical pass-through of everything
     in variable_dictionary.py).
  2. Provide a factory function that builds a fresh, UNFIT sklearn
     ColumnTransformer on demand -- train_pipeline.py calls this factory
     once per CV fold, so nothing here is ever fit on data that later
     leaks into a validation/test split.

RAW VS DERIVED LIFESTYLE FEATURES -- read before changing MODELING_* lists
-----------------------------------------------------------------------
variable_dictionary.py's 26 raw lifestyle variables include structurally
redundant triplets (e.g. PAQ605 Y/N -> PAQ610 days -> PAD615 minutes are
three columns describing one underlying construct, with PAQ610/PAD615
NaN whenever PAQ605=No by design). Feeding all three into a model,
alongside the derived totals build_dataset.py already computed, would be
redundant, would complicate SHAP interpretation (three correlated columns
"explaining" one behavior), and would reintroduce the exact
structural-missingness-vs-genuine-missingness ambiguity build_dataset.py's
skip-pattern-aware derivations were built to resolve.

DECISION: the model receives the DERIVED summary features
(total_vigorous_min_wk, smoking_status, drinking_status, etc.), not the
raw questionnaire items underneath them. Raw items remain in
data/processed/analytic_dataset.csv untouched, for audit/appendix
purposes, but are not part of MODELING_BASELINE_FEATURES /
MODELING_LIFESTYLE_FEATURES below. This cuts the lifestyle feature count
from 26 raw variables to 11 clean ones.
"""

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RANDOM_STATE = 42
TEST_SIZE = 0.10  # held-out final test set; CV happens on the remaining 90%


# ---------------------------------------------------------------------------
# MODELING FEATURE SETS
# ---------------------------------------------------------------------------
# ATTENTION: These are the lists train_pipeline.py actually uses to build
# the baseline-vs-lifestyle-augmented ablation. "Baseline" = demographic +
# anthropometric only. "Lifestyle-augmented" = baseline + lifestyle.
# get_feature_matrix() below is how you pull either set out of a dataframe.

MODELING_BASELINE_FEATURES = [
    "RIDAGEYR",     # numeric
    "is_female",    # binary, derived from RIAGENDR (see recode_binary_yn)
    "RIDRETH3",     # categorical -- one-hot
    "DMDEDUC2",     # categorical -- one-hot; NaN for ages 18-19 (structural, see build_dataset.py)
    "INDFMPIR",     # numeric
    "BMXWT",        # numeric
    "BMXHT",        # numeric
    "BMXBMI",       # numeric
    "BMXWAIST",     # numeric
]

MODELING_LIFESTYLE_FEATURES = [
    "total_vigorous_min_wk",    # numeric, derived (replaces 6 raw PAQ columns)
    "total_moderate_min_wk",    # numeric, derived (replaces 6 raw PAQ columns)
    "sedentary_min_day",        # numeric (PAD680, recoded)
    "drinking_status",          # categorical -- one-hot (never/former/current)
    "alq130_avg_drinks_filled", # numeric, structural-zero-filled
    "smoking_status",           # categorical -- one-hot (never/former/current)
    "smoked_recently",          # binary, derived from SMDANY
    "SLD012",                   # numeric, sleep hours weekdays
    "SLD013",                   # numeric, sleep hours weekends
    "trouble_sleeping",         # binary, derived from SLQ050
    "SLQ120",                   # numeric (0-4 frequency scale), daytime sleepiness
]

# ATTENTION: train_pipeline.py should build X using
# MODELING_BASELINE_FEATURES for the baseline arm and
# MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES for the
# lifestyle-augmented arm -- that pairing IS the incremental-value ablation.

CATEGORICAL_FEATURES = {"RIDRETH3", "DMDEDUC2", "drinking_status", "smoking_status"}
# Everything else in the two lists above is treated as numeric.

TARGET_COLUMN = "hypertension_primary"  # = hypertension_b, see build_dataset.py


# ---------------------------------------------------------------------------
# DETERMINISTIC RECODING (safe to do once, upfront -- not fold-dependent)
# ---------------------------------------------------------------------------
# ATTENTION: These recodes are pure lookup-table remaps with no dependence
# on the training data's distribution (unlike imputation/scaling), so
# there's no leakage risk in applying them once outside the CV loop --
# this is different from the one-hot encoder below, which IS built fresh
# per fold to stay consistent with the fold-fitted-transformer rule.

def recode_binary_yn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode 1/2-coded Yes/No NHANES variables into clearly-named 0/1 columns.

    ATTENTION -- BUG THIS FIXES (confirmed against real data): the original implementation used
    (df["COL"] == 2).astype("Int64"), which silently converts genuinely
    missing values to 0. In pandas, `NaN == 2` evaluates to False, not NaN
    -- so a person who never answered SMDANY doesn't become "unknown,"
    they become indistinguishable from someone who explicitly said "no."
    Confirmed on real data: SMDANY has 594 missing values (6.6% of the
    dataset) that were being silently coded as "did not smoke recently."
    SLQ050 has 7 missing (0.1%). RIAGENDR has 0 missing in this dataset
    (interviewer-observed, rarely if ever missing) so this specific bug
    had no practical impact there, but the fix is applied uniformly for
    correctness and to protect against any future data pull where it
    might.

    ATTENTION -- SECOND REAL BUG, FOUND DURING THE ACTUAL TUNING RUN: the
    first fix used .map({...}).astype("Int64") (pandas' nullable integer
    extension dtype, which represents missing values as pandas.NA). This
    correctly preserves missingness in isolation, but scikit-learn's
    ColumnTransformer explicitly REJECTS Int64/pandas.NA columns passed
    through as "remainder" columns -- confirmed via a real crash:
    "ValueError: ... has dtype Int64 and uses pandas.NA ... Storing this
    output in a numpy array can cause errors." This didn't surface during
    isolated testing of this function (which only checked pd.isna() on the
    output directly) -- it only appeared once the recoded columns were fed
    through an actual ColumnTransformer, during the first real tuning run.
    Lesson: dtype-sensitive fixes need an end-to-end pipeline test, not
    just a unit check on the function in isolation.

    FIX: drop the .astype("Int64") entirely. .map() applied to a column
    containing NaN already produces plain float64 with np.nan for missing
    (pandas' standard behavior -- int and NaN can't coexist in a classic
    numpy int dtype, so pandas upcasts to float automatically). float64/
    np.nan is fully compatible with ColumnTransformer and SimpleImputer,
    and preserves the exact same missingness semantics as the Int64
    version -- verified: pd.isna() and .isna().sum() counts are identical
    either way, only the dtype representation differs.
    """
    df = df.copy()
    df["is_female"] = df["RIAGENDR"].map({1: 0, 2: 1}).astype("float64")
    df["smoked_recently"] = df["SMDANY"].map({1: 1, 2: 0}).astype("float64")
    df["trouble_sleeping"] = df["SLQ050"].map({1: 1, 2: 0}).astype("float64")
    return df


# ---------------------------------------------------------------------------
# COLUMN TRANSFORMER FACTORY (fold-fitted -- call fresh per fold)
# ---------------------------------------------------------------------------

def build_column_transformer(feature_list: list) -> ColumnTransformer:
    """
    Build a FRESH, UNFIT ColumnTransformer for a given feature list.

    ATTENTION: This is a FACTORY, not a shared instance. Call it once per
    CV fold inside train_pipeline.py (e.g. `ct = build_column_transformer(
    features); ct.fit(X_train_fold)`), never reuse one already-fit
    instance across folds or across train/test -- that would leak fold
    membership information (specifically, which categories exist in which
    fold, and how many) across the split boundary the whole leakage-safe
    design exists to protect.

    Deliberately does NOT include imputation or scaling -- those are
    train_pipeline.py's responsibility, fit per-fold, per the
    file-responsibility split. This transformer's only job is one-hot
    encoding the categorical columns; numeric columns pass through
    unscaled and with NaNs intact (train_pipeline.py's imputer handles
    them next, inside the same per-fold Pipeline).

    handle_unknown='ignore' on the encoder means a category seen only in
    a validation/test fold (not in that fold's training data) becomes an
    all-zero row rather than raising an error -- appropriate here since
    our categories are small and fixed by the NHANES codebook, not
    data-driven.
    """
    categorical_cols = [c for c in feature_list if c in CATEGORICAL_FEATURES]
    numeric_cols = [c for c in feature_list if c not in CATEGORICAL_FEATURES]

    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="passthrough",  # numeric_cols pass through unchanged, in order
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# FEATURE MATRIX HELPER
# ---------------------------------------------------------------------------

def get_feature_matrix(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Select a feature list out of a dataframe. Trivial, but centralizes the
    "which columns does the model actually see" logic in one place rather
    than scattering df[[...]] selections across train_pipeline.py.
    """
    return df[feature_list].copy()


# ---------------------------------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------

def load_and_split(
    processed_path: str = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Load the analytic dataset, filter to phenotype-eligible rows, apply the
    deterministic binary recodes, and produce a stratified train/test split.

    ATTENTION -- DELIBERATELY NOT PARAMETERIZED BY TARGET COLUMN: this
    function always filters and stratifies on TARGET_COLUMN (hypertension_
    primary, phenotype B), regardless of which phenotype a caller ultimately
    analyzes downstream. This is intentional, not an oversight: for a
    phenotype sensitivity analysis (e.g. re-scoring under hypertension_a),
    the whole point is to vary ONLY the label definition -- if the
    train/test PARTICIPANT split also changed because a different column
    was used to stratify, differences in results would be confounded
    between "the phenotype changed" and "different people ended up in the
    test set." Confirmed on real data: stratifying on hypertension_a vs
    hypertension_primary with the SAME random_state still produces ~10%
    different train-set membership (725 of 7221 rows), because stratified
    splitting internally depends on the label distribution being split on.
    Filtering on TARGET_COLUMN specifically (rather than target-agnostic
    has_bp logic) is safe because all three phenotype definitions (A/B/C)
    share the exact same eligibility criterion from build_dataset.py --
    a row has a phenotype value if and only if it has a valid BP reading.

    Callers that need a specific phenotype as y (e.g. train_pipeline.py's
    --target-column flag) should pull that column from the returned
    train_df/test_df themselves -- all three phenotype columns are present
    in both, this function just controls which rows go where.

    Returns (train_df, test_df) -- full dataframes, not yet reduced to a
    specific feature list, so train_pipeline.py can call get_feature_matrix()
    with either MODELING_BASELINE_FEATURES or the lifestyle-augmented
    superset on the SAME split.
    """
    if processed_path is None:
        processed_path = os.path.join(PROCESSED_DIR, "analytic_dataset.csv")

    df = pd.read_csv(processed_path)

    n_before = len(df)
    df = df[df[TARGET_COLUMN].notna()].copy()
    n_after = len(df)
    print(f"[load_and_split] Dropped {n_before - n_after} rows with no valid "
          f"BP reading (phenotype-eligible: {n_after})")

    df = recode_binary_yn(df)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    print(f"[load_and_split] Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(f"[load_and_split] Train prevalence (primary phenotype): "
          f"{train_df[TARGET_COLUMN].mean():.3f}, "
          f"Test: {test_df[TARGET_COLUMN].mean():.3f}")

    return train_df, test_df


if __name__ == "__main__":
    # Smoke test: confirm the split runs cleanly and feature lists resolve
    # against real columns, without needing train_pipeline.py to exist yet.
    train_df, test_df = load_and_split()

    print("\n--- Feature list sanity check ---")
    missing_baseline = [c for c in MODELING_BASELINE_FEATURES if c not in train_df.columns]
    missing_lifestyle = [c for c in MODELING_LIFESTYLE_FEATURES if c not in train_df.columns]
    print(f"Missing baseline columns: {missing_baseline}")
    print(f"Missing lifestyle columns: {missing_lifestyle}")

    print("\n--- Example: building baseline vs lifestyle-augmented matrices ---")
    X_train_baseline = get_feature_matrix(train_df, MODELING_BASELINE_FEATURES)
    X_train_full = get_feature_matrix(
        train_df, MODELING_BASELINE_FEATURES + MODELING_LIFESTYLE_FEATURES
    )
    print(f"Baseline X shape: {X_train_baseline.shape}")
    print(f"Lifestyle-augmented X shape: {X_train_full.shape}")

    print("\n--- Example: fitting the column transformer on the baseline set ---")
    ct = build_column_transformer(MODELING_BASELINE_FEATURES)
    X_transformed = ct.fit_transform(X_train_baseline)
    print(f"Transformed baseline X shape: {X_transformed.shape}")
    print(f"Output feature names: {ct.get_feature_names_out().tolist()}")
