"""
build_dataset.py

PURPOSE
-------
Deterministic, participant-level dataset construction ONLY. This script:
  1. Loads all 9 downloaded NHANES modules
  2. Validates expected columns are present
  3. Merges everything on SEQN
  4. Applies cohort eligibility filters (interviewed+examined, age 18+)
  5. Recodes NHANES sentinel missing-value codes to NaN (via variable_dictionary)
  6. Derives BP averages (sys_avg/dia_avg) from the 3 oscillometric readings
  7. Derives the hypertension phenotype -- THREE versions (A/B/C) are computed
     and preserved, with hypertension_primary = Option B, so a sensitivity
     analysis (A vs B vs C) is cheap to run later without re-merging anything.
  8. Derives skip-pattern-aware constructs for smoking, alcohol, and physical
     activity, so a structural "not applicable" (e.g. never-smoker) is never
     confused with genuine missingness (e.g. refused to answer) downstream.
  9. Writes ONE analysis-ready table to data/processed/, plus diagnostic
     reports (cohort flow, missingness audit, phenotype breakdown) to
     outputs/tables/.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-------------------------------------------
No imputation, no scaling, no one-hot encoding, no SMOTE, no feature
selection, no train/test split. Everything "learned from data" belongs in
preprocessing.py's fold-fitted transformers, not here -- this script's job
ends at "one clean, fully-documented participant table." Mixing the two
responsibilities is exactly the kind of thing that causes silent leakage.

USAGE
-----
    python src/build_dataset.py

OUTPUT
------
    data/processed/analytic_dataset.csv       -- one row per eligible participant
    outputs/tables/cohort_flow.md             -- N at each filtering step
    outputs/tables/missingness_audit.csv      -- per-variable N/missing/%missing/range
    outputs/tables/phenotype_breakdown.md     -- N positive/negative under A/B/C

NOTE ON TESTING
----------------
This script has NOT been run against the real data yet (no local access to
the downloaded .xpt files at write time). Run it, then paste back the
console output / diagnostic files so we can review the actual cohort
numbers together before moving on to preprocessing.py.
"""

import os
import pandas as pd
import numpy as np

from variable_dictionary import (
    ALL_VARIABLES,
    get_missing_codes,
    get_variables_by_category,
    CATEGORY_PHENOTYPE,
    CATEGORY_DEMOGRAPHIC,
    CATEGORY_ANTHROPOMETRIC,
    CATEGORY_LIFESTYLE,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")

# ATTENTION: single source of truth for which raw file backs which module.
# Mirrors download_nhanes.py's MODULES dict -- if that changes, this must too.
MODULE_FILES = {
    "P_DEMO": "P_DEMO.xpt",
    "P_BPXO": "P_BPXO.xpt",
    "P_BMX": "P_BMX.xpt",
    "P_BPQ": "P_BPQ.xpt",
    "P_PAQ": "P_PAQ.xpt",
    "P_ALQ": "P_ALQ.xpt",
    "P_SLQ": "P_SLQ.xpt",
    "P_SMQ": "P_SMQ.xpt",
    "P_SMQRTU": "P_SMQRTU.xpt",
}

# ATTENTION: cohort-definition variables that must be pulled in even though
# they are NOT in ALL_VARIABLES (they're filters, not features -- see
# variable_dictionary.py's docstring notes on RIDSTATR).
COHORT_FILTER_COLS = ["RIDSTATR"]

# Survey-design columns: NOT used as model predictors, but preserved in the
# processed dataset so a survey-weighted sensitivity analysis remains
# possible without rebuilding the dataset.
SURVEY_DESIGN_COLS = ["WTINTPRP", "WTMECPRP", "SDMVPSU", "SDMVSTRA"]

AGE_FLOOR = 18


# ---------------------------------------------------------------------------
# STEP 1-2: LOAD + VALIDATE
# ---------------------------------------------------------------------------

def load_module(module_code: str) -> pd.DataFrame:
    """
    Load one NHANES .xpt file into a DataFrame.

    ATTENTION: pandas' built-in read_sas(format='xport') handles NHANES's
    SAS Transport files natively -- no pyreadstat dependency needed for this
    step (pyreadstat stays in requirements.txt as a fallback/alternative,
    but isn't required for the standard .xpt format NHANES uses).
    """
    path = os.path.join(RAW_DIR, MODULE_FILES[module_code])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run download_nhanes.py first."
        )
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    # NHANES SEQN is a float on read (SAS numeric); cast to int for clean merges.
    df["SEQN"] = df["SEQN"].astype(int)
    df = _fix_sas_zero_artifact(df)
    return df


def _fix_sas_zero_artifact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Snap SAS Transport's near-zero float artifact to a clean 0.0.

    ATTENTION -- REAL BUG THIS FIXES: pandas.read_sas(format='xport')
    sometimes decodes a legitimate value of 0 (e.g. ALQ121=0 "never in the
    last year", PAD680=0 minutes sedentary, INDFMPIR=0.00) as a tiny
    denormalized float (~5.397605346934028e-79) instead of exact 0.0. This
    is a known artifact of how the XPORT binary format's numeric encoding
    gets converted to IEEE754 doubles -- it is NOT a missing-value sentinel
    and NOT a data quality issue in the underlying NHANES data itself.

    Left unfixed, this silently breaks any `== 0` comparison downstream --
    confirmed via real data: it caused derive_alcohol_features()'s "former
    drinker" (ALQ121==0) branch to never fire, misclassifying every
    past-year-abstainer as a current drinker. Verified via
    `df[col] == 5.397605346934028e-79` that this affects every module with
    a 0-valued response category (P_ALQ, P_SLQ, P_SMQ, P_DEMO, P_PAQ),
    not just ALQ121 -- hence fixing it generically at load time here,
    rather than patching each downstream comparison individually.
    """
    numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns
    for col in numeric_cols:
        near_zero = df[col].abs() < 1e-6
        df.loc[near_zero, col] = 0.0
    return df


def validate_columns(df: pd.DataFrame, expected_cols: list, module_code: str) -> None:
    """
    Confirm every expected variable actually exists in the loaded file.

    ATTENTION: This is a deliberate fail-loud check. A silently-missing
    column (e.g. from a codebook typo, or NHANES renaming something between
    releases) should crash the script here, not surface three steps later as
    a confusing KeyError in preprocessing.py.
    """
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{module_code}: expected columns not found in file: {missing}"
        )


def load_all_modules() -> dict:
    """
    Load and validate all 9 modules. Returns {module_code: DataFrame}.
    """
    dfs = {}
    for module_code in MODULE_FILES:
        df = load_module(module_code)
        # Which ALL_VARIABLES entries belong to this module -- reuse the
        # same prefix-matching logic as the summary-table generator so we
        # don't hardcode two different variable->module mappings.
        expected = [code for code in ALL_VARIABLES if _module_of(code) == module_code]
        # Cohort filter cols only live in P_DEMO
        if module_code == "P_DEMO":
            expected = expected + COHORT_FILTER_COLS + SURVEY_DESIGN_COLS
        validate_columns(df, expected, module_code)
        dfs[module_code] = df
        print(f"[loaded] {module_code}: {len(df)} rows, {len(df.columns)} columns")
    return dfs


def _module_of(code: str) -> str:
    """
    ATTENTION: Mirrors the mapping used in the variable_selection_summary.md
    generator. If ALL_VARIABLES gains a variable this function doesn't
    recognize, validate_columns will simply not check for it -- so keep this
    in sync whenever variable_dictionary.py changes.
    """
    if code.startswith("BPQ"):
        return "P_BPQ"
    if code.startswith("PA"):
        return "P_PAQ"
    if code in ("RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"):
        return "P_DEMO"
    if code.startswith("BMX"):
        return "P_BMX"
    if code.startswith("BPXO"):
        return "P_BPXO"
    if code.startswith("ALQ"):
        return "P_ALQ"
    if code.startswith("SLD") or code.startswith("SLQ"):
        return "P_SLQ"
    if code.startswith("SMQ0") or code == "SMQ040":
        return "P_SMQ"
    if code == "SMDANY":
        return "P_SMQRTU"
    raise ValueError(f"Unrecognized variable code: {code}")


# ---------------------------------------------------------------------------
# STEP 3: MERGE
# ---------------------------------------------------------------------------

def merge_all_modules(dfs: dict) -> pd.DataFrame:
    """
    Merge all 9 modules on SEQN.

    ATTENTION: Uses P_DEMO as the base and LEFT-merges everything else onto
    it. This is deliberate: P_DEMO contains every NHANES participant
    (including those who weren't examined), so starting there and filtering
    down explicitly in apply_cohort_filters() keeps the "who got excluded
    and why" logic visible and auditable, rather than silently losing rows
    via an inner join before we've had a chance to count them.
    """
    merged = dfs["P_DEMO"].copy()
    for module_code, df in dfs.items():
        if module_code == "P_DEMO":
            continue
        merged = merged.merge(df, on="SEQN", how="left", validate="one_to_one")
    return merged


# ---------------------------------------------------------------------------
# STEP 4: COHORT FILTERS
# ---------------------------------------------------------------------------

def apply_cohort_filters(df: pd.DataFrame, cohort_flow: list) -> pd.DataFrame:
    """
    Apply eligibility filters, logging N at each step into cohort_flow.

    ATTENTION: Order matters here for the audit trail to read sensibly:
    start from the full P_DEMO population, then narrow by
    interview/exam status, then by age. Each step's count is recorded
    BEFORE filtering, matching the "Initial -> ... -> final" flow diagram
    format requested for the paper's cohort-characterization section.
    """
    cohort_flow.append(("Initial P_DEMO participants (all ages)", len(df)))

    df = df[df["RIDSTATR"] == 2].copy()  # both interviewed and MEC examined
    cohort_flow.append(("Interviewed AND MEC examined (RIDSTATR=2)", len(df)))

    df = df[df["RIDAGEYR"] >= AGE_FLOOR].copy()
    cohort_flow.append((f"Age >= {AGE_FLOOR}", len(df)))

    return df


# ---------------------------------------------------------------------------
# STEP 5: RECODE MISSING VALUES
# ---------------------------------------------------------------------------

def recode_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NHANES sentinel codes (Refused/Don't know) with NaN for every
    variable in ALL_VARIABLES, using variable_dictionary.get_missing_codes().

    ATTENTION: Only touches columns actually in ALL_VARIABLES -- cohort
    filter columns (RIDSTATR) and survey-design columns (WTINTPRP etc.) are
    deliberately left untouched here, since they aren't in the dictionary
    and don't have sentinel-code semantics the same way.
    """
    df = df.copy()
    for code in ALL_VARIABLES:
        if code not in df.columns:
            continue
        missing_codes = get_missing_codes(code)
        if missing_codes:
            df[code] = df[code].replace(missing_codes, np.nan)
    return df


# ---------------------------------------------------------------------------
# STEP 6: DERIVE BP AVERAGES
# ---------------------------------------------------------------------------

def derive_bp_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average available oscillometric BP readings into sys_avg / dia_avg.

    ATTENTION: Uses row-wise mean of whichever of the 3 readings are
    non-null (NHANES analytic guidance: average available readings, don't
    just use reading 1). A participant with only 1 or 2 valid readings
    still gets an average; a participant with 0 valid readings gets NaN
    (and will be excluded when the phenotype is computed, since neither
    the measured-BP nor medication condition can be evaluated without at
    least one signal).
    """
    df = df.copy()
    sys_cols = ["BPXOSY1", "BPXOSY2", "BPXOSY3"]
    dia_cols = ["BPXODI1", "BPXODI2", "BPXODI3"]
    df["sys_avg"] = df[sys_cols].mean(axis=1, skipna=True)
    df["dia_avg"] = df[dia_cols].mean(axis=1, skipna=True)
    return df


# ---------------------------------------------------------------------------
# STEP 7: DERIVE PHENOTYPE (A / B / C)
# ---------------------------------------------------------------------------

def derive_phenotype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all three phenotype definitions, with hypertension_primary = B.

    Definitions:
      A - measured only:       sys_avg >= 130 OR dia_avg >= 80
      B - measured OR meds:    A OR BPQ050A == 1 (currently on BP medication)
      C - B OR self-report:    B OR BPQ020 == 1 (ever told had hypertension)

    ATTENTION: All three columns are kept in the output specifically so a
    phenotype sensitivity analysis (A vs B vs C) is a cheap re-run later,
    not a re-merge. hypertension_primary is what the main pipeline should
    use; hypertension_a / hypertension_c are for the appendix robustness
    check.

    A row is only phenotype-eligible if sys_avg/dia_avg are non-null (i.e.
    at least one valid BP reading existed) -- rows failing this become NaN
    across all three phenotype columns rather than defaulting to negative.
    """
    df = df.copy()

    has_bp = df["sys_avg"].notna() & df["dia_avg"].notna()
    measured_positive = (df["sys_avg"] >= 130) | (df["dia_avg"] >= 80)

    df["hypertension_a"] = np.where(has_bp, measured_positive.astype("Int64"), pd.NA)

    on_medication = (df["BPQ050A"] == 1)
    df["hypertension_b"] = np.where(
        has_bp,
        (measured_positive | on_medication.fillna(False)).astype("Int64"),
        pd.NA,
    )

    self_reported = (df["BPQ020"] == 1)
    df["hypertension_c"] = np.where(
        has_bp,
        (measured_positive | on_medication.fillna(False) | self_reported.fillna(False)).astype("Int64"),
        pd.NA,
    )

    df["hypertension_primary"] = df["hypertension_b"]
    return df


# ---------------------------------------------------------------------------
# STEP 8a: SMOKING -- SKIP-PATTERN-AWARE DERIVATION
# ---------------------------------------------------------------------------

def derive_smoking_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive smoking_status categorical: 'never' / 'former' / 'current' / NaN.

    ATTENTION: SMQ040 is only asked when SMQ020==1 (skip pattern). A blank
    SMQ040 when SMQ020==2 means "never smoker" (structural), NOT "unknown
    current status." Handling this explicitly here means preprocessing.py's
    imputation never sees this as a generic missing value to guess at.

    Logic:
      SMQ020 == 2                    -> 'never'
      SMQ020 == 1 and SMQ040 in {1,2} -> 'current' (every day / some days)
      SMQ020 == 1 and SMQ040 == 3     -> 'former'
      SMQ020 is NaN (refused/DK)      -> NaN (genuine missingness, preserved)
      SMQ020 == 1 and SMQ040 is NaN   -> NaN (refused/DK on the follow-up --
                                          genuinely unknown, not structural)
    """
    df = df.copy()

    def _status(row):
        if pd.isna(row["SMQ020"]):
            return np.nan
        if row["SMQ020"] == 2:
            return "never"
        if row["SMQ020"] == 1:
            if row["SMQ040"] in (1, 2):
                return "current"
            if row["SMQ040"] == 3:
                return "former"
            return np.nan  # SMQ040 refused/DK -- genuine missingness
        return np.nan

    df["smoking_status"] = df.apply(_status, axis=1)
    return df


# ---------------------------------------------------------------------------
# STEP 8b: ALCOHOL -- SKIP-PATTERN-AWARE DERIVATION
# ---------------------------------------------------------------------------

def derive_alcohol_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive drinking_status categorical and a structurally-correct avg-drinks
    feature.

    ATTENTION: Two nested skip patterns here, both structural not missing:
      - ALQ111 == 2 (never had a drink, ever) -> section ends. ALQ121 and
        ALQ130 are both blank BY DESIGN, not unknown.
      - ALQ111 == 1 but ALQ121 == 0 (never in the last 12 months) -> ALQ130
        is blank BY DESIGN (can't report an average for a period with zero
        drinking days).

    drinking_status: 'never' / 'former' (drank lifetime, not in past year) /
    'current' / NaN.
    alq130_avg_drinks_filled: the avg-drinks value, with structural zeros
    filled in for never/former drinkers (0 is the semantically correct value
    for "average drinks on drinking days" when there were no drinking days),
    while genuine refused/DK on ALQ130 itself remains NaN.
    """
    df = df.copy()

    def _status(row):
        if pd.isna(row["ALQ111"]):
            return np.nan
        if row["ALQ111"] == 2:
            return "never"
        if row["ALQ111"] == 1:
            if pd.isna(row["ALQ121"]):
                return np.nan  # refused/DK on frequency -- genuine missingness
            if row["ALQ121"] == 0:
                return "former"
            return "current"
        return np.nan

    df["drinking_status"] = df.apply(_status, axis=1)

    def _avg_drinks(row):
        if row["drinking_status"] in ("never", "former"):
            return 0.0
        if row["drinking_status"] == "current":
            return row["ALQ130"]  # NaN here = genuine refused/DK on ALQ130 itself
        return np.nan

    df["alq130_avg_drinks_filled"] = df.apply(_avg_drinks, axis=1)
    return df


# ---------------------------------------------------------------------------
# STEP 8c: PHYSICAL ACTIVITY -- SKIP-PATTERN-AWARE DERIVATION
# ---------------------------------------------------------------------------

def _derive_activity_minutes_per_week(
    df: pd.DataFrame, yn_col: str, days_col: str, minutes_col: str, out_col: str
) -> pd.DataFrame:
    """
    Shared helper for all 5 PAQ Y/N -> days -> minutes triplets.

    ATTENTION: When the Y/N question is 'No', the days/minutes follow-ups
    are structurally not applicable -> 0 minutes/week (not missing). When
    the Y/N question itself is refused/DK (already NaN from recoding), the
    derived total is NaN (genuinely unknown, not zero).
    """
    def _compute(row):
        if pd.isna(row[yn_col]):
            return np.nan
        if row[yn_col] == 2:  # No
            return 0.0
        if row[yn_col] == 1:  # Yes
            if pd.isna(row[days_col]) or pd.isna(row[minutes_col]):
                return np.nan  # follow-up refused/DK -- genuine missingness
            return row[days_col] * row[minutes_col]
        return np.nan

    df[out_col] = df.apply(_compute, axis=1)
    return df


def derive_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive weekly-minutes summary features from the 5 PAQ Y/N->days->minutes
    triplets, plus combine into vigorous/moderate/sedentary totals.

    ATTENTION: Raw PAQ605-PAD680 columns are preserved in the output
    untouched -- these derived features are IN ADDITION, not a replacement.
    preprocessing.py / train_pipeline.py can decide whether the model
    actually consumes the raw 16 variables, these 7 derived ones, or both;
    that's a modeling choice, not something to lock in here.
    """
    df = df.copy()
    df = _derive_activity_minutes_per_week(df, "PAQ605", "PAQ610", "PAD615", "vig_work_min_wk")
    df = _derive_activity_minutes_per_week(df, "PAQ620", "PAQ625", "PAD630", "mod_work_min_wk")
    df = _derive_activity_minutes_per_week(df, "PAQ635", "PAQ640", "PAD645", "walk_bike_min_wk")
    df = _derive_activity_minutes_per_week(df, "PAQ650", "PAQ655", "PAD660", "vig_rec_min_wk")
    df = _derive_activity_minutes_per_week(df, "PAQ665", "PAQ670", "PAD675", "mod_rec_min_wk")

    # Combined totals -- NaN-safe sum (treats NaN as 0 only when at least one
    # of the two components is known; both-NaN stays NaN).
    df["total_vigorous_min_wk"] = df[["vig_work_min_wk", "vig_rec_min_wk"]].sum(
        axis=1, min_count=1
    )
    df["total_moderate_min_wk"] = df[
        ["mod_work_min_wk", "mod_rec_min_wk", "walk_bike_min_wk"]
    ].sum(axis=1, min_count=1)
    # PAD680 (sedentary minutes/day) is already a direct variable -- no
    # derivation needed beyond the sentinel-code recoding already applied.
    df["sedentary_min_day"] = df["PAD680"]

    return df


# ---------------------------------------------------------------------------
# DIAGNOSTIC OUTPUTS
# ---------------------------------------------------------------------------

def write_cohort_flow(cohort_flow: list) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, "cohort_flow.md")
    with open(path, "w") as f:
        f.write("# Cohort Flow\n\n| Step | N |\n|---|---|\n")
        for step, n in cohort_flow:
            f.write(f"| {step} | {n} |\n")
    print(f"[wrote] {path}")


def write_missingness_audit(df: pd.DataFrame) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    rows = []
    for code, info in ALL_VARIABLES.items():
        if code not in df.columns:
            continue
        col = df[code]
        n_total = len(col)
        n_missing = col.isna().sum()
        row = {
            "variable": code,
            "category": info["category"],
            "label": info["label"],
            "n_observed": n_total - n_missing,
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / n_total, 1) if n_total else np.nan,
        }
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().any():
            row["min"] = numeric.min()
            row["max"] = numeric.max()
        else:
            row["min"] = np.nan
            row["max"] = np.nan
        row["n_unique"] = col.nunique(dropna=True)
        rows.append(row)
    audit_df = pd.DataFrame(rows)
    path = os.path.join(TABLES_DIR, "missingness_audit.csv")
    audit_df.to_csv(path, index=False)
    print(f"[wrote] {path}")


def write_phenotype_breakdown(df: pd.DataFrame) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, "phenotype_breakdown.md")
    with open(path, "w") as f:
        f.write("# Phenotype Breakdown\n\n")
        for col, label in [
            ("hypertension_a", "A - measured BP only"),
            ("hypertension_b", "B - measured OR medication (PRIMARY)"),
            ("hypertension_c", "C - B OR self-reported diagnosis"),
        ]:
            n_eligible = df[col].notna().sum()
            n_pos = (df[col] == 1).sum()
            n_neg = (df[col] == 0).sum()
            prevalence = round(100 * n_pos / n_eligible, 1) if n_eligible else np.nan
            f.write(f"## {label}\n")
            f.write(f"- Phenotype-eligible N (valid BP reading): {n_eligible}\n")
            f.write(f"- Positive: {n_pos}\n")
            f.write(f"- Negative: {n_neg}\n")
            f.write(f"- Prevalence: {prevalence}%\n\n")

        # Component breakdown for hypertension_b specifically -- how many
        # positives come from measured BP vs medication-only.
        has_bp = df["sys_avg"].notna() & df["dia_avg"].notna()
        measured_pos = has_bp & ((df["sys_avg"] >= 130) | (df["dia_avg"] >= 80))
        med_only_pos = has_bp & (~measured_pos) & (df["BPQ050A"] == 1)
        f.write("## Component breakdown (within hypertension_b positives)\n")
        f.write(f"- Measured-BP positive: {measured_pos.sum()}\n")
        f.write(f"- Medication-only positive (normal measured BP, on meds): {med_only_pos.sum()}\n")
    print(f"[wrote] {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    cohort_flow = []

    print("--- Loading modules ---")
    dfs = load_all_modules()

    print("\n--- Merging on SEQN ---")
    merged = merge_all_modules(dfs)
    print(f"Merged shape: {merged.shape}")

    print("\n--- Applying cohort filters ---")
    cohort = apply_cohort_filters(merged, cohort_flow)

    print("\n--- Recoding missing-value sentinels ---")
    cohort = recode_missing_values(cohort)

    print("\n--- Deriving BP averages ---")
    cohort = derive_bp_averages(cohort)

    print("\n--- Deriving phenotype (A/B/C) ---")
    cohort = derive_phenotype(cohort)
    cohort_flow.append(("Phenotype-eligible (valid BP reading)", int(cohort["hypertension_primary"].notna().sum())))

    print("\n--- Deriving skip-pattern-aware lifestyle features ---")
    cohort = derive_smoking_status(cohort)
    cohort = derive_alcohol_features(cohort)
    cohort = derive_activity_features(cohort)

    print("\n--- Writing outputs ---")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "analytic_dataset.csv")
    cohort.to_csv(out_path, index=False)
    print(f"[wrote] {out_path} ({cohort.shape[0]} rows, {cohort.shape[1]} columns)")

    write_cohort_flow(cohort_flow)
    write_missingness_audit(cohort)
    write_phenotype_breakdown(cohort)

    print("\n--- Cohort flow summary ---")
    for step, n in cohort_flow:
        print(f"  {step}: {n}")

    return cohort


if __name__ == "__main__":
    build_dataset()
