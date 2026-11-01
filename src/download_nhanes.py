"""
download_nhanes.py

PURPOSE
-------
Programmatically download the NHANES 2017-March 2020 pre-pandemic cycle files
this project needs, so the raw dataset is fully reproducible from a single
script run rather than manual point-and-click downloads.

For each NHANES module in MODULES below, this script fetches:
  1. The .xpt data file  -> data/raw/{code}.xpt
  2. The .htm codebook   -> docs/codebooks/{code}.htm

Codebooks matter as much as the data here: they define what each numeric
code means (e.g. category labels) and, critically, which codes represent
"Refused" (commonly 7/77/777...) or "Don't know" (commonly 9/99/999...).
Those must be recoded to NaN downstream in variable_dictionary.py /
build_dataset.py -- if you skip reading the codebook for a module, values
like "999 drinks per day" will silently leak into the data as if real.

Run this once to populate data/raw/ and docs/codebooks/. Safe to re-run:
existing files are skipped unless FORCE_REDOWNLOAD is set to True.

USAGE
-----
    python src/download_nhanes.py

OUTPUT
------
    data/raw/<code>.xpt        (one per module)
    docs/codebooks/<code>.htm  (one per module)
"""

import os
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# NOTE: All 2017-2020 pre-pandemic files live under this base path on CDC's
# site, using the "P_" prefix convention for this specific cycle.
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CODEBOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "codebooks")

# If True, re-download files even if they already exist locally.
FORCE_REDOWNLOAD = False

# ATTENTION: This is the single source of truth for which NHANES modules
# this project pulls in. Add/remove modules here -- do not hardcode file
# lists anywhere else in the pipeline. Each entry maps a short internal
# label to the NHANES file code (the "P_XXX" convention for this cycle).
#
# Categorized by role in the study design (see variable_dictionary.py for
# how individual variables within each module get selected/labeled):
MODULES = {
    # --- Core cohort / phenotype definition ---
    "demographics": "P_DEMO",
    "blood_pressure_exam": "P_BPXO",     # oscillometric BP exam readings
    "body_measures": "P_BMX",            # BMI, weight, height
    "bp_cholesterol_q": "P_BPQ",         # BP/cholesterol awareness, meds, diagnosis

    # --- Lifestyle predictor modules (the "lifestyle habits" scope) ---
    "physical_activity": "P_PAQ",
    "alcohol_use": "P_ALQ",
    "sleep_disorders": "P_SLQ",
    "smoking_cigarette": "P_SMQ",
    "smoking_recent_tobacco": "P_SMQRTU",
}

# ATTENTION: Modules considered but explicitly EXCLUDED from this project's
# scope (see docs/variable_selection_record.md). Left here as documentation
# so it's clear these were a deliberate decision, not an oversight:
#   - P_DBQ (Diet Behavior & Nutrition) -- "food" was scoped out separately
#     from "lifestyle" by design.
#   - Full Dietary component (24-hr recall / WWEIA) -- out of scope, far
#     larger/more complex data structure than the Questionnaire component.
#   - P_SMQFAM / P_SMQSHS (household/secondhand smoke) -- environmental
#     exposure, not the participant's own lifestyle behavior.
#   - Limited Access Data -- requires NCHS Research Data Center (RDC)
#     application; not feasible within this project's timeline.


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def _download_file(url: str, dest_path: str) -> bool:
    """
    Download a single file from `url` to `dest_path`.

    ATTENTION: Network calls can fail for many reasons (CDC server hiccup,
    rate limiting, connectivity). This function is deliberately defensive:
    it never raises on a failed download -- it prints a warning and returns
    False so the caller can decide how to handle partial failures instead
    of the whole script crashing on module #7 of 9.
    """
    if os.path.exists(dest_path) and not FORCE_REDOWNLOAD:
        print(f"  [skip] {os.path.basename(dest_path)} already exists")
        return True

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        print(f"  [ok]   {os.path.basename(dest_path)} ({len(data):,} bytes)")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  [FAIL] {url} -> {e}")
        return False


def download_module(label: str, code: str) -> dict:
    """
    Download both the .xpt data file and .htm codebook for one NHANES module.

    ATTENTION: Returns a small status dict rather than just printing, so
    download_all() can produce a final summary of what succeeded/failed --
    this matters because a silently-missing module is exactly the kind of
    thing that causes a confusing bug three files downstream in
    build_dataset.py.
    """
    xpt_url = f"{BASE_URL}/{code}.xpt"
    htm_url = f"{BASE_URL}/{code}.htm"

    xpt_dest = os.path.join(RAW_DATA_DIR, f"{code}.xpt")
    htm_dest = os.path.join(CODEBOOK_DIR, f"{code}.htm")

    print(f"[{label}] {code}")
    xpt_ok = _download_file(xpt_url, xpt_dest)
    htm_ok = _download_file(htm_url, htm_dest)

    return {"label": label, "code": code, "data_ok": xpt_ok, "codebook_ok": htm_ok}


def download_all() -> None:
    """
    Download every module listed in MODULES, then print a pass/fail summary.

    ATTENTION: This is the function you actually call. If you add a new
    module to MODULES above, you do not need to touch this function --
    it iterates the dict automatically.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CODEBOOK_DIR, exist_ok=True)

    results = []
    for label, code in MODULES.items():
        result = download_module(label, code)
        results.append(result)
        time.sleep(0.5)  # be polite to CDC's server between requests

    print("\n--- Download summary ---")
    failures = [r for r in results if not (r["data_ok"] and r["codebook_ok"])]
    if not failures:
        print(f"All {len(results)} modules downloaded successfully.")
    else:
        print(f"{len(failures)} of {len(results)} modules had a failure:")
        for r in failures:
            print(f"  - {r['label']} ({r['code']}): "
                  f"data_ok={r['data_ok']}, codebook_ok={r['codebook_ok']}")
        print("Re-run this script to retry only the missing files "
              "(existing successful downloads are skipped).")


if __name__ == "__main__":
    download_all()
