"""
variable_dictionary.py

PURPOSE
-------
Single source of truth mapping raw NHANES variable codes to:
  1. A human-readable name (for tables/figures/paper text)
  2. Which numeric codes must be recoded to NaN before modeling
     (NHANES uses "Refused" / "Don't know" sentinel codes, not blanks)
  3. Which feature category the variable belongs to (phenotype,
     demographic, anthropometric, or lifestyle) -- this grouping is what
     enables the baseline-vs-lifestyle-augmented ablation experiment.

WHY THIS FILE EXISTS SEPARATELY FROM build_dataset.py
------------------------------------------------------
This is the one file in the pipeline that requires reading actual NHANES
codebooks and making a judgment call per variable. Keeping it isolated
means: (a) it's the one place to review/audit before trusting anything
downstream, (b) build_dataset.py stays pure merge/recode logic with no
hardcoded magic numbers scattered through it.

STATUS -- READ BEFORE USING
----------------------------
ALL NINE modules VERIFIED against actual CDC codebooks: P_BPQ, P_PAQ,
P_DEMO, P_BPXO, P_BMX, P_ALQ, P_SLQ, P_SMQ, P_SMQRTU. This is the frozen
feature universe for the project (see docs/variable_selection_record.md
for the scope decision excluding Laboratory/biomarker data).

COHORT NOTE: age floor is 18+ (matches P_PAQ's adult section eligible
sample). DMDEDUC2 (education) is only asked of ages 20+, so 18-19 year-olds
will have structural (expected) missingness on that one variable -- this is
a documented, deliberate tradeoff, not a bug (18+ was kept, rather than
restricting to 20+, to match the physical-activity module's adult sample).

GENERAL NHANES MISSING-VALUE CONVENTION (confirmed pattern across every
INTERVIEW/QUESTIONNAIRE module verified so far -- always check the actual
codebook rather than assuming):
  - Single-digit categorical questions:  7 = Refused,   9 = Don't know
  - Two-digit numeric questions:        77 = Refused,   99 = Don't know
  - Four-digit numeric questions (e.g. minutes): 7777 = Refused, 9999 = Don't know
  - "." (blank) = Missing / not asked (e.g. skip pattern)

EXCEPTION -- EXAM/TECHNICIAN-MEASURED DATA (P_BPXO, P_BMX): these modules
have NO Refused/Don't know codes at all. A technician takes the
measurement; there's no one to refuse or not know. Missingness there is
only blank (".") meaning the reading wasn't obtained (e.g. participant
declined, equipment issue, physical limitation). Confirmed directly from
both codebooks below.
"""

# ---------------------------------------------------------------------------
# FEATURE CATEGORIES
# ---------------------------------------------------------------------------
# ATTENTION: These category labels are used directly by preprocessing.py to
# build the "demographic/anthropometric baseline" vs "lifestyle-augmented"
# feature sets for the incremental-value ablation experiment. If you add a
# variable below, you MUST assign it one of these categories or the ablation
# logic will silently drop it from both feature sets.
CATEGORY_PHENOTYPE = "phenotype"            # defines the hypertension outcome itself
CATEGORY_DEMOGRAPHIC = "demographic"        # age, sex, race/ethnicity, education, income
CATEGORY_ANTHROPOMETRIC = "anthropometric"  # BMI, weight, height, waist circumference
CATEGORY_LIFESTYLE = "lifestyle"            # activity, alcohol, sleep, smoking


# ---------------------------------------------------------------------------
# P_BPQ -- Blood Pressure & Cholesterol Questionnaire
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPQ.htm
# ---------------------------------------------------------------------------
# ATTENTION: These are PHENOTYPE variables, not predictors. BPQ020/BPQ030
# capture self-reported hypertension diagnosis; BPQ040A/BPQ050A capture
# antihypertensive medication use. Whether/how medication status factors
# into your phenotype definition (e.g. does someone on BP meds with
# currently-controlled measured BP count as a positive case?) is a research
# design decision to make explicitly and document -- do not let it default
# silently based on how these get merged.
BPQ_VARIABLES = {
    "BPQ020": {
        "label": "Ever told had high blood pressure",
        "category": CATEGORY_PHENOTYPE,
        "missing_codes": [7, 9],   # 7=Refused, 9=Don't know; single-digit categorical
    },
    "BPQ030": {
        "label": "Told had high blood pressure 2+ times",
        "category": CATEGORY_PHENOTYPE,
        "missing_codes": [7, 9],
    },
    "BPQ040A": {
        "label": "Taking prescription for hypertension",
        "category": CATEGORY_PHENOTYPE,
        "missing_codes": [7, 9],
    },
    "BPQ050A": {
        "label": "Now taking prescribed medicine for high blood pressure",
        "category": CATEGORY_PHENOTYPE,
        "missing_codes": [7, 9],
    },
}


# ---------------------------------------------------------------------------
# P_PAQ -- Physical Activity Questionnaire
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_PAQ.htm
# ---------------------------------------------------------------------------
# ATTENTION: Note the field-width pattern confirmed directly from this
# codebook: Yes/No items use 7/9, day-count items use 77/99, and
# minutes-per-day items use 7777/9999. This is the pattern referenced in
# the module docstring above -- P_PAQ is the module that pattern was
# confirmed against.
PAQ_VARIABLES = {
    "PAQ605": {"label": "Vigorous work activity (Y/N)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7, 9]},
    "PAQ610": {"label": "Number of days vigorous work", "category": CATEGORY_LIFESTYLE, "missing_codes": [77, 99]},
    "PAD615": {"label": "Minutes vigorous-intensity work (per typical day)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7777, 9999]},
    "PAQ620": {"label": "Moderate work activity (Y/N)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7, 9]},
    "PAQ625": {"label": "Number of days moderate work", "category": CATEGORY_LIFESTYLE, "missing_codes": [77, 99]},
    "PAD630": {"label": "Minutes moderate-intensity work (per typical day)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7777, 9999]},
    "PAQ635": {"label": "Walk or bicycle for transportation (Y/N)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7, 9]},
    "PAQ640": {"label": "Number of days walk or bicycle", "category": CATEGORY_LIFESTYLE, "missing_codes": [77, 99]},
    "PAD645": {"label": "Minutes walk/bicycle for transportation (per typical day)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7777, 9999]},
    "PAQ650": {"label": "Vigorous recreational activities (Y/N)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7, 9]},
    "PAQ655": {"label": "Days vigorous recreational activities", "category": CATEGORY_LIFESTYLE, "missing_codes": [77, 99]},
    "PAD660": {"label": "Minutes vigorous recreational activities (per typical day)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7777, 9999]},
    "PAQ665": {"label": "Moderate recreational activities (Y/N)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7, 9]},
    "PAQ670": {"label": "Days moderate recreational activities", "category": CATEGORY_LIFESTYLE, "missing_codes": [77, 99]},
    "PAD675": {"label": "Minutes moderate recreational activities (per typical day)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7777, 9999]},
    "PAD680": {"label": "Minutes sedentary activity (per typical day)", "category": CATEGORY_LIFESTYLE, "missing_codes": [7777, 9999]},
}


# ---------------------------------------------------------------------------
# P_DEMO -- Demographics
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.htm
# ---------------------------------------------------------------------------
# ATTENTION: RIDSTATR (Interview/Examination status) is NOT included here as a
# feature -- it's a cohort-inclusion filter. Only RIDSTATR=2 ("Both interviewed
# and MEC examined") should be kept in build_dataset.py's cohort construction;
# RIDSTATR=1 means interview-only, no exam data (including no BP readings) exists.
# ATTENTION: RIDAGEYR is topcoded at 80 (all ages 80+ recorded as exactly 80,
# per NHANES disclosure policy) -- worth a one-line note in the paper's cohort
# description, not a bug to fix.
DEMO_VARIABLES = {
    "RIAGENDR": {
        "label": "Gender",
        "category": CATEGORY_DEMOGRAPHIC,
        "missing_codes": [],  # no Refused/Don't know possible; interviewer-observed
        # NOTE: this is THE variable flagged in the earlier code review as being
        # fed into models as raw numeric 1/2 without proper encoding. Values are
        # 1=Male, 2=Female. preprocessing.py should treat this as categorical
        # (one-hot or binary-encode), not as a continuous numeric feature.
    },
    "RIDAGEYR": {
        "label": "Age in years at screening",
        "category": CATEGORY_DEMOGRAPHIC,
        "missing_codes": [],  # no Refused/Don't know; calculated from DOB
    },
    "RIDRETH3": {
        "label": "Race/Hispanic origin (with Non-Hispanic Asian category)",
        "category": CATEGORY_DEMOGRAPHIC,
        "missing_codes": [],
        # Categories: 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White,
        # 4=Non-Hispanic Black, 6=Non-Hispanic Asian, 7=Other/Multi-Racial.
        # Note code 5 is intentionally unused (not a data error if absent).
    },
    "DMDEDUC2": {
        "label": "Education level (Adults 20+)",
        "category": CATEGORY_DEMOGRAPHIC,  # folded into baseline, not a separate SES arm
        "missing_codes": [7, 9],  # 7=Refused, 9=Don't know
        # Only asked of ages 20+; younger participants will be NaN here by design
        # (structural missingness, not a data quality issue) -- cohort floor is
        # 18+, so this will show up for the 18-19 age band. Document, don't "fix."
    },
    "INDFMPIR": {
        "label": "Ratio of family income to poverty",
        "category": CATEGORY_DEMOGRAPHIC,
        "missing_codes": [],  # blank = missing; no Refused/Don't know sentinel
        # ATTENTION: topcoded at 5.0 (any ratio >=5.00 is recorded as exactly 5)
        # for disclosure protection -- same pattern as RIDAGEYR's age-80 topcode.
        # Worth a one-line note in the paper alongside the age topcoding mention.
    },
}


# ---------------------------------------------------------------------------
# P_BMX -- Body Measures
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.htm
# ---------------------------------------------------------------------------
# ATTENTION: Same as P_BPXO -- this is exam/technician-measured data, so
# there are NO Refused/Don't know sentinel codes. Missing values are blank
# (".") only, meaning the measurement wasn't obtained (e.g. participant used
# a wheelchair, medical appliance interfered, etc. -- see BMIWT/BMIHT/etc.
# "comment" variables in the full codebook if you ever need to distinguish
# *why* a measurement is missing; not pulled in here since they're QA
# metadata, not predictive features).
BMX_VARIABLES = {
    "BMXWT": {"label": "Weight (kg)", "category": CATEGORY_ANTHROPOMETRIC, "missing_codes": []},
    "BMXHT": {"label": "Standing height (cm)", "category": CATEGORY_ANTHROPOMETRIC, "missing_codes": []},
    "BMXBMI": {"label": "Body Mass Index (kg/m^2)", "category": CATEGORY_ANTHROPOMETRIC, "missing_codes": []},
    "BMXWAIST": {"label": "Waist circumference (cm)", "category": CATEGORY_ANTHROPOMETRIC, "missing_codes": []},
}


# ---------------------------------------------------------------------------
# P_BPXO -- Blood Pressure Exam (oscillometric)
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.htm
# ---------------------------------------------------------------------------
# ATTENTION: This is exam data (technician-measured), not interview data --
# there are NO Refused/Don't know sentinel codes here. Missing values are
# genuinely just blank ("."), meaning the reading wasn't obtained. This is
# where your sys_avg/dia_avg phenotype threshold variables come from -- three
# consecutive readings per participant, meant to be averaged (NHANES analytic
# guidance recommends averaging available readings, not just using reading 1).
BPXO_VARIABLES = {
    "BPXOSY1": {"label": "Systolic - 1st oscillometric reading", "category": CATEGORY_PHENOTYPE, "missing_codes": []},
    "BPXODI1": {"label": "Diastolic - 1st oscillometric reading", "category": CATEGORY_PHENOTYPE, "missing_codes": []},
    "BPXOSY2": {"label": "Systolic - 2nd oscillometric reading", "category": CATEGORY_PHENOTYPE, "missing_codes": []},
    "BPXODI2": {"label": "Diastolic - 2nd oscillometric reading", "category": CATEGORY_PHENOTYPE, "missing_codes": []},
    "BPXOSY3": {"label": "Systolic - 3rd oscillometric reading", "category": CATEGORY_PHENOTYPE, "missing_codes": []},
    "BPXODI3": {"label": "Diastolic - 3rd oscillometric reading", "category": CATEGORY_PHENOTYPE, "missing_codes": []},
    # BPAOARM (arm selected) and BPAOCSZ (cuff size) are deliberately excluded --
    # these are procedural/QA variables about how the measurement was taken,
    # not predictive features.
}


# ---------------------------------------------------------------------------
# P_ALQ -- Alcohol Use
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_ALQ.htm
# ---------------------------------------------------------------------------
# ATTENTION: These three variables are the standard combination used in
# NHANES-based literature to classify drinking status (never/former/
# moderate/heavy) -- ALQ111 (lifetime), ALQ121 (past-12mo frequency),
# ALQ130 (avg drinks/day). Deliberately excluded the more granular binge-
# drinking sub-questions (ALQ142, ALQ270, ALQ280, ALQ290, ALQ151, ALQ170)
# to keep the feature set focused; revisit only if binge drinking becomes
# a specific research question.
ALQ_VARIABLES = {
    "ALQ111": {
        "label": "Ever had a drink of any kind of alcohol",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [7, 9],  # single-digit categorical
    },
    "ALQ121": {
        "label": "Past 12 months: how often drink alcohol",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [77, 99],  # two-digit frequency scale (0-10 real values)
    },
    "ALQ130": {
        "label": "Avg # alcoholic drinks/day - past 12 months",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [777, 999],  # three-digit here, not four -- confirmed from codebook
    },
}

# P_SLQ -- Sleep Disorders
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SLQ.htm
# ---------------------------------------------------------------------------
# ATTENTION: SLD012/SLD013 are DERIVED variables (calculated from reported
# bed/wake times, not asked directly) -- their "missing_codes" list is
# deliberately EMPTY. Codes 2 ("less than 3 hours") and 14 ("14 hours or
# more") are real topcoded values, NOT missing-value sentinels -- do not
# add them to missing_codes or you'll silently delete valid short/long
# sleepers from the data.
SLQ_VARIABLES = {
    "SLD012": {
        "label": "Sleep hours - weekdays/workdays",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [],  # derived variable; only blank = missing (see ATTENTION above)
    },
    "SLD013": {
        "label": "Sleep hours - weekends",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [],
    },
    "SLQ050": {
        "label": "Ever told doctor had trouble sleeping",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [7, 9],
    },
    "SLQ120": {
        "label": "How often feel overly sleepy during day",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [7, 9],
    },
}

# P_SMQ -- Smoking, Cigarette Use
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SMQ.htm
# ---------------------------------------------------------------------------
# ATTENTION: SMQ020 + SMQ040 is the exact combination the codebook's own
# Analytic Notes recommend for determining current smoking status (18+):
# "combine the responses for the preliminary question (SMQ020) on '100
# cigarettes in lifetime,' with responses to the follow-up question on
# 'now smokes cigarettes' (SMQ040)."
SMQ_VARIABLES = {
    "SMQ020": {
        "label": "Smoked at least 100 cigarettes in life",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [7, 9],
    },
    "SMQ040": {
        "label": "Do you now smoke cigarettes (every day/some days/not at all)",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [7, 9],
        # Values: 1=Every day, 2=Some days, 3=Not at all. Only asked of
        # SMQ020=Yes respondents -- structural missingness for never-smokers.
    },
}

# P_SMQRTU -- Smoking, Recent Tobacco Use
# STATUS: VERIFIED against https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SMQRTU.htm
# ---------------------------------------------------------------------------
# ATTENTION: SMDANY is a pre-built NHANES summary variable ("any tobacco
# product used in past 5 days," combining smoke/smokeless/nicotine-
# replacement questions). Deliberately used instead of the ~15 granular
# per-product variables (SMQ690A-K, SMQ710-863) underneath it -- pulling
# those in would be redundant with both SMDANY and the P_SMQ cigarette
# variables above.
SMQRTU_VARIABLES = {
    "SMDANY": {
        "label": "Used any tobacco product in the last 5 days",
        "category": CATEGORY_LIFESTYLE,
        "missing_codes": [7, 9],
    },
}


# ---------------------------------------------------------------------------
# COMBINED DICTIONARY
# ---------------------------------------------------------------------------
# ATTENTION: build_dataset.py imports ALL_VARIABLES (not the individual
# per-module dicts above) as its single source of truth. Adding a new
# module: define its dict above, then add it to this merge below --
# nothing else in the pipeline needs to change.
ALL_VARIABLES = {
    **BPQ_VARIABLES,
    **PAQ_VARIABLES,
    **DEMO_VARIABLES,
    **BMX_VARIABLES,
    **BPXO_VARIABLES,
    **ALQ_VARIABLES,
    **SLQ_VARIABLES,
    **SMQ_VARIABLES,
    **SMQRTU_VARIABLES,
}


def get_variables_by_category(category: str) -> list:
    """
    Return the list of variable codes belonging to a given feature category.

    ATTENTION: This is the function preprocessing.py calls to build the
    baseline (demographic + anthropometric) vs. lifestyle-augmented feature
    sets for the incremental-value ablation -- e.g.
    get_variables_by_category(CATEGORY_LIFESTYLE) gives you exactly the
    columns to add on top of the baseline set.
    """
    return [code for code, info in ALL_VARIABLES.items()
            if info["category"] == category]


def get_missing_codes(variable_code: str) -> list:
    """
    Return the sentinel codes that must be recoded to NaN for one variable.

    ATTENTION: Called by build_dataset.py during merge/clean. Returns an
    empty list (not an error) for variables with no missing_codes entry --
    review those on purpose, since a truly code-free variable is unusual
    for NHANES *interview* data (though normal for exam/technician data
    like P_BPXO and P_BMX, per the module docstring above).
    """
    return ALL_VARIABLES.get(variable_code, {}).get("missing_codes", [])
