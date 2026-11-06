"""
cohort_table.py

PURPOSE
-------
Descriptive characteristics of the analytic cohort (the 8,024
phenotype-eligible adults), overall and by hypertension status under the
primary Phenotype B and the measured-BP-only Phenotype A.

The cohort is loaded with preprocessing.load_and_split(), so the
eligibility filter and binary recodes are exactly those used for modeling;
development and test rows are recombined because this is a description of
the whole analytic cohort, not of either partition.

All summaries are unweighted (survey weights are not used anywhere in this
study), so they describe the analytic cohort, not the U.S. population.
Percentages use non-missing values as the denominator; the "Missing"
column counts missing values in the overall cohort.

USAGE
-----
    python src/cohort_table.py

OUTPUT
------
    outputs/tables/cohort_characteristics.csv
    paper/appendix_tables/cohort_characteristics.tex
"""

import os
import re
import pandas as pd

from preprocessing import load_and_split

ROOT = os.path.join(os.path.dirname(__file__), "..")
TABLES_DIR = os.path.join(ROOT, "outputs", "tables")
PAPER_DIR = os.path.join(ROOT, "paper", "appendix_tables")

RACE_LABELS = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
               4: "Non-Hispanic Black", 6: "Non-Hispanic Asian",
               7: "Other/multiracial"}
EDUC_LABELS = {1: "Less than 9th grade", 2: "9th--11th grade",
               3: "High school/GED", 4: "Some college/AA", 5: "College graduate+"}
STATUS_LABELS = {"current": "Current", "former": "Former", "never": "Never"}


def group_columns(df: pd.DataFrame) -> list:
    """(column label, subset) pairs for the table columns."""
    return [
        ("Overall", df),
        ("B: yes", df[df.hypertension_primary == 1]),
        ("B: no", df[df.hypertension_primary == 0]),
        ("A: yes", df[df.hypertension_a == 1]),
        ("A: no", df[df.hypertension_a == 0]),
    ]


def mean_sd(s: pd.Series, dp: int = 1) -> str:
    s = s.dropna()
    return f"{s.mean():.{dp}f} ({s.std():.{dp}f})"


def median_iqr(s: pd.Series, dp: int = 0) -> str:
    s = s.dropna()
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{q2:.{dp}f} [{q1:.{dp}f}, {q3:.{dp}f}]"


def n_pct(s: pd.Series, value) -> str:
    s = s.dropna()
    n = int((s == value).sum())
    return f"{n:,} ({100 * n / len(s):.1f})"


def build_rows(df: pd.DataFrame) -> list:
    """List of (section, row label, function(subset)->str, overall missing)."""
    rows = []

    def add(section, label, fn, col):
        rows.append((section, label, fn, int(df[col].isna().sum()) if col else None))

    add("Demographic/SES", "Age, years, mean (SD)", lambda d: mean_sd(d.RIDAGEYR), "RIDAGEYR")
    add("Demographic/SES", "Female, n (\\%)", lambda d: n_pct(d.is_female, 1), "is_female")
    add("Demographic/SES", "Race/ethnicity, n (\\%)", None, "RIDRETH3")
    for k, v in RACE_LABELS.items():
        add("Demographic/SES", f"\\quad {v}", lambda d, k=k: n_pct(d.RIDRETH3, k), None)
    add("Demographic/SES", "Education, n (\\%)", None, "DMDEDUC2")
    for k, v in EDUC_LABELS.items():
        add("Demographic/SES", f"\\quad {v}", lambda d, k=k: n_pct(d.DMDEDUC2, k), None)
    add("Demographic/SES", "Income-to-poverty ratio, med. [IQR]",
        lambda d: median_iqr(d.INDFMPIR, 2), "INDFMPIR")

    add("Anthropometric", "Weight, kg, mean (SD)", lambda d: mean_sd(d.BMXWT), "BMXWT")
    add("Anthropometric", "Height, cm, mean (SD)", lambda d: mean_sd(d.BMXHT), "BMXHT")
    add("Anthropometric", "BMI, kg/m$^2$, mean (SD)", lambda d: mean_sd(d.BMXBMI), "BMXBMI")
    add("Anthropometric", "Waist, cm, mean (SD)",
        lambda d: mean_sd(d.BMXWAIST), "BMXWAIST")

    add("Lifestyle", "Vigorous activity, min/wk, med. [IQR]",
        lambda d: median_iqr(d.total_vigorous_min_wk), "total_vigorous_min_wk")
    add("Lifestyle", "Moderate activity, min/wk, med. [IQR]",
        lambda d: median_iqr(d.total_moderate_min_wk), "total_moderate_min_wk")
    add("Lifestyle", "Sedentary, min/day, med. [IQR]",
        lambda d: median_iqr(d.sedentary_min_day), "sedentary_min_day")
    add("Lifestyle", "Drinking status, n (\\%)", None, "drinking_status")
    for k, v in STATUS_LABELS.items():
        add("Lifestyle", f"\\quad {v}", lambda d, k=k: n_pct(d.drinking_status, k), None)
    add("Lifestyle", "Drinks/day, med. [IQR]",
        lambda d: median_iqr(d.alq130_avg_drinks_filled, 1), "alq130_avg_drinks_filled")
    add("Lifestyle", "Smoking status, n (\\%)", None, "smoking_status")
    for k, v in STATUS_LABELS.items():
        add("Lifestyle", f"\\quad {v}", lambda d, k=k: n_pct(d.smoking_status, k), None)
    add("Lifestyle", "Recent tobacco use, n (\\%)",
        lambda d: n_pct(d.smoked_recently, 1), "smoked_recently")
    add("Lifestyle", "Weekday sleep, h, mean (SD)", lambda d: mean_sd(d.SLD012), "SLD012")
    add("Lifestyle", "Weekend sleep, h, mean (SD)", lambda d: mean_sd(d.SLD013), "SLD013")
    add("Lifestyle", "Trouble sleeping, n (\\%)",
        lambda d: n_pct(d.trouble_sleeping, 1), "trouble_sleeping")
    add("Lifestyle", "Daytime sleepiness, n (\\%)",
        lambda d: n_pct(d.SLQ120.where(d.SLQ120.isna(), d.SLQ120 >= 3), True), "SLQ120")
    return rows


def main() -> None:
    train_df, test_df = load_and_split()
    df = pd.concat([train_df, test_df])
    groups = group_columns(df)
    rows = build_rows(df)

    records = []
    for section, label, fn, missing in rows:
        rec = {"section": section, "characteristic": label}
        for name, sub in groups:
            rec[name] = fn(sub) if fn else ""
        rec["Missing"] = "" if missing is None else f"{missing:,}"
        records.append(rec)
    out = pd.DataFrame(records)
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv = out.copy()
    csv["characteristic"] = (csv.characteristic.str.replace("\\quad ", "  ", regex=False)
                             .str.replace("\\%", "%", regex=False)
                             .str.replace("$^2$", "^2", regex=False)
                             .str.replace("--", "-", regex=False))
    csv.to_csv(os.path.join(TABLES_DIR, "cohort_characteristics.csv"), index=False)

    header_n = " & ".join(f"\\bfseries {name}" for name, _ in groups)
    count_row = " & ".join(f"$n = {len(sub):,}$".replace(",", "{,}") for _, sub in groups)
    lines = [f"\\bfseries Characteristic & {header_n} & \\bfseries Missing \\\\",
             f" & {count_row} & \\\\", "\\midrule"]
    current = None
    for _, r in out.iterrows():
        if r.section != current:
            if current is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\itshape {r.section}}} \\\\")
            current = r.section
        cells = [r[name] for name, _ in groups] + [r.Missing]
        # thousands separators only (a comma followed by three digits)
        cells = [re.sub(r",(?=\d{3}\b)", "{,}", c) for c in cells]
        lines.append(f"{r.characteristic} & " + " & ".join(cells) + " \\\\")

    caption = ("Characteristics of the analytic cohort, overall and by hypertension "
               "status under the primary Phenotype B (measured $\\geq$130/80~mmHg or "
               "antihypertensive medication) and Phenotype A (measured BP only). "
               "Summaries are unweighted and describe the analytic cohort, not the "
               "U.S.\\ population. Percentages use non-missing values; ``Missing'' "
               "counts missing values in the overall cohort. Age is top-coded at 80 "
               "and the income-to-poverty ratio at 5 in the public NHANES release. "
               "Daytime sleepiness is the share reporting feeling overly sleepy "
               "often or almost always.")
    body = "\n".join([
        "\\begin{table*}[htbp]",
        "\\floatconts",
        "  {tab:cohort_characteristics}",
        f"  {{\\caption{{{caption}}}}}",
        "  {\\scriptsize\\setlength{\\tabcolsep}{2pt}\\begin{tabular}{@{}lcccccc@{}}",
        "  \\toprule",
        *[f"  {l}" for l in lines],
        "  \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table*}",
    ])
    os.makedirs(PAPER_DIR, exist_ok=True)
    with open(os.path.join(PAPER_DIR, "cohort_characteristics.tex"), "w",
              encoding="utf-8") as f:
        f.write(body + "\n")
    print(out[["characteristic", "Overall", "B: yes", "B: no", "Missing"]].to_string())


if __name__ == "__main__":
    main()
