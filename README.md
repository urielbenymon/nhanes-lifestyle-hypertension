# Does Lifestyle Information Improve Hypertension Prediction?

Code for **"Does Lifestyle Information Improve Hypertension Prediction? A Leakage-Controlled Evaluation on NHANES"** (Uriel Benymon, ML4H 2026 Findings Track).

This study asks whether a block of routinely available lifestyle information (physical activity, alcohol use, smoking, and sleep) improves machine-learning prediction of hypertension beyond demographic, socioeconomic, and anthropometric predictors. It uses the NHANES 2017–March 2020 pre-pandemic cycle (adults 18+, N = 8,024) and five model families: Logistic Regression, Random Forest, Gradient Boosting, Bagged Decision Trees, and XGBoost.

**Design**

- Two feature arms are compared on the same participants and split: a baseline arm and a lifestyle-augmented arm.
- Preprocessing (one-hot encoding, median imputation, scaling, and SMOTE) is fit inside each training fold only.
- Hyperparameters are tuned by cross-validation on a 90% development set and then frozen.
- Held-out results are reported with paired bootstrap confidence intervals.
- Cross-model agreement in SHAP feature importance is analyzed, with matched-feature-space and shuffled-feature controls.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The paper's results were produced with Python 3.14 and the exact package versions pinned in `requirements.txt`. They ran on Windows 11 with an AMD Ryzen Z1 Extreme CPU and 16 GB RAM, without a GPU.

## Data

No manual download is needed. `src/download_nhanes.py` fetches every NHANES file used, plus its codebook, directly from the CDC:
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020

NHANES data are public and de-identified. Raw and processed data are not stored in this repository.

---

## Reproducing the results

Run all commands from the repository root.

### 1. Main pipeline

```bash
python src/download_nhanes.py        # raw .xpt files + codebooks from CDC
python src/build_dataset.py          # merge modules, build cohort and phenotypes A/B/C
python src/preprocessing.py          # smoke test of the fixed 90/10 split and feature sets
python src/train_pipeline.py --tune  # CV hyperparameter search on the development set; saves frozen models
python src/evaluate.py               # held-out metrics, paired bootstrap CIs, SHAP
python src/smote_sensitivity.py      # SMOTE vs. no-SMOTE, frozen hyperparameters

# Phenotype A (measured BP only) sensitivity analysis, reusing the frozen hyperparameters
python src/train_pipeline.py --target-column hypertension_a --run-label phenotype_a
python src/evaluate.py       --target-column hypertension_a --run-label phenotype_a

python src/make_figures.py           # figures + LaTeX cohort table
```

### 2. Supplementary analyses

Each of these is a separate script. None of them retunes models or overwrites the primary results.

```bash
python src/shap_agreement_controls.py                          # matched-space / predictor / domain SHAP agreement
python src/shap_agreement_controls.py --run-label phenotype_a
python src/feature_redundancy.py                               # lifestyle vs. baseline redundancy (development set only)
python src/calibration_analysis.py                             # Brier, calibration, decision curves

python src/robustness_refits.py --part splits --n-seeds 1      # seed-42 check: reproduces the main held-out AUCs
python src/robustness_refits.py --part ablations               # age-only, age+BMI, demographic-only, ... subsets
python src/robustness_refits.py --part untreated               # adults definitely not on BP medication
python src/robustness_refits.py --part trainseeds              # training-seed variability, original split fixed
python src/robustness_refits.py --part splits                  # exploratory repeated splits (see note below)

python src/shap_refit_controls.py --part shuffled              # shuffled-lifestyle SHAP control
python src/shap_refit_controls.py --part nosmote               # SHAP without SMOTE
```

**Note on repeated splits.** Hyperparameters were tuned on the original (seed-42) development set. For other seeds, part of each new test set contributed to that tuning, so these results show split sensitivity conditional on the original tuning. They are not independent validation. The `trainseeds` analysis keeps the original split fixed and is the cleaner check.

### 3. Paper tables

```bash
python src/cohort_table.py           # cohort characteristics (Appendix A)
python src/make_appendix_tables.py   # formats appendix tables from outputs/tables/ into paper/appendix_tables/
```

`outputs/untuned_primary/` holds the development-CV and held-out results from the initial run with library-default hyperparameters, before tuning. The paper cites them when describing how the test set was used.

### Approximate runtime

| Step | Time |
|---|---|
| Hyperparameter search (all models, both arms) | about 19 min |
| SHAP | about 20–30 min per run; most of it is the Bagged Decision Tree permutation explainer |
| Each refit in `robustness_refits.py` | about 20 s per seed |
| All other steps | seconds to a few minutes |

---

## Repository layout

```
src/                      pipeline and analysis scripts
outputs/tables/           all result tables (CSV) used in the paper
outputs/figures/          all figures (PDF)
outputs/shap_values/      raw SHAP arrays (300 test participants x features), per arm/model/run
docs/                     variable selection record
```

`data/`, `docs/codebooks/`, `outputs/models/`, and `paper/` are created when the scripts run and are not tracked.

### Where results come from

| Result | Script | Output file(s) in `outputs/tables/` |
|---|---|---|
| Cohort flow and phenotype prevalence | `build_dataset.py` | `cohort_flow.md`, `phenotype_breakdown.md`, `missingness_audit.csv` |
| Hyperparameter search and selected values | `train_pipeline.py --tune` | `tuning_results.csv`, `best_hyperparameters.csv` |
| Development-set CV performance | `train_pipeline.py` | `cv_results.csv` |
| Held-out performance and paired lifestyle-minus-baseline differences | `evaluate.py` | `test_metrics.csv`, `paired_comparison.csv` |
| Cross-model SHAP agreement (top-10 Jaccard, Spearman) | `evaluate.py` | `shap_importance.csv`, `shap_stability.csv`, `shap_rank_correlation.csv` |
| Phenotype A sensitivity | `train_pipeline.py` / `evaluate.py --run-label phenotype_a` | `*_phenotype_a.csv` |
| SMOTE sensitivity | `smote_sensitivity.py` | `smote_sensitivity_*.csv` |
| Matched-space, predictor-level, and domain-level SHAP agreement | `shap_agreement_controls.py` | `shap_agreement_controls*.csv`, `shap_domain_importance*.csv` |
| Lifestyle–baseline redundancy | `feature_redundancy.py` | `redundancy_*.csv` |
| Calibration and decision curves | `calibration_analysis.py` | `calibration_metrics.csv`, `calibration_paired_deltas.csv`, `decision_curve.csv` |
| Feature-subset ablations | `robustness_refits.py --part ablations` | `ablation_results.csv` |
| Untreated-adult subgroup | `robustness_refits.py --part untreated` | `untreated_results.csv` |
| Training-seed and repeated-split variability | `robustness_refits.py` | `train_seed_*.csv`, `repeated_splits_*.csv` |
| Shuffled-lifestyle and no-SMOTE SHAP controls | `shap_refit_controls.py` | `shap_shuffled_control_summary.csv`, `*_no_smote.csv` |
| Cohort characteristics | `cohort_table.py` | `cohort_characteristics.csv` |
| Library-default (untuned) results cited for test-set history | initial untuned run of `train_pipeline.py` / `evaluate.py` | `outputs/untuned_primary/` (`cv_results.csv`, `test_metrics.csv`, `paired_comparison.csv`) |

All figures are written to `outputs/figures/` by `make_figures.py`, `calibration_analysis.py`, and `feature_redundancy.py`. Appendix LaTeX tables are written to `paper/appendix_tables/` by `cohort_table.py` and `make_appendix_tables.py` (not tracked).

---

## Notes on scope

- **Prediction, not causation.** Estimates describe predictive performance within the analytic cohort. NHANES survey weights are retained in the processed data but not used, so results are not nationally representative. SHAP values describe what the models learned, not causal effects.
- **Outcome.** The primary outcome (Phenotype B) is measured BP ≥ 130/80 mmHg or current antihypertensive medication. Medication use and self-reported diagnosis define the phenotypes and are never used as predictors.
- **Variable selection.** Variable selection was prespecified. `docs/variable_selection_record.md` records the inclusion or exclusion decision for every variable considered.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@inproceedings{benymon2026does,
  title={Does Lifestyle Information Improve Hypertension Prediction? A Leakage-Controlled Evaluation on {NHANES}},
  author={Uriel Benymon},
  booktitle={Machine Learning for Health 2026},
  year={2026},
  url={https://openreview.net/forum?id=BW9QSU8cTo}
}
```

## License

MIT (see `LICENSE`).
