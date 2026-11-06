# Variable Selection Record

This record documents every variable considered for the study: what was included, what was
excluded, and why. It covers the nine NHANES 2017–March 2020 modules used, variable by
variable, followed by the NHANES components that were not used.

**Included variables: 45** (10 phenotype, 5 demographic/SES, 4 anthropometric, 26 raw
lifestyle). The machine-readable list is in `src/variable_dictionary.py`. The 26 raw
lifestyle items are consolidated into 11 derived model predictors in `src/build_dataset.py`
and `src/preprocessing.py`.

---

## Part 1 — Variable-by-variable record for the 9 downloaded modules

### P_BPQ — Blood Pressure & Cholesterol Questionnaire

| Variable | Status | Reason |
|---|---|---|
| `BPQ020` | Included | Hypertension diagnosis — phenotype |
| `BPQ030` | Included | Hypertension diagnosed 2+ times — phenotype |
| `BPQ040A` | Included | Told to take BP medication — phenotype/medication status |
| `BPQ050A` | Included | Currently taking BP medication — phenotype/medication status |
| `BPD035` | Excluded | Age first told had hypertension — a timing detail, not needed for phenotype or lifestyle framing |
| `BPQ080` | Excluded | Cholesterol diagnosis — out of scope for the lifestyle-vs-baseline hypertension question |
| `BPQ060` | Excluded | Ever had cholesterol checked — cholesterol-scope, same reasoning |
| `BPQ070` | Excluded | Time since last cholesterol check — cholesterol-scope |
| `BPQ090D` | Excluded | Told to take cholesterol medication — cholesterol-scope |
| `BPQ100D` | Excluded | Currently taking cholesterol medication — cholesterol-scope |

### P_PAQ — Physical Activity Questionnaire

All 16 adult-section variables included. No exclusions — this module's full adult content maps directly onto the lifestyle-activity predictor set.

### P_DEMO — Demographics

| Variable | Status | Reason |
|---|---|---|
| `RIAGENDR` | Included | Core demographic baseline |
| `RIDAGEYR` | Included | Core demographic baseline |
| `RIDRETH3` | Included | Core demographic baseline |
| `DMDEDUC2` | Included | SES, included in the demographic/SES baseline |
| `INDFMPIR` | Included | SES, folded into demographic baseline |
| `SDDSRVYR` | Excluded | Release-cycle identifier, constant across all rows — not a feature |
| `RIDSTATR` | Excluded | Cohort-inclusion filter (handled in `build_dataset.py`), not a predictor |
| `RIDAGEMN` | Excluded | Age in months — only populated for infants 0–24mo, N/A to 18+ cohort |
| `RIDRETH1` | Excluded | Older race/ethnicity variable, superseded by `RIDRETH3` (redundant) |
| `RIDEXMON` | Excluded | Six-month exam period — data-collection artifact, not a participant characteristic |
| `DMDBORN4` | Excluded | Country of birth — not scoped into the demographic/SES baseline decision |
| `DMDYRUSZ` | Excluded | Years in US — conditional on `DMDBORN4`, same reasoning |
| `DMDMARTZ` | Excluded | Marital status — not part of the prespecified demographic/SES baseline; a candidate for future work |
| `RIDEXPRG` | Excluded | Pregnancy status — only released for women 20–44, heavy structural missingness for the rest of the cohort |
| `SIALANG`, `SIAPROXY`, `SIAINTRP`, `FIALANG`, `FIAPROXY`, `FIAINTRP`, `MIALANG`, `MIAPROXY`, `MIAINTRP`, `AIALANGA` | Excluded | Interview administration logistics (language used, proxy/interpreter flags) — not participant characteristics |
| `WTINTPRP`, `WTMECPRP` | Excluded as predictors | Survey sample weights — retained in the processed dataset but not used; estimates describe the analytic cohort rather than the weighted U.S. population |
| `SDMVPSU`, `SDMVSTRA` | Excluded as predictors | Masked variance units for complex-survey variance estimation — same caveat as sample weights above |

### P_BMX — Body Measures

| Variable | Status | Reason |
|---|---|---|
| `BMXWT` | Included | Core anthropometric baseline |
| `BMXHT` | Included | Core anthropometric baseline |
| `BMXBMI` | Included | Core anthropometric baseline |
| `BMXWAIST` | Included | Core anthropometric baseline; well-established hypertension-relevant measure |
| `BMDSTATS` | Excluded | Component status/QC flag, not a feature |
| `BMIWT`, `BMIRECUM`, `BMIHEAD`, `BMIHT`, `BMILEG`, `BMIARML`, `BMIARMC`, `BMIWAIST`, `BMIHIP` | Excluded | All "comment" codes describing measurement problems (e.g. "clothing worn," "could not obtain") — QC metadata, not real values |
| `BMXRECUM` | Excluded | Recumbent length — infants only (birth–47 months) |
| `BMXHEAD` | Excluded | Head circumference — infants only (birth–6 months) |
| `BMDBMIC` | Excluded | Child/youth BMI category — ages 2–19 only, doesn't cover full adult cohort |
| `BMXLEG`, `BMXARML`, `BMXARMC` | Excluded | Upper leg/arm length and circumference — anthropometric but not prioritized; not core to hypertension literature the way BMI/waist are |
| `BMXHIP` | Excluded | Hip circumference — would enable waist-to-hip ratio; a candidate for future work |

### P_BPXO — Blood Pressure Exam (oscillometric)

| Variable | Status | Reason |
|---|---|---|
| `BPXOSY1/2/3`, `BPXODI1/2/3` | Included | Phenotype — averaged into sys_avg/dia_avg |
| `BPAOARM` | Excluded | Which arm was used — procedural, not a feature |
| `BPAOCSZ` | Excluded | Cuff size code — procedural |
| `BPXOPLS1/2/3` | Excluded | Pulse readings — a physiological measure outside the demographic/anthropometric/lifestyle scope; a candidate for future work |

### P_ALQ — Alcohol Use

| Variable | Status | Reason |
|---|---|---|
| `ALQ111` | Included | Lifetime drinking status |
| `ALQ121` | Included | Past-12-month frequency |
| `ALQ130` | Included | Average drinks/day |
| `ALQ142`, `ALQ270`, `ALQ280`, `ALQ290` | Excluded | Binge-drinking frequency follow-ups — excluded to keep the feature set focused; revisit only if binge drinking becomes a specific research question |
| `ALQ151` | Excluded | Lifetime heavy-drinking history — same reasoning |
| `ALQ170CK` | Excluded | Not a real variable — a skip-logic check item |
| `ALQ170` | Excluded | Past-30-day binge frequency — same reasoning as the ALQ142/270/280/290 group |

### P_SLQ — Sleep Disorders

| Variable | Status | Reason |
|---|---|---|
| `SLD012`, `SLD013` | Included | Derived sleep-duration (weekday/weekend) — core sleep habit |
| `SLQ050` | Included | Diagnosed trouble sleeping |
| `SLQ120` | Included | Daytime sleepiness |
| `SLQ300`, `SLQ310`, `SLQ320`, `SLQ330` | Excluded | Raw bed/wake-time strings — superseded by the derived `SLD012`/`SLD013` |
| `SLQ030`, `SLQ040` | Excluded | Snoring / snort-or-stop-breathing symptoms — reads more as sleep-disorder (apnea) screening than a lifestyle habit |

### P_SMQ — Smoking, Cigarette Use

| Variable | Status | Reason |
|---|---|---|
| `SMQ020` | Included | Ever smoked 100+ cigarettes |
| `SMQ040` | Included | Current smoking status — codebook's own recommended pairing with SMQ020 |
| `SMD030` | Excluded | Age started smoking regularly — timing detail |
| `SMQ050Q`, `SMQ050U` | Excluded | Time since quit smoking — detail beyond current-status scope |
| `SMD057` | Excluded | Cigarettes/day when quit — same reasoning |
| `SMQ078` | Excluded | Time to first cigarette after waking — a nicotine-dependence severity measure beyond current-status scope; a candidate for future work |
| `SMD641`, `SMD650` | Excluded | Past-30-day smoking frequency/quantity — overlaps with the simpler `SMQ040` categorical |
| `SMD100FL`, `SMD100MN` | Excluded | Filter/menthol cigarette type — product detail, not behavior |
| `SMQ670` | Excluded | Tried to quit in past 12 months — a behavior-change measure beyond current-status scope |
| `SMQ621`, `SMD630` | Excluded | Youth-only lifetime cigarette count / age first smoked — ages 12–17 only, N/A to 18+ cohort |
| `SMAQUEX2` | Excluded | Questionnaire mode flag — procedural |

### P_SMQRTU — Smoking, Recent Tobacco Use

| Variable | Status | Reason |
|---|---|---|
| `SMDANY` | Included | Pre-built summary variable: any tobacco product used in past 5 days |
| `SMQ681` | Excluded | Recent smoked-tobacco flag — redundant with `SMDANY` and `P_SMQ` |
| `SMQ690A`–`SMQ690K` (product checklist) | Excluded | Per-product past-5-day use indicators — granular detail underneath `SMDANY` |
| `SMQ710/720/725/740/770/800/817/830/845/849` | Excluded | Per-product frequency/quantity/timing details |
| `SMQ851`, `SMQ863` | Excluded | Smokeless / nicotine-replacement recent use — subsumed into `SMDANY` |
| `SMAQUEX` | Excluded | Questionnaire mode flag — procedural |

---

## Part 2 — NHANES components not used

### Excluded by scope

| Component | Reason |
|---|---|
| **Dietary component** (24-hr recall / WWEIA) | Diet was treated separately from the lifestyle block; 24-hour recall data require a distinct processing pipeline |
| **Laboratory component** (cholesterol, glucose, insulin, etc.) | Outside the question of routinely available lifestyle information; fasting-subsample measures would also introduce substantial missingness |
| **P_INQ** (Income questionnaire) | Redundant with `INDFMPIR`, already sourced from `P_DEMO` |
| **Youth-only files** (`P_PAQY`, `P_ALQY_R`, and the youth-only variables inside `P_SMQ`) | Cohort is 18+, matching `P_PAQ`'s adult section |
| **Limited Access / Non-Public data** | Not used; the study relies only on publicly available files |

### Other questionnaire modules

Listed for completeness. These modules were not used; most fall outside the demographic/anthropometric/lifestyle scope:

| Component | Note |
|---|---|
| Acculturation (`P_ACQ`) | Not explored |
| Audiometry (`P_AUQ`) | Not relevant to hypertension/lifestyle |
| Cardiovascular Health (`P_CDQ`) | Angina/CVD symptom questions — arguably comorbidity-relevant, not explored |
| Consumer Behavior (`P_CBQPFA`/`P_CBQPFC`) | Not relevant |
| Current Health Status (`P_HSQ`) | Self-rated general health — not explored |
| Dermatology (`P_DEQ`) | Not relevant |
| **Diabetes (`P_DIQ`)** | Comorbidity/covariate; outside the current scope and a candidate for future work |
| Diet Behavior & Nutrition (`P_DBQ`) | Diet was treated separately from the lifestyle block |
| Early Childhood (`P_ECQ`) | N/A to adult cohort |
| Food Security (`P_FSQ`) | SES-adjacent, not selected |
| Health Insurance (`P_HIQ`) | SES-adjacent, not selected |
| Hepatitis (`P_HEQ`) | Not relevant |
| Hospital Utilization (`P_HUQ`) | Healthcare access, not lifestyle |
| Immunization (`P_IMQ`) | Not relevant |
| Kidney Conditions (`P_KIQ_U`) | Comorbidity-adjacent, not selected |
| Medical Conditions (`P_MCQ`) | Broad comorbidity questionnaire — could inform exclusion criteria, not currently used |
| **Mental Health / Depression Screener (`P_DPQ`)** | Plausible stress–hypertension pathway; outside the current scope and a candidate for future work |
| Occupation (`P_OCQ`) | SES-adjacent, not selected |
| Oral Health (`P_OHQ`) | Not relevant |
| Osteoporosis (`P_OSQ`) | Not relevant |
| Pesticide Use (`P_PUQMEC`) | Environmental exposure, out of scope |
| Prescription Medications (`P_RXQ_RX`/`RXQ_DRUG`) | Could reveal broader medication burden/polypharmacy — not explored |
| Preventive Aspirin Use (`P_RXQASA`) | Not relevant |
| Reproductive Health (`P_RHQ`) | Not relevant to hypertension/lifestyle focus |
| Weight History (`P_WHQ`/`P_WHQMEC`) | Self-reported weight history; excluded in favor of the measured `BMX` values |

---

## Candidates for future expansion

1. `BPXOPLS1/2/3` (pulse)
2. `BMXHIP` (enables waist-to-hip ratio)
3. `DMDMARTZ` (marital status)
4. `SMQ078` (nicotine dependence)
5. `P_DIQ` (diabetes status)
6. `P_DPQ` (depression screener)
