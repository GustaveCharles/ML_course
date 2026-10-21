# CS-433 Project 1 — heart disease survey classification

A NumPy-only binary classifier for the course's BRFSS survey dataset. The six required regression methods are in `implementations.py`. `model.py` prepares the survey answers and evaluates predictions; `run.py` trains the chosen model and writes `submission.csv`. The two-page report is in `report.pdf` with its LaTeX source in `report.tex`.

## Data and requirements

Use Python 3 with NumPy. Download the course-supplied `dataset_to_release.zip` from [the CS-433 Project 1 data folder](https://github.com/epfml/ML_course/blob/main/projects/project1/data/dataset_to_release.zip) and place it at `data/dataset_to_release.zip`. The script extracts its CSV files to `data/dataset/` when needed. Alternatively, place `x_train.csv`, `y_train.csv`, and `x_test.csv` directly in `data/dataset/`. The data archive and extracted files are intentionally not committed here.

## Run

From the repository root:

```sh
python3 run.py --validate
python3 plot_results.py
python3 run.py
```

The first command creates `validation_results.json`; the second regenerates the report figure and needs Matplotlib only for plotting. The last command fits the final model on all labeled rows and writes `submission.csv` with the original test IDs. Neither command uploads to AIcrowd. Only the course dataset and NumPy are used for training.

## Validation and features

The stratified outer split uses NumPy seed 433: 262,508 training rows and
65,627 held-out rows. An inner 80/20 split of the outer training rows, seed
619, gives 210,006 fit and 52,502 development rows. Feature sets, the L2
coefficient, and the number of updates were compared on development rows.
The final comparison against the original classifier and ridge regression
uses the outer holdout. This is one development split and one holdout, not
cross-validation. All imputation and standardization statistics are fitted
only on the corresponding fit rows. Each classification threshold maximizes
F1 on fit rows, without reading development or holdout labels.

The selected model is regularized logistic regression with L2 coefficient
0.0001, 900 full-batch updates, and step size 0.5. Its 156-column design
matrix includes the intercept. `run.py` refits it and its threshold on all
labeled rows before predicting the test rows. `--validate` reproduces the
feature and parameter comparisons in `validation_results.json`.
The revised submission received F1 0.438 and AUC 0.880 on AIcrowd; model
selection used the local development set, not these competition scores.

Numeric inputs are age, BMI, physical and mental unhealthy days, and days when
poor health limited usual activities. BRFSS code 88 means zero days for the
three day-count variables; other invalid or absent values use the
training median and a missing-value flag. Categorical variables use one-hot
encoding for explicitly listed valid values; unknown and unanswered values
yield all-zero indicators. The nested feature groups are:

| Group | Input columns |
| --- | --- |
| Demographic | `_AGE80`, `SEX` |
| Clinical | `_BMI5`, `PHYSHLTH`, `MENTHLTH`, `GENHLTH`, `BPHIGH4`, `TOLDHI2`, `CVDSTRK3`, `CHCKIDNY`, `DIABETE3`, `DIFFWALK` |
| Lifestyle and socioeconomic | `_SMOKER3`, `EXERANY2`, `INCOME2`, `EDUCA`, `HLTHPLN1`, `CHECKUP1` |
| Additional health and context | `POORHLTH`, `PERSDOC2`, `MEDCOST`, `BLOODCHO`, `ASTHMA3`, `CHCOCNCR`, `CHCCOPD1`, `HAVARTH3`, `ADDEPEV2`, `MARITAL`, `EMPLOY1`, `RENTHOM1`, `VETERAN3`, `BLIND`, `DECIDE`, `DIFFDRES`, `DIFFALON`, `_PACAT1` |
| Extended core | `BPMEDS`, `CHCSCNCR`, `ASTHNOW`, `QLACTLM2`, `USEEQUIP`, `INTERNET`, `SMOKE100`, `USENOW3`, `FLUSHOT6`, `PNEUVAC3`, `_CHLDCNT`, `_AGEG5YR` |

The selected model adds eight simple nonlinear terms: age squared, BMI
squared, age times BMI, and age interactions with sex, blood pressure,
cholesterol, diabetes, and smoking. These follow the polynomial feature idea
from Lab 3. Each term is standardized using fit rows only. `model.py`
specifies the valid codes and numeric ranges for every field. The ID column
is never used as a feature.

The committed `submission.csv` has SHA-256 `48f5d0601c40848c6cad3edc391fe55794426fdc5a7dc2ed33e7928a5a7b5e50`; a fresh `python3 run.py` with the supplied dataset produced the same bytes.

The report source uses the included `latex-example-paper/IEEEtran.cls`. To rebuild the PDF from the repository root, run `pdflatex report.tex` twice. The public grading tests expect a URL ending in `/tree/<commit SHA>` so the exact submitted revision is fixed.
