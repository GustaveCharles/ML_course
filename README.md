# CS-433 Project 1 — heart disease survey classification

A NumPy-only binary classifier for the course's BRFSS survey dataset. The six required regression methods are in `implementations.py`. `model.py` prepares the survey answers and evaluates predictions; `run.py` trains the chosen model and writes `submission.csv`.

## Data and requirements

Use Python 3 with NumPy. Download the course-supplied `dataset_to_release.zip` from [the CS-433 Project 1 data folder](https://github.com/epfml/ML_course/blob/main/projects/project1/data/dataset_to_release.zip) and place it at `data/dataset_to_release.zip`. The script extracts its CSV files to `data/dataset/` when needed. Alternatively, place `x_train.csv`, `y_train.csv`, and `x_test.csv` directly in `data/dataset/`. The data archive and extracted files are intentionally not committed here.

## Run

From the repository root:

```sh
python3 run.py --validate
python3 run.py
```

The validation command creates `validation_results.json`. The second command fits the final model on all labeled rows and writes `submission.csv` with the original test IDs. Neither command uploads to AIcrowd. Only the course dataset and NumPy are used for training.
