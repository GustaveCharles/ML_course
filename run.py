"""Reproduce Project 1 validation results or the AIcrowd submission CSV."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from helpers import create_csv_submission
from implementations import ridge_regression
from model import (
    FEATURE_NAMES,
    choose_f1_threshold,
    fit_logistic,
    load_features,
    load_labels,
    make_design,
    metrics,
    predict_scores,
    stratified_split,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset"
ARCHIVE = ROOT / "data" / "dataset_to_release.zip"
BASE_GROUPS = ("demographic", "clinical", "lifestyle")
FINAL_GROUPS = BASE_GROUPS + ("additional_health", "extended_core")
FINAL_LAMBDA = 0.0001
FINAL_ITERATIONS = 900


def ensure_data():
    """Use the supplied CSVs, extracting the course archive when needed."""
    if not (DATA / "x_train.csv").exists():
        if not ARCHIVE.exists():
            raise FileNotFoundError(
                "Place the CSVs in data/dataset/ or dataset_to_release.zip in data/"
            )
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(ROOT / "data")


def validate(x, y):
    """Select features on development rows, then check one fixed holdout."""
    train, validation = stratified_split(y, seed=433)
    fit_relative, dev_relative = stratified_split(y[train], seed=619)
    fit_rows, dev_rows = train[fit_relative], train[dev_relative]
    results = {
        "outer_seed": 433,
        "development_seed": 619,
        "fit_rows": len(fit_rows),
        "development_rows": len(dev_rows),
        "holdout_rows": len(validation),
        "positive_fraction": float(np.mean(y == 1)),
        "development": {},
        "holdout": {},
    }
    candidates = (
        ("baseline", BASE_GROUPS, False, 0.0001, 300),
        ("nonlinear", BASE_GROUPS, True, 0.0001, 300),
        ("additional_health", FINAL_GROUPS[:-1], False, 0.0001, 300),
        ("additional_health_nonlinear", FINAL_GROUPS[:-1], True, 0.0001, 300),
        ("extended", FINAL_GROUPS, False, 0.0001, 300),
        ("extended_nonlinear", FINAL_GROUPS, True, 0.0001, 300),
        ("extended_long", FINAL_GROUPS, True, 0.0001, 900),
        ("extended_stronger_l2", FINAL_GROUPS, True, 0.001, 900),
        ("extended_weaker_l2", FINAL_GROUPS, True, 0.00001, 900),
    )
    for name, groups, engineered, lambda_, iterations in candidates:
        tx_fit, tx_dev = make_design(x[fit_rows], x[dev_rows], groups, engineered)
        w, loss = fit_logistic(y[fit_rows], tx_fit, lambda_, iterations)
        threshold = choose_f1_threshold(y[fit_rows], predict_scores(tx_fit, w))
        results["development"][name] = {
            "features": tx_fit.shape[1],
            "lambda": lambda_,
            "iterations": iterations,
            "train_log_loss": float(loss),
            "threshold": threshold,
            **metrics(y[dev_rows], predict_scores(tx_dev, w), threshold),
        }
        del tx_fit, tx_dev, w

    tx_fit, tx_dev = make_design(x[fit_rows], x[dev_rows], FINAL_GROUPS, True)
    ridge_development = {}
    for lambda_ in (0.001, 0.01, 0.1):
        w, loss = ridge_regression((y[fit_rows] == 1).astype(float), tx_fit, lambda_)
        threshold = choose_f1_threshold(y[fit_rows], tx_fit @ w)
        ridge_development[str(lambda_)] = {
            "train_mse": float(loss),
            "threshold": threshold,
            **metrics(y[dev_rows], tx_dev @ w, threshold),
        }
    results["ridge_development"] = ridge_development
    best_ridge_lambda = max(
        ridge_development, key=lambda value: ridge_development[value]["f1"]
    )
    del tx_fit, tx_dev

    y_val = y[validation]
    results["holdout"]["majority"] = metrics(y_val, np.full(len(y_val), -1.0))
    tx_train, tx_val = make_design(x[train], x[validation], BASE_GROUPS)
    w, loss = fit_logistic(y[train], tx_train)
    threshold = choose_f1_threshold(y[train], predict_scores(tx_train, w))
    results["holdout"]["baseline"] = {
        "features": tx_train.shape[1],
        "train_log_loss": float(loss),
        "threshold": threshold,
        **metrics(y_val, predict_scores(tx_val, w), threshold),
    }
    del tx_train, tx_val, w

    tx_train, tx_val = make_design(x[train], x[validation], FINAL_GROUPS, True)
    ridge_w, ridge_loss = ridge_regression(
        (y[train] == 1).astype(float), tx_train, float(best_ridge_lambda)
    )
    ridge_threshold = choose_f1_threshold(y[train], tx_train @ ridge_w)
    results["holdout"]["ridge_same_features"] = {
        "lambda": float(best_ridge_lambda),
        "features": tx_train.shape[1],
        "train_mse": float(ridge_loss),
        "threshold": ridge_threshold,
        **metrics(y_val, tx_val @ ridge_w, ridge_threshold),
    }
    w, loss = fit_logistic(y[train], tx_train, FINAL_LAMBDA, FINAL_ITERATIONS)
    threshold = choose_f1_threshold(y[train], predict_scores(tx_train, w))
    scores = predict_scores(tx_val, w)
    selected = {
        "features": tx_train.shape[1],
        "lambda": FINAL_LAMBDA,
        "iterations": FINAL_ITERATIONS,
        "train_log_loss": float(loss),
        "threshold": threshold,
        **metrics(y_val, scores, threshold),
        "fixed_half_probability": metrics(y_val, scores, threshold=0.0),
        "age_subgroups": {},
    }
    order = np.argsort(-scores, kind="mergesort")
    true_positives = np.cumsum(y_val[order] == 1)
    sample = np.unique(np.linspace(0, len(scores) - 1, 251, dtype=int))
    positive = y_val == 1
    chosen = scores >= threshold
    default = scores >= 0.0
    selected["precision_recall_curve"] = {
        "recall": np.r_[0.0, true_positives[sample] / positive.sum()].tolist(),
        "precision": np.r_[1.0, true_positives[sample] / (sample + 1)].tolist(),
        "chosen_precision": float(np.count_nonzero(chosen & positive) / chosen.sum()),
        "chosen_recall": selected["recall"],
        "default_precision": float(
            np.count_nonzero(default & positive) / default.sum()
        ),
        "default_recall": selected["fixed_half_probability"]["recall"],
    }
    age = x[validation, FEATURE_NAMES.index("_AGE80")]
    for label, mask in (
        ("under_50", age < 50),
        ("50_to_64", (age >= 50) & (age < 65)),
        ("65_and_over", age >= 65),
    ):
        selected["age_subgroups"][label] = {
            "rows": int(mask.sum()),
            "positive_fraction": float(np.mean(y_val[mask] == 1)),
            **metrics(y_val[mask], scores[mask], threshold),
        }
    results["holdout"]["selected"] = selected
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate", action="store_true", help="run the fixed holdout study"
    )
    args = parser.parse_args()
    ensure_data()
    _, x_train = load_features(DATA / "x_train.csv")
    y_train = load_labels(DATA / "y_train.csv")
    if len(x_train) != len(y_train):
        raise ValueError("Training features and labels differ in length")

    if args.validate:
        results = validate(x_train, y_train)
        path = ROOT / "validation_results.json"
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        return

    test_ids, x_test = load_features(DATA / "x_test.csv")
    tx_train, tx_test = make_design(x_train, x_test, FINAL_GROUPS, True)
    w, loss = fit_logistic(y_train, tx_train, FINAL_LAMBDA, FINAL_ITERATIONS)
    threshold = choose_f1_threshold(y_train, predict_scores(tx_train, w))
    predictions = np.where(predict_scores(tx_test, w) >= threshold, 1, -1)
    output = ROOT / "submission.csv"
    create_csv_submission(test_ids, predictions, output)
    print(
        f"Wrote {len(predictions)} predictions to {output} "
        f"(training loss {loss:.6f}, logit threshold {threshold:.6f})"
    )


if __name__ == "__main__":
    main()
