"""Small NumPy-only feature and evaluation pipeline for Project 1.

The transformations follow Lab 1 standardization and Lab 5 logistic
regression. Fitted means and standard deviations come from training rows only.
"""

import csv
from pathlib import Path

import numpy as np

from implementations import reg_logistic_regression

NUMERIC = {
    "_AGE80": (18, 80),
    "_BMI5": (10, 100),
    "PHYSHLTH": (0, 30),
    "MENTHLTH": (0, 30),
    "POORHLTH": (0, 30),
}
CATEGORICAL = {
    "SEX": (1, 2),
    "GENHLTH": (1, 2, 3, 4, 5),
    "BPHIGH4": (1, 2, 3, 4),
    "TOLDHI2": (1, 2),
    "CVDSTRK3": (1, 2),
    "CHCKIDNY": (1, 2),
    "DIABETE3": (1, 2, 3, 4),
    "DIFFWALK": (1, 2),
    "_SMOKER3": (1, 2, 3, 4),
    "EXERANY2": (1, 2),
    "INCOME2": (1, 2, 3, 4, 5, 6, 7, 8),
    "EDUCA": (1, 2, 3, 4, 5, 6),
    "HLTHPLN1": (1, 2),
    "CHECKUP1": (1, 2, 3, 4),
    "PERSDOC2": (1, 2, 3),
    "MEDCOST": (1, 2),
    "BLOODCHO": (1, 2),
    "ASTHMA3": (1, 2),
    "CHCOCNCR": (1, 2),
    "CHCCOPD1": (1, 2),
    "HAVARTH3": (1, 2),
    "ADDEPEV2": (1, 2),
    "MARITAL": (1, 2, 3, 4, 5, 6),
    "EMPLOY1": (1, 2, 3, 4, 5, 6, 7, 8),
    "RENTHOM1": (1, 2, 3),
    "VETERAN3": (1, 2),
    "BLIND": (1, 2),
    "DECIDE": (1, 2),
    "DIFFDRES": (1, 2),
    "DIFFALON": (1, 2),
    "_PACAT1": (1, 2, 3, 4),
    "BPMEDS": (1, 2),
    "CHCSCNCR": (1, 2),
    "ASTHNOW": (1, 2),
    "QLACTLM2": (1, 2),
    "USEEQUIP": (1, 2),
    "INTERNET": (1, 2),
    "SMOKE100": (1, 2),
    "USENOW3": (1, 2, 3),
    "FLUSHOT6": (1, 2),
    "PNEUVAC3": (1, 2),
    "_CHLDCNT": (1, 2, 3, 4, 5, 6),
    "_AGEG5YR": tuple(range(1, 14)),
}
GROUPS = {
    "demographic": ("_AGE80", "SEX"),
    "clinical": (
        "_BMI5",
        "PHYSHLTH",
        "MENTHLTH",
        "GENHLTH",
        "BPHIGH4",
        "TOLDHI2",
        "CVDSTRK3",
        "CHCKIDNY",
        "DIABETE3",
        "DIFFWALK",
    ),
    "lifestyle": (
        "_SMOKER3",
        "EXERANY2",
        "INCOME2",
        "EDUCA",
        "HLTHPLN1",
        "CHECKUP1",
    ),
    "additional_health": (
        "POORHLTH",
        "PERSDOC2",
        "MEDCOST",
        "BLOODCHO",
        "ASTHMA3",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "MARITAL",
        "EMPLOY1",
        "RENTHOM1",
        "VETERAN3",
        "BLIND",
        "DECIDE",
        "DIFFDRES",
        "DIFFALON",
        "_PACAT1",
    ),
    "extended_core": (
        "BPMEDS",
        "CHCSCNCR",
        "ASTHNOW",
        "QLACTLM2",
        "USEEQUIP",
        "INTERNET",
        "SMOKE100",
        "USENOW3",
        "FLUSHOT6",
        "PNEUVAC3",
        "_CHLDCNT",
        "_AGEG5YR",
    ),
}
FEATURE_NAMES = tuple(
    dict.fromkeys(name for group in GROUPS.values() for name in group)
)


def load_features(path):
    """Read only the needed columns of a supplied Project 1 CSV file."""
    path = Path(path)
    with path.open(newline="") as stream:
        header = next(csv.reader(stream))
    indices = [header.index("Id")] + [header.index(name) for name in FEATURE_NAMES]
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=indices)
    return raw[:, 0].astype(int), raw[:, 1:]


def load_labels(path):
    """Read the provided -1/+1 labels in CSV order."""
    return np.genfromtxt(path, delimiter=",", skip_header=1, usecols=1, dtype=int)


def _clean_numeric(values, name):
    """Decode BRFSS special values before train-only standardization."""
    values = values.copy()
    if name in ("PHYSHLTH", "MENTHLTH", "POORHLTH"):
        values[values == 88] = 0  # BRFSS code for zero unhealthy days
    lower, upper = NUMERIC[name]
    valid = np.isfinite(values) & (values >= lower) & (values <= upper)
    values[~valid] = np.nan
    return values


def make_design(
    train_raw,
    other_raw,
    groups=("demographic", "clinical", "lifestyle"),
    engineered=False,
):
    """Build intercept, train-standardized numeric and one-hot categorical features.

    Missing numeric values receive the training median plus a missing flag.
    Missing/unknown categories are represented by all-zero one-hot columns.
    """
    selected = set(name for group in groups for name in GROUPS[group])
    train_parts = [np.ones((len(train_raw), 1))]
    other_parts = [np.ones((len(other_raw), 1))]
    scaled = {}
    for column, name in enumerate(FEATURE_NAMES):
        if name not in selected:
            continue
        a, b = train_raw[:, column], other_raw[:, column]
        if name in NUMERIC:
            a, b = _clean_numeric(a, name), _clean_numeric(b, name)
            median = np.nanmedian(a)
            missing_a, missing_b = ~np.isfinite(a), ~np.isfinite(b)
            a = np.where(missing_a, median, a)
            b = np.where(missing_b, median, b)
            mean, std = a.mean(), a.std()
            if std == 0:
                std = 1.0
            train_parts.extend(
                ((a - mean)[:, None] / std, missing_a[:, None].astype(float))
            )
            other_parts.extend(
                ((b - mean)[:, None] / std, missing_b[:, None].astype(float))
            )
            scaled[name] = ((a - mean) / std, (b - mean) / std)
        else:
            categories = np.asarray(CATEGORICAL[name])
            train_parts.append((a[:, None] == categories).astype(float))
            other_parts.append((b[:, None] == categories).astype(float))
    if engineered:
        age_train, age_other = scaled["_AGE80"]
        bmi_train, bmi_other = scaled["_BMI5"]
        terms = [
            (age_train**2, age_other**2),
            (bmi_train**2, bmi_other**2),
            (age_train * bmi_train, age_other * bmi_other),
        ]
        for name, valid_codes in (
            ("SEX", (1,)),
            ("BPHIGH4", (1,)),
            ("TOLDHI2", (1,)),
            ("DIABETE3", (1, 2)),
            ("_SMOKER3", (1, 2)),
        ):
            column = FEATURE_NAMES.index(name)
            train_flag = np.isin(train_raw[:, column], valid_codes)
            other_flag = np.isin(other_raw[:, column], valid_codes)
            terms.append((age_train * train_flag, age_other * other_flag))
        for a, b in terms:
            mean, std = a.mean(), a.std()
            if std == 0:
                std = 1.0
            train_parts.append(((a - mean) / std)[:, None])
            other_parts.append(((b - mean) / std)[:, None])
    return np.column_stack(train_parts), np.column_stack(other_parts)


def fit_logistic(y, tx, lambda_=0.0001, iterations=300, gamma=0.5):
    """Use the Lab 5 regularized logistic method on 0/1 labels."""
    w, loss = reg_logistic_regression(
        (y == 1).astype(float), tx, lambda_, np.zeros(tx.shape[1]), iterations, gamma
    )
    return w, loss


def predict_scores(tx, w):
    """Return logistic scores; ranking is equivalent to sigmoid probability."""
    return tx @ w


def choose_f1_threshold(y, scores):
    """Choose a score cutoff maximizing F1 on training rows only."""
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    true_positives = np.cumsum(y[order] == 1)
    predicted_positives = np.arange(1, len(y) + 1)
    f1 = 2 * true_positives / (predicted_positives + np.count_nonzero(y == 1))
    is_last_tie = np.r_[ranked_scores[:-1] != ranked_scores[1:], True]
    best = np.argmax(np.where(is_last_tie, f1, -1))
    return float(ranked_scores[best])


def auc(y, scores):
    """Rank-based ROC area, with tied scores receiving half credit."""
    positive = y == 1
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(y) + 1, dtype=float)
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(y)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        ranks[start:stop] = (start + stop + 1) / 2
    n_pos = positive.sum()
    n_neg = len(y) - n_pos
    return (ranks[positive[order]].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(y, scores, threshold=0.0):
    """Accuracy, balanced accuracy, positive-class F1 and ROC AUC."""
    pred = scores >= threshold
    true = y == 1
    tp = np.count_nonzero(pred & true)
    fp = np.count_nonzero(pred & ~true)
    fn = np.count_nonzero(~pred & true)
    tn = np.count_nonzero(~pred & ~true)
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "auc": auc(y, scores),
        "recall": tp / (tp + fn),
    }


def stratified_split(y, seed=433, validation_fraction=0.2):
    """Fixed disjoint train/validation indices, stratified by label."""
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in (-1, 1):
        rows = rng.permutation(np.flatnonzero(y == label))
        cut = round(len(rows) * validation_fraction)
        validation.append(rows[:cut])
        train.append(rows[cut:])
    return np.concatenate(train), np.concatenate(validation)
