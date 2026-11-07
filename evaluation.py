"""Metric computation + plotting helpers shared by train_models.py and app.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def auc_score(y_true, proba, n_classes: int) -> float:
    """ROC-AUC that also works for multi-class (one-vs-rest, macro average).

    If the evaluation file happens to miss some classes, AUC is computed over
    the classes that are present (probabilities re-normalised); with a single
    class present AUC is undefined and NaN is returned.
    """
    y_true = np.asarray(y_true)
    present = np.unique(y_true)
    if len(present) < 2:
        return float("nan")
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        if len(present) == n_classes:
            return float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                              labels=np.arange(n_classes))
            )
        sub = proba[:, present]
        sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
        if len(present) == 2:
            return float(roc_auc_score((y_true == present[1]).astype(int), sub[:, 1]))
        return float(
            roc_auc_score(y_true, sub, multi_class="ovr", average="macro", labels=present)
        )
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_pred, proba, n_classes: int, average: str = "macro") -> dict:
    """The six metrics required by the assignment."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_score(y_true, proba, n_classes),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))


def confusion_figure(cm: np.ndarray, class_names, normalize: bool = False, title: str = ""):
    """Heat-map of a confusion matrix (optionally row-normalised)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    shown = cm.astype(float)
    if normalize:
        shown = shown / np.clip(shown.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        shown, annot=True, fmt=".2f" if normalize else ".0f", cmap="YlGnBu",
        xticklabels=class_names, yticklabels=class_names, cbar=False,
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    if title:
        ax.set_title(title, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def metrics_table(rows: dict[str, dict]) -> pd.DataFrame:
    """{model name: metric dict} -> tidy DataFrame in the required column order."""
    df = pd.DataFrame(rows).T[METRIC_ORDER]
    df.index.name = "ML Model Name"
    return df
