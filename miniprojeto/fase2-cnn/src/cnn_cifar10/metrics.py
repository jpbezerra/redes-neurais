"""Métricas de avaliação: globais e por classe."""

from __future__ import annotations

import numpy as np
from sklearn import metrics


def get_scores(targets, predictions) -> dict[str, float]:
    """Métricas globais pedidas no enunciado: acurácia, precision, recall (+ f1)."""
    return {
        "accuracy": metrics.accuracy_score(targets, predictions),
        "balanced_accuracy": metrics.balanced_accuracy_score(targets, predictions),
        "precision": metrics.precision_score(targets, predictions, average="weighted", zero_division=0),
        "recall": metrics.recall_score(targets, predictions, average="weighted", zero_division=0),
        "f1_score": metrics.f1_score(targets, predictions, average="weighted", zero_division=0),
    }


def get_per_class_accuracy(targets, predictions, class_names: list[str] | None = None) -> dict[str, float]:
    """Taxa de acerto (recall) por classe, conforme pedido no enunciado."""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    classes = np.unique(targets)
    result = {}
    for c in classes:
        mask = targets == c
        acc = (predictions[mask] == targets[mask]).mean() if mask.sum() > 0 else float("nan")
        label = class_names[c] if class_names is not None else str(c)
        result[label] = float(acc)
    return result


def classification_report_dict(targets, predictions, class_names: list[str] | None = None) -> dict:
    """Relatório completo (precision/recall/f1 por classe) como dict, útil para logar/salvar."""
    return metrics.classification_report(
        targets, predictions, target_names=class_names, output_dict=True, zero_division=0
    )
