"""Gradient Boosting (sklearn) e XGBoost — baseline nao-neural de comparacao,
exigido desde o inicio junto com o MLP (secao de modelos do enunciado).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class GradientBoostingHyperparams:
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    criterion: str = "friedman_mse"
    min_samples_leaf: int = 1
    max_depth: int = 3
    seed: int = 0


def treinar_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray,
                               hp: GradientBoostingHyperparams) -> GradientBoostingClassifier:
    modelo = GradientBoostingClassifier(
        loss="log_loss",  # equivalente a 'deviance' nas versoes recentes do sklearn
        learning_rate=hp.learning_rate,
        n_estimators=hp.n_estimators,
        subsample=hp.subsample,
        criterion=hp.criterion,
        min_samples_leaf=hp.min_samples_leaf,
        max_depth=hp.max_depth,
        random_state=hp.seed,
    )
    modelo.fit(X_train, y_train)
    return modelo


@dataclass
class XGBoostHyperparams:
    learning_rate: float = 0.1
    n_estimators: int = 100
    max_depth: int = 3
    subsample: float = 1.0
    scale_pos_weight: float = 1.0  # alternativa ao oversampling (ver data.py)
    seed: int = 0


def treinar_xgboost(X_train: np.ndarray, y_train: np.ndarray, hp: XGBoostHyperparams):
    import xgboost as xgb
    modelo = xgb.XGBClassifier(
        learning_rate=hp.learning_rate,
        n_estimators=hp.n_estimators,
        max_depth=hp.max_depth,
        subsample=hp.subsample,
        scale_pos_weight=hp.scale_pos_weight,
        random_state=hp.seed,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    modelo.fit(X_train, y_train)
    return modelo
