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
    # Opcoes de melhoria. Os padroes sao os do proprio XGBoost, entao com eles
    # o modelo e identico ao das execucoes ja registradas.
    colsample_bytree: float = 1.0
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    eval_metric: str = "logloss"
    early_stopping_rounds: int | None = None  # exige X_val/y_val em treinar_xgboost


def treinar_xgboost(X_train: np.ndarray, y_train: np.ndarray, hp: XGBoostHyperparams,
                    X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
    """Com `early_stopping_rounds`, `n_estimators` vira o teto: o numero de
    arvores e escolhido pela metrica na validacao (o teste nao participa)."""
    import xgboost as xgb
    modelo = xgb.XGBClassifier(
        learning_rate=hp.learning_rate,
        n_estimators=hp.n_estimators,
        max_depth=hp.max_depth,
        subsample=hp.subsample,
        scale_pos_weight=hp.scale_pos_weight,
        colsample_bytree=hp.colsample_bytree,
        min_child_weight=hp.min_child_weight,
        reg_lambda=hp.reg_lambda,
        random_state=hp.seed,
        eval_metric=hp.eval_metric,
        early_stopping_rounds=hp.early_stopping_rounds,
    )
    if hp.early_stopping_rounds is None:
        modelo.fit(X_train, y_train)
    else:
        if X_val is None or y_val is None:
            raise ValueError("early_stopping_rounds exige X_val e y_val.")
        modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return modelo
