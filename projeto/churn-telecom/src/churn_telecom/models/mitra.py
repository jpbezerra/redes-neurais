"""Mitra (Amazon) — foundation model tabular, disponivel via o extra
`[mitra]` do AutoGluon (`pip install "autogluon.tabular[mitra]"`), ja que
nao existe um pacote `mitra` legitimo e isolado no PyPI (ver smoke test da
modelos avançados — o pacote `mitra` la e uma lib de algebra linear nao relacionada).

So importa `autogluon`/torch dentro da funcao (lazy import) — instalacao
e treino real so no Kaggle (arvore de dependencias grande: PyTorch,
LightGBM, CatBoost e outros).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MitraHyperparams:
    time_limit_segundos: int = 600
    seed: int = 0


def treinar_mitra(X_train: np.ndarray, y_train: np.ndarray, colunas: list[str], hp: MitraHyperparams):
    """AutoGluon trabalha com DataFrame + nome da coluna-alvo, nao com
    arrays soltos como os outros wrappers — reconstroi um DataFrame aqui
    para caber na API do TabularPredictor."""
    from autogluon.tabular import TabularPredictor

    df_train = pd.DataFrame(X_train, columns=colunas)
    df_train["_churn_target"] = y_train

    preditor = TabularPredictor(label="_churn_target", problem_type="binary", eval_metric="roc_auc")
    preditor.fit(
        df_train, hyperparameters={"MITRA": {}},
        time_limit=hp.time_limit_segundos,
    )
    return preditor


def prever_mitra(preditor, X: np.ndarray, colunas: list[str]) -> np.ndarray:
    df = pd.DataFrame(X, columns=colunas)
    return preditor.predict_proba(df)[1].to_numpy()
