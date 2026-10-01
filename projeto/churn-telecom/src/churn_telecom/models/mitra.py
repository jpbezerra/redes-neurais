"""Mitra (Amazon) — foundation model tabular, disponivel via o extra
`[mitra]` do AutoGluon (`pip install "autogluon.tabular[mitra]"`), ja que
nao existe um pacote `mitra` legitimo e isolado no PyPI (o `mitra` do PyPI
e uma lib de algebra linear nao relacionada).

So importa `autogluon`/torch dentro da funcao (lazy import) — treino real
so no Kaggle (precisa de GPU e bastante RAM).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MitraHyperparams:
    time_limit_segundos: int = 600
    num_gpus: int = 1                 # 0 para CPU (muito lento)
    max_memory_usage_ratio: float = 1.5  # AutoGluon e conservador demais na estimativa de RAM do Mitra
    seed: int = 0


def treinar_mitra(X_train: np.ndarray, y_train: np.ndarray, colunas: list[str],
                  hp: MitraHyperparams, path: str | None = None):
    """AutoGluon trabalha com DataFrame + nome da coluna-alvo; reconstroi
    um DataFrame aqui para caber na API do TabularPredictor."""
    from autogluon.tabular import TabularPredictor

    df_train = pd.DataFrame(X_train, columns=colunas)
    df_train["_churn_target"] = y_train

    preditor = TabularPredictor(label="_churn_target", problem_type="binary",
                                eval_metric="roc_auc", path=path)
    preditor.fit(
        df_train,
        hyperparameters={"MITRA": {"seed": hp.seed}},
        time_limit=hp.time_limit_segundos,
        num_gpus=hp.num_gpus,
        ag_args_fit={"ag.max_memory_usage_ratio": hp.max_memory_usage_ratio},
    )
    return preditor


def prever_mitra(preditor, X: np.ndarray, colunas: list[str]) -> np.ndarray:
    df = pd.DataFrame(X, columns=colunas)
    return preditor.predict_proba(df)[1].to_numpy()


def salvar_mitra(preditor, destino) -> None:
    """`TabularPredictor.save()` nao aceita caminho — copia o diretorio do
    preditor para `destino` via `clone`."""
    preditor.clone(path=str(destino), dirs_exist_ok=True)
