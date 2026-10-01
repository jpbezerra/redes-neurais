"""TabPFN v2 — foundation model para dados tabulares (pacote `tabpfn` do
PyPI, confirmado como o modelo correto no smoke test da modelos avançados).

So importa `tabpfn`/torch dentro da funcao (lazy import) — este modulo
pode ser importado neste ambiente mesmo sem as dependencias instaladas;
o treino de verdade so acontece no Kaggle (ou outro ambiente com GPU e
rede liberada para o download do checkpoint pre-treinado do TabPFN).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabPFNHyperparams:
    device: str = "cuda"  # TabPFN e pesado em CPU; usar GPU no Kaggle
    n_estimators: int = 8  # TabPFN v2 usa um pequeno ensemble interno por padrao
    seed: int = 0


def treinar_tabpfn(X_train: np.ndarray, y_train: np.ndarray, hp: TabPFNHyperparams):
    """TabPFN nao tem 'epocas' — e um modelo pre-treinado (in-context
    learning): o 'treino' e so passar o conjunto de treino como contexto.
    Por isso nao ha historico de curva de treino aqui (diferente de
    MLP/STab)."""
    from tabpfn import TabPFNClassifier

    modelo = TabPFNClassifier(device=hp.device, n_estimators=hp.n_estimators, random_state=hp.seed)
    modelo.fit(X_train, y_train)
    return modelo


def prever_tabpfn(modelo, X: np.ndarray) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]
