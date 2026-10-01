"""TabKAN — KAN adaptado para dados tabulares (pacote `tabkan` do PyPI,
confirmado como o modelo correto no smoke test da modelos avançados; depende de
`pykan`, `fkan`, `rkan` e `torch>=2.0`).

So importa `tabkan`/torch dentro da funcao (lazy import) — treino real so
no Kaggle. A API exata do pacote (classe `TabKANClassifier` ou
equivalente) deve ser conferida no Kaggle no momento do uso, ja que nao
foi possivel instalar e inspecionar aqui — este wrapper assume a
convencao mais comum de bibliotecas sklearn-like (`fit`/`predict_proba`)
usada por quase todo o ecossistema de KAN tabular; ajustar se a API real
divergir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabKANHyperparams:
    hidden_width: int = 10
    grid: int = 5
    k: int = 3
    lr: float = 1e-3
    epochs: int = 100
    seed: int = 0


def treinar_tabkan(X_train: np.ndarray, y_train: np.ndarray, hp: TabKANHyperparams):
    """Assume API sklearn-like. Se a classe real do pacote tiver nome ou
    assinatura diferente, ajustar aqui no Kaggle (ver docstring do modulo)."""
    from tabkan import TabKANClassifier

    modelo = TabKANClassifier(
        hidden_width=hp.hidden_width, grid=hp.grid, k=hp.k,
        lr=hp.lr, epochs=hp.epochs, random_state=hp.seed,
    )
    modelo.fit(X_train, y_train)
    return modelo


def prever_tabkan(modelo, X: np.ndarray) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]
