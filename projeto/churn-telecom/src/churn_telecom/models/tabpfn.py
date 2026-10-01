"""TabPFN v2 — foundation model para dados tabulares (pacote `tabpfn`).

A versao do modelo e fixada explicitamente em **v2** (a pedida no
enunciado). Isso importa por dois motivos: (1) versoes recentes do pacote
usam por padrao checkpoints mais novos (v2.5+), e (2) esses checkpoints
novos sao "gated" — exigem aceitar licenca e um TABPFN_TOKEN, o que quebra
execucao nao-interativa (Kaggle "Save & Run All"). Os pesos v2 nao sao
gated e baixam direto.

Imports lazy — modulo importavel sem `tabpfn`/torch instalados.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabPFNHyperparams:
    device: str = "cuda"   # TabPFN e pesado em CPU; usar GPU no Kaggle
    n_estimators: int = 8  # ensemble interno do TabPFN
    versao: str = "v2"
    seed: int = 0


def treinar_tabpfn(X_train: np.ndarray, y_train: np.ndarray, hp: TabPFNHyperparams):
    """TabPFN nao tem 'epocas' — e pre-treinado (in-context learning): o
    'treino' e passar o conjunto de treino como contexto. Sem historico."""
    from tabpfn import TabPFNClassifier

    try:
        from tabpfn.constants import ModelVersion
        modelo = TabPFNClassifier.create_default_for_version(
            ModelVersion(hp.versao),
            device=hp.device, n_estimators=hp.n_estimators, random_state=hp.seed,
        )
    except (ImportError, AttributeError):
        # versoes antigas do pacote (<2.x) so tem v2 — construtor direto
        modelo = TabPFNClassifier(device=hp.device, n_estimators=hp.n_estimators,
                                  random_state=hp.seed)
    modelo.fit(X_train, y_train)
    return modelo


def prever_tabpfn(modelo, X: np.ndarray) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]
