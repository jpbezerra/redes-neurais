"""Utilitarios de predicao: correcao de prior, media de modelos e
persistencia das probabilidades (para ensembles/consolidacao sem retreinar)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def corrigir_prior(p: np.ndarray, prior_treino: float, prior_alvo: float = 0.5) -> np.ndarray:
    """Reescala probabilidades de um modelo treinado com proporcao de churn
    `prior_treino` para a proporcao `prior_alvo` (Bayes: so multiplica as
    odds por uma constante — monotonico, entao o KS nao muda).

    Serve para colocar modelos treinados SEM oversampling (TabPFN/Mitra,
    que sofrem com linhas duplicadas) na mesma escala dos modelos treinados
    com treino balanceado, onde o limiar 0.5 faz sentido."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    odds = p / (1 - p) * ((1 - prior_treino) / prior_treino) * (prior_alvo / (1 - prior_alvo))
    return odds / (1 + odds)


def media_probabilidades(lista: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(lista), axis=0)


def salvar_predicoes(results_dir: str | Path, model_id: str, p_val, p_test) -> Path:
    """Grava `predicoes.npz` (probabilidades de validacao e teste) dentro de
    results/{model_id}/ — pequeno (poucos KB), vai junto no push."""
    destino = Path(results_dir) / model_id / "predicoes.npz"
    np.savez_compressed(destino, p_val=np.asarray(p_val), p_test=np.asarray(p_test))
    return destino
