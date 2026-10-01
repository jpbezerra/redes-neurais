"""KAN — Kolmogorov-Arnold Network (pacote `pykan` do PyPI, confirmado
como o modelo correto no smoke test da modelos avançados, versao mais recente
disponivel no momento da checagem: 0.2.8).

So importa `kan`/torch dentro da funcao (lazy import) — treino real so no
Kaggle. A API do pykan (`kan.KAN`) usa `width` (lista de tamanhos de
camada, incluindo entrada e saida), `grid` (resolucao da grade de
splines) e `k` (ordem do spline) em vez dos hiperparametros classicos de
MLP — documentado explicitamente para nao confundir com `MLPHyperparams`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KANHyperparams:
    hidden_width: int = 10  # navalha de Occam: comeca pequeno, igual ao MLP da baselines
    grid: int = 5
    k: int = 3
    lr: float = 1e-2
    steps: int = 200
    seed: int = 0


def treinar_kan(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, hp: KANHyperparams, device: str = "cpu"):
    import torch
    from kan import KAN

    torch.manual_seed(hp.seed)
    n_features = X_train.shape[1]
    modelo = KAN(width=[n_features, hp.hidden_width, 1], grid=hp.grid, k=hp.k, seed=hp.seed, device=device)

    dataset = {
        "train_input": torch.from_numpy(X_train.astype(np.float32)).to(device),
        "train_label": torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1).to(device),
        "test_input": torch.from_numpy(X_val.astype(np.float32)).to(device),
        "test_label": torch.from_numpy(y_val.astype(np.float32)).reshape(-1, 1).to(device),
    }
    # pykan usa BCEWithLogits implicitamente via loss_fn customizada para classificacao;
    # aqui usamos MSE na saida sigmoidal, documentado como escolha pragmatica
    # (a API nativa do pykan e voltada a regressao/function-fitting).
    historico = modelo.fit(dataset, opt="Adam", lr=hp.lr, steps=hp.steps, loss_fn=torch.nn.BCEWithLogitsLoss())
    return modelo, historico


def prever_kan(modelo, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch
    with torch.no_grad():
        logit = modelo(torch.from_numpy(X.astype(np.float32)).to(device))
        return torch.sigmoid(logit).cpu().numpy().reshape(-1)
