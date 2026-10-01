"""TabKAN — KANs para dados tabulares (pacote `tabkan` do PyPI, v1.0.x).

API real do pacote (conferida inspecionando o wheel publicado): nao existe
classe sklearn-like tipo `TabKANClassifier`; o pacote expoe modulos
PyTorch (`ChebyshevKAN`, `FourierKAN`, `SplineKAN`, ...) com construtor
`(layers, orders)` — `layers` = [entrada, ocultas..., saida] e `orders` =
grau do polinomio de cada camada KAN (a ultima camada e um Linear comum).

Usamos `ChebyshevKAN` (variante principal do artigo) e um loop de treino
proprio com Adam + BCEWithLogits + early stopping na validacao — o
`.fit()` nativo do pacote usa L-BFGS em full-batch e reporta sqrt(loss),
o que nao combina com a metrica de classificacao usada no resto do projeto.

Imports de torch/tabkan sao lazy — modulo importavel sem as dependencias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabKANHyperparams:
    hidden_width: int = 10   # navalha de Occam, igual ao MLP baseline
    degree: int = 3          # grau do polinomio de Chebyshev
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 10
    seed: int = 0


def treinar_tabkan(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   hp: TabKANHyperparams, device: str = "cpu", verbose: bool = True):
    import copy
    import torch
    from tabkan import ChebyshevKAN

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    modelo = ChebyshevKAN(layers=[X_train.shape[1], hp.hidden_width, 1],
                          orders=[hp.degree]).to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_train.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1).to(device)
    Xv = torch.from_numpy(X_val.astype(np.float32)).to(device)
    yv = torch.from_numpy(y_val.astype(np.float32)).reshape(-1, 1).to(device)

    historico, melhor, melhor_estado, falhas = [], float("inf"), None, 0
    for epoca in range(hp.max_epochs):
        modelo.train()
        perm = torch.randperm(len(Xt), device=device)
        soma = 0.0
        for i in range(0, len(Xt), hp.batch_size):
            idx = perm[i:i + hp.batch_size]
            opt.zero_grad()
            loss = loss_fn(modelo(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            soma += loss.item() * len(idx)
        modelo.eval()
        with torch.no_grad():
            val_loss = loss_fn(modelo(Xv), yv).item()
        historico.append({"epoca": epoca, "train_loss": soma / len(Xt), "val_loss": val_loss})
        if val_loss < melhor - 1e-6:
            melhor, melhor_estado, falhas = val_loss, copy.deepcopy(modelo.state_dict()), 0
        else:
            falhas += 1
        if verbose and epoca % 10 == 0:
            print(f"  epoca {epoca:4d} | val_loss={val_loss:.4f} falhas={falhas}/{hp.patience}", flush=True)
        if falhas >= hp.patience:
            break
    modelo.load_state_dict(melhor_estado)
    return modelo, historico


def prever_tabkan(modelo, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch
    modelo.eval()
    with torch.no_grad():
        logit = modelo(torch.from_numpy(X.astype(np.float32)).to(device))
        return torch.sigmoid(logit).cpu().numpy().reshape(-1)
