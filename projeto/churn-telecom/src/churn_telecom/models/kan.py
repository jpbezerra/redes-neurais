"""KAN — Kolmogorov-Arnold Network (pacote `pykan`).

O `.fit()` nativo do pykan (L-BFGS/Adam full-batch, loss reportada como
sqrt) divergiu para NaN logo no primeiro passo no Kaggle (KS=0.04,
AUROC=nan, historico todo NaN). Por isso o modelo do pykan e treinado aqui
com um loop proprio — mini-batch, Adam, gradient clipping e early stopping
na loss de validacao, o mesmo protocolo de STab/TabKAN/MLP — e o NaN e
tratado como falha explicita em vez de virar metrica silenciosa.

Opcoes de melhoria (desligadas por padrao — com os valores padrao o treino
e identico ao das execucoes ja registradas): `early_stop_metric="val_auroc"`,
`scheduler="plateau"` e `weight_decay` (mesma semantica de `stab.py`).

Imports lazy (torch/kan) — modulo importavel sem as dependencias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KANHyperparams:
    hidden_width: int = 8     # navalha de Occam: comeca pequeno
    grid: int = 3             # resolucao da grade dos splines
    k: int = 3                # ordem do spline
    lr: float = 3e-3
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 8
    max_grad_norm: float = 1.0
    seed: int = 0
    weight_decay: float = 0.0
    early_stop_metric: str = "val_loss"    # 'val_loss' | 'val_auroc'
    scheduler: str = "none"                # 'none' | 'plateau'


def treinar_kan(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, hp: KANHyperparams,
                device: str = "cpu", verbose: bool = False):
    import copy
    import tempfile
    import torch
    from kan import KAN
    from ..metrics import auroc, criterio_parada, cross_entropy

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    modelo = KAN(width=[X_train.shape[1], hp.hidden_width, 1], grid=hp.grid, k=hp.k,
                 seed=hp.seed, device=device, save_act=False,
                 ckpt_path=tempfile.mkdtemp(prefix="kan_ckpt_"))
    opt = torch.optim.Adam(modelo.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(hp.patience // 3, 1))
             if hp.scheduler == "plateau" else None)
    bce = torch.nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_train.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1).to(device)
    Xv = torch.from_numpy(X_val.astype(np.float32)).to(device)

    historico, melhor, melhor_estado, falhas = [], float("inf"), None, 0
    for epoca in range(hp.max_epochs):
        modelo.train()
        perm = torch.randperm(len(Xt), device=device)
        soma = 0.0
        for i in range(0, len(Xt), hp.batch_size):
            idx = perm[i:i + hp.batch_size]
            opt.zero_grad()
            perda = bce(modelo(Xt[idx]), yt[idx])
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), hp.max_grad_norm)
            opt.step()
            soma += perda.item() * len(idx)
        modelo.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(modelo(Xv)).cpu().numpy().ravel()
        val_loss = cross_entropy(y_val, p_val)
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"KAN divergiu (val_loss={val_loss}) na epoca {epoca}")
        score = criterio_parada(hp.early_stop_metric, y_val, p_val)
        historico.append({"epoch": epoca, "train_loss": soma / len(Xt), "val_loss": val_loss,
                          "val_auroc": auroc(y_val, p_val), "lr": opt.param_groups[0]["lr"]})
        if sched is not None:
            sched.step(score)
        if score < melhor - 1e-6:
            melhor, melhor_estado, falhas = score, copy.deepcopy(modelo.state_dict()), 0
        else:
            falhas += 1
        if verbose and epoca % 10 == 0:
            print(f"  epoca {epoca:4d} | val_loss={val_loss:.4f} falhas={falhas}/{hp.patience}", flush=True)
        if falhas >= hp.patience:
            break
    modelo.load_state_dict(melhor_estado)
    return modelo, historico


def prever_kan(modelo, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch
    modelo.eval()
    with torch.no_grad():
        logit = modelo(torch.from_numpy(X.astype(np.float32)).to(device))
        return torch.sigmoid(logit).cpu().numpy().reshape(-1)
