"""STab — Transformer para dados tabulares.

Nota honesta (ver reports/tables/modelos_avancados_smoke_test.md):
nao existe um pacote publico consolidado chamado "STab" no PyPI (o nome
`stab` la e um gerador de sites estaticos, sem relacao). Em vez de depender
de um pacote de proveniencia incerta, esta e uma implementacao PROPRIA e
enxuta de um Transformer para dados tabulares (cada feature vira um
"token", projetado para `dim`, passa por `depth` camadas de self-attention
multi-cabeca, e um token [CLS] agrega a predicao final) — a mesma ideia
geral de modelos como TabTransformer/FT-Transformer, e cobre o espaco de
hiperparametros pedido no enunciado (dim, depth, heads, attn_dropout,
ff_dropout, lr, weight_decay, batch_size).

Os hiperparametros `U`, `cases` e `sample_size` citados no enunciado nao
tem definicao padrao na literatura publica — ficam documentados aqui como
nao implementados (em vez de inventar um significado): `U` e tratado como
sinonimo de `dim` (tamanho do embedding), `cases`/`sample_size` nao tem
efeito nesta implementacao (podem ser retomados se o enunciado original
dos slides trouxer uma definicao mais especifica).

So importa torch dentro das funcoes (lazy import) para o modulo poder ser
importado neste ambiente mesmo sem torch instalado — o treino de verdade
so acontece no Kaggle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STabHyperparams:
    dim: int = 32
    depth: int = 2
    heads: int = 4
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 10
    seed: int = 0


def _build_model(n_features: int, hp: "STabHyperparams"):
    import torch
    import torch.nn as nn

    class STabEncoder(nn.Module):
        """Cada feature numerica (apos o pre-processamento, ja tudo
        float) vira um token via projecao linear propria; um token [CLS]
        aprendido agrega a predicao apos `depth` camadas de self-attention."""

        def __init__(self, n_features: int, hp: STabHyperparams):
            super().__init__()
            self.token_proj = nn.Linear(1, hp.dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hp.dim) * 0.02)
            self.pos_embed = nn.Parameter(torch.randn(1, n_features + 1, hp.dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hp.dim, nhead=hp.heads, dim_feedforward=hp.dim * 4,
                dropout=hp.ff_dropout, activation="gelu", batch_first=True,
            )
            layer.self_attn.dropout = hp.attn_dropout
            self.encoder = nn.TransformerEncoder(layer, num_layers=hp.depth)
            self.head = nn.Linear(hp.dim, 1)

        def forward(self, x):
            # x: [B, n_features] -> tokens [B, n_features, 1] -> [B, n_features, dim]
            tokens = self.token_proj(x.unsqueeze(-1))
            cls = self.cls_token.expand(x.size(0), -1, -1)
            seq = torch.cat([cls, tokens], dim=1) + self.pos_embed
            encoded = self.encoder(seq)
            logit = self.head(encoded[:, 0, :]).squeeze(-1)
            return logit

    torch.manual_seed(hp.seed)
    return STabEncoder(n_features, hp)


def treinar_stab(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  hp: STabHyperparams, device: str = "cpu", verbose: bool = False):
    import torch
    import torch.nn as nn

    dev = torch.device(device)
    modelo = _build_model(X_train.shape[1], hp).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32))
    Xv = torch.from_numpy(X_val.astype(np.float32)).to(dev)
    yv_np = y_val

    historico = []
    melhor_val_loss = float("inf")
    melhor_estado = None
    falhas = 0
    n = len(Xt)

    for epoca in range(hp.max_epochs):
        modelo.train()
        perm = torch.randperm(n)
        perda_ep = 0.0
        n_batches = max(n // hp.batch_size, 1)
        for b in range(n_batches):
            idx = perm[b * hp.batch_size:(b + 1) * hp.batch_size]
            if len(idx) == 0:
                continue
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            opt.zero_grad()
            logit = modelo(xb)
            perda = bce(logit, yb)
            perda.backward()
            opt.step()
            perda_ep += perda.item()

        modelo.eval()
        with torch.no_grad():
            logit_val = modelo(Xv)
            p_val = torch.sigmoid(logit_val).cpu().numpy()
        from ..metrics import cross_entropy
        val_loss = cross_entropy(yv_np, p_val)
        historico.append({"epoch": epoca, "train_loss": perda_ep / n_batches, "val_loss": val_loss})

        if val_loss < melhor_val_loss - 1e-6:
            melhor_val_loss = val_loss
            melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            falhas = 0
        else:
            falhas += 1
        if verbose and epoca % max(hp.max_epochs // 20, 1) == 0:
            print(f"  epoca {epoca:>4} | val_loss={val_loss:.4f} falhas={falhas}/{hp.patience}")
        if falhas >= hp.patience:
            break

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    return modelo, historico


def prever_stab(modelo, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch
    modelo.eval()
    with torch.no_grad():
        logit = modelo(torch.from_numpy(X.astype("float32")).to(device))
        return torch.sigmoid(logit).cpu().numpy()
