"""Avaliacao robusta: repete o split de 3 etapas com varias sementes e
reporta media +- desvio do KS/AUROC por modelo.

Motivo: o teste tem ~1760 linhas (~470 churners), entao o erro-padrao do KS
de um unico split e ~0.02 — maior que as diferencas entre quase todos os
modelos. Um unico numero de teste nao permite dizer qual modelo e melhor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import assert_sem_vazamento, split_three_stage
from .features import Preprocessor
from .metrics import auroc, ks_statistic


def avaliar_multiplas_particoes(df, fabricas: dict, seeds=range(5), incluir_ensemble: bool = True,
                                verbose: bool = True) -> pd.DataFrame:
    """`fabricas`: {nome: fn(Xtr, ytr, Xva, yva, seed) -> (p_val, p_test_fn)}.
    Na pratica cada fabrica recebe os dados e devolve uma funcao `prever(X)`.
    Retorna DataFrame longo: seed, modelo, ks, auroc (+ linha 'ensemble')."""
    linhas = []
    for seed in seeds:
        split = split_three_stage(df, seed=seed)
        assert_sem_vazamento(split)
        tr, va, te = df.iloc[split.train_idx], df.iloc[split.val_idx], df.iloc[split.test_idx]
        prep = Preprocessor().fit(tr)
        Xtr, _ = prep.transform(tr); Xva, _ = prep.transform(va); Xte, _ = prep.transform(te)
        ytr, yva, yte = prep.transform_target(tr), prep.transform_target(va), prep.transform_target(te)
        preds = {}
        for nome, fab in fabricas.items():
            try:
                prever = fab(Xtr, ytr, Xva, yva, seed)
                preds[nome] = prever(Xte)
            except Exception as e:  # um modelo falhar nao derruba a avaliacao
                print(f"[seed {seed}] {nome} falhou: {type(e).__name__}: {e}")
                continue
            linhas.append({"seed": seed, "modelo": nome,
                           "ks": ks_statistic(yte, preds[nome])["ks"], "auroc": auroc(yte, preds[nome])})
            if verbose:
                print(f"[seed {seed}] {nome:10s} ks={linhas[-1]['ks']:.4f} auroc={linhas[-1]['auroc']:.4f}", flush=True)
        if incluir_ensemble and len(preds) > 1:
            pm = np.mean(np.vstack(list(preds.values())), axis=0)
            linhas.append({"seed": seed, "modelo": "ensemble_media",
                           "ks": ks_statistic(yte, pm)["ks"], "auroc": auroc(yte, pm)})
    return pd.DataFrame(linhas)


def resumir(df_long: pd.DataFrame) -> pd.DataFrame:
    g = df_long.groupby("modelo")[["ks", "auroc"]].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return g.sort_values("ks_mean", ascending=False).reset_index()
