"""Referencias para interpretar o KS do projeto.

1. Regressao logistica nas 5 particoes (mesmo protocolo de todos os modelos):
   C e transformacao escolhidos SO pelo AUROC de validacao.
2. Sorte da divisao: o protocolo de uma divisao aleatoria unica 80/20 (como no
   ranking do semestre anterior), repetido N vezes, mostra quanto o KS de teste
   varia so pela divisao.

Uso: python -m churn_telecom.referencia   (roda local em poucos minutos, CPU)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import rodada8 as r8
from .metrics import auroc, ks_statistic

RODADA = "rodada9"
NOME_LOGISTICA = "logistica_referencia"
GRADE_C = (0.01, 0.03, 0.1, 0.3, 1.0)


def _ks(y, p) -> float:
    return ks_statistic(y, p)["ks"]


def treinar_logistica(part: r8.Particao):
    """Grade C x transformacao, treino sem copias; escolha pelo AUROC de validacao."""
    from sklearn.linear_model import LogisticRegression
    melhor = None
    for transf in ("nenhuma", "log"):
        X = part.dados("neural", transf)
        for C in GRADE_C:
            m = LogisticRegression(C=C, max_iter=5000).fit(X["nat"], part.y["nat"])
            pv = m.predict_proba(X["val"])[:, 1]
            a = auroc(part.y["val"], pv)
            if melhor is None or a > melhor[0]:
                melhor = (a, pv, m.predict_proba(X["test"])[:, 1], {"C": C, "transformacao": transf, "treino": "nat"})
    return melhor[1], melhor[2], melhor[3]


def registrar_logistica(df: pd.DataFrame, base: Path, seeds=range(5)) -> None:
    for s in seeds:
        if (r8.pasta_de(base, NOME_LOGISTICA, s, RODADA) / "metadata.json").exists():
            continue
        part = r8.preparar_particao(df, s)
        pv, pt, hp = treinar_logistica(part)
        r8.registrar(NOME_LOGISTICA, part, pv, pt, base, hp=hp,
                     dataset={"algoritmo": "logistica", "familia": "Regressão logística", "codificacao": "neural",
                              "transformacao": hp["transformacao"], "treino": "nat",
                              "n_linhas_treino": int(len(part.y["nat"]))},
                     paciencia={"sem_epocas": "solver convexo (lbfgs)"},
                     extra={"etapa": "I_referencia_logistica"}, rodada=RODADA)
        print(f"[I] {NOME_LOGISTICA} p{s} hp={hp}")


def simular_divisao_unica(df: pd.DataFrame, n: int = 300, test_size: float = 0.2) -> np.ndarray:
    """KS de teste da logistica em `n` divisoes aleatorias estratificadas treino/teste
    (protocolo de divisao unica, sem oversampling, treino = todo o resto)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d = d.dropna().drop(columns=[c for c in d.columns if c.lower() == "customerid"])
    y = d.pop("Churn").astype(str).isin(["Yes", "True", "1"]).astype(int).values
    X = pd.get_dummies(d, drop_first=True).astype(float).values
    out = []
    for s in range(n):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=s)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(Xtr), ytr)
        out.append(_ks(yte, m.predict_proba(sc.transform(Xte))[:, 1]))
    return np.array(out)


def figura_sorte(ks: np.ndarray, destino: Path, ks_projeto: float, ranking: list[float]) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ks, bins=30, color="#4C78A8", edgecolor="white", linewidth=1)
    tinta = "#444444"
    for x, txt, ls in ((ks_projeto, f"melhor média do projeto\n(5 partições) {ks_projeto:.3f}", "-"),
                       (max(ranking), f"1º do ranking {max(ranking):.3f}", "--")):
        ax.axvline(x, color=tinta, linestyle=ls, linewidth=1.5)
        ax.text(x, ax.get_ylim()[1] * 0.97, " " + txt, va="top", fontsize=9, color=tinta)
    ax.set_xlabel("KS de teste (regressão logística, uma divisão aleatória 80/20)")
    ax.set_ylabel("divisões")
    ax.set_title(f"Só a divisão muda o KS: {len(ks)} divisões, média {ks.mean():.3f}, "
                 f"{np.mean(ks >= max(ranking)):.1%} ≥ {max(ranking):.3f}", fontsize=10)
    for e in ("top", "right"):
        ax.spines[e].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


RANKING_SEMESTRE_ANTERIOR = [0.5864, 0.5736, 0.5728, 0.561, 0.561, 0.556, 0.5531, 0.5513, 0.5512, 0.551,
                             0.5429, 0.5422, 0.542, 0.5414]

if __name__ == "__main__":
    from .data import load_raw
    raiz = Path(__file__).resolve().parents[2]
    df = load_raw(raiz / "data")
    registrar_logistica(df, raiz / "results")
    ks = simular_divisao_unica(df)
    assert 0.45 < ks.mean() < 0.60 and ks.std() > 0.005, ks.mean()   # sanidade
    tab = pd.DataFrame({"divisao": range(len(ks)), "ks_teste": ks})
    tab.to_csv(raiz / "reports" / "tables" / "sorte_divisao_unica_logistica.csv", index=False)
    resumo = {"media": ks.mean(), "desvio": ks.std(), "p10": np.quantile(ks, .1), "p90": np.quantile(ks, .9),
              "maximo": ks.max(), "frac_ge_0586": np.mean(ks >= 0.5864), "frac_ge_056": np.mean(ks >= 0.56)}
    print({k: round(float(v), 4) for k, v in resumo.items()})
    from .relatorio import gerar
    melhores = gerar([raiz / "results"], raiz)
    figura_sorte(ks, raiz / "reports" / "figures" / "sorte_divisao_unica.png", float(melhores.ks_mean.max()),
                 RANKING_SEMESTRE_ANTERIOR)
