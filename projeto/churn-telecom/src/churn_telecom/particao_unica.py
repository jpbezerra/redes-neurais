"""Relatorio complementar: cada particao lida como se fosse a unica divisao.

Cada uma das 5 particoes vira um "experimento de divisao unica", para
comparar com resultados reportados numa so divisao treino/teste. Nao substitui o relatorio principal
(`relatorio.py`, media em 5 particoes); so acrescenta tabelas/figuras com o
prefixo `particao_unica_`.

Tres leituras, da mais honesta para a mais otimista:
1. Selecao pela validacao: em cada particao, o modelo de maior AUROC de
   validacao (como um grupo com uma so divisao faria sem olhar o teste).
2. Melhor de cada algoritmo (pela media) visto particao a particao.
3. Maior KS de teste por algoritmo entre todas as configuracoes e particoes
   (escolhido olhando o teste: so referencia, otimista por construcao).

Uso: python -m churn_telecom.particao_unica
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .relatorio import _hp_texto, carregar_5_particoes, melhores_por_familia

AZUL, CINZA, TINTA, TINTA2, FUNDO, GRADE = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"


def _completas(longo: pd.DataFrame) -> pd.DataFrame:
    """So configuracoes com as 5 particoes (as mesmas que o relatorio principal compara)."""
    n = longo.groupby(["rodada", "modelo"]).particao.transform("nunique")
    return longo[n == 5]


def selecao_pela_validacao(longo: pd.DataFrame) -> pd.DataFrame:
    """Por particao: modelo individual de maior AUROC de validacao. Ficam fora os
    ensembles (membros escolhidos pela validacao media das 5 particoes) e os
    modelos re-treinados em treino+validacao (AUROC de validacao inflado)."""
    c = longo[(longo.familia != "Ensemble") & longo.auroc_validacao.notna()
              & ~longo.modelo.str.contains("treino_mais_validacao|trval")]
    sel = c.loc[c.groupby("particao").auroc_validacao.idxmax()]
    sel = sel[["particao", "familia", "rodada", "modelo", "auroc_validacao", "ks", "auroc"]].reset_index(drop=True)
    # referencia otimista: maior KS de teste de qualquer configuracao naquela particao
    mx = longo.loc[longo.groupby("particao").ks.idxmax()].set_index("particao")
    return sel.assign(ks_maximo_na_particao=sel.particao.map(mx.ks),
                      modelo_do_maximo=sel.particao.map(mx.modelo))


def melhor_de_cada_por_particao(longo: pd.DataFrame, melhores: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for _, r in melhores.iterrows():
        ks = longo[(longo.rodada == r.rodada) & (longo.modelo == r.modelo)].set_index("particao").ks.sort_index()
        linhas.append({"familia": r.familia, "modelo": r.modelo, "rodada": r.rodada,
                       **{f"ks_p{int(p)}": v for p, v in ks.items()},
                       "ks_min": ks.min(), "ks_mediana": ks.median(), "ks_max": ks.max(),
                       "particao_do_max": int(ks.idxmax()), "ks_media_5": ks.mean()})
    return pd.DataFrame(linhas).sort_values("ks_max", ascending=False).reset_index(drop=True)


def maior_ks_por_algoritmo(longo: pd.DataFrame) -> pd.DataFrame:
    m = longo.loc[longo.groupby("familia").ks.idxmax()].sort_values("ks", ascending=False)
    m = m.assign(hiperparametros=m.hiperparametros.map(_hp_texto),
                 configuracoes_e_particoes_olhadas=m.familia.map(longo.groupby("familia").size()))
    return m[["familia", "ks", "auroc", "particao", "rodada", "modelo", "configuracoes_e_particoes_olhadas",
              "hiperparametros"]].reset_index(drop=True)


def _eixo(ax):
    ax.set_facecolor(FUNDO)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TINTA2, length=0)
    ax.grid(axis="x", color=GRADE, lw=0.8)
    ax.set_axisbelow(True)


def figura_por_particao(tab: pd.DataFrame, destino: Path) -> None:
    import matplotlib.pyplot as plt
    t = tab.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(t) + 1.6), facecolor=FUNDO)
    _eixo(ax)
    cols = [c for c in t.columns if c.startswith("ks_p")]
    for y, r in t.iterrows():
        v = r[cols].astype(float).values
        ax.plot([v.min(), v.max()], [y, y], color=CINZA, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(v, [y] * len(v), s=46, color=AZUL, edgecolor=FUNDO, linewidth=1.5, zorder=3)
        ax.plot([r.ks_media_5] * 2, [y - 0.22, y + 0.22], color=TINTA, lw=2, zorder=4)
        ax.text(v.max() + 0.003, y, f"{v.max():.3f} (p{r.particao_do_max})", va="center", fontsize=8.5,
                color=TINTA)
    ax.set_yticks(range(len(t)), t.familia)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xlim(tab.ks_min.min() - 0.01, tab.ks_max.max() + 0.025)
    ax.set_xlabel("KS no teste — cada ponto é uma partição lida como divisão única; traço preto = média das 5",
                  color=TINTA2, fontsize=9)
    ax.set_title("Melhor configuração de cada algoritmo, partição por partição", loc="left", color=TINTA,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, facecolor=FUNDO)
    plt.close(fig)


def figura_selecao(sel: pd.DataFrame, destino: Path) -> None:
    """Uma linha por particao: KS do modelo escolhido pela validacao (o resultado
    honesto de divisao unica) e o maior KS de teste naquela particao (otimista)."""
    import matplotlib.pyplot as plt
    t = sel.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(t) + 1.8), facecolor=FUNDO)
    _eixo(ax)
    for y, r in t.iterrows():
        ax.plot([r.ks, r.ks_maximo_na_particao], [y, y], color=GRADE, lw=2, zorder=1)
        ax.scatter([r.ks_maximo_na_particao], [y], s=40, color=CINZA, edgecolor=FUNDO, linewidth=1.5, zorder=2,
                   label="maior KS de teste na partição (escolhido olhando o teste)" if y == 0 else None)
        ax.scatter([r.ks], [y], s=56, color=AZUL, edgecolor=FUNDO, linewidth=1.5, zorder=3,
                   label="modelo escolhido pela validação" if y == 0 else None)
        ax.text(max(r.ks, r.ks_maximo_na_particao) + 0.002, y, f"{r.ks:.4f} ({r.modelo})", va="center",
                fontsize=8, color=TINTA)
    ax.axvline(sel.ks.mean(), color=TINTA2, ls="--", lw=1)
    ax.text(sel.ks.mean(), len(t) - 0.45, f" média {sel.ks.mean():.4f}", color=TINTA2, fontsize=8.5, va="bottom")
    ax.set_yticks(range(len(t)), [f"partição {int(p)}" for p in t.particao])
    ax.set_ylim(-0.6, len(t) - 0.2)
    ax.set_xlim(min(sel.ks.min(), sel.ks_maximo_na_particao.min()) - 0.01,
                max(sel.ks.max(), sel.ks_maximo_na_particao.max()) + 0.06)
    ax.set_xlabel("KS no teste de cada partição lida como divisão única", color=TINTA2)
    ax.set_title("Resultado de divisão única em cada partição", loc="left", color=TINTA, fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=TINTA2)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, facecolor=FUNDO)
    plt.close(fig)


def gerar(results_dir: Path, out_dir: Path) -> dict:
    longo = _completas(carregar_5_particoes(results_dir).drop_duplicates(["rodada", "modelo", "particao"]))
    melhores = melhores_por_familia(longo)
    t, f = out_dir / "reports" / "tables", out_dir / "reports" / "figures"
    saidas = {"selecao_pela_validacao": selecao_pela_validacao(longo),
              "melhor_de_cada_algoritmo": melhor_de_cada_por_particao(longo, melhores),
              "maior_ks_por_algoritmo": maior_ks_por_algoritmo(longo)}
    for nome, df in saidas.items():
        df.to_csv(t / f"particao_unica_{nome}.csv", index=False)
    figura_por_particao(saidas["melhor_de_cada_algoritmo"], f / "particao_unica_melhor_de_cada_algoritmo.png")
    figura_selecao(saidas["selecao_pela_validacao"], f / "particao_unica_selecao_pela_validacao.png")
    return saidas


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[2]
    s = gerar(raiz / "results", raiz)
    sel, mel = s["selecao_pela_validacao"], s["melhor_de_cada_algoritmo"]
    assert len(sel) == 5 and sel.particao.tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(mel[[f"ks_p{i}" for i in range(5)]].mean(axis=1), mel.ks_media_5)
    pd.set_option("display.width", 200)
    for nome, df in s.items():
        print(f"\n== {nome}\n", df.drop(columns=["hiperparametros"], errors="ignore").round(4).to_string(index=False))
