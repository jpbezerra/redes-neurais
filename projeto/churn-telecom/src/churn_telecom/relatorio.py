"""Melhor configuracao de cada algoritmo (familia), com hiperparametros —
tabela + figuras para a apresentacao.

Fonte: `results/avaliacao_5_particoes/{rodada}/{nome}_particao{k}/metadata.json`
(todas as rodadas com avaliacao em 5 particoes). So entram configuracoes com
as 5 particoes completas; a pasta `vazamento/` fica de fora (demonstracao de
vazamento, nao e um modelo valido). A escolha do "melhor" de cada familia e
pelo KS medio no teste — e o que se apresenta; nao e usada para escolher nada
dentro dos experimentos.

Uso: `python -m churn_telecom.relatorio` (gera em reports/).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ordem importa: prefixos mais especificos primeiro
FAMILIAS = [("tabpfn_v25", "TabPFN 2.5"), ("tabpfn", "TabPFN v2"), ("tabkan", "TabKAN"), ("kan", "KAN"),
            ("catboost", "CatBoost"), ("xgb", "XGBoost"), ("gb", "Gradient Boosting"), ("mlp", "MLP"),
            ("stab", "STab"), ("ensemble", "Ensemble"), ("stacking", "Ensemble")]
EXCLUIR_RODADAS = {"vazamento"}
# hiperparametros que nao descrevem o modelo (mostrados a parte ou irrelevantes)
HP_OMITIR = {"seed", "device", "eval_metric", "max_epochs", "patience", "early_stopping_rounds",
             "early_stop_metric", "max_grad_norm", "cat_features", "scale_pos_weight", "loss_metric",
             "solver", "criterion", "pos_weight", "versao"}


def familia_de(nome: str) -> str:
    return next((f for p, f in FAMILIAS if nome.startswith(p)), "Outro")


def carregar_5_particoes(results_dir: str | Path = "results") -> pd.DataFrame:
    linhas = []
    for meta in Path(results_dir).glob("avaliacao_5_particoes/*/*/metadata.json"):
        rodada = meta.parent.parent.name
        if rodada in EXCLUIR_RODADAS:
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        d = m.get("dataset", {})
        nome = d.get("nome_na_tabela") or meta.parent.name.rsplit("_particao", 1)[0]
        linhas.append({"rodada": rodada, "modelo": nome, "particao": d.get("particao"),
                       "familia": d.get("familia") or familia_de(nome),
                       "ks": m["metrics"]["ks"], "auroc": m["metrics"]["auroc"],
                       "acuracia_limiar_validacao": m["metrics"].get("acuracia_limiar_validacao"),
                       "hiperparametros": m.get("hyperparameters", {}),
                       "paciencia": d.get("paciencia", {}),
                       "treino": d.get("treino") or d.get("balanceamento_treino")
                       or m.get("hyperparameters", {}).get("contexto")   # TabPFN: contexto do in-context learning
                       or ("-" if familia_de(nome) == "Ensemble" else "duplicacao"),
                       "codificacao": d.get("codificacao", ""), "transformacao": d.get("transformacao", "nenhuma")})
    return pd.DataFrame(linhas)


def _hp_texto(hp: dict) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):   # membros de ensemble
            return "+".join(Path(x).name.rsplit("_particao", 1)[0] for x in v)
        if isinstance(v, (list, tuple)):
            return "x".join(map(str, v))
        return str(v)
    return ", ".join(f"{k}={fmt(v)}" for k, v in hp.items() if k not in HP_OMITIR and v is not None)


def melhores_por_familia(longo: pd.DataFrame) -> pd.DataFrame:
    g = (longo.groupby(["familia", "rodada", "modelo"])
         .agg(n=("ks", "size"), ks_mean=("ks", "mean"), ks_std=("ks", "std"), auroc_mean=("auroc", "mean"),
              acuracia_limiar_validacao_mean=("acuracia_limiar_validacao", "mean"))
         .reset_index())
    g = g[g.n == 5]
    melhor = g.loc[g.groupby("familia")["ks_mean"].idxmax()].sort_values("ks_mean", ascending=False)
    # hiperparametros/protocolo da particao 0 da configuracao vencedora
    p0 = longo[longo.particao == 0].set_index(["rodada", "modelo"])
    extra = []
    for _, r in melhor.iterrows():
        x = p0.loc[(r.rodada, r.modelo)]
        x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
        it = x.paciencia.get("melhor_epoca") or x.paciencia.get("melhor_iteracao") or \
            x.hiperparametros.get("n_estimators") or ""
        extra.append({"treino": x.treino, "codificacao": x.codificacao, "transformacao": x.transformacao,
                      "iteracoes_particao0": it, "hiperparametros": _hp_texto(x.hiperparametros),
                      "configuracoes_testadas": int((g.familia == r.familia).sum())})
    return pd.concat([melhor.reset_index(drop=True), pd.DataFrame(extra)], axis=1)


def figura_ks(melhores: pd.DataFrame, destino: Path, meta_ks: float | None = 0.59):
    import matplotlib.pyplot as plt
    azul, tinta, tinta2 = "#2a78d6", "#0b0b0b", "#52514e"
    m = melhores.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(m) + 1.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    # pontos (nao barras): o eixo nao comeca em zero, e barra truncada exagera diferencas
    ax.errorbar(m.ks_mean, m.familia, xerr=m.ks_std, fmt="o", color=azul, ms=8, ecolor=tinta2,
                elinewidth=1, capsize=3)
    for y, (ks, sd, nome) in enumerate(zip(m.ks_mean, m.ks_std, m.modelo)):
        ax.text(ks + sd + 0.003, y, f"{ks:.3f}  ({nome})", va="center", fontsize=8.5, color=tinta)
    if meta_ks:
        ax.axvline(meta_ks, color=tinta2, ls="--", lw=1)
        ax.text(meta_ks, len(m) - 0.4, f" meta {meta_ks}", color=tinta2, fontsize=8.5, va="bottom")
    lo = max(0, m.ks_mean.min() - 0.06)
    ax.set_xlim(lo, max(meta_ks or 0, (m.ks_mean + m.ks_std).max()) + 0.07)
    ax.set_xlabel("KS no teste (média ± desvio em 5 partições)", color=tinta2)
    ax.set_title("Melhor configuração de cada algoritmo", loc="left", color=tinta, fontsize=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=tinta2, length=0)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def gerar(results_dirs=("results",), out_dir: str | Path = ".", meta_ks: float | None = 0.59) -> pd.DataFrame:
    longo = pd.concat([carregar_5_particoes(d) for d in results_dirs], ignore_index=True)
    longo = longo.drop_duplicates(["rodada", "modelo", "particao"])
    melhores = melhores_por_familia(longo)
    out = Path(out_dir)
    (out / "reports" / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    melhores.to_csv(out / "reports" / "tables" / "melhores_por_algoritmo.csv", index=False)
    figura_ks(melhores, out / "reports" / "figures" / "melhores_por_algoritmo_ks.png", meta_ks)
    return melhores


if __name__ == "__main__":
    m = gerar()
    print(m[["familia", "rodada", "modelo", "ks_mean", "ks_std", "configuracoes_testadas", "hiperparametros"]]
          .to_string(index=False))
