"""Rodada 8 — baseline com paciencia, Optuna, transformacoes de variaveis.

Orientacao do professor: (1) nenhum algoritmo treina com numero fixo de
epocas/arvores — todos param por PACIENCIA na validacao (o teto existe so
como trava de seguranca e fica registrado se for atingido); (2) baseline com
todas as variaveis e todos os algoritmos; (3) so depois, transformacoes das
continuas (log/faixas); (4) tabela com o melhor de cada algoritmo.

Cada (configuracao, particao) vira uma pasta
`results/avaliacao_5_particoes/rodada8/{nome}_particao{k}/` com
metadata.json + predicoes.npz (+ modelo). Se a pasta ja existe, o treino e
pulado — o notebook pode ser re-executado e continua de onde parou.

Diferencas de protocolo em relacao as rodadas 1-7 (iguais para todos os
algoritmos desta rodada, entao a comparacao interna e justa):
- a paciencia olha a validacao SEM as copias do oversampling (as copias so
  repetem linhas e deixam a metrica de validacao mais ruidosa);
- o Preprocessor e ajustado no treino sem copias (quantis/medias da
  distribuicao real do treino).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import assert_sem_vazamento, split_three_stage
from .features import Preprocessor
from .metrics import auroc, full_report

RODADA = "rodada8"
PACIENCIA_REDES = 30      # epocas sem melhora do AUROC de validacao
PACIENCIA_ARVORES = 200   # arvores sem melhora
TETO_EPOCAS = 3000        # trava de seguranca (registrada se atingida)
TETO_ARVORES = 20000
TETO_ARVORES_GB = 3000    # sklearn GB ajusta o teto inteiro antes de escolher (mais lento)


# ---------------------------------------------------------------- dados
@dataclass
class Particao:
    seed: int
    y: dict                      # 'dup', 'nat', 'val', 'test', 'trval_dup', 'trval_nat'
    X: dict = field(default_factory=dict)   # (modo, transformacao) -> dict como `y`
    colunas: dict = field(default_factory=dict)
    _dfs: dict = field(default_factory=dict, repr=False)

    def dados(self, modo: str, transformacao: str = "nenhuma") -> dict:
        chave = (modo, transformacao)
        if chave not in self.X:
            prep = Preprocessor(modo=modo, transformacao=transformacao).fit(self._dfs["nat"])
            self.X[chave] = {k: prep.transform(d)[0] for k, d in self._dfs.items()}
            self.colunas[chave] = prep.transform(self._dfs["nat"].head(2))[1]
        return self.X[chave]


def preparar_particao(df: pd.DataFrame, seed: int) -> Particao:
    s = split_three_stage(df, seed=seed)
    assert_sem_vazamento(s)
    dfs = {
        "dup": df.iloc[s.train_idx], "nat": df.iloc[s.train_no_oversample_idx],
        "val": df.iloc[s.val_no_oversample_idx], "test": df.iloc[s.test_idx],
        "trval_dup": df.iloc[np.concatenate([s.train_idx, s.val_idx])],
        "trval_nat": df.iloc[np.concatenate([s.train_no_oversample_idx, s.val_no_oversample_idx])],
    }
    y = {k: (d["Churn"].values == "Yes").astype(np.float64) for k, d in dfs.items()}
    return Particao(seed=seed, y=y, _dfs=dfs)


# ---------------------------------------------------------------- algoritmos
# Cada algoritmo: treinar(hp, Xtr, ytr, Xva, yva, seed, ctx) -> (prever, info, modelo)
# `info` traz o numero de iteracoes escolhido pela paciencia e se o teto foi atingido.

def _hist_info(hist, teto):
    df = pd.DataFrame(hist)
    melhor = int(df["val_auroc"].idxmax()) + 1
    return {"epocas_treinadas": len(df), "melhor_epoca": melhor, "teto_atingido": len(df) >= teto}, df


def _hp(cls, hp: dict, **base):
    """Monta o dataclass `cls`: `base` (padrao da rodada) sobrescrito por `hp`;
    chaves que o dataclass nao tem vao para `outros` (repassadas a biblioteca)."""
    from dataclasses import fields
    nomes = {f.name for f in fields(cls)}
    kw = {**base, **{k: v for k, v in hp.items() if k in nomes}}
    resto = {k: v for k, v in hp.items() if k not in nomes}
    if resto:
        if "outros" not in nomes:
            raise TypeError(f"{cls.__name__} nao aceita {sorted(resto)}")
        kw["outros"] = {**kw.get("outros", {}), **resto}
    return cls(**kw)


def _mlp(hp, Xtr, ytr, Xva, yva, seed, ctx):
    from .models.mlp import MLPHyperparams, treinar_mlp
    h = dict(hp)
    camadas = tuple([h.pop("largura")] * h.pop("n_camadas"))
    p = _hp(MLPHyperparams, h, hidden_layer_sizes=camadas, max_epochs=TETO_EPOCAS, patience=PACIENCIA_REDES,
            early_stop_metric="val_auroc", seed=seed)
    m, hist = treinar_mlp(Xtr, ytr, Xva, yva, p)
    info, dfh = _hist_info(hist, TETO_EPOCAS)
    return (lambda X: m.predict_proba(X)[:, 1]), {**info, "historico": dfh}, m


_BASE_REDES = dict(max_epochs=TETO_EPOCAS, patience=PACIENCIA_REDES, early_stop_metric="val_auroc",
                   scheduler="plateau")


def _stab(hp, Xtr, ytr, Xva, yva, seed, ctx):
    from .models.stab import STabHyperparams, prever_stab, treinar_stab
    p = _hp(STabHyperparams, hp, **_BASE_REDES, max_grad_norm=1.0, seed=seed)
    m, hist = treinar_stab(Xtr, ytr, Xva, yva, p, device=ctx["device"])
    info, dfh = _hist_info(hist, TETO_EPOCAS)
    return (lambda X: prever_stab(m, X, ctx["device"])), {**info, "historico": dfh}, m


def _tabkan(hp, Xtr, ytr, Xva, yva, seed, ctx):
    from .models.tabkan import TabKANHyperparams, prever_tabkan, treinar_tabkan
    p = _hp(TabKANHyperparams, hp, **_BASE_REDES, max_grad_norm=1.0, seed=seed)
    m, hist = treinar_tabkan(Xtr, ytr, Xva, yva, p, device=ctx["device"], verbose=False)
    info, dfh = _hist_info(hist, TETO_EPOCAS)
    return (lambda X: prever_tabkan(m, X, ctx["device"])), {**info, "historico": dfh}, m


def _kan(hp, Xtr, ytr, Xva, yva, seed, ctx):
    from .models.kan import KANHyperparams, prever_kan, treinar_kan
    p = _hp(KANHyperparams, hp, **_BASE_REDES, seed=seed)
    m, hist = treinar_kan(Xtr, ytr, Xva, yva, p, device=ctx["device"])
    info, dfh = _hist_info(hist, TETO_EPOCAS)
    return (lambda X: prever_kan(m, X, ctx["device"])), {**info, "historico": dfh}, m


def _xgb(hp, Xtr, ytr, Xva, yva, seed, ctx):
    from .models.boosting import XGBoostHyperparams, treinar_xgboost
    n_fixo = ctx.get("n_iter_fixo")
    p = _hp(XGBoostHyperparams, hp, n_estimators=TETO_ARVORES, eval_metric="auc", seed=seed,
            early_stopping_rounds=PACIENCIA_ARVORES)
    if n_fixo:
        p.n_estimators, p.early_stopping_rounds = n_fixo, None
    m = treinar_xgboost(Xtr, ytr, p, None if n_fixo else Xva, None if n_fixo else yva)
    n = n_fixo or int(m.best_iteration) + 1
    return (lambda X: m.predict_proba(X)[:, 1]), {"melhor_iteracao": n, "teto_atingido": n >= TETO_ARVORES}, m


def _gb(hp, Xtr, ytr, Xva, yva, seed, ctx):
    from .models.boosting import GradientBoostingHyperparams, treinar_gradient_boosting
    n_fixo = ctx.get("n_iter_fixo")
    p = _hp(GradientBoostingHyperparams, hp, n_estimators=TETO_ARVORES_GB, seed=seed,
            early_stopping_rounds=PACIENCIA_ARVORES)
    if n_fixo:
        p.n_estimators, p.early_stopping_rounds = n_fixo, None
    m = treinar_gradient_boosting(Xtr, ytr, p, Xva, yva)
    n = n_fixo or m.melhor_iteracao_
    return (lambda X: m.predict_proba(X)[:, 1]), {"melhor_iteracao": n, "teto_atingido": n >= TETO_ARVORES_GB}, m


def _catboost(hp, Xtr, ytr, Xva, yva, seed, ctx):
    from .models.boosting import CatBoostHyperparams, prever_catboost, treinar_catboost
    cats = tuple(i for i, c in enumerate(ctx["colunas"]) if c.endswith(("__ord", "__faixa")))
    p = _hp(CatBoostHyperparams, hp, iterations=TETO_ARVORES, early_stopping_rounds=PACIENCIA_ARVORES,
            cat_features=cats, seed=seed)
    m = treinar_catboost(Xtr, ytr, p, Xva, yva)
    n = int(m.get_best_iteration()) + 1
    return (lambda X: prever_catboost(m, X, cats)), {"melhor_iteracao": n, "teto_atingido": n >= TETO_ARVORES}, m


def _tabpfn(hp, Xtr, ytr, Xva, yva, seed, ctx):
    from .models.tabpfn import TabPFNHyperparams, prever_tabpfn, treinar_tabpfn
    m = treinar_tabpfn(Xtr, ytr, _hp(TabPFNHyperparams, hp, device=ctx["device"], seed=seed))
    return (lambda X: prever_tabpfn(m, X)), {"sem_epocas": "in-context learning"}, None


@dataclass
class Algoritmo:
    nome: str
    familia: str
    modo: str                 # codificacao do Preprocessor
    treino: str               # 'dup' (oversampling do enunciado) | 'nat' (TabPFN: sem copias)
    treinar: callable
    padrao: dict              # baseline: hiperparametros padrao + paciencia
    espaco: callable | None   # trial -> dict (sem epocas/arvores: a paciencia decide)
    n_trials: int = 0


def _espaco_mlp(t):
    return dict(n_camadas=t.suggest_int("n_camadas", 1, 3), largura=t.suggest_int("largura", 8, 256, log=True),
                activation=t.suggest_categorical("activation", ["relu", "tanh"]),
                alpha=t.suggest_float("alpha", 1e-6, 1e-1, log=True),
                learning_rate_init=t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                batch_size=t.suggest_categorical("batch_size", [32, 64, 128, 256]))


def _espaco_gb(t):
    return dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=t.suggest_int("max_depth", 2, 6),
                min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 100, log=True),
                subsample=t.suggest_float("subsample", 0.5, 1.0))


def _espaco_xgb(t):
    return dict(learning_rate=t.suggest_float("learning_rate", 0.005, 0.2, log=True),
                max_depth=t.suggest_int("max_depth", 2, 8),
                subsample=t.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                min_child_weight=t.suggest_float("min_child_weight", 0.1, 50, log=True),
                reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100, log=True))


def _espaco_catboost(t):
    return dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                depth=t.suggest_int("depth", 3, 8),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 30, log=True))


def _espaco_stab(t):
    return dict(dim=t.suggest_categorical("dim", [16, 32, 64, 128]), depth=t.suggest_int("depth", 1, 4),
                heads=t.suggest_categorical("heads", [2, 4, 8]),
                attn_dropout=t.suggest_float("attn_dropout", 0.0, 0.4),
                ff_dropout=t.suggest_float("ff_dropout", 0.0, 0.4),
                lr=t.suggest_float("lr", 1e-4, 3e-3, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                batch_size=t.suggest_categorical("batch_size", [64, 128, 256]),
                embedding=t.suggest_categorical("embedding", ["compartilhada", "por_feature"]))


def _espaco_tabkan(t):
    return dict(hidden_width=t.suggest_int("hidden_width", 4, 64, log=True), degree=t.suggest_int("degree", 2, 6),
                lr=t.suggest_float("lr", 1e-4, 1e-2, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                batch_size=t.suggest_categorical("batch_size", [64, 128, 256]))


def _espaco_kan(t):
    return dict(hidden_width=t.suggest_int("hidden_width", 4, 16), grid=t.suggest_int("grid", 3, 10),
                k=t.suggest_int("k", 2, 3), lr=t.suggest_float("lr", 1e-3, 1e-2, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                batch_size=t.suggest_categorical("batch_size", [128, 256]))


ALGORITMOS = {a.nome: a for a in [
    # padrao = mesmos valores dos baselines/rodadas anteriores; so a parada muda
    Algoritmo("mlp", "MLP", "neural", "dup", _mlp,
              dict(n_camadas=1, largura=10, activation="relu", alpha=1e-4, learning_rate_init=1e-3, batch_size=64),
              _espaco_mlp, 40),
    Algoritmo("gb", "Gradient Boosting", "arvores", "dup", _gb, dict(learning_rate=0.1, max_depth=3), _espaco_gb, 50),
    Algoritmo("xgb", "XGBoost", "arvores", "dup", _xgb, dict(learning_rate=0.1, max_depth=3), _espaco_xgb, 100),
    Algoritmo("catboost", "CatBoost", "arvores", "dup", _catboost, dict(learning_rate=0.05, depth=6),
              _espaco_catboost, 60),
    Algoritmo("stab", "STab", "neural", "dup", _stab, dict(dim=32, depth=2, heads=4), _espaco_stab, 30),
    Algoritmo("tabkan", "TabKAN", "neural", "dup", _tabkan, dict(hidden_width=10, degree=3), _espaco_tabkan, 40),
    Algoritmo("kan", "KAN", "neural", "dup", _kan, dict(hidden_width=8, grid=3, k=3), _espaco_kan, 20),
    Algoritmo("tabpfn", "TabPFN v2", "neural", "nat", _tabpfn, dict(n_estimators=8), None),
]}


# ---------------------------------------------------------------- execucao
def _limiar_acuracia(y, p):
    grade = np.linspace(0.02, 0.98, 97)
    return float(grade[np.argmax([np.mean((p >= t) == y) for t in grade])])


def _salvar_modelo(modelo, pasta: Path):
    if modelo is None:
        return
    if hasattr(modelo, "state_dict"):
        import torch
        torch.save(modelo.state_dict(), pasta / "model.pt")
    else:
        import joblib
        joblib.dump(modelo, pasta / "model.joblib")


def pasta_de(base: Path, nome: str, seed: int, rodada: str = RODADA) -> Path:
    return Path(base) / "avaliacao_5_particoes" / rodada / f"{nome}_particao{seed}"


META_HP = ("treino", "transformacao")   # escolhas de dados que a busca ampla trata como hiperparametro


def treinar_e_avaliar(nome: str, alg: Algoritmo, hp: dict, part: Particao, *, results_dirs: list[Path],
                      device: str, transformacao: str = "nenhuma", treino_final_trval: bool = False,
                      n_iter_fixo: int | None = None, extra: dict | None = None,
                      rodada: str = RODADA) -> dict:
    """Treina uma configuracao numa particao e grava em results_dirs[0].
    Se a pasta ja existe em qualquer `results_dirs`, so le o metadata (retomada)."""
    for base in results_dirs:
        meta = pasta_de(base, nome, part.seed, rodada) / "metadata.json"
        if meta.exists():
            return json.loads(meta.read_text(encoding="utf-8"))

    hp_modelo = {k: v for k, v in hp.items() if k not in META_HP}
    transformacao = hp.get("transformacao", transformacao)
    X = part.dados(alg.modo, transformacao)
    chave_tr = ("trval_" if treino_final_trval else "") + hp.get("treino", alg.treino)
    ctx = {"device": device, "colunas": part.colunas[(alg.modo, transformacao)], "n_iter_fixo": n_iter_fixo}
    t0 = time.time()
    prever, info, modelo = alg.treinar(hp_modelo, X[chave_tr], part.y[chave_tr], X["val"], part.y["val"],
                                       part.seed, ctx)
    pv, pt = prever(X["val"]), prever(X["test"])
    hist = info.pop("historico", None)

    info["segundos"] = round(time.time() - t0, 1)
    dataset = {"algoritmo": alg.nome, "familia": alg.familia, "codificacao": alg.modo,
               "transformacao": transformacao, "treino": chave_tr, "n_linhas_treino": int(len(part.y[chave_tr]))}
    return registrar(nome, part, pv, pt, results_dirs[0], hp=hp, dataset=dataset, paciencia=info,
                     extra=extra, historico=hist, modelo=modelo, rodada=rodada)


def registrar(nome: str, part: Particao, pv, pt, base: Path, *, hp: dict, dataset: dict, paciencia: dict,
              extra: dict | None = None, historico=None, modelo=None, rodada: str = RODADA) -> dict:
    """Grava metadata.json + predicoes.npz (+ history.csv, modelo) de uma
    configuracao numa particao. Usado tambem pelos ensembles."""
    limiar = _limiar_acuracia(part.y["val"], pv)
    metr = full_report(part.y["test"], pt)
    metr.update(acuracia_limiar_05=float(np.mean((pt >= 0.5) == part.y["test"])), limiar_validacao=limiar,
                acuracia_limiar_validacao=float(np.mean((pt >= limiar) == part.y["test"])))
    meta = {
        "model_id": f"{nome}_particao{part.seed}", "framework": dataset["familia"], "hyperparameters": hp,
        "metrics": metr,
        "dataset": {"rodada": rodada, "nome_na_tabela": nome, "particao": part.seed, **dataset,
                    "metricas_validacao": {"ks_validacao": full_report(part.y["val"], pv)["ks"],
                                           "auroc_validacao": auroc(part.y["val"], pv)},
                    "paciencia": paciencia, **(extra or {})},
    }
    pasta = pasta_de(base, nome, part.seed, rodada)
    pasta.mkdir(parents=True)
    (pasta / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    np.savez_compressed(pasta / "predicoes.npz", p_val=pv, p_test=pt)
    if historico is not None:
        historico.to_csv(pasta / "history.csv", index=False)
    _salvar_modelo(modelo, pasta)
    return meta


def otimizar(alg: Algoritmo, part: Particao, device: str, seed_sampler: int = 0,
             timeout: float | None = None):
    """Optuna (TPE) na particao `part`: objetivo = AUROC da validacao sem copias,
    cada trial treinado com paciencia. Devolve (melhores_hp, DataFrame dos trials, importancias)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = part.dados(alg.modo)
    ctx = {"device": device, "colunas": part.colunas[(alg.modo, "nenhuma")]}

    def objetivo(trial):
        hp = alg.espaco(trial)
        prever, info, _ = alg.treinar(hp, X[alg.treino], part.y[alg.treino], X["val"], part.y["val"], part.seed, ctx)
        trial.set_user_attr("iteracoes", info.get("melhor_epoca", info.get("melhor_iteracao")))
        return auroc(part.y["val"], prever(X["val"]))

    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed_sampler))
    estudo.enqueue_trial(alg.padrao)   # o padrao entra como 1o trial: a busca nunca fica abaixo dele na validacao
    estudo.optimize(objetivo, n_trials=alg.n_trials, timeout=timeout, catch=(Exception,))
    try:
        imp = optuna.importance.get_param_importances(estudo)
    except Exception:
        imp = {}
    return estudo.best_params, estudo.trials_dataframe(), imp


# ---------------------------------------------------------------- resumo
def carregar(results_dirs: list[Path], rodada: str = RODADA) -> pd.DataFrame:
    """Uma linha por (configuracao, particao) desta rodada, de todas as bases."""
    linhas, vistos = [], set()
    for base in results_dirs:
        for meta in sorted((Path(base) / "avaliacao_5_particoes" / rodada).glob("*/metadata.json")):
            if meta.parent.name in vistos:
                continue
            vistos.add(meta.parent.name)
            m = json.loads(meta.read_text(encoding="utf-8"))
            d = m["dataset"]
            linhas.append({"modelo": d["nome_na_tabela"], "seed": d["particao"], "algoritmo": d["algoritmo"],
                           "familia": d["familia"], "etapa": d.get("etapa", ""),
                           "ks": m["metrics"]["ks"], "auroc": m["metrics"]["auroc"],
                           "acuracia_limiar_validacao": m["metrics"]["acuracia_limiar_validacao"],
                           "auroc_validacao": d["metricas_validacao"]["auroc_validacao"],
                           "iteracoes": d["paciencia"].get("melhor_epoca", d["paciencia"].get("melhor_iteracao")),
                           "teto_atingido": d["paciencia"].get("teto_atingido", False),
                           "pasta": str(meta.parent)})
    return pd.DataFrame(linhas)


def resumir(longo: pd.DataFrame) -> pd.DataFrame:
    g = longo.groupby(["modelo", "algoritmo", "familia", "etapa"]).agg(
        n_particoes=("ks", "size"), ks_mean=("ks", "mean"), ks_std=("ks", "std"),
        auroc_mean=("auroc", "mean"), acuracia_limiar_validacao_mean=("acuracia_limiar_validacao", "mean"),
        auroc_validacao_mean=("auroc_validacao", "mean"), iteracoes_mediana=("iteracoes", "median"),
        teto_atingido=("teto_atingido", "sum"))
    return g.reset_index().sort_values("ks_mean", ascending=False)


def comparar_pareado(longo: pd.DataFrame, novo: str, referencia: str) -> dict:
    """Diferenca de KS por particao (novo - referencia), IC 95% t com n-1 g.l."""
    from scipy import stats
    a = longo[longo.modelo == novo].set_index("seed")["ks"]
    b = longo[longo.modelo == referencia].set_index("seed")["ks"]
    d = (a - b).dropna()
    if len(d) < 2:
        return {"novo": novo, "referencia": referencia, "n": len(d)}
    meia = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"novo": novo, "referencia": referencia, "n": len(d), "delta_ks": d.mean(),
            "ic95_inf": d.mean() - meia, "ic95_sup": d.mean() + meia,
            "veredito": "melhor" if d.mean() - meia > 0 else "pior" if d.mean() + meia < 0 else "ruido"}


if __name__ == "__main__":
    # self-check rapido (sem torch): GB com paciencia e Preprocessor com faixas
    from .data import load_raw
    df = load_raw(Path(__file__).resolve().parents[2] / "data")
    part = preparar_particao(df, 0)
    for t in ("nenhuma", "log", "faixas_adiciona", "faixas_substitui"):
        for modo in ("neural", "arvores"):
            X = part.dados(modo, t)
            assert X["test"].shape[1] == len(part.colunas[(modo, t)]) and np.isfinite(X["test"]).all(), (modo, t)
    prever, info, _ = _gb(dict(learning_rate=0.2, max_depth=3), part.dados("arvores")["dup"], part.y["dup"],
                          part.dados("arvores")["val"], part.y["val"], 0, {})
    a = auroc(part.y["test"], prever(part.dados("arvores")["test"]))
    assert 1 <= info["melhor_iteracao"] < TETO_ARVORES_GB and a > 0.8, (info, a)
    print("rodada8 self-check ok:", info, f"auroc_teste={a:.4f}")
