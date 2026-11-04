"""Rodada 9 — busca ampla de hiperparametros.

O que muda em relacao a busca da rodada 8 (etapa B):

1. **Espaco completo.** Cada algoritmo expoe (quase) todos os
   hiperparametros da biblioteca, e as escolhas de dados viram
   hiperparametros tambem: `treino` (duplicacao do enunciado vs. sem copias),
   `transformacao` (nenhuma/log/faixas), a propria paciencia e o criterio de
   parada (AUROC vs. cross-entropy de validacao).
2. **Objetivo por validacao cruzada.** A rodada 8 escolhia pelo AUROC de UMA
   validacao (25% dos dados): o vencedor da busca e favorecido pela sorte
   daquele pedaco (foi o que fez a busca da rodada 3 "ganhar" na validacao e
   perder no teste). Aqui o objetivo e o AUROC medio em 3 folds estratificados
   de treino+validacao da particao 0 (cada fold de treino tem o mesmo tamanho
   do treino original; a paciencia olha o fold de fora). O teste nao participa.
3. **Retomavel.** O estudo Optuna fica num arquivo journal
   (`reports/tables/rodada9_optuna_{alg}.journal`): se a sessao cair, os trials
   ja feitos sao recarregados e a busca continua.

Depois da busca, a melhor configuracao e avaliada nas 5 particoes com o
protocolo normal (treina no treino, paciencia na validacao, teste uma vez).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import rodada8 as r8
from .features import Preprocessor
from .metrics import auroc

RODADA = "rodada9"
N_FOLDS = 3

N_TRIALS = {"mlp": 60, "gb": 60, "xgb": 150, "catboost": 80, "stab": 40, "tabkan": 50, "kan": 25, "tabpfn": 25}


# ---------------------------------------------------------------- espacos
def _dados(t, alg):
    """Escolhas de dados como hiperparametro."""
    hp = {}
    if alg.nome != "tabpfn":   # TabPFN com copias = memorizacao (rodada 4: -0.038 de KS)
        hp["treino"] = t.suggest_categorical("treino", ["dup", "nat"])
    opcoes = ["nenhuma", "faixas_adiciona", "faixas_substitui"] + (["log"] if alg.modo == "neural" else [])
    hp["transformacao"] = t.suggest_categorical("transformacao", opcoes)
    return hp


def _redes(t):
    return dict(patience=t.suggest_categorical("patience", [15, 30, 60]),
                early_stop_metric=t.suggest_categorical("early_stop_metric", ["val_auroc", "val_loss"]))


def _arvores(t, nome="early_stopping_rounds"):
    return {nome: t.suggest_categorical(nome, [100, 200, 400])}


def espaco_mlp(t):
    return dict(n_camadas=t.suggest_int("n_camadas", 1, 4), largura=t.suggest_int("largura", 8, 512, log=True),
                activation=t.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
                alpha=t.suggest_float("alpha", 1e-7, 1e-1, log=True),
                learning_rate_init=t.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
                batch_size=t.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512]), **_redes(t))


def espaco_gb(t):
    return dict(learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 2, 8),
                min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 200, log=True),
                subsample=t.suggest_float("subsample", 0.4, 1.0),
                max_features=t.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5, 0.75]),
                **_arvores(t))


def espaco_xgb(t):
    hp = dict(learning_rate=t.suggest_float("learning_rate", 0.003, 0.3, log=True),
              max_depth=t.suggest_int("max_depth", 2, 10),
              subsample=t.suggest_float("subsample", 0.4, 1.0),
              colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
              colsample_bylevel=t.suggest_float("colsample_bylevel", 0.2, 1.0),
              colsample_bynode=t.suggest_float("colsample_bynode", 0.2, 1.0),
              min_child_weight=t.suggest_float("min_child_weight", 0.01, 100, log=True),
              reg_lambda=t.suggest_float("reg_lambda", 1e-4, 100, log=True),
              reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10, log=True),
              gamma=t.suggest_float("gamma", 1e-8, 10, log=True),
              max_delta_step=t.suggest_float("max_delta_step", 0, 10),
              scale_pos_weight=t.suggest_float("scale_pos_weight", 0.5, 4, log=True),
              max_bin=t.suggest_categorical("max_bin", [32, 64, 128, 256, 512]),
              grow_policy=t.suggest_categorical("grow_policy", ["depthwise", "lossguide"]), **_arvores(t))
    if hp["grow_policy"] == "lossguide":
        hp["max_leaves"] = t.suggest_int("max_leaves", 4, 256, log=True)
    return hp


def espaco_catboost(t):
    hp = dict(learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True),
              depth=t.suggest_int("depth", 2, 10),
              l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.1, 100, log=True),
              random_strength=t.suggest_float("random_strength", 1e-3, 10, log=True),
              bagging_temperature=t.suggest_float("bagging_temperature", 0, 10),
              border_count=t.suggest_categorical("border_count", [32, 64, 128, 254]),
              one_hot_max_size=t.suggest_categorical("one_hot_max_size", [2, 4, 8, 16]),
              auto_class_weights=t.suggest_categorical("auto_class_weights", [None, "Balanced", "SqrtBalanced"]),
              grow_policy=t.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
              **_arvores(t))
    if hp["grow_policy"] != "SymmetricTree":
        hp["min_data_in_leaf"] = t.suggest_int("min_data_in_leaf", 1, 200, log=True)
    if hp["grow_policy"] == "Lossguide":
        hp["max_leaves"] = t.suggest_int("max_leaves", 4, 64, log=True)
    return hp


def espaco_stab(t):
    return dict(dim=t.suggest_categorical("dim", [16, 32, 64, 128]), depth=t.suggest_int("depth", 1, 6),
                heads=t.suggest_categorical("heads", [1, 2, 4, 8]),
                attn_dropout=t.suggest_float("attn_dropout", 0.0, 0.5),
                ff_dropout=t.suggest_float("ff_dropout", 0.0, 0.5),
                lr=t.suggest_float("lr", 3e-5, 3e-3, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-7, 1e-1, log=True),
                batch_size=t.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
                embedding=t.suggest_categorical("embedding", ["compartilhada", "por_feature"]),
                scheduler=t.suggest_categorical("scheduler", ["none", "plateau"]),
                max_grad_norm=t.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0]), **_redes(t))


def espaco_tabkan(t):
    return dict(hidden_width=t.suggest_int("hidden_width", 2, 128, log=True), degree=t.suggest_int("degree", 1, 8),
                lr=t.suggest_float("lr", 3e-5, 3e-2, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-7, 1e-1, log=True),
                batch_size=t.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
                scheduler=t.suggest_categorical("scheduler", ["none", "plateau"]),
                max_grad_norm=t.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0]), **_redes(t))


def espaco_kan(t):
    return dict(hidden_width=t.suggest_int("hidden_width", 2, 32, log=True), grid=t.suggest_int("grid", 3, 15),
                k=t.suggest_int("k", 1, 4), lr=t.suggest_float("lr", 3e-4, 3e-2, log=True),
                weight_decay=t.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
                batch_size=t.suggest_categorical("batch_size", [64, 128, 256, 512]),
                scheduler=t.suggest_categorical("scheduler", ["none", "plateau"]),
                max_grad_norm=t.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0]), **_redes(t))


def espaco_tabpfn(t):
    return dict(n_estimators=t.suggest_categorical("n_estimators", [4, 8, 16, 32]),
                softmax_temperature=t.suggest_float("softmax_temperature", 0.5, 1.5),
                average_before_softmax=t.suggest_categorical("average_before_softmax", [False, True]),
                balance_probabilities=t.suggest_categorical("balance_probabilities", [False, True]))


ESPACOS = {"mlp": espaco_mlp, "gb": espaco_gb, "xgb": espaco_xgb, "catboost": espaco_catboost,
           "stab": espaco_stab, "tabkan": espaco_tabkan, "kan": espaco_kan, "tabpfn": espaco_tabpfn}


def espaco_completo(alg):
    return lambda t: {**_dados(t, alg), **ESPACOS[alg.nome](t)}


# ---------------------------------------------------------------- validacao cruzada
class FoldsCV:
    """3 folds estratificados de treino+validacao (sem copias) de uma particao.
    Cada fold: Preprocessor ajustado no fold de treino; versao 'dup' com a
    classe minoritaria reamostrada (com reposicao) SO dentro do fold de treino."""

    def __init__(self, part: r8.Particao, n_folds: int = N_FOLDS, seed: int = 0):
        from sklearn.model_selection import StratifiedKFold
        self.df = part._dfs["trval_nat"].reset_index(drop=True)
        y = (self.df["Churn"].values == "Yes").astype(int)
        self.folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(self.df, y))
        self.seed, self._cache = seed, {}

    def dados(self, modo: str, transformacao: str, treino: str):
        chave = (modo, transformacao, treino)
        if chave not in self._cache:
            saida = []
            for i, (tr, va) in enumerate(self.folds):
                d_tr, d_va = self.df.iloc[tr], self.df.iloc[va]
                if treino == "dup":
                    rng = np.random.default_rng(self.seed + i)
                    pos = np.where(d_tr["Churn"].values == "Yes")[0]
                    neg = np.where(d_tr["Churn"].values != "Yes")[0]
                    idx = np.concatenate([neg, rng.choice(pos, size=len(neg), replace=True)])
                    d_fit = d_tr.iloc[rng.permutation(idx)]
                else:
                    d_fit = d_tr
                prep = Preprocessor(modo=modo, transformacao=transformacao).fit(d_tr)
                y_of = lambda d: (d["Churn"].values == "Yes").astype(np.float64)
                saida.append((prep.transform(d_fit)[0], y_of(d_fit), prep.transform(d_va)[0], y_of(d_va),
                              prep.transform(d_tr.head(2))[1]))
            self._cache[chave] = saida
        return self._cache[chave]


def _storage(caminho: Path):
    import optuna
    try:   # optuna >= 4
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        return optuna.storages.JournalStorage(JournalFileBackend(str(caminho), lock_obj=JournalFileOpenLock(str(caminho))))
    except ImportError:
        return optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(str(caminho), lock_obj=optuna.storages.JournalFileOpenLock(str(caminho))))


def otimizar_cv(alg: r8.Algoritmo, cv: FoldsCV, device: str, journal: Path, n_trials: int,
                warm_start: list[dict] = (), timeout: float | None = None):
    """Busca TPE com objetivo = AUROC medio nos folds. Retoma do `journal`.
    `warm_start`: configuracoes enfileiradas no inicio (ex.: melhor da rodada 8)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo = optuna.create_study(study_name=alg.nome, storage=_storage(journal), load_if_exists=True,
                                 direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True))
    feitos = [t for t in estudo.trials if t.state.is_finished()]
    if not feitos:
        for hp in warm_start:
            estudo.enqueue_trial(hp, skip_if_exists=True)
    espaco = espaco_completo(alg)

    def objetivo(trial):
        hp = espaco(trial)
        hp_modelo = {k: v for k, v in hp.items() if k not in r8.META_HP}
        aucs, iters = [], []
        for Xtr, ytr, Xva, yva, cols in cv.dados(alg.modo, hp["transformacao"], hp.get("treino", alg.treino)):
            prever, info, _ = alg.treinar(hp_modelo, Xtr, ytr, Xva, yva, 0, {"device": device, "colunas": cols})
            aucs.append(auroc(yva, prever(Xva)))
            iters.append(info.get("melhor_epoca", info.get("melhor_iteracao")))
            trial.report(float(np.mean(aucs)), len(aucs) - 1)
        trial.set_user_attr("auroc_folds", aucs)
        trial.set_user_attr("iteracoes_folds", iters)
        return float(np.mean(aucs))

    restantes = n_trials - len(feitos)
    if restantes > 0:
        estudo.optimize(objetivo, n_trials=restantes, timeout=timeout, catch=(Exception,))
    return estudo


def importancias(estudo) -> dict:
    """fANOVA sobre os trials completos (quanto da variacao do objetivo cada
    hiperparametro explica). Vazio se houver trials de menos."""
    import optuna
    try:
        return optuna.importance.get_param_importances(estudo)
    except Exception:
        return {}


def efeito_por_valor(estudo, top: float = 0.2) -> pd.DataFrame:
    """Para cada hiperparametro categorico: AUROC CV medio por valor e a
    frequencia do valor entre os `top` melhores trials — mostra QUAL valor ajuda."""
    df = estudo.trials_dataframe()
    df = df[df.state == "COMPLETE"]
    if df.empty:
        return pd.DataFrame()
    corte = df["value"].quantile(1 - top)
    linhas = []
    for c in [c for c in df.columns if c.startswith("params_")]:
        col = df[c].astype(str)
        if col.nunique() > 8:   # numerico continuo: resumido pela fANOVA
            continue
        for v, g in df.groupby(col):
            linhas.append({"hiperparametro": c[7:], "valor": v, "n_trials": len(g), "auroc_cv_medio": g["value"].mean(),
                           "frac_no_top": float((g["value"] >= corte).mean())})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    # self-check (sem torch): espacos amostram e a CV roda com GB/XGB em 2 trials
    import tempfile
    from .data import load_raw
    df = load_raw(Path(__file__).resolve().parents[2] / "data")
    part = r8.preparar_particao(df, 0)
    cv = FoldsCV(part)
    with tempfile.TemporaryDirectory() as d:
        for nome in ("gb", "xgb"):
            alg = r8.ALGORITMOS[nome]
            j = Path(d) / f"{nome}.journal"
            e = otimizar_cv(alg, cv, "cpu", j, n_trials=2, warm_start=[{**alg.padrao, "treino": "dup",
                                                                         "transformacao": "nenhuma"}])
            e2 = otimizar_cv(alg, cv, "cpu", j, n_trials=3)   # retomada: so 1 trial novo
            ok = [t for t in e2.trials if t.state.name == "COMPLETE"]
            assert len(e2.trials) == 3 and ok and 0.8 < e2.best_value < 0.9, [t.state for t in e2.trials]
            print(nome, "ok:", round(e2.best_value, 4), e2.best_params)
    print("busca_ampla self-check ok")
