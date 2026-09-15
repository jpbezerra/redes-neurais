"""Gera `notebooks/01_train_lstm_bitcoin.ipynb`.

Editar um .ipynb à mão é frágil (JSON grande, outputs embutidos), então o
notebook é gerado por script — mesmo padrão adotado no Mini-projeto 1. Rodar
de novo regenera o arquivo do zero.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_train_lstm_bitcoin.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# Mini-projeto 2 — LSTM para prever o preço do Bitcoin

Enunciado: empregar LSTM para prever o preço do Bitcoin de dezembro de 2014 a
maio de 2018, usando os primeiros 80% dos registros para treinar e os 20%
finais para teste.

Este notebook segue a mesma organização do Mini-projeto 1: o código fica no
pacote `src/lstm_bitcoin/` e o notebook apenas orquestra experimentos e
reporta resultados.

**O que é feito aqui, em ordem:**

1. Carregar a série e olhar para ela antes de modelar (seção 1)
2. Estabelecer o **baseline ingênuo** — o número que qualquer modelo precisa bater (seção 2)
3. Baseline LSTM reproduzindo o tutorial do enunciado (seção 3)
4. O diagnóstico que muda tudo: nível vs. variação (seção 4)
5. Busca manual por levas (seção 5)
6. Busca automática com **Optuna** (seção 6)
7. Tarefa secundária: prever a **direção** do movimento, com matriz de confusão (seção 7)
8. Validação na base moderna do Kaggle (seção 8)
9. Tabela de hiperparâmetros e conclusões (seção 9)
"""),

    md("## 0. Setup do ambiente"),
    code("""
#@title Setup (Colab, Kaggle ou local)
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/jpbezerra/redes-neurais.git"   # ajuste se necessario
REPO_DIR = "redes-neurais"
SUBPATH  = "miniprojeto/mp2-lstm-bitcoin"

def _sh(cmd):
    print("$", cmd); subprocess.run(cmd, shell=True, check=False)

IN_COLAB  = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle")

if IN_COLAB or IN_KAGGLE:
    base = Path("/content") if IN_COLAB else Path("/kaggle/working")
    root = base / REPO_DIR
    if root.exists():
        _sh(f"cd {root} && git pull --ff-only")
    else:
        _sh(f"git clone {REPO_URL} {root}")
    PROJECT_ROOT = root / SUBPATH
    _sh(f"pip install -q -e {PROJECT_ROOT}")
    _sh("pip install -q optuna")
else:
    PROJECT_ROOT = Path.cwd().parent

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
"""),
    code("""
#@title Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from lstm_bitcoin.config import ExperimentConfig
from lstm_bitcoin.data import load_dataset, prepare_splits
from lstm_bitcoin.train import fit, fit_or_load
from lstm_bitcoin.metrics import (regression_scores, naive_baseline_scores,
                                  confusion_matrix, confusion_matrix_percent, format_confusion)
from lstm_bitcoin.checkpointing import load_all_metadata, hyperparameter_table
from lstm_bitcoin.utils import (set_seed, get_device, plot_forecast, plot_training_curves,
                                plot_confusion, plot_residuals)

device = get_device()
print("device:", device)
"""),

    md("""
## 1. A série, antes de qualquer modelo

Olhar para os dados antes de modelar não é formalidade: a característica que
esta célula revela é a que decide todo o resto do projeto.
"""),
    code("""
#@title Carregar e inspecionar
df = load_dataset("tutorial", data_dir=DATA_DIR)
print(f"{len(df)} dias, de {df.Date.min().date()} a {df.Date.max().date()}")
display(df.head(3))
display(df[["Open","High","Low","Close","Volume"]].describe().round(2))

# Divisao temporal do enunciado: 80% treino / 20% teste (o treino ainda cede
# 10% finais para validacao, usada no early stopping).
n = len(df); n_test = int(n*0.2); n_val = int((n-n_test)*0.1); n_train = n-n_test-n_val

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df.Date[:n_train], df.Close[:n_train], lw=1.2, label=f"treino ({n_train}d)")
ax.plot(df.Date[n_train:n_train+n_val], df.Close[n_train:n_train+n_val], lw=1.2, label=f"validacao ({n_val}d)")
ax.plot(df.Date[n_train+n_val:], df.Close[n_train+n_val:], lw=1.2, label=f"teste ({n_test}d)")
ax.set_ylabel("Preco (USD)"); ax.legend(frameon=False, ncol=3)
ax.set_title("Bitcoin — divisao temporal (nunca aleatoria)", fontsize=11, loc="left")
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(FIGURES_DIR/"serie_divisao.png", bbox_inches="tight"); plt.show()

print(f"faixa de preco no TREINO: US$ {df.Close[:n_train].min():,.0f} a US$ {df.Close[:n_train].max():,.0f}")
print(f"faixa de preco no TESTE : US$ {df.Close[n_train+n_val:].min():,.0f} a US$ {df.Close[n_train+n_val:].max():,.0f}")
"""),
    md("""
### O problema que essa célula revela

O treino vive entre **US$ 120 e US$ 2.698**. O teste vive entre **US$ 3.617 e
US$ 19.650**. As duas faixas **não se sobrepõem**: o menor preço do teste é
maior que o maior preço do treino.

Isso é fatal para um modelo que prevê o *nível* do preço. A rede aprende a
produzir saídas dentro da faixa que viu, e no teste é obrigada a extrapolar
para valores 7× maiores. Nenhuma escolha de normalização conserta isso — é
uma característica do problema, não do modelo.

Guarde essa observação: ela explica o resultado da seção 3 e motiva a 4.
"""),

    md("""
## 2. O baseline ingênuo — o número que precisa ser batido

Antes de treinar qualquer rede, é obrigatório saber quanto vale o palpite mais
burro possível: **"amanhã o preço será igual ao de hoje"**.

Séries financeiras são próximas de um passeio aleatório, o que torna esse
baseline surpreendentemente forte. Um modelo que não o supera não aprendeu
nada — ele apenas aprendeu a copiar o último valor com um dia de atraso, o que
produz um gráfico lindo e um RMSE enganoso.
"""),
    code("""
#@title Baseline ingenuo
close = df.Close.to_numpy()
split = n_train + n_val
y_true_naive = close[split:]
y_pred_naive = close[split-1:-1]      # valor do dia anterior

naive = regression_scores(y_true_naive, y_pred_naive)
subiu = (np.diff(close[split-1:]) > 0)
print("BASELINE INGENUO (repete o preco de ontem)")
for k, v in naive.items(): print(f"  {k:>6}: {v:,.4f}")
print(f"\\n  dias de alta no teste: {subiu.mean():.1%}  <- baseline da tarefa de direcao")
"""),

    md("""
## 3. Baseline LSTM — reproduzindo o tutorial do enunciado

Configuração próxima da do tutorial citado: janela de 60 dias, apenas o preço
de fechamento como feature, normalização MinMax e previsão do **nível** do
preço.
"""),
    code("""
#@title Baseline: prever o NIVEL do preco
baseline_config = ExperimentConfig(
    run_name="baseline_nivel",
    window=60, features=("Close",), target="Close",
    scaler="minmax", diff_target=False,
    model_type="lstm", hidden_size=64, num_layers=2,
    optimizer="adam", learning_rate=1e-3, loss="mse",
    batch_size=32, num_epochs=100, patience=15,
    notes="Reproduz o tutorial do enunciado: janela 60, MinMax, previsao do nivel do preco.",
)

splits = prepare_splits(df, baseline_config)
set_seed(baseline_config.seed)
experiment_results = {}
experiment_results["baseline_nivel"] = fit_or_load(
    baseline_config, splits, device, results_dir=RESULTS_DIR)

res = experiment_results["baseline_nivel"]
print({k: round(v,4) if isinstance(v,float) else v for k,v in res["test_scores"].items()})
"""),
    code("""
#@title Grafico: o que o baseline realmente faz
r = experiment_results["baseline_nivel"]
if "test_result" in r and "y_pred" in r.get("test_result", {}):
    tr = r["test_result"]
    plot_forecast(splits["test_dates"], tr["y_true"], tr["y_pred"], naive=tr.get("anchor"),
                  title="Baseline (nivel) — previsao vs. real",
                  save_path=FIGURES_DIR/"baseline_nivel.png"); plt.show()
    plot_training_curves(r["history"], save_path=FIGURES_DIR/"baseline_curvas.png"); plt.show()
"""),
    md("""
### Leitura do baseline

Compare o RMSE com o do baseline ingênuo da seção 2. O campo `beats_naive`
responde diretamente: o modelo supera ou não o palpite trivial?

Se `r2` vier **negativo**, o modelo é literalmente pior do que ter respondido
a média do conjunto em todos os dias — e isso acontece justamente pela
extrapolação diagnosticada na seção 1.
"""),

    md("""
## 4. O ajuste decisivo: prever a variação, não o nível

A solução para a extrapolação é mudar o que se pede ao modelo.

**Nível:** "qual será o preço amanhã?" — o alvo em 2018 é um número que nunca
apareceu no treino.

**Variação:** "quanto o preço vai mudar de hoje para amanhã?" — subir 3% em
2016 e subir 3% em 2018 são o mesmo número, ainda que o preço seja 10× maior.
A variação é aproximadamente **estacionária**, e é isso que permite que o que
foi aprendido no treino continue valendo no teste.

O preço é reconstruído depois: `preço previsto = último preço + variação prevista`.

Esta célula compara as duas formulações mantendo todo o resto igual.
"""),
    code("""
#@title Nivel vs. variacao, com o resto identico
comparacao = []
for nome, kw in [("nivel",    dict(scaler="minmax",   diff_target=False)),
                 ("nivel_log",dict(scaler="minmax",   diff_target=False, log_price=True)),
                 ("variacao", dict(scaler="standard", diff_target=True)),
                 ("retorno",  dict(scaler="standard", diff_target=True, log_price=True))]:
    cfg = ExperimentConfig(run_name=f"formul_{nome}", window=60, hidden_size=64, num_layers=2,
                           num_epochs=60, patience=12, grad_clip=1.0,
                           notes=f"Comparacao de formulacao do alvo: {nome}.", **kw)
    s = prepare_splits(df, cfg); set_seed(cfg.seed)
    r = fit_or_load(cfg, s, device, results_dir=RESULTS_DIR)
    experiment_results[cfg.run_name] = r
    sc = r["test_scores"]
    comparacao.append({"formulacao": nome, "rmse": sc["rmse"], "mae": sc["mae"],
                       "r2": sc["r2"], "dir_acc": sc.get("directional_accuracy"),
                       "bate_ingenuo": sc.get("beats_naive")})

pd.DataFrame(comparacao).round(4)
"""),

    md("""
## 5. Busca manual por levas

Mesma metodologia do Mini-projeto 1: cada leva testa uma alavanca isolada
sobre a melhor configuração conhecida, e o vencedor vira a base da leva
seguinte. Isso produz o "por quê" que uma busca automática não entrega.

A **janela** é o hiperparâmetro mais característico de série temporal e não
tem equivalente no projeto anterior: ela define quantos dias de passado o
modelo enxerga para decidir o próximo.
"""),
    code("""
#@title Leva 1 — variacoes isoladas sobre a melhor formulacao
BASE = dict(scaler="standard", diff_target=True, grad_clip=1.0,
            num_epochs=60, patience=12, features=("Close",))

leva1 = [
    ExperimentConfig(run_name="l1_window15",  window=15,  hidden_size=64, **BASE,
                     notes="Janela curta: so as ultimas 2 semanas importam?"),
    ExperimentConfig(run_name="l1_window30",  window=30,  hidden_size=64, **BASE,
                     notes="Janela de 1 mes."),
    ExperimentConfig(run_name="l1_window90",  window=90,  hidden_size=64, **BASE,
                     notes="Janela longa: 3 meses de contexto."),
    ExperimentConfig(run_name="l1_hidden16",  window=60, hidden_size=16, **BASE,
                     notes="Capacidade menor — a serie e curta, talvez 64 seja demais."),
    ExperimentConfig(run_name="l1_hidden128", window=60, hidden_size=128, **BASE,
                     notes="Capacidade maior."),
    ExperimentConfig(run_name="l1_gru",       window=60, hidden_size=64, model_type="gru", **BASE,
                     notes="GRU: versao simplificada da LSTM, menos parametros."),
    ExperimentConfig(run_name="l1_rnn",       window=60, hidden_size=64, model_type="rnn", **BASE,
                     notes="RNN simples: a referencia que a LSTM veio superar."),
    ExperimentConfig(run_name="l1_ohlcv",     window=60, hidden_size=64,
                     **{**BASE, "features": ("Open","High","Low","Close","Volume")},
                     notes="Usa OHLCV completo em vez de so o fechamento."),
]

for cfg in leva1:
    if cfg.run_name in experiment_results: continue
    s = prepare_splits(df, cfg); set_seed(cfg.seed)
    experiment_results[cfg.run_name] = fit_or_load(cfg, s, device, results_dir=RESULTS_DIR)

pd.DataFrame({k: v["test_scores"] for k,v in experiment_results.items()}).T.sort_values("rmse").round(4)
"""),
    md("""
### Espaço para a sua análise da leva 1

Depois de rodar, registre aqui: qual janela venceu e o que isso diz sobre a
memória útil da série? A LSTM superou a GRU e a RNN simples? Adicionar OHLCV
ajudou ou o volume só trouxe ruído?

A leva 2 deve combinar os vencedores, como no Mini-projeto 1.
"""),

    md("""
## 6. Busca automática com Optuna

A busca manual explica *por quê*; o Optuna cobre *muito mais espaço*. Ele
propõe combinações, observa o resultado e concentra as próximas propostas nas
regiões promissoras (amostrador TPE), abortando cedo os treinos que vão mal
(pruning).

Dois cuidados no código: a otimização usa **apenas a validação** — o teste é
tocado uma única vez, no final, para o modelo escolhido — e o estudo pode ser
salvo em SQLite para sobreviver à queda da sessão.
"""),
    code("""
#@title Rodar o estudo
from lstm_bitcoin.tuning import run_study, best_config, study_dataframe, importance_table

base_tune = ExperimentConfig(run_name="tune", scaler="standard", diff_target=True,
                             num_epochs=40, patience=8)

study = run_study(df, base_tune, device,
                  n_trials=50,                       # aumente se houver tempo de GPU
                  study_name="lstm_btc",
                  storage=f"sqlite:///{RESULTS_DIR}/optuna.db")

print("melhor val_loss:", study.best_value)
print("melhores parametros:", study.best_params)
study_dataframe(study).head(15)
"""),
    code("""
#@title Importancia dos hiperparametros (fANOVA)
imp = importance_table(study)
display(imp)

if not imp.empty:
    fig, ax = plt.subplots(figsize=(7,3.4))
    ax.barh(imp.hiperparametro[::-1], imp.importancia[::-1], color="#2F6F9F")
    ax.set_xlabel("Importancia relativa"); ax.grid(axis="y", alpha=0)
    ax.set_title("O que mais explicou a variacao de desempenho", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(FIGURES_DIR/"optuna_importancia.png", bbox_inches="tight"); plt.show()
"""),
    code("""
#@title Treinar e registrar o melhor config encontrado
cfg_best = best_config(study, base_tune, run_name="optuna_best")
cfg_best.num_epochs, cfg_best.patience = 120, 20   # treino final mais longo
cfg_best.notes = "Melhor configuracao encontrada pelo Optuna, retreinada por mais epocas."

s_best = prepare_splits(df, cfg_best); set_seed(cfg_best.seed)
experiment_results["optuna_best"] = fit_or_load(cfg_best, s_best, device, results_dir=RESULTS_DIR)
print({k: round(v,4) if isinstance(v,float) else v
       for k,v in experiment_results["optuna_best"]["test_scores"].items()})
"""),

    md("""
## 7. Tarefa secundária: prever a direção do movimento

Previsão de preço é regressão e não produz matriz de confusão. Mas a pergunta
que de fato importa — *o preço sobe ou desce amanhã?* — é classificação
binária, e essa sim produz matriz de confusão.

Vale fazer as duas porque um RMSE baixo pode esconder um modelo inútil: se ele
só copia o preço de ontem, acerta o nível e erra metade das direções. A
acurácia direcional é o teste real de poder preditivo.

O baseline aqui é a **taxa da classe majoritária** (seção 2): se 52% dos dias
foram de alta, responder "alta" sempre acerta 52%.
"""),
    code("""
#@title Treinar o classificador de direcao
cfg_dir = ExperimentConfig(
    run_name="direcao_baseline", task="direction",
    window=30, hidden_size=64, num_layers=2,
    scaler="standard", diff_target=True, grad_clip=1.0,
    num_epochs=80, patience=15,
    notes="Classificacao binaria alta/baixa do dia seguinte — gera a matriz de confusao.",
)
s_dir = prepare_splits(df, cfg_dir); set_seed(cfg_dir.seed)
experiment_results["direcao_baseline"] = fit_or_load(cfg_dir, s_dir, device, results_dir=RESULTS_DIR)

sc = experiment_results["direcao_baseline"]["test_scores"]
print({k: round(v,4) for k,v in sc.items()})
print(f"\\nGanho sobre o baseline majoritario: "
      f"{(sc['accuracy']-sc['majority_rate'])*100:+.2f} p.p.")
"""),
    code("""
#@title Matriz de confusao — em numero e em percentual
cm = np.array(experiment_results["direcao_baseline"]["test_result"]["confusion_matrix"])

print(format_confusion(cm)); print()
print("PERCENTUAL POR COLUNA (precision):")
print(np.round(confusion_matrix_percent(cm, "pred"), 1))

plot_confusion(cm, title="Direcao do preco — matriz de confusao",
               save_path=FIGURES_DIR/"matriz_confusao.png"); plt.show()

display(pd.DataFrame(cm, index=["real baixa","real alta"], columns=["prev. baixa","prev. alta"]))
display(pd.DataFrame(np.round(confusion_matrix_percent(cm,"true"),2),
                     index=["real baixa","real alta"],
                     columns=["prev. baixa %","prev. alta %"]))
"""),

    md("""
## 8. Validação na base moderna (Kaggle)

O recorte do enunciado termina em maio de 2018. Verificar se o que funcionou
ali continua valendo num período diferente é o teste mais honesto de
generalização — e é a resposta pronta caso perguntem "isso funciona hoje?".

Baixe `mczielinski/bitcoin-historical-data` (1 minuto, desde 2012, atualizado
diariamente) e coloque o CSV em `data/`. A função reamostra para diário
seguindo a convenção OHLCV.
"""),
    code("""
#@title Repetir o melhor config na base do Kaggle
KAGGLE_CSV = DATA_DIR / "btcusd_1-min_data.csv"

if KAGGLE_CSV.exists():
    df_k = load_dataset("kaggle", data_dir=DATA_DIR, csv_path=KAGGLE_CSV)
    print(f"{len(df_k)} dias, de {df_k.Date.min().date()} a {df_k.Date.max().date()}")

    cfg_k = ExperimentConfig(**{**cfg_best.to_dict(),
                                "run_name": "kaggle_best", "dataset": "kaggle",
                                "notes": "Melhor config do Optuna aplicada a base moderna do Kaggle."})
    s_k = prepare_splits(df_k, cfg_k); set_seed(cfg_k.seed)
    experiment_results["kaggle_best"] = fit_or_load(cfg_k, s_k, device, results_dir=RESULTS_DIR)
    print({k: round(v,4) if isinstance(v,float) else v
           for k,v in experiment_results["kaggle_best"]["test_scores"].items()})
else:
    print(f"Base do Kaggle nao encontrada em {KAGGLE_CSV}.")
    print("Baixe com:  kaggle datasets download -d mczielinski/bitcoin-historical-data")
"""),

    md("""
## 9. Tabela de hiperparâmetros e resultados

`hyperparameter_table` monta a tabela do relatório final mostrando **apenas os
hiperparâmetros que de fato variaram** entre as execuções, junto das métricas
de teste. Colunas constantes são omitidas para a tabela não virar um muro de
valores repetidos.
"""),
    code("""
#@title Tabela final de hiperparametros variados
tabela = hyperparameter_table(RESULTS_DIR, only_varied=True)
display(tabela)
tabela.to_csv(RESULTS_DIR/"tabela_hiperparametros.csv", index=False)
print(f"Salva em {RESULTS_DIR/'tabela_hiperparametros.csv'}")
"""),
    code("""
#@title Todas as execucoes salvas
dfm = load_all_metadata(RESULTS_DIR)
if not dfm.empty:
    cols = [c for c in ["run_name","task","metrics.test_rmse","metrics.test_mae",
                        "metrics.test_directional_accuracy","metrics.test_accuracy",
                        "metrics.epochs_trained"] if c in dfm.columns]
    display(dfm[cols].sort_values(cols[2] if len(cols)>2 else "run_name"))
"""),

    md("""
## 10. Conclusões

Preencher ao final, cobrindo:

- **O baseline ingênuo foi superado?** Por quanto? Se não foi, o modelo não tem valor prático.
- **Nível vs. variação:** qual a diferença medida, e por quê?
- **Qual hiperparâmetro mais importou** — comparar a intuição das levas manuais com o ranking fANOVA do Optuna.
- **Acurácia direcional:** ficou acima da classe majoritária? Quanto?
- **A matriz de confusão é simétrica?** Se o modelo erra muito mais num sentido, ele tem viés de alta ou de baixa — o que é esperado numa série que subiu muito no período de teste.
- **Generalizou para a base moderna?** Se o desempenho caiu, o modelo aprendeu o regime de 2014-2018, não o Bitcoin.
- **Limites honestos:** séries financeiras são próximas de passeio aleatório; ganhos pequenos sobre o ingênuo são o resultado esperado, e qualquer coisa espetacular quase sempre indica vazamento temporal.
"""),
]


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Notebook gerado com {len(CELLS)} células em {NB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
