"""Gera `notebooks/01_train_lstm_bitcoin.ipynb`.

Editar um .ipynb à mão é frágil (JSON grande, outputs embutidos), então o
notebook é gerado por script — mesmo padrão adotado no Mini-projeto 1. Rodar
de novo regenera o arquivo do zero.
"""

from __future__ import annotations

import hashlib
import json
import sys
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

    md("""
## 0. Setup do ambiente

**No Kaggle, antes de rodar:** ative *Settings → Internet → On* (o notebook
clona o repositório e baixa o CSV) e, se for rodar o Optuna com muitos trials,
*Accelerator → GPU*. Depois use **"Save Version" → "Save & Run All (Commit)"**,
que executa num kernel gerenciado em segundo plano e não depende da aba ficar
aberta.
"""),
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
RESULTS_DIR = PROJECT_ROOT / "results"          # so results/{model_id}/ — uma pasta por execucao
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR  = PROJECT_ROOT / "reports" / "tables"   # CSVs do relatorio
STUDIES_DIR = PROJECT_ROOT / "reports" / "studies"  # banco do Optuna
for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, STUDIES_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RESULTS_DIR :", RESULTS_DIR, "(so execucoes)")
print("reports/    : figures | tables | studies")

# Checagem de rede: no Kaggle a internet vem DESLIGADA por padrao, e sem ela o
# clone e o download do CSV falham de formas confusas. Melhor falhar aqui, com
# uma mensagem clara.
if IN_COLAB or IN_KAGGLE:
    import socket
    try:
        socket.create_connection(("raw.githubusercontent.com", 443), timeout=5).close()
        print("internet: OK")
    except OSError:
        print("\\n!!! SEM INTERNET. No Kaggle: Settings -> Internet -> On, e rode de novo.")
"""),
    code("""
#@title Imports
import json
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
## 4. O ajuste decisivo: qual alvo é realmente estacionário

A extrapolação da seção 1 se resolve mudando o que se pede ao modelo. Mas
**existem dois modos de fazer isso, e só um funciona** — a primeira execução
deste notebook provou isso na prática.

**Variação em dólares** (`diff_target=True`): prever quantos dólares o preço
muda. Parece estacionário, mas não é: um movimento de 3% valia US$ 20 em 2015
e US$ 500 em 2018.

**Retorno percentual** (`diff_target=True` + `log_price=True`): prever a
variação *relativa*. Subir 3% é o mesmo número em qualquer nível de preço.

Medindo o desvio-padrão do alvo em cada partição:

| Alvo | Treino | Teste | Razão |
|---|---|---|---|
| Variação em US$ | 26,2 | 608,2 | **23,2×** |
| Log-retorno | 0,044 | 0,055 | **1,23×** |

Como o scaler é ajustado só no treino, com a variação em dólares o modelo
aprende numa escala 23× menor que a do teste. Por isso **todos os experimentos
da primeira execução empataram em RMSE ≈ 607**, exatamente o valor do baseline
ingênuo: a rede convergiu para prever "não muda" e virou o baseline.

A célula abaixo mede as quatro formulações com as métricas de diagnóstico
novas, que são o que expõe esse problema.
"""),
    code("""
#@title Comparacao das formulacoes do alvo
comparacao = []
for nome, kw in [("nivel",        dict(scaler="minmax",   diff_target=False)),
                 ("nivel_log",    dict(scaler="minmax",   diff_target=False, log_price=True)),
                 ("variacao_usd", dict(scaler="standard", diff_target=True)),
                 ("log_retorno",  dict(scaler="standard", diff_target=True, log_price=True))]:
    cfg = ExperimentConfig(run_name=f"formul_{nome}", window=30, hidden_size=32, num_layers=1,
                           num_epochs=150, patience=30, grad_clip=1.0,
                           notes=f"Comparacao de formulacao do alvo: {nome}.", **kw)
    s = prepare_splits(df, cfg); set_seed(cfg.seed)
    r = fit_or_load(cfg, s, device, results_dir=RESULTS_DIR)
    experiment_results[cfg.run_name] = r
    sc = r["test_scores"]
    comparacao.append({"formulacao": nome, "rmse": sc["rmse"], "r2": sc["r2"],
                       "dir_acc": sc.get("directional_accuracy"),
                       "razao_mov": sc.get("movement_ratio"),
                       "corr_mov": sc.get("movement_corr"),
                       "bate_ingenuo": sc.get("beats_naive")})

pd.DataFrame(comparacao).round(4)
"""),
    md("""
### Como ler `razao_mov` e `corr_mov` — as métricas que importam

Estas duas colunas são novas, e são elas que separam um modelo real de um
impostor:

- **`razao_mov`** = desvio do movimento previsto ÷ desvio do movimento real.
  Perto de **0** significa que o modelo prevê "amanhã ≈ hoje" — ele *virou* o
  baseline ingênuo, e o RMSE empatado não é coincidência, é o mesmo modelo.
  Perto de **1** significa que ele arrisca movimentos da magnitude certa.
- **`corr_mov`** = correlação entre movimento previsto e real. É aqui que mora
  o poder preditivo; RMSE baixo com correlação zero não vale nada.

Na primeira execução, `razao_mov` ficou em **0,015** — o modelo previa
movimentos 67× menores que os reais, com correlação *negativa*. Um RMSE de 607
que parecia ótimo escondia um modelo que não tinha aprendido nada.

**Por que o RMSE sozinho engana tanto aqui:** prever "não muda" é a estratégia
que *minimiza* o erro quadrático numa série quase aleatória. O modelo não
falhou em otimizar — ele otimizou perfeitamente para a métrica errada.
"""),

    md("""
## 5. Busca manual por levas

A primeira execução mostrou que **variar janela, capacidade e tipo de célula
não mudou nada**: todas as 8 configurações da leva 1 deram RMSE entre 607,2 e
607,8. Isso não significa que esses hiperparâmetros não importam — significa
que, com o alvo errado, nenhum deles conseguia importar.

Agora, com log-retorno, a leva 1 investiga o que de fato move a agulha. E há
um achado da exploração que vale destacar: **learning rate e janela curta são
o que faz o modelo arriscar previsões** em vez de colapsar no baseline.
"""),
    code("""
#@title Leva 1 — o que faz o modelo SAIR do baseline
BASE = dict(scaler="standard", diff_target=True, log_price=True, grad_clip=1.0,
            num_epochs=150, patience=30, features=("Close",))

leva1 = [
    # Janela: o quanto de passado importa para um retorno diario
    ExperimentConfig(run_name="l1_w5",   window=5,  hidden_size=16, num_layers=1, learning_rate=3e-3, **BASE,
                     notes="Janela muito curta + lr alto: a combinacao que fez o modelo prever movimento de verdade."),
    ExperimentConfig(run_name="l1_w10",  window=10, hidden_size=16, num_layers=1, learning_rate=3e-3, **BASE,
                     notes="Janela de 2 semanas."),
    ExperimentConfig(run_name="l1_w30",  window=30, hidden_size=32, num_layers=1, learning_rate=1e-3, **BASE,
                     notes="Janela de 1 mes, lr padrao."),
    ExperimentConfig(run_name="l1_w60",  window=60, hidden_size=32, num_layers=1, learning_rate=1e-3, **BASE,
                     notes="Janela do tutorial."),
    # Learning rate: a alavanca que decide se a rede arrisca ou colapsa no baseline
    ExperimentConfig(run_name="l1_lr_alto", window=10, hidden_size=16, num_layers=1, learning_rate=5e-3, **BASE,
                     notes="lr ainda maior — testa o limite antes de desestabilizar."),
    ExperimentConfig(run_name="l1_lr_baixo", window=10, hidden_size=16, num_layers=1, learning_rate=3e-4, **BASE,
                     notes="lr baixo: esperado colapsar para 'nao muda' (razao_mov ~ 0)."),
    # Capacidade
    ExperimentConfig(run_name="l1_h8",   window=5, hidden_size=8,  num_layers=1, learning_rate=3e-3, **BASE,
                     notes="Capacidade minima — a serie tem so 857 janelas de treino."),
    ExperimentConfig(run_name="l1_h64",  window=5, hidden_size=64, num_layers=1, learning_rate=3e-3, **BASE,
                     notes="Capacidade maior."),
    # Familias de celula recorrente
    ExperimentConfig(run_name="l1_gru",  window=5, hidden_size=16, num_layers=1, learning_rate=3e-3,
                     **{**BASE, "model_type": "gru"}, notes="GRU: versao simplificada da LSTM."),
    ExperimentConfig(run_name="l1_rnn",  window=5, hidden_size=16, num_layers=1, learning_rate=3e-3,
                     **{**BASE, "model_type": "rnn"}, notes="RNN simples: a referencia que a LSTM veio superar."),
    # Features
    ExperimentConfig(run_name="l1_ohlcv", window=5, hidden_size=16, num_layers=1, learning_rate=3e-3,
                     **{**BASE, "features": ("Open","High","Low","Close","Volume")},
                     notes="OHLCV completo: volume costuma carregar sinal de volatilidade."),
]

for cfg in leva1:
    if cfg.run_name in experiment_results: continue
    s = prepare_splits(df, cfg); set_seed(cfg.seed)
    experiment_results[cfg.run_name] = fit_or_load(cfg, s, device, results_dir=RESULTS_DIR)

cols = ["rmse","r2","directional_accuracy","movement_ratio","movement_corr","beats_naive"]
tab = pd.DataFrame({k: v["test_scores"] for k,v in experiment_results.items()}).T
display(tab[[c for c in cols if c in tab.columns]].sort_values("movement_corr", ascending=False).round(4))

print("\\nComo ler: RMSE menor NAO e o criterio. Procure linhas com movement_ratio")
print("entre ~0.2 e ~1.5 E movement_corr positiva — sao os modelos que arriscaram")
print("previsoes de magnitude plausivel e acertaram. Razao ~0 = virou o baseline.")
"""),
    md("""
### Espaço para a sua análise da leva 1

Depois de rodar, registre: qual learning rate fez `razao_mov` sair de ~0? A
janela curta venceu como esperado? A LSTM superou GRU e RNN simples, ou as
três empatam (o que sugeriria que a memória longa não ajuda em retorno
diário)? O volume do OHLCV acrescentou sinal?

A leva 2 deve combinar os vencedores — e o critério de "vencedor" aqui é
`movement_corr`, não RMSE.
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

**Ressalva importante sobre o que o Optuna está otimizando.** Ele minimiza a
perda de validação, que é MSE sobre o retorno. Numa série quase aleatória, a
configuração que minimiza MSE é frequentemente a que **menos arrisca** — ou
seja, o Optuna tende a convergir para o baseline ingênuo disfarçado. Foi
exatamente o que aconteceu na primeira execução: o melhor trial deu
`razao_mov = 0,015`.

Por isso, ao olhar os resultados, confira `movement_ratio` e `movement_corr`
do modelo escolhido. Se a razão vier perto de zero, o vencedor do Optuna é um
modelo que aprendeu a não responder, e a configuração da leva 1 com maior
`movement_corr` é a escolha mais honesta para o relatório.
"""),
    code("""
#@title Rodar o estudo
from lstm_bitcoin.tuning import run_study, best_config, study_dataframe, importance_table

base_tune = ExperimentConfig(run_name="tune", scaler="standard",
                             diff_target=True, log_price=True,   # log-retorno: ver secao 4
                             num_epochs=120, patience=25, grad_clip=1.0)

# Espaco ajustado pelo que a leva 1 revelou: janelas curtas e learning rate alto
# sao o que faz o modelo arriscar previsoes em vez de colapsar no baseline.
ESPACO = {
    "window":        ("categorical", [5, 10, 15, 20, 30, 45]),
    "hidden_size":   ("categorical", [8, 16, 32, 64]),
    "num_layers":    ("int", 1, 2),
    "dropout":       ("float", 0.0, 0.3),
    "learning_rate": ("loguniform", 5e-4, 1e-2),
    "batch_size":    ("categorical", [16, 32, 64]),
    "weight_decay":  ("loguniform", 1e-7, 1e-3),
    "model_type":    ("categorical", ["lstm", "gru"]),
    "grad_clip":     ("categorical", [0.5, 1.0, 5.0]),
}

study = run_study(df, base_tune, device,
                  n_trials=60,                       # aumente se houver tempo de GPU
                  space=ESPACO,
                  study_name="lstm_btc_logret",
                  storage=f"sqlite:///{STUDIES_DIR}/optuna.db")

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
# O run_name e DINAMICO: optuna_{tarefa}_{n_trials}t_{hash dos params vencedores},
# ex. "optuna_regression_50t_a3f9c1". Isso evita que um segundo estudo (mais
# trials, outro espaco, ou a tarefa de direcao) seja silenciosamente ignorado
# pelo fit_or_load, que reaproveita qualquer execucao com o mesmo nome.
cfg_best = best_config(study, base_tune)
cfg_best.num_epochs, cfg_best.patience = 120, 20   # treino final mais longo

print("run_name gerado:", cfg_best.run_name)

s_best = prepare_splits(df, cfg_best); set_seed(cfg_best.seed)
experiment_results[cfg_best.run_name] = fit_or_load(cfg_best, s_best, device, results_dir=RESULTS_DIR)
print({k: round(v,4) if isinstance(v,float) else v
       for k,v in experiment_results[cfg_best.run_name]["test_scores"].items()})
"""),

    md("""
## 7. Tarefa secundária: prever a direção do movimento

Previsão de preço é regressão e não produz matriz de confusão. Mas a pergunta
que de fato importa — *o preço sobe ou desce amanhã?* — é classificação
binária, e essa sim produz matriz de confusão.

Vale fazer as duas porque um RMSE baixo pode esconder um modelo inútil: se ele
só copia o preço de ontem, acerta o nível e erra metade das direções. A
acurácia direcional é o teste real de poder preditivo.

O baseline aqui é a **taxa da classe majoritária** (seção 2): se 52,8% dos
dias foram de alta, responder "alta" sempre acerta 52,8%.

**Esta é a parte do projeto que funcionou.** Na exploração, a classificação de
direção chegou a **74% de acurácia contra 52,8% do baseline — mais de 20
pontos percentuais de ganho**, enquanto a regressão mal empatava com o palpite
ingênuo.

O contraste vale ser explicado no relatório, porque é conceitualmente
interessante: prever *quanto* o preço muda é quase impossível, mas prever *para
onde* ele vai é bem mais tratável. A magnitude do movimento é dominada por
ruído; o sinal está na direção. E a autocorrelação de −0,22 no lag 1 do
log-retorno (mede-se no treino) é justamente o tipo de estrutura que uma rede
consegue capturar para acertar o sinal sem acertar o tamanho.
"""),
    code("""
#@title Treinar o classificador de direcao
# Varias configs: a tarefa de direcao respondeu MUITO melhor que a de regressao,
# entao vale comparar algumas em vez de treinar so uma.
dir_configs = [
    ExperimentConfig(run_name="dir_w5_h16",  task="direction", window=5,  hidden_size=16, num_layers=1,
                     learning_rate=3e-3, scaler="standard", diff_target=True, log_price=True,
                     grad_clip=1.0, num_epochs=150, patience=30,
                     notes="Direcao com janela curta — a melhor config da exploracao."),
    ExperimentConfig(run_name="dir_w10_h16", task="direction", window=10, hidden_size=16, num_layers=1,
                     learning_rate=3e-3, scaler="standard", diff_target=True, log_price=True,
                     grad_clip=1.0, num_epochs=150, patience=30,
                     notes="Direcao com janela de 2 semanas."),
    ExperimentConfig(run_name="dir_w30_h32", task="direction", window=30, hidden_size=32, num_layers=1,
                     learning_rate=1e-3, scaler="standard", diff_target=True, log_price=True,
                     grad_clip=1.0, num_epochs=150, patience=30,
                     notes="Direcao com janela de 1 mes."),
    ExperimentConfig(run_name="dir_ohlcv",   task="direction", window=5,  hidden_size=16, num_layers=1,
                     learning_rate=3e-3, scaler="standard", diff_target=True, log_price=True,
                     grad_clip=1.0, num_epochs=150, patience=30,
                     features=("Open","High","Low","Close","Volume"),
                     notes="Direcao com OHLCV — volume carrega sinal de volatilidade."),
]

for cfg in dir_configs:
    if cfg.run_name in experiment_results: continue
    sd = prepare_splits(df, cfg); set_seed(cfg.seed)
    experiment_results[cfg.run_name] = fit_or_load(cfg, sd, device, results_dir=RESULTS_DIR)

dir_tab = pd.DataFrame({c.run_name: experiment_results[c.run_name]["test_scores"]
                        for c in dir_configs}).T
display(dir_tab[["accuracy","precision","recall","f1_score","majority_rate"]].round(4))

MELHOR_DIR = dir_tab["accuracy"].idxmax()
sc = experiment_results[MELHOR_DIR]["test_scores"]
print(f"\\nMelhor: {MELHOR_DIR}")
print(f"Ganho sobre o baseline majoritario: {(sc['accuracy']-sc['majority_rate'])*100:+.2f} p.p.")
"""),
    code("""
#@title Matriz de confusao — em numero e em percentual
cm = np.array(experiment_results[MELHOR_DIR]["test_result"]["confusion_matrix"])

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
                                "run_name": f"kaggle_{cfg_best.run_name}", "dataset": "kaggle",
                                "notes": "Melhor config do Optuna aplicada a base moderna do Kaggle."})
    s_k = prepare_splits(df_k, cfg_k); set_seed(cfg_k.seed)
    experiment_results[cfg_k.run_name] = fit_or_load(cfg_k, s_k, device, results_dir=RESULTS_DIR)
    print({k: round(v,4) if isinstance(v,float) else v
           for k,v in experiment_results[cfg_k.run_name]["test_scores"].items()})
else:
    print(f"Base do Kaggle nao encontrada em {KAGGLE_CSV}.")
    print("Baixe com:  kaggle datasets download -d mczielinski/bitcoin-historical-data")
"""),


    md("""
## 9. E o relatório?

Tabelas, gráficos comparativos e conclusões **não ficam aqui** — ficam em
`02_results_report.ipynb`.

O motivo é prático. Este notebook é caro: roda horas no Kaggle e cada saída é
um registro que se perde se algo der errado. O notebook de relatório só lê
`results/`, roda em segundos e pode ser reexecutado à vontade — ajustando um
gráfico, renomeando uma coluna, reescrevendo uma conclusão — sem retreinar
nada.

Ter as duas coisas separadas evita o pior cenário: duas versões da mesma
tabela que divergem silenciosamente porque você corrigiu uma e esqueceu a
outra.

Depois que este notebook terminar, rode:

```bash
python scripts/make_report_notebook.py   # se quiser regenerar
jupyter notebook notebooks/02_results_report.ipynb
```
"""),

    md("""
## 9.1 Devolver os resultados ao GitHub (push direto do Kaggle)

No Kaggle tudo que este notebook gera fica dentro do clone em
`/kaggle/working/redes-neurais/...`, que é **efêmero**: some quando a sessão
encerra. Em vez de baixar um zip e copiar à mão, a célula abaixo **commita e dá
push direto para o repositório**.

### Configuração (uma vez só)

1. **Gere um token no GitHub.** Em *Settings → Developer settings → Personal
   access tokens → Fine-grained tokens*, crie um token com acesso apenas ao
   repositório `redes-neurais` e permissão **Contents: Read and write**.
   Defina uma validade curta (30 ou 90 dias).
2. **Guarde no Kaggle.** No notebook, *Add-ons → Secrets → Add a new secret*,
   com label `GITHUB_TOKEN` e o token como valor. Marque a caixa para anexar
   ao notebook.
3. Confirme que *Settings → Internet* está **On**.

O token fica no cofre do Kaggle, não no código — por isso ele pode ir para o
GitHub sem risco. A célula nunca imprime o token, e a URL com credencial é
removida do `git remote` ao final.

### Por que um branch separado

O push vai para um branch `kaggle-results` (configurável), não para a `main`.
Assim os resultados chegam ao GitHub sem risco de conflito com o que você tem
localmente, e você decide quando mesclar — com um `git merge kaggle-results`
ou por um Pull Request no GitHub.
"""),
    code("""
#@title Commit + push dos resultados para o GitHub
# Requer: Secret GITHUB_TOKEN (Add-ons -> Secrets) e Internet ligada.
BRANCH   = "kaggle-results"       # branch de destino; nao usa a main de proposito
GIT_USER = "Kaggle Bot"
GIT_MAIL = "kaggle@users.noreply.github.com"

def _run(cmd, cwd=None, secret=None):
    # Roda um comando e devolve (codigo, saida), mascarando o token no log.
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if secret:
        out = out.replace(secret, "***")
    return r.returncode, out

if not (IN_KAGGLE or IN_COLAB):
    print("Rodando local — nada a enviar, os arquivos ja estao no seu repositorio.")
else:
    token = None
    if IN_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        except Exception as e:
            print("Nao consegui ler o Secret GITHUB_TOKEN:", type(e).__name__)
    else:
        from getpass import getpass
        token = getpass("GitHub token: ")

    if not token:
        print("\\nSEM TOKEN — pulei o push. Configure Add-ons -> Secrets -> GITHUB_TOKEN.")
        print("Os resultados continuam em", RESULTS_DIR, "(efemeros!)")
    else:
        REPO = root                      # raiz do clone feito no setup
        slug = REPO_URL.split("github.com/")[-1].removesuffix(".git")
        auth = f"https://x-access-token:{token}@github.com/{slug}.git"

        _run(f'git config user.name "{GIT_USER}"', cwd=REPO)
        _run(f'git config user.email "{GIT_MAIL}"', cwd=REPO)

        # Branch dedicado: reaproveita se ja existir no remoto.
        _run(f"git fetch origin {BRANCH}", cwd=REPO)
        code_, _ = _run(f"git checkout {BRANCH}", cwd=REPO)
        if code_ != 0:
            _run(f"git checkout -B {BRANCH}", cwd=REPO)

        # Adiciona so o que interessa (results/ e reports/ do mp2).
        sub = SUBPATH
        _run(f"git add {sub}/results {sub}/reports", cwd=REPO)  # reports = figures+tables+studies

        code_, status = _run("git status --porcelain", cwd=REPO)
        if not status:
            print("Nada novo para commitar.")
        else:
            n = len(status.splitlines())
            msg = f"results(mp2): execucao do Kaggle — {n} arquivos"
            _run(f'git commit -m "{msg}"', cwd=REPO)
            code_, out = _run(f"git push {auth} HEAD:{BRANCH}", cwd=REPO, secret=token)
            if code_ == 0:
                print(f"PUSH OK -> branch '{BRANCH}' ({n} arquivos)")
                print(f"   https://github.com/{slug}/tree/{BRANCH}/{sub}/results")
                print(f"\\n   Localmente:  git fetch origin && git merge origin/{BRANCH}")
            else:
                print("FALHA no push:"); print(out[:800])

        # Nao deixa a URL com token no .git/config.
        _run("git remote set-url origin " + REPO_URL, cwd=REPO)
"""),
    code("""
#@title (alternativa) Empacotar para download manual
# Use se preferir nao configurar token: copia tudo para a raiz de /kaggle/working,
# que e o que aparece na aba Output da versao.
import shutil

if IN_KAGGLE or IN_COLAB:
    OUT = Path("/kaggle/working") if IN_KAGGLE else Path("/content/saida")
    OUT.mkdir(parents=True, exist_ok=True)

    for origem, destino in [(RESULTS_DIR, OUT/"results"), (FIGURES_DIR, OUT/"figures")]:
        if origem.exists():
            if destino.exists(): shutil.rmtree(destino)
            shutil.copytree(origem, destino)
            print(f"{destino}  ({sum(1 for _ in destino.rglob('*') if _.is_file())} arquivos)")

    shutil.make_archive(str(OUT/"mp2_resultados"), "zip", root_dir=OUT, base_dir="results")
    print("\\nmp2_resultados.zip pronto — aba Output da versao.")
else:
    print("Rodando local: resultados em", RESULTS_DIR)
"""),

    md("""
## 10. Conclusões

Estão em `02_results_report.ipynb`, seção 8 — junto das tabelas e gráficos que
as sustentam, num notebook que você reexecuta em segundos para mantê-las em dia.
"""),
]


def _key(cell: dict) -> str:
    """Identidade de uma célula: o hash do seu código-fonte.

    Duas células com o mesmo código são "a mesma célula", ainda que tenham
    mudado de posição no notebook. É isso que permite preservar a saída de uma
    célula quando outra, antes dela, foi editada.
    """
    return hashlib.sha1("".join(cell["source"]).encode("utf-8")).hexdigest()


def merge_outputs(novas: list[dict], nb_path: Path) -> tuple[list[dict], int]:
    """Transporta as saídas já executadas do notebook em disco para as células novas.

    O gerador reconstrói a estrutura do notebook a cada execução, mas o arquivo
    em disco pode conter **saídas caras** — gráficos, tabelas e logs de treinos
    que levaram horas no Kaggle. Descartá-las a cada regeneração significaria
    perder o registro visual da execução.

    A regra é conservadora e simples: uma célula de código só herda a saída
    anterior se o seu **código for byte a byte idêntico**. Se o código mudou, a
    saída antiga passou a não corresponder ao que a célula faz, então ela é
    descartada — é o comportamento correto, porque uma saída que não bate com o
    código é pior que nenhuma saída.
    """
    if not nb_path.exists():
        return novas, 0

    try:
        antigo = json.loads(nb_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  (nao consegui ler o notebook anterior: {exc}; gerando limpo)")
        return novas, 0

    # Mapa código -> (saídas, execution_count) das células já executadas.
    salvas = {}
    for c in antigo.get("cells", []):
        if c.get("cell_type") == "code" and c.get("outputs"):
            salvas[_key(c)] = (c["outputs"], c.get("execution_count"))

    preservadas = 0
    resultado = []
    for c in novas:
        c = dict(c)
        if c["cell_type"] == "code":
            achou = salvas.get(_key(c))
            if achou:
                c["outputs"], c["execution_count"] = achou
                preservadas += 1
        resultado.append(c)
    return resultado, preservadas


def main(limpar: bool = False) -> int:
    """Gera o notebook, preservando as saídas das células cujo código não mudou.

    `--limpar` força um notebook sem saída nenhuma (útil antes de commitar uma
    versão enxuta, ou quando as saídas antigas viraram ruído).
    """
    celulas = CELLS
    preservadas = 0
    if not limpar:
        celulas, preservadas = merge_outputs(CELLS, NB_PATH)

    nb = {
        "cells": celulas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    total_code = sum(1 for c in celulas if c["cell_type"] == "code")
    print(f"Notebook gerado com {len(celulas)} células em {NB_PATH}")
    if limpar:
        print("  (--limpar: saidas descartadas)")
    else:
        print(f"  saidas preservadas: {preservadas}/{total_code} células de código")
        if preservadas < total_code:
            print(f"  {total_code - preservadas} célula(s) mudaram de código — saida descartada,")
            print("  rode-as de novo para atualizar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(limpar="--limpar" in sys.argv))
