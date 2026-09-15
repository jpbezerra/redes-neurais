# Mini-projeto 2 — LSTM para prever o preço do Bitcoin

Enunciado: empregar **LSTM** para prever o preço do Bitcoin de dezembro de
2014 a maio de 2018, usando os primeiros 80% dos registros para treinar e os
20% finais para teste.

> **Status:** base do projeto pronta e testada de ponta a ponta; os
> experimentos ainda não foram executados. Os resultados abaixo marcados como
> *diagnóstico* vêm de treinos curtos de validação do pipeline, não dos
> experimentos finais.

## Estrutura

```
mp2-lstm-bitcoin/
├── README.md
├── pyproject.toml           # pip install -e .
├── requirements.txt
├── src/lstm_bitcoin/        # código reutilizável
│   ├── config.py            # ExperimentConfig: todos os hiperparâmetros de um experimento
│   ├── data.py              # carga das bases, divisão temporal, escala e janelas deslizantes
│   ├── model.py             # RecurrentForecaster: LSTM / GRU / RNN parametrizáveis
│   ├── metrics.py           # regressão (RMSE/MAE/MAPE/R²) + classificação + matriz de confusão
│   ├── train.py             # loop de treino com early stopping + avaliação em dólares
│   ├── checkpointing.py     # skill `model-saver` + tabela de hiperparâmetros do relatório
│   ├── tuning.py            # busca automática com Optuna (TPE + pruning + importância)
│   └── utils.py             # seed, device e os gráficos padrão do relatório
├── notebooks/
│   └── 01_train_lstm_bitcoin.ipynb   # orquestra os experimentos
├── scripts/
│   └── make_notebook.py     # gera o notebook programaticamente
├── data/                    # CSVs baixados automaticamente (não versionados)
├── reports/figures/         # gráficos do relatório
└── results/                 # results/{model_id}/ — uma pasta por execução
```

Mesma separação do Mini-projeto 1: `src/` tem a implementação testável,
`notebooks/` orquestra e reporta.

## Como rodar

### Local

```bash
cd miniprojeto/mp2-lstm-bitcoin
python -m venv .venv && source .venv/bin/activate   # ou .venv\Scripts\activate no Windows
pip install -e .
jupyter notebook notebooks/01_train_lstm_bitcoin.ipynb
```

### Colab / Kaggle

Abra o notebook e rode a célula de setup — ela clona o repositório, instala o
pacote e ajusta os caminhos. No Kaggle, prefira **"Save Version" → "Save & Run
All (Commit)"**: o modo commit roda num kernel gerenciado em segundo plano e
não depende da aba do navegador ficar aberta.

Este projeto é **muito mais leve que a Fase 2 do Mini-projeto 1**: a série tem
1.273 pontos (contra 50.000 imagens), então cada treino leva segundos a poucos
minutos em CPU. GPU só compensa para rodar o Optuna com muitos trials.

---

# As bases de dados

O professor não disponibilizou base. Foram selecionadas duas, com papéis
diferentes.

## Base principal — `tutorial`

O `btc.csv` do repositório [brynmwangy/predicting-bitcoin-prices-using-LSTM](https://github.com/brynmwangy/predicting-bitcoin-prices-using-LSTM),
que é exatamente o material creditado nos slides do enunciado. Verificado:
**1.273 linhas, de 01/12/2014 a 26/05/2018, sem nenhum dia faltando** — bate
com o período pedido.

Colunas: `Date, Symbol, Open, High, Low, Close, Volume From, Volume To`.
Baixada automaticamente por `data.load_tutorial()`.

Usar a mesma base do enunciado garante que o resultado seja comparável ao de
referência e que a comparação com outros grupos seja justa.

## Base secundária — `kaggle`

[mczielinski/bitcoin-historical-data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data):
BTC/USD do Bitstamp em resolução de **1 minuto desde janeiro de 2012**,
atualizada diariamente por [GitHub Actions](https://github.com/mczielinski/kaggle-bitcoin).
`data.load_kaggle()` reamostra para diário seguindo a convenção OHLCV.

Serve para responder à pergunta que o recorte do enunciado não responde: *o
que funciona em 2014-2018 continua funcionando depois?* O período pós-2018 tem
um regime de preços completamente diferente, então é um teste de
generalização honesto.

Baixe com `kaggle datasets download -d mczielinski/bitcoin-historical-data` e
coloque o CSV em `data/`.

### Outras consideradas

[Crypto Market Data: 50+ Coins Daily OHLCV](https://www.kaggle.com/datasets/abdullahkhan70/daily-multi-year-ohlcv-crypto-market-data)
seria interessante para usar outras criptomoedas como features adicionais, mas
foge do escopo. As bases de 5 minutos ou 5 segundos têm granularidade fina
demais — o enunciado é sobre previsão diária.

---

# Decisões metodológicas

## A divisão é temporal, nunca aleatória

Em série temporal, embaralhar antes de dividir faz o modelo treinar com dados
de 2018 e ser testado em 2015 — ele aprende o futuro e prevê o passado. O
resultado fica espetacular e completamente falso.

Aqui o treino é sempre o começo da série e o teste sempre o fim, conforme o
enunciado (80/20). O treino ainda cede seus 10% finais para validação, usada
no early stopping e na seleção de hiperparâmetros — o teste é tocado uma única
vez, no fim.

## O scaler é ajustado só no treino

Ajustar no conjunto inteiro faria o mínimo e o máximo do teste vazarem. Como o
preço vai de ~US$ 300 a ~US$ 19.000 na série, esse vazamento seria enorme.

## O baseline ingênuo é obrigatório

O palpite **"amanhã o preço será igual ao de hoje"** é difícil de bater, porque
séries financeiras são próximas de um passeio aleatório. Qualquer modelo que
não o supere não aprendeu nada: apenas copiou o último valor com um dia de
atraso.

Toda avaliação de regressão reporta `naive_rmse` e `beats_naive` ao lado do
RMSE. Isso é o que separa um relatório honesto de um gráfico bonito e vazio.

---

# O diagnóstico central: nível vs. variação

Este é o achado que orienta o projeto inteiro, e ele aparece já na inspeção
dos dados, antes de qualquer modelo:

| Partição | Faixa de preço |
|---|---|
| Treino (918 dias) | US$ 120 – US$ 2.698 |
| Teste (254 dias) | US$ 3.617 – US$ 19.650 |

**As faixas não se sobrepõem.** O menor preço do teste é maior que o maior
preço do treino. Um modelo que prevê o *nível* precisa extrapolar para valores
7× acima de tudo que viu — e redes neurais não extrapolam bem.

A correção é mudar o alvo: prever a **variação** entre dias consecutivos em
vez do nível. Subir 3% em 2016 e subir 3% em 2018 são o mesmo número, mesmo
com o preço 10× maior — a variação é aproximadamente estacionária. O preço é
reconstruído depois como `último preço observado + variação prevista`.

Medição com treinos curtos (25 épocas, `hidden_size=32`), tudo o mais igual:

| Formulação | RMSE | R² | Bate o ingênuo? |
|---|---|---|---|
| Nível + MinMax (o do tutorial) | 5.485 | −1,49 | não |
| Nível + log + MinMax | 4.679 | −0,81 | não |
| **Variação + Standard** | **607,5** | **0,970** | **sim** |
| Variação em log (retorno) | 609,2 | 0,969 | quase |

*(Números de diagnóstico do pipeline, não do experimento final.)*

O R² negativo das duas primeiras linhas significa literalmente que o modelo é
pior do que ter respondido a média em todos os dias. Nenhuma escolha de
normalização resolve — é o alvo que estava errado.

Esse resultado é o análogo, neste projeto, do que a leva 3 foi no Mini-projeto
1: um fracasso que reorienta a busca inteira.

---

# As três coisas pedidas para o relatório

## 1. Tabela de hiperparâmetros variados

`checkpointing.hyperparameter_table(results_dir)` monta a tabela do relatório
lendo todos os `metadata.json` salvos. Com `only_varied=True` (padrão), ela
**omite as colunas constantes** e mostra só o que de fato foi investigado,
junto das métricas de teste, ordenada pelo RMSE.

A célula da seção 9 do notebook também exporta em
`results/tabela_hiperparametros.csv`.

## 2. Matriz de confusão em número e em percentual

Previsão de preço é regressão e não produz matriz de confusão. Por isso o
projeto tem uma **tarefa secundária de classificação**: prever se o preço sobe
ou desce no dia seguinte (`task="direction"`). Ela usa o mesmo pipeline e a
mesma arquitetura — mudam só a última camada e a função de perda.

`metrics.confusion_matrix` dá a contagem absoluta e
`metrics.confusion_matrix_percent` converte nas três leituras:

- `normalize="true"` — cada linha soma 100%: *"das vezes que o preço realmente subiu, em quantas o modelo acertou?"* (recall)
- `normalize="pred"` — cada coluna soma 100%: *"das vezes que o modelo disse alta, em quantas ele acertou?"* (precision)
- `normalize="all"` — a matriz inteira soma 100%

`utils.plot_confusion` desenha as duas leituras na mesma figura: cada célula
mostra o número absoluto em cima e o percentual da linha embaixo. E o
`metadata.json` de cada execução de direção já guarda as três versões, então o
relatório não depende de recalcular nada.

Exemplo real do teste de validação do pipeline (LSTM, janela 30, 20 épocas):

```
CONTAGEM  (linha = real, coluna = previsto)
                      baixa        alta
  real baixa           104          29
  real alta             39          82

PERCENTUAL POR LINHA  (cada linha soma 100%)
  real baixa         78,2%       21,8%
  real alta          32,2%       67,8%
```

Acurácia 73,2% contra uma classe majoritária de 52,4%. A assimetria entre as
linhas (78,2% contra 67,8%) já mostra que o modelo é mais confiável quando
prevê baixa — exatamente o tipo de leitura que só a matriz de confusão dá.

## 3. Optuna

`tuning.run_study()` roda a busca automática. Detalhes da implementação:

- **Amostrador TPE**: modela a distribuição dos bons e maus resultados e concentra as propostas onde a razão entre elas é alta.
- **Pruning** (`MedianPruner`): aborta trials que vão claramente mal, multiplicando quantas configurações cabem no mesmo tempo.
- **Otimiza a validação, nunca o teste.** Otimizar no teste seria escolher o modelo que teve sorte com aquele conjunto.
- **Persistência opcional em SQLite** (`storage="sqlite:///optuna.db"`), para o estudo sobreviver à queda da sessão do Kaggle e ser retomado.
- **`importance_table()`** dá o ranking fANOVA — o análogo automático da análise "o que a busca revelou sobre cada hiperparâmetro" feita à mão no Mini-projeto 1.

O espaço padrão (`DEFAULT_SPACE`) cobre janela, tamanho e número de camadas,
dropout, learning rate, batch size, weight decay, tipo de célula (LSTM/GRU),
scaler e grad clipping.

**As duas buscas são complementares e o relatório deve ter as duas:** levas
manuais para entender o efeito isolado de cada alavanca, Optuna para achar a
melhor combinação dentro da região promissora.

---

# Hiperparâmetros investigados

Específicos de série temporal, sem equivalente no Mini-projeto 1:

- **`window`** — quantos dias de passado o modelo enxerga. É o hiperparâmetro mais característico do problema.
- **`horizon`** — quantos dias à frente prever.
- **`diff_target`** / **`log_price`** — a formulação do alvo, discutida acima.
- **`scaler`** — MinMax, Standard ou nenhum.

De arquitetura e otimização:

- **`model_type`** — LSTM, GRU ou RNN simples. Comparar as três é interessante: a LSTM foi inventada para resolver o esquecimento da RNN, e a GRU é uma simplificação dela.
- **`hidden_size`**, **`num_layers`**, **`dropout`**, **`bidirectional`**, **`fc_layers`**
- **`optimizer`**, **`learning_rate`**, **`weight_decay`**, **`lr_schedule`**
- **`grad_clip`** — redes recorrentes reaplicam os mesmos pesos a cada passo de tempo, então o gradiente pode crescer geometricamente ao longo da janela. Cortar a norma evita a explosão.

---

# Roteiro sugerido

1. Inspecionar a série e constatar a não sobreposição das faixas (seção 1)
2. Fixar o baseline ingênuo (seção 2)
3. Rodar o baseline do tutorial e ver que ele não bate o ingênuo (seção 3)
4. Corrigir a formulação do alvo (seção 4)
5. Levas manuais: janela, capacidade, LSTM vs GRU vs RNN, OHLCV vs só fechamento (seção 5)
6. Optuna sobre a região promissora (seção 6)
7. Classificação de direção + matriz de confusão (seção 7)
8. Repetir o melhor na base do Kaggle (seção 8)
9. Tabela de hiperparâmetros e conclusões (seções 9 e 10)

## Uma expectativa honesta sobre os resultados

Séries de preço de ativos financeiros são próximas de um passeio aleatório.
**Ganhos pequenos sobre o baseline ingênuo são o resultado esperado e
correto** — e uma acurácia direcional de 55-60% já seria um bom resultado.

Se algum experimento produzir uma previsão quase perfeita, a primeira hipótese
deve ser vazamento temporal, não sucesso. Os três lugares onde isso costuma
acontecer, e que este código evita explicitamente: divisão aleatória, scaler
ajustado no conjunto inteiro, e janelas recortadas antes da divisão.
