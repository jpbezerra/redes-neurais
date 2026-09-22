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
│   ├── 01_train_lstm_bitcoin.ipynb   # treina — caro, roda no Kaggle
│   └── 02_results_report.ipynb       # só lê results/ — rápido, gera o relatório
├── scripts/
│   ├── make_notebook.py         # gera o notebook de treino
│   └── make_report_notebook.py  # gera o notebook de relatório
├── data/                    # CSVs baixados automaticamente (não versionados)
├── reports/figures/         # gráficos do relatório
└── results/                 # results/{model_id}/ — uma pasta por execução
```

Mesma separação do Mini-projeto 1: `src/` tem a implementação testável,
`notebooks/` orquestra e reporta.

### Por que dois notebooks

`01_train` **treina**: é caro (horas de GPU no Kaggle) e a saída de cada célula
é um registro que não se quer perder à toa. `02_results_report` **só lê**
`results/`: roda em segundos, não precisa de GPU, e reconstrói todas as tabelas
e figuras do relatório a partir dos `metadata.json` já salvos.

A vantagem prática é que o relatório pode ser refeito quantas vezes for
preciso — ajustando um gráfico, renomeando uma coluna — sem retreinar nada,
porque o registro real dos resultados é a pasta `results/`, não a saída de uma
célula.

### Os geradores são incrementais

Os dois notebooks são gerados por script, e os geradores **preservam as saídas
das células cujo código não mudou** (comparação por hash do fonte). Editar uma
célula descarta a saída só dela; as outras 18 continuam com o resultado da
execução anterior.

```bash
python scripts/make_notebook.py            # regenera preservando saídas
python scripts/make_report_notebook.py
python scripts/make_notebook.py --limpar   # descarta todas as saídas
```

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
pacote e ajusta os caminhos. No Kaggle, ative *Settings → Internet → On* e
prefira **"Save Version" → "Save & Run All (Commit)"**: o modo commit roda num
kernel gerenciado em segundo plano e não depende da aba do navegador ficar
aberta.

#### Devolver os resultados ao GitHub

O Kaggle tem um botão nativo *File → Link to GitHub*, mas ele versiona **apenas
o arquivo .ipynb** — não os `results/` nem os gráficos, que é justamente o que
interessa aqui.

Por isso a seção 9.1 do notebook faz o push por código. Configuração, uma vez
só: gere um *fine-grained token* no GitHub com permissão **Contents: Read and
write** restrita a este repositório, e guarde no Kaggle em *Add-ons → Secrets*
com o label `GITHUB_TOKEN`. O token fica no cofre do Kaggle, nunca no código, e
a URL com credencial é removida do `git remote` ao final da célula.

O push vai para o branch `kaggle-results`, não para a `main` — assim os
resultados chegam sem risco de conflito, e você mescla quando quiser com
`git fetch origin && git merge origin/kaggle-results`. Quem preferir não
configurar token tem, logo abaixo, a célula alternativa que empacota tudo num
zip na aba Output.

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

# O diagnóstico central: qual alvo é estacionário

Este é o achado que orienta o projeto, e a primeira execução completa o
refinou de forma importante.

## O problema de partida

| Partição | Faixa de preço |
|---|---|
| Treino (918 dias) | US$ 120 – US$ 2.698 |
| Teste (254 dias) | US$ 3.617 – US$ 19.650 |

**As faixas não se sobrepõem.** Prevendo o *nível*, a rede precisa extrapolar
7× acima de tudo que viu — e redes não extrapolam bem.

## A correção tem dois modos, e só um funciona

A solução é prever a mudança em vez do nível. Mas **variação em dólares** e
**retorno percentual** são coisas diferentes:

| Alvo | Desvio no treino | Desvio no teste | Razão |
|---|---|---|---|
| Variação em US$ | 26,2 | 608,2 | **23,2×** |
| Log-retorno | 0,044 | 0,055 | **1,23×** |

A variação em dólares **não é estacionária**: um movimento de 3% valia US$ 20
em 2015 e US$ 500 em 2018. Como o scaler é ajustado só no treino, o modelo
aprende numa escala 23× menor que a do teste.

Foi exatamente o que aconteceu na primeira execução: **as 14 execuções de
regressão empataram em RMSE ≈ 607**, o valor do baseline ingênuo. Variar
janela (5 a 90), capacidade (16 a 128) e célula (LSTM/GRU/RNN) produziu uma
faixa de 0,1%. Com o alvo errado, nenhum hiperparâmetro conseguia importar.

A configuração agora usa `diff_target=True` **junto com** `log_price=True`.

## As métricas que expuseram o problema

Um RMSE de 607 parecia ótimo e escondia um modelo que não aprendeu nada. Duas
métricas novas em `train.evaluate_split` tornam isso visível:

- **`movement_ratio`** — desvio do movimento previsto ÷ desvio do real. Perto
  de 0 significa que o modelo previu "amanhã ≈ hoje": ele *virou* o baseline.
  Na primeira execução deu **0,015** (movimentos 67× menores que os reais).
- **`movement_corr`** — correlação entre movimento previsto e real. Era
  **negativa**.

O motivo é conceitual: numa série quase aleatória, prever "não muda" é a
estratégia que *minimiza* o erro quadrático. O modelo não falhou em otimizar —
otimizou perfeitamente para a métrica errada.

Por isso a seleção do melhor modelo não usa RMSE. O critério é: entre os
modelos com `movement_ratio` entre 0,2 e 2,0 e R² > 0,5, o de maior
`movement_corr`. As duas guardas são necessárias — sem o piso vence quem
colapsou, sem o teto vence o `baseline_nivel`, que tem razão 4,4 e correlação
0,12 mas RMSE de 3.940 e R² negativo.

## Regressão e direção têm dificuldades muito diferentes

O contraste é o achado mais interessante do projeto:

| Tarefa | Resultado | Baseline | Ganho |
|---|---|---|---|
| Regressão (quanto muda) | RMSE 603 | 607,9 | marginal |
| **Direção (para onde vai)** | **~74% de acurácia** | 52,8% | **+21 p.p.** |

Prever *quanto* o preço muda é dominado por ruído; prever *para onde* ele vai é
tratável. A autocorrelação do log-retorno no treino é **−0,22 no lag 1** —
existe estrutura, mas pouca, e ela aparece no sinal, não na magnitude.

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
- **`run_name` dinâmico.** `best_config()` gera um nome no formato `optuna_{tarefa}_{n_trials}t_{hash}` (ex.: `optuna_regression_50t_a3f9c1`). O hash deriva dos hiperparâmetros vencedores, então dois estudos que chegam à mesma configuração recebem o mesmo nome e estudos diferentes recebem nomes diferentes. Isso não é cosmético: `fit_or_load` reaproveita qualquer execução com o mesmo `run_name`, então um nome fixo faria um segundo estudo — com mais trials ou outra tarefa — carregar silenciosamente o resultado do primeiro.

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
