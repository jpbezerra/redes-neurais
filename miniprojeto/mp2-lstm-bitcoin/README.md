# Mini-projeto 2 — LSTM para prever o preço do Bitcoin

Enunciado: empregar **LSTM** para prever o preço do Bitcoin, avaliando com **MSE, RMSE e POCID**. O trabalho parte da base do tutorial do enunciado (dezembro de 2014 a maio de 2018, primeiros 80% para treino e 20% finais para teste) e usa também a base disponibilizada pelo professor no Classroom (agosto de 2017 a agosto de 2023) para os resultados finais de classificação.

> **Status:** concluído. Todos os experimentos foram executados e o relatório está em `notebooks/02_results_report.ipynb`.

## Resultado em uma página

O achado principal não é um modelo que acerta o preço, e sim entender por que ele não acerta.

| Pergunta | Base | Resultado | Referência |
|---|---|---|---|
| Prever o **nível** do preço | tutorial | RMSE US$ 3.850 | baseline ingênuo US$ 607,9 |
| Prever o **log-retorno** do dia seguinte | tutorial | melhor RMSE US$ 606,5 (POCID entre 48% e 54%) | baseline ingênuo US$ 607,9 |
| Prever **sobe ou desce** (alvo corrigido) | professor | **48,7%** de acurácia (LSTM/GRU, média de 10 sementes) | classe majoritária do teste 53,8% |
| Modelos clássicos (alvo corrigido) | professor | regressão logística 48,1%, random forest 49,3%, gradient boosting 51,2% | classe majoritária 53,8% |
| **Walk-forward** (5 períodos, 2020–2023) | professor | 51,2% em média | majoritária 53,9% |
| **GAN** como data augmentation | professor | 47,0% → 47,8% (+0,87 p.p., ±2,79 entre sementes) | indistinguível de ruído |
| Horizontes de 3, 5 e 10 dias | professor | 10 dias: +1,1 p.p. (desvio entre sementes 0,022) | majoritária de cada horizonte |

Em todos os casos, nenhum modelo supera a classe majoritária do teste de forma consistente. Séries diárias de preço do Bitcoin se aproximam de um passeio aleatório.

> **Sobre os números "alvo corrigido".** Durante o trabalho foi encontrado um bug na construção do alvo de direção (ver a seção *O bug no alvo de direção*). Os resultados de classificação anteriores à correção (acurácias de 60% a 77%) são **inválidos** e não aparecem em nenhuma conclusão. Valem apenas os números marcados como "alvo corrigido".

## Estrutura

```
mp2-lstm-bitcoin/
├── README.md
├── pyproject.toml           # pip install -e .
├── requirements.txt
├── src/lstm_bitcoin/        # código reutilizável
│   ├── config.py            # ExperimentConfig: todos os hiperparâmetros de um experimento
│   ├── data.py              # carga das bases, divisão temporal, escala e janelas deslizantes
│   ├── features.py          # features adicionais (volatilidade, retornos defasados, candle)
│   ├── model.py             # RecurrentForecaster: LSTM / GRU / RNN parametrizáveis
│   ├── metrics.py           # regressão (RMSE/MAE/MAPE/R²) + classificação + matriz de confusão
│   ├── train.py             # loop de treino com early stopping + avaliação em dólares
│   ├── gan.py               # GAN recorrente para gerar sequências OHLC sintéticas
│   ├── checkpointing.py     # skill `model-saver` + tabela de hiperparâmetros do relatório
│   ├── tuning.py            # busca automática com Optuna (TPE + pruning + importância)
│   └── utils.py             # seed, device e os gráficos padrão do relatório
├── notebooks/
│   ├── 01_train_lstm_bitcoin.ipynb   # treina (caro, roda no Kaggle) + Optuna
│   ├── 02_results_report.ipynb       # só lê results/ e reports/tables/: gera o relatório
│   └── 03_gan_augmentation.ipynb     # experimento de dados sintéticos com GAN
├── scripts/                 # geradores de notebooks e experimentos em linha de comando
│   ├── make_notebook.py         # gera o notebook 01
│   ├── make_report_notebook.py  # gera o notebook 02
│   ├── make_gan_notebook.py     # gera o notebook 03
│   ├── run_local.py             # roda levas de experimentos em CPU, em segundo plano
│   ├── walk_forward.py          # validação temporal walk-forward
│   ├── abstencao.py             # abstenção seletiva (zona morta)
│   └── gan_augmentation.py      # treino do GAN e comparação real vs. real + sintético
├── data/
│   └── btc_prof_2023.csv    # base do professor (Classroom): 2.176 dias, 17/08/2017 a 01/08/2023
├── reports/
│   ├── figures/             # gráficos do relatório (PNG)
│   └── tables/              # tabelas do relatório (CSV)
└── results/                 # results/{model_id}/ — uma pasta por execução (metadata.json, histórico, pesos)
```

A base do tutorial (`btc.csv`) é baixada automaticamente por `data.load_tutorial()` e não é versionada. A base do professor fica em `data/` e é lida por `data.load_professor()`.

### Para que serve cada notebook

- **`01_train_lstm_bitcoin.ipynb`** *treina*. É caro (horas de GPU no Kaggle), e cada execução é salva em `results/{model_id}/` com um `metadata.json`, de modo que o registro real dos resultados é a pasta `results/`, não a saída de uma célula.
- **`02_results_report.ipynb`** *só lê* `results/` e `reports/tables/`. Roda em segundos, sem GPU, e reconstrói todas as tabelas e figuras do relatório. Pode ser refeito quantas vezes for preciso sem retreinar nada. Contém: o panorama das execuções, por que a regressão de preço colapsa, a jornada da classificação de direção, o bug no alvo e o resultado honesto depois da correção.
- **`03_gan_augmentation.ipynb`** reporta o experimento de aumento de dados com GAN: a técnica e a expectativa, a sanidade das features sintéticas, a comparação real vs. real + sintético e a conclusão.

### Os geradores de notebooks são incrementais

Os notebooks são gerados por script, e os geradores **preservam as saídas das células cujo código não mudou** (comparação por hash do fonte). Editar uma célula descarta a saída só dela.

```bash
python scripts/make_notebook.py            # regenera o notebook 01 preservando saídas
python scripts/make_report_notebook.py     # regenera o notebook 02
python scripts/make_gan_notebook.py        # regenera o notebook 03
python scripts/make_notebook.py --limpar   # descarta todas as saídas
```

## Como rodar

### Local

```bash
cd miniprojeto/mp2-lstm-bitcoin
python -m venv .venv && source .venv/bin/activate   # ou .venv\Scripts\activate no Windows
pip install -e .
jupyter notebook notebooks/02_results_report.ipynb   # relatório (não precisa treinar)
```

Para reproduzir os experimentos, a ordem é: `01_train_lstm_bitcoin.ipynb` (ou `scripts/run_local.py` para levas em CPU), depois `scripts/walk_forward.py`, `scripts/abstencao.py` e `scripts/gan_augmentation.py` (o desenho experimental de cada um está na docstring do topo do arquivo), e por fim os geradores `make_report_notebook.py` e `make_gan_notebook.py` para reconstruir os notebooks 02 e 03.

### Colab / Kaggle

Abra o notebook e rode a célula de setup: ela clona o repositório, instala o pacote e ajusta os caminhos. No Kaggle, ative *Settings → Internet → On* e prefira **"Save Version" → "Save & Run All (Commit)"**: o modo commit roda num kernel gerenciado em segundo plano e não depende da aba do navegador ficar aberta.

#### Devolver os resultados ao GitHub

O botão *File → Link to GitHub* do Kaggle versiona só o arquivo `.ipynb`, não os `results/` nem os gráficos. Por isso a seção 9.1 do notebook 01 faz o push por código. Configuração, uma vez só: gere um *fine-grained token* no GitHub com permissão **Contents: Read and write** restrita a este repositório, e guarde no Kaggle em *Add-ons → Secrets* com o label `GITHUB_TOKEN`. O push vai para o branch `kaggle-results`, não para a `main`, e você mescla quando quiser com `git fetch origin && git merge origin/kaggle-results`. Quem preferir não configurar token tem uma célula alternativa que empacota tudo num zip na aba Output.

O projeto é leve: a série tem de 1.273 a 2.176 pontos, então cada treino leva segundos a poucos minutos em CPU. GPU só compensa para rodar o Optuna com muitos trials.

---

# As bases de dados

Foram usadas duas bases, com papéis diferentes.

## Base do tutorial — `tutorial`

O `btc.csv` do repositório [brynmwangy/predicting-bitcoin-prices-using-LSTM](https://github.com/brynmwangy/predicting-bitcoin-prices-using-LSTM), o material creditado nos slides do enunciado. **1.273 linhas, de 01/12/2014 a 26/05/2018, sem nenhum dia faltando.** Colunas: `Date, Symbol, Open, High, Low, Close, Volume From, Volume To`. Baixada automaticamente por `data.load_tutorial()`.

Usada para reproduzir o tutorial e para toda a parte de **regressão** (prever quanto o preço muda). Com o corte 80/20, o treino fica com 1.018 dias (US$ 120 a US$ 4.948, dos quais os 10% finais são validação) e o teste com **254 dias** (US$ 3.617 a US$ 19.650).

**Atenção à ordem do arquivo.** O `btc.csv` vem do dado mais recente para o mais antigo, e o tutorial **não reordena**. Seguindo o tutorial à risca, o modelo treina com 2015–2018 e é testado em dez/2014–ago/2015: ele prevê o passado, num trecho calmo (US$ 120 a US$ 378) que já estava dentro do treino, e o gráfico fica enganosamente bom. Este projeto ordena a série cronologicamente.

## Base do professor — `professor`

`data/btc_prof_2023.csv`, disponibilizada no Classroom: **2.176 dias, de 17/08/2017 a 01/08/2023**, sem lacunas (todos os intervalos entre dias consecutivos são de exatamente 1 dia). Colunas: `date, open, high, low, close, number_of_trades`, renomeadas para o padrão do projeto por `data.load_professor()`.

Usada nos **resultados finais de classificação** (sobe ou desce), no walk-forward, na abstenção, nos horizontes e no GAN. Com o corte 80/20, o teste tem **435 dias** (US$ 15.781 a US$ 31.801), inteiramente **dentro** da faixa de preço do treino (US$ 3.189 a US$ 67.526), ao contrário da base do tutorial.

Outras bases (a de 1 minuto do Kaggle, `mczielinski/bitcoin-historical-data`) continuam suportadas por `data.load_kaggle()`, mas os resultados finais deste relatório usam a base do professor.

---

# Decisões metodológicas

## A divisão é temporal, nunca aleatória

Em série temporal, embaralhar antes de dividir faz o modelo treinar com dados do futuro e ser testado no passado. O resultado fica espetacular e completamente falso. O treino é sempre o começo da série e o teste sempre o fim (80/20). O treino ainda cede seus 10% finais para validação, usada no early stopping e na seleção de hiperparâmetros: o teste é tocado uma única vez, no fim.

## O scaler é ajustado só no treino

Ajustar no conjunto inteiro faria o mínimo e o máximo do teste vazarem para o treino.

## O baseline ingênuo é obrigatório

O palpite **"amanhã o preço será igual ao de hoje"** é difícil de bater, porque séries financeiras são próximas de um passeio aleatório. Qualquer modelo que não o supere não aprendeu nada. Toda avaliação de regressão reporta `naive_rmse` e `beats_naive` ao lado do RMSE. Na classificação, a referência equivalente é a **classe majoritária** do teste, e o relatório mostra também a majoritária do treino (responder sempre "alta" daria 46,2% no teste da base do professor).

---

# O diagnóstico central: qual alvo é estacionário (base do tutorial)

| Partição | Faixa de preço |
|---|---|
| Treino (sem a validação) | US$ 120 – US$ 2.698 |
| Teste (254 dias) | US$ 3.617 – US$ 19.650 |

**As faixas não se sobrepõem.** Prevendo o *nível*, a rede precisa extrapolar cerca de 7× acima de tudo que viu, e redes neurais não extrapolam bem (o RMSE chega a US$ 3.850).

Prever a mudança em vez do nível ajuda, mas **variação em dólares** e **retorno percentual** são coisas diferentes:

| Alvo | Razão de escala teste/treino |
|---|---|
| Variação em US$ | **23,2×** (não estacionária) |
| Log-retorno | **1,23×** |

A configuração usa `diff_target=True` **junto com** `log_price=True`.

## As métricas que expuseram o colapso

Com o log-retorno, o RMSE de todos os modelos ficou colado no do baseline ingênuo (≈ US$ 607). Duas métricas, em `train.evaluate_split`, mostram o motivo:

- **`movement_ratio` (`razao_mov`)**: desvio do movimento previsto ÷ desvio do real. Perto de 0 significa que o modelo previu "amanhã ≈ hoje": ele *virou* o baseline.
- **`movement_corr` (`corr_mov`)**: correlação entre movimento previsto e real.

Numa série quase aleatória, prever "não muda" é a estratégia que *minimiza* o erro quadrático. O modelo não falhou em otimizar: otimizou perfeitamente para a métrica errada.

Variar janela (3 a 60 dias), capacidade (16 a 128 unidades), camadas (1 a 3) e célula (LSTM, GRU, RNN) manteve o RMSE dentro de 1% do baseline. O teto está nos dados, não na arquitetura.

## Métricas de regressão (teste de 254 dias)

| Modelo | RMSE (US$) | MAE (US$) | POCID |
|---|---|---|---|
| Baseline ingênuo | 607,9 | 405,7 | — |
| LSTM, nível do preço | 3.850,6 | 2.851,4 | 48,4% |
| LSTM, log do nível | 1.217,5 | 835,1 | 50,4% |
| LSTM, log-retorno | 608,0 | 406,0 | 50,0% |
| LSTM (janela 10, 64 un.) | 606,5 | 406,2 | 52,8% |
| GRU (janela 10, 64 un.) | 607,2 | 406,5 | 51,6% |
| RNN simples (janela 10, 64 un.) | 608,7 | 407,6 | 53,1% |

O melhor modelo vence o ingênuo por US$ 1,4 de RMSE e perde no MAE. Um RMSE baixo não garante um modelo útil: o POCID, entre 48% e 54%, é praticamente cara ou coroa.

---

# Classificação de direção

Previsão de preço é regressão e não produz matriz de confusão. Por isso o projeto tem uma **tarefa de classificação**: prever se o preço sobe ou desce no dia seguinte (`task="direction"`). Usa o mesmo pipeline e a mesma arquitetura; mudam só a última camada (sigmoide) e a função de perda (entropia cruzada binária).

`metrics.confusion_matrix` dá a contagem absoluta e `metrics.confusion_matrix_percent` converte nas três leituras: `normalize="true"` (cada linha soma 100%, o recall), `normalize="pred"` (cada coluna soma 100%, a precisão) e `normalize="all"`.

## O bug no alvo de direção

O primeiro resultado parecia excelente: **72,9%** (GRU, média de 10 sementes) na base do tutorial e **72,6%** na base do professor, contra uma majoritária de cerca de 53%. Mas os modelos clássicos (regressão logística, random forest, gradient boosting) ficavam em cerca de 50% no mesmo alvo: uma diferença de mais de 20 p.p. entre famílias de modelos é suspeita.

A auditoria encontrou o erro em `make_windows`: a condição `futuro > último`, aplicada sobre arrays **já diferenciados**, respondia *"a variação de amanhã é maior que a de hoje?"* em vez de *"o preço sobe de D para D+1?"*. A primeira é fácil (depois de uma variação grande, o Bitcoin tende a ter uma variação menor); a segunda não tem padrão. A correção lê o sinal da variação na **escala original**, antes do escalador, em `data.py`.

## Resultado depois da correção (base do professor)

Com a pergunta certa, a rede não encontra padrão e tende a responder "alta" todos os dias. Numa execução em que isso acontece, a acurácia é **46,2%**, exatamente a proporção de dias de alta no teste (201 de 435).

| Modelo | Acurácia |
|---|---|
| Classe majoritária do teste | 53,8% |
| Gradient boosting | 51,2% |
| Random forest | 49,3% |
| **LSTM/GRU (alvo corrigido, média de 10 sementes)** | **48,7%** |
| Regressão logística | 48,1% |
| Dummy (maioria do treino, "alta") | 46,2% |

Por que abaixo de 50%? No treino, 53% dos dias são de alta; no teste, 54% são de queda. O modelo aprende "responda alta" e o mercado muda justamente no teste.

## O que foi tentado para sair do teto

| Tentativa | Resultado |
|---|---|
| Mudar o alvo (nível → log-retorno) | Resolveu a extrapolação, mas o modelo passou a prever "não muda" |
| 19 configurações manuais (janela, unidades, camadas, célula, learning rate, perda) | Todas a menos de 1% do ingênuo |
| Optuna | Ver abaixo |
| Mais features (OHLC, volatilidade, retornos defasados) | Ganho aparente de +7 p.p., mas era o bug |
| Horizontes de 3, 5 e 10 dias | Nenhum bate a majoritária de forma consistente (10 dias: +1,1 p.p., dentro do ruído) |
| Zona morta / abstenção | Ganho instável; 49,3% |
| Walk-forward (retreinar em 5 períodos, 2020–2023) | 51,2% em média, 2,7 p.p. abaixo da majoritária (53,9%); em 3 das 5 dobras o modelo só repete uma classe |
| GAN (dados sintéticos) | +0,87 p.p. (±2,79), dentro do ruído |

## GAN como data augmentation

`gan.py` treina um GAN recorrente que gera sequências OHLC em log-retorno. Os dados sintéticos acertam a escala das médias, mas são menos dispersos que os reais e só 6,5% dos dias gerados são de alta (contra 53% no real). As perdas do gerador e do discriminador se estabilizam sem que nenhuma vá a zero: não há colapso do GAN. Comparando 10 sementes, a acurácia vai de 47,0% (real) para 47,8% (real + sintético); o ganho médio de +0,87 p.p. é puxado pela semente 3 (+7,6 p.p.) e é indistinguível de ruído. Sem GAN, 9 das 10 sementes colapsam em 46,2%.

O classificador desse experimento é uma **LSTM menor (2 camadas × 32 unidades, janela de 3 dias)**, diferente da GRU de 64 unidades usada nos resultados da tabela acima. Por isso a referência do experimento é 47,0%, e não 48,7%.

---

# Optuna

`tuning.run_study()` roda a busca automática:

- **Amostrador TPE**: modela a distribuição dos bons e maus resultados e concentra as propostas onde a razão entre elas é alta.
- **Pruning** (`MedianPruner`): aborta trials que vão claramente mal.
- **Otimiza a validação, nunca o teste.**
- **Persistência opcional em SQLite** (`storage="sqlite:///optuna.db"`), para o estudo sobreviver à queda da sessão do Kaggle.
- **`importance_table()`** dá o ranking fANOVA dos hiperparâmetros.
- **`run_name` dinâmico**, no formato `optuna_{tarefa}_{n_trials}t_{hash}`: o hash deriva dos hiperparâmetros vencedores, para que estudos diferentes não reaproveitem o resultado um do outro (`fit_or_load` reutiliza qualquer execução com o mesmo `run_name`).

Foram feitos dois estudos para classificação de direção, **ambos antes da correção do bug** (portanto otimizaram a pergunta errada):

| Estudo | Trials | Resultado |
|---|---|---|
| Base do tutorial | 15 | GRU, janela 10, 64 unidades, 1 camada, OHLC, lr ≈ 0,0039 |
| Base do professor | 33 | GRU, janela 2, 64 unidades, 1 camada, 9 features, lr ≈ 0,0058 |

**Três configurações aparecem nos resultados**, e não devem ser confundidas:

1. **GRU, janela 10, OHLC** (Optuna de 15 trials): é a dos resultados da tabela de classificação corrigida (48,7%).
2. **GRU, janela 2, 9 features** (Optuna de 33 trials): não entrou nos resultados finais.
3. **LSTM 2 × 32, janela 3**: usada no GAN e no walk-forward.

---

# Hiperparâmetros investigados

Específicos de série temporal:

- **`window`**: quantos dias de passado o modelo enxerga.
- **`horizon`**: quantos dias à frente prever.
- **`diff_target`** / **`log_price`**: a formulação do alvo.
- **`scaler`**: MinMax, Standard ou nenhum.

De arquitetura e otimização:

- **`model_type`**: LSTM, GRU ou RNN simples.
- **`hidden_size`**, **`num_layers`**, **`dropout`**, **`bidirectional`**, **`fc_layers`**
- **`optimizer`**, **`learning_rate`**, **`weight_decay`**, **`lr_schedule`**
- **`grad_clip`**: redes recorrentes reaplicam os mesmos pesos a cada passo de tempo, então o gradiente pode crescer geometricamente ao longo da janela. Cortar a norma evita a explosão.

`checkpointing.hyperparameter_table(results_dir)` monta a tabela do relatório lendo todos os `metadata.json`; com `only_varied=True` (padrão), omite as colunas constantes. A tabela também é exportada em `reports/tables/tabela_hiperparametros.csv`.

---

# Tabelas do relatório (`reports/tables/`)

| Arquivo | Conteúdo | Status |
|---|---|---|
| `resumo_regressao.csv` | métricas de todas as execuções de regressão | válido |
| `tabela_hiperparametros.csv` | hiperparâmetros variados × métricas de teste | válido |
| `baselines_classicos_alvo_corrigido.csv` | acurácia dos modelos clássicos e da rede, alvo corrigido | válido |
| `abstencao_alvo_corrigido.csv` | abstenção seletiva com o alvo corrigido | válido |
| `zona_morta_retorno.csv` | abstenção por zona morta de retorno | válido |
| `horizonte_previsao.csv` | acurácia por horizonte (1, 3, 5 e 10 dias) | válido |
| `walk_forward_professor.csv` | walk-forward na base do professor, alvo corrigido | válido (`walk_forward_prof.csv` é uma cópia idêntica) |
| `gan_real_vs_aumentado.csv` | 10 sementes, real vs. real + sintético | válido |
| `gan_sanidade.csv` | média e desvio das features reais vs. sintéticas | válido |
| `gan_curva_treino.csv` | perdas do gerador e do discriminador | válido |
| `resumo_direcao.csv` | execuções de direção | **parcialmente inválido**: as linhas `dir*`, `feat_*` e `fx_*` são anteriores à correção do bug |
| `walk_forward.csv`, `walk_forward_final.csv`, `abstencao_wf.csv`, `ensemble_abstencao_wf.csv` | validações anteriores à correção | **inválidos** (alvo com bug, acurácias de 65% a 88%); mantidos apenas como histórico, não usar |

---

# Expectativa honesta sobre os resultados

Séries de preço de ativos financeiros são próximas de um passeio aleatório. Ganhos pequenos sobre o baseline ingênuo são o resultado esperado e correto. Se algum experimento produzir uma previsão quase perfeita, a primeira hipótese deve ser vazamento temporal ou erro na construção do alvo, não sucesso. Foi exatamente o que aconteceu com os 72–73% de acurácia, que eram um bug.

Os lugares onde isso costuma acontecer, e que este código evita explicitamente: divisão aleatória, scaler ajustado no conjunto inteiro, janelas recortadas antes da divisão e alvo mal construído.

Ideias para continuar: incluir informação externa (volume, notícias, dados on-chain, outros ativos), testar dados intradiários, refazer o Optuna com o alvo corrigido e avaliar com uma métrica de retorno financeiro.
