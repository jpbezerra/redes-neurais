# Projeto Final — Previsão de Churn em Telecom

**Status:** Fase 4 concluída (engenharia de features). Próxima: Fase 5
(modelos avançados — STab, TabPFNv2, KAN, TabKAN, Mitra).

## Dataset

`data/telco_customer_churn.csv` (não versionado) — mesmo schema do dataset
Kaggle [`customers-churned-in-telecom-services`](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services/data)
citado no enunciado da disciplina: 7.043 clientes, 19 variáveis independentes
+ `customerID` + `Churn` (alvo).

Achados da EDA real (Fase 0): `reports/tables/eda_fase0_achados.md`,
notebook em `notebooks/00_eda.ipynb`.

## Pipeline de dados (Fase 1)

`src/churn_telecom/data.py` implementa o particionamento em 3 etapas exigido
pelo enunciado: split por classe (50/25/25), reamostragem com repetição da
classe minoritária **só em treino e validação** (teste sempre com a
distribuição real, ~73.5%/26.5%), recombinação e embaralhamento.
`assert_sem_vazamento` confere que nenhuma linha original aparece em mais de
uma partição.

`src/churn_telecom/features.py` tem o pré-processamento (imputação de
`TotalCharges`, encoding binário/one-hot, normalização z-score) ajustado
**só no treino**.

`src/churn_telecom/metrics.py` implementa a métrica principal (KS) e as
secundárias (MSE, cross-entropy, matriz de confusão, AUROC, precision,
recall, F1). O KS tem self-test embutido, validado contra `scipy.ks_2samp`
e casos analíticos de separação perfeita/nula.

Notebook: `notebooks/01_data_pipeline.ipynb`.

## Baselines (Fase 2)

MLP (1 camada, 10 unidades — navalha de Occam) + Gradient Boosting + XGBoost,
cada um nas duas estratégias de balanceamento (oversampling vs.
class_weight/scale_pos_weight), avaliados pela métrica principal (KS, com
curva no estilo do enunciado) e pelas secundárias.

**Nota de ambiente:** o MLP usa `sklearn.neural_network.MLPClassifier` em
vez de PyTorch — a instalação de PyTorch trava neste ambiente (as
dependências CUDA que o pacote genérico do PyPI traz mesmo sem GPU são
pesadas demais, e o índice CPU-only oficial do PyTorch está fora da rede
disponível aqui). O MLPClassifier cobre os mesmos hiperparâmetros pedidos
(camadas/unidades, ativação, otimizador, regularização L2) com early
stopping manual por patience (`src/churn_telecom/models/mlp.py`). Se mais
controle de arquitetura for necessário mais adiante, reavaliar PyTorch
rodando no Kaggle.

**Resultado:** KS entre 0.51 e 0.52 em teste real (nunca balanceado
artificialmente), AUROC ~0.83, convergente entre as 3 famílias de modelo —
consistente com benchmarks publicados para este dataset, auditado contra
vazamento (ver `notebooks/02_baselines.ipynb`, seção 3).

Notebook: `notebooks/02_baselines.ipynb`.

## Busca de hiperparâmetros (Fase 3)

Leva 1 via Optuna (20 trials cada), otimizando **KS na validação**, em MLP
e Gradient Boosting. O teste só é tocado uma vez no final, para reportar a
métrica honesta do vencedor de cada leva.

**Nota de ambiente:** o storage SQLite nativo do Optuna falha com erro de
I/O quando o arquivo `.db` fica num drive de rede montado (é o caso desta
pasta) — SQLite precisa de locking de arquivo que o mount não garante.
Cada estudo roda em memória e os trials são exportados para CSV
(`reports/tables/fase3_trials_*.csv`).

**Achado honesto:** o melhor trial de Gradient Boosting chegou a KS=0.571
na *validação*, mas caiu para KS≈0.510 no *teste* — e o MLP teve o mesmo
padrão em escala menor. Isso é overfitting de hiperparâmetro (poucos trials
+ um único split de validação). A conclusão desta leva é que **a busca não
trouxe ganho real sobre os baselines da Fase 2** — documentado em detalhe,
em vez de reportar só o número de validação (que pareceria um resultado
melhor do que realmente é).

Notebook: `notebooks/03_hyperparam_search.ipynb`.

## Engenharia de features (Fase 4)

Três features derivadas (identificadas na EDA da Fase 0) testadas
**isoladamente** contra o baseline de Gradient Boosting da Fase 2, com
hiperparâmetros fixos para isolar o efeito de cada uma:
`charges_per_tenure` (`TotalCharges/(tenure+1)`), `n_servicos_adicionais`
(contagem de serviços extras) e `tenure_bucket` (faixas de tenure).

**Achado honesto:** só `charges_per_tenure` teve ganho isolado positivo na
validação (+0.0046), mas o ganho **não se confirmou no teste** — o modelo
com a feature teve KS=0.516, abaixo do baseline puro da Fase 2 (KS=0.524).
Mesmo padrão de ruído de validação já visto na Fase 3. Nenhuma feature
derivada foi incorporada à referência do projeto — o baseline
`gb_churn_baseline_oversample` da Fase 2 continua sendo o melhor resultado.

Notebook: `notebooks/04_feature_engineering.ipynb`.

## Pendência de limpeza (ação manual sua)

`results/_to_delete/` tem arquivos de um estudo Optuna com SQLite que
corrompeu no mount de rede (ver nota acima) — não consigo apagar arquivos
nesta pasta por padrão. Pode deletar essa subpasta quando quiser, ela não é
referenciada por nenhum código.

## Estrutura

```
churn-telecom/
├── data/                  # CSV bruto (não versionado)
├── src/churn_telecom/     # pacote importável (local e Kaggle)
│   ├── data.py            # carregamento bruto + split de 3 etapas
│   ├── features.py        # pré-processamento (fit só no treino)
│   ├── metrics.py         # KS (principal) + métricas secundárias
│   ├── checkpointing.py   # salvamento no padrão results/{model_id}/
│   └── models/            # mlp.py, boosting.py (GB + XGBoost)
├── scripts/                # geradores de notebook + runners (não versionado, ver .gitignore)
├── notebooks/
├── results/                # artefatos de experimento (model-saver)
└── reports/{figures,tables}
```

## Como rodar

```bash
python3 scripts/eda_fase0.py                 # EDA (tabelas)
python3 scripts/make_eda_notebook.py          # gera notebooks/00_eda.ipynb
python3 scripts/fase1_pipeline.py             # pipeline de dados (diagnósticos)
python3 scripts/make_pipeline_notebook.py     # gera notebooks/01_data_pipeline.ipynb
python3 scripts/fase2_baselines.py            # treina e salva os baselines
python3 scripts/make_fase2_notebook.py        # gera notebooks/02_baselines.ipynb
python3 scripts/fase3_optuna.py mlp 20        # leva de busca Optuna (MLP)
python3 scripts/fase3_optuna.py gb 20         # leva de busca Optuna (Gradient Boosting)
python3 scripts/make_fase3_notebook.py        # gera notebooks/03_hyperparam_search.ipynb
python3 scripts/fase4_features.py             # testa features derivadas (isoladas + combinação vencedora)
python3 scripts/make_fase4_notebook.py        # gera notebooks/04_feature_engineering.ipynb
python3 src/churn_telecom/metrics.py          # self-test do KS
```

## Convenções

Seguindo os mesmos padrões de `miniprojeto/mp1-cifar10` e
`miniprojeto/mp2-lstm-bitcoin`: pacote por fase em `src/`, notebooks gerados
por script, experimentos salvos via skill model-saver em `results/`, nada de
jargão de sessão nos nomes (sempre descritivo), disciplina anti-vazamento
(scaler/encoder ajustado só no treino), levas sucessivas de experimentos.
