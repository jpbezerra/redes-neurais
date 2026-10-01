# Projeto Final — Previsão de Churn em Telecom

**Status:** Fase 1 concluída (pipeline de dados: split em 3 etapas +
anti-vazamento + pré-processamento). Próxima: Fase 2 (baseline MLP +
Gradient Boosting).

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
e casos analíticos de separação perfeita/nula — rode com
`python3 src/churn_telecom/metrics.py`.

Notebook correspondente: `notebooks/01_data_pipeline.ipynb`.

## Estrutura

```
churn-telecom/
├── data/                  # CSV bruto (não versionado)
├── src/churn_telecom/     # pacote importável (local e Kaggle)
│   ├── data.py            # carregamento bruto + split de 3 etapas
│   ├── features.py        # pré-processamento (fit só no treino)
│   ├── metrics.py         # KS (principal) + métricas secundárias
│   └── models/
├── scripts/                # geradores de notebook + runners (não versionado, ver .gitignore)
├── notebooks/
├── results/                # artefatos de experimento (model-saver: pesos + metadata.json + history.csv)
└── reports/{figures,tables}
```

## Como rodar

```bash
python3 scripts/eda_fase0.py              # EDA (tabelas)
python3 scripts/make_eda_notebook.py      # gera notebooks/00_eda.ipynb
python3 scripts/fase1_pipeline.py         # pipeline de dados (diagnósticos)
python3 scripts/make_pipeline_notebook.py # gera notebooks/01_data_pipeline.ipynb
python3 src/churn_telecom/metrics.py      # self-test do KS
```

## Convenções

Seguindo os mesmos padrões de `miniprojeto/mp1-cifar10` e
`miniprojeto/mp2-lstm-bitcoin`: pacote por fase em `src/`, notebooks gerados
por script, experimentos salvos via skill model-saver em `results/`, nada de
jargão de sessão nos nomes (sempre descritivo), disciplina anti-vazamento
(scaler/encoder ajustado só no treino), levas sucessivas de experimentos.
