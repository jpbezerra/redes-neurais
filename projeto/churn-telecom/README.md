# Projeto Final — Previsão de Churn em Telecom

**Status:** código da modelos avançados pronto para rodar no Kaggle (5 modelos
avançados implementados: STab, TabPFN v2, KAN, TabKAN, Mitra). Próxima:
rodar `notebooks/06_modelos_avancados_kaggle.ipynb` no Kaggle e trazer os resultados
de volta para `results/`, depois para a consolidação final.

## Dataset

`data/telco_customer_churn.csv` (não versionado) — mesmo schema do dataset
Kaggle [`customers-churned-in-telecom-services`](https://www.kaggle.com/datasets/kapturovalexander/customers-churned-in-telecom-services/data)
citado no enunciado da disciplina: 7.043 clientes, 19 variáveis independentes
+ `customerID` + `Churn` (alvo).

Achados da EDA real (EDA): `reports/tables/eda_achados.md`,
notebook em `notebooks/00_eda.ipynb`.

## Pipeline de dados (pipeline de dados)

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

## Baselines (baselines)

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

## Busca de hiperparâmetros (busca de hiperparâmetros)

Leva 1 via Optuna (20 trials cada), otimizando **KS na validação**, em MLP
e Gradient Boosting. O teste só é tocado uma vez no final, para reportar a
métrica honesta do vencedor de cada leva.

**Nota de ambiente:** o storage SQLite nativo do Optuna falha com erro de
I/O quando o arquivo `.db` fica num drive de rede montado (é o caso desta
pasta) — SQLite precisa de locking de arquivo que o mount não garante.
Cada estudo roda em memória e os trials são exportados para CSV
(`reports/tables/busca_hiperparametros_trials_*.csv`).

**Achado honesto:** o melhor trial de Gradient Boosting chegou a KS=0.571
na *validação*, mas caiu para KS≈0.510 no *teste* — e o MLP teve o mesmo
padrão em escala menor. Isso é overfitting de hiperparâmetro (poucos trials
+ um único split de validação). A conclusão desta leva é que **a busca não
trouxe ganho real sobre os baselines da baselines** — documentado em detalhe,
em vez de reportar só o número de validação (que pareceria um resultado
melhor do que realmente é).

Notebook: `notebooks/03_hyperparam_search.ipynb`.

## Engenharia de features (engenharia de features)

Três features derivadas (identificadas na EDA da EDA) testadas
**isoladamente** contra o baseline de Gradient Boosting da baselines, com
hiperparâmetros fixos para isolar o efeito de cada uma:
`charges_per_tenure` (`TotalCharges/(tenure+1)`), `n_servicos_adicionais`
(contagem de serviços extras) e `tenure_bucket` (faixas de tenure).

**Achado honesto:** só `charges_per_tenure` teve ganho isolado positivo na
validação (+0.0046), mas o ganho **não se confirmou no teste** — o modelo
com a feature teve KS=0.516, abaixo do baseline puro da baselines (KS=0.524).
Mesmo padrão de ruído de validação já visto na busca de hiperparâmetros. Nenhuma feature
derivada foi incorporada à referência do projeto — o baseline
`gb_churn_baseline_oversample` da baselines continua sendo o melhor resultado.

Notebook: `notebooks/04_feature_engineering.ipynb`.

## Modelos avançados (modelos avançados) — smoke test

Antes de treinar STab, TabPFN v2, KAN, TabKAN e Mitra, um smoke test de
instalação (`scripts/smoke_test_modelos_avancados.py`) — **nenhum é viável neste
ambiente**:

- **PyTorch não instala aqui**: o wheel do PyPI tem 554,6 MB, o download
  não retoma entre tentativas, e 3 tentativas de 170s não completaram. O
  índice CPU-only oficial do PyTorch está fora da rede disponível.
- **STab e Mitra são colisões de nome no PyPI** — os pacotes `stab` e
  `mitra` existentes ali não são os modelos do enunciado (um gerador de
  sites estáticos e uma lib de álgebra linear, respectivamente). O Mitra
  real só existe via `autogluon.tabular[mitra]`, ainda mais pesado.
- **TabPFN v2, KAN (pykan) e TabKAN** existem e são os modelos corretos,
  mas todos dependem de PyTorch — mesmo bloqueio.

**Recomendação (documentada em detalhe no notebook): migrar a modelos avançados para
o Kaggle Notebook**, que já vem com PyTorch + GPU — o pacote
`src/churn_telecom/` é importável lá diretamente, bastando subir o CSV
como dataset.

Notebook: `notebooks/05_modelos_avancados_smoke_test.ipynb`.

## Modelos avançados — implementação (modelos avançados, para rodar no Kaggle)

Cada modelo tem seu wrapper em `src/churn_telecom/models/`, no mesmo
padrão de `mlp.py`/`boosting.py` (dataclass de hiperparâmetros +
`treinar_*`/`prever_*`), mas com **imports pesados só dentro das funções**
(lazy import) — por isso os módulos importam normalmente aqui mesmo sem
PyTorch instalado; o treino de verdade só roda no Kaggle:

- **`models/stab.py`** — **implementação própria** de Transformer
  tabular. Não existe um pacote público legítimo chamado "STab" (o nome
  `stab` no PyPI é um gerador de sites estáticos, sem relação — ver smoke
  test). Em vez de depender de proveniência incerta, implementei um
  encoder Transformer enxuto (feature → token, `depth` camadas de
  self-attention, token [CLS] agregando a predição), cobrindo o espaço de
  hiperparâmetros do enunciado (`dim`, `depth`, `heads`, `attn_dropout`,
  `ff_dropout`, `lr`, `weight_decay`, `batch_size`). Os hiperparâmetros
  `U`, `cases` e `sample_size` do enunciado não têm definição padrão
  publicamente disponível — documentados como não implementados em vez de
  inventar um significado.
- **`models/tabpfn.py`** — wrapper do pacote `tabpfn` real (confirmado no
  smoke test). Sem curva de treino (é in-context learning, não treino por
  época).
- **`models/kan.py`** — wrapper do `pykan` real, usando a API nativa
  (`width`, `grid`, `k`).
- **`models/tabkan.py`** — wrapper do pacote `tabkan`; a API exata não
  pôde ser inspecionada aqui (não instala neste ambiente) — assume
  convenção sklearn-like (`fit`/`predict_proba`), **a confirmar e ajustar
  no Kaggle** se divergir.
- **`models/mitra.py`** — via `autogluon.tabular[mitra]` (o Mitra real da
  Amazon), já que o pacote `mitra` isolado do PyPI é uma lib de álgebra
  linear não relacionada (outra colisão de nome, ver smoke test).

**`scripts/treina_modelos_avancados_kaggle.py`** treina os 5 em sequência,
cada um dentro de um `try/except` — se um falhar lá (API divergente,
pacote desatualizado), os outros continuam e o erro fica documentado, em
vez de travar tudo.

**`notebooks/06_modelos_avancados_kaggle.ipynb`** é o notebook pronto para
**subir no Kaggle** (gerado aqui, mas não executado aqui — não roda sem
PyTorch). Segue o mesmo padrão de setup já usado em
`miniprojeto/mp2-lstm-bitcoin`: clona o repositório do GitHub e instala o
pacote via `pip install -e`, em vez de depender de subir `src/` como
Kaggle Dataset separado. `load_raw` também baixa o CSV automaticamente se
não existir em `data/` (mesma conveniência do mini-projeto 2), então não é
preciso subir o dataset do Kaggle manualmente também.

Checklist de uso:
1. Ative *Settings → Internet → On* e *Accelerator → GPU*.
2. Rode as células na ordem — a primeira clona
   `https://github.com/jpbezerra/redes-neurais.git` e instala o pacote
   `churn-telecom` (ajuste `REPO_URL` na célula se o repositório remoto
   tiver outro endereço).
3. As células seguintes instalam os pacotes que faltam (`tabpfn`, `pykan`,
   `tabkan`, `autogluon.tabular[mitra]`), treinam os 5 modelos e salvam
   cada um em `results/{model_id}/` (mesmo padrão model-saver de sempre).
4. Use "Save Version" → "Save & Run All (Commit)" para rodar em segundo
   plano sem depender da aba ficar aberta.
5. Baixe a pasta `results/` do commit e cole em `results/` aqui — o
   notebook de consolidação final só lê `results/`, não retreina nada.

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
python3 scripts/eda.py                 # EDA (tabelas)
python3 scripts/make_eda_notebook.py          # gera notebooks/00_eda.ipynb
python3 scripts/pipeline_dados.py             # pipeline de dados (diagnósticos)
python3 scripts/make_pipeline_notebook.py     # gera notebooks/01_data_pipeline.ipynb
python3 scripts/treina_baselines.py            # treina e salva os baselines
python3 scripts/make_baselines_notebook.py        # gera notebooks/02_baselines.ipynb
python3 scripts/busca_hiperparametros.py mlp 20        # leva de busca Optuna (MLP)
python3 scripts/busca_hiperparametros.py gb 20         # leva de busca Optuna (Gradient Boosting)
python3 scripts/make_busca_hiperparametros_notebook.py        # gera notebooks/03_hyperparam_search.ipynb
python3 scripts/engenharia_features.py             # testa features derivadas (isoladas + combinação vencedora)
python3 scripts/make_engenharia_features_notebook.py        # gera notebooks/04_feature_engineering.ipynb
python3 scripts/smoke_test_modelos_avancados.py           # smoke test de instalação dos modelos avançados
python3 scripts/make_smoke_test_notebook.py        # gera notebooks/05_modelos_avancados_smoke_test.ipynb
python3 scripts/make_kaggle_notebook.py # gera notebooks/06_modelos_avancados_kaggle.ipynb (subir no Kaggle, não rodar aqui)
python3 src/churn_telecom/metrics.py          # self-test do KS

# No Kaggle (apos subir os datasets, ver secao acima):
python3 scripts/treina_modelos_avancados_kaggle.py   # ou rode notebooks/06_modelos_avancados_kaggle.ipynb celula a celula
```

## Convenções

Seguindo os mesmos padrões de `miniprojeto/mp1-cifar10` e
`miniprojeto/mp2-lstm-bitcoin`: pacote por etapa em `src/`, notebooks gerados
por script, experimentos salvos via skill model-saver em `results/`, nada de
jargão de sessão nos nomes (sempre descritivo), disciplina anti-vazamento
(scaler/encoder ajustado só no treino), levas sucessivas de experimentos.
