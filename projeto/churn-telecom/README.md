# Projeto Final — Previsão de Churn em Telecom

**Status:** primeira rodada de modelos avançados executada no Kaggle
(`notebooks/05_modelos_avancados_kaggle.ipynb`, resultados em `results/`).
Próxima: rodar `notebooks/06_modelos_melhorados.ipynb` no Kaggle (rodada de
melhorias — ver seção "Modelos avançados — rodada de melhorias").

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

**`notebooks/05_modelos_avancados_kaggle.ipynb`** é o notebook pronto para
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

## Modelos avançados — rodada de melhorias (para rodar no Kaggle)

`notebooks/06_modelos_melhorados.ipynb` (gerado por
`scripts/make_modelos_melhorados_notebook.py`). Parte do que a rodada
anterior mostrou — na avaliação em 5 partições STab e TabKAN lideraram, e
TabPFN/Mitra só foram bem com contexto sem duplicatas:

- **STab**: embedding por feature (FT-Transformer), parada/escolha da época
  pelo AUROC de validação, ReduceLROnPlateau, clipping; duas capacidades,
  a melhor (na validação) re-treinada com 5 sementes (bagging);
- **TabKAN / KAN**: mesmo critério de parada + scheduler; mais largura
  (TabKAN) e grade mais fina + weight decay (KAN);
- **TabPFN v2**: contexto sem duplicatas com 32 estimadores internos;
- **Mitra**: contexto sem duplicatas, 4 estimadores internos e 100 passos de
  fine-tuning (o padrão do AutoGluon é 1 estimador e 50 passos);
- **XGBoost**: configuração regularizada com o nº de árvores escolhido por
  early stopping no AUROC de validação (sem busca — a busca Optuna anterior
  sobreajustou à validação);
- **Ensembles**: média no espaço logit, membros escolhidos pelo AUROC de
  validação, incluindo as predições já salvas da rodada anterior;
- **Avaliação robusta pareada** (5 partições) de original vs. melhorado.

**Auditoria:** nada é reexecutado nem sobrescrito. As melhorias são opções
novas nos wrappers de `src/churn_telecom/models/`, com padrões que
reproduzem exatamente a rodada anterior; cada experimento grava um
`model_id` novo e o notebook aborta se o ID já existir em `results/`. Os
resultados vão para a branch `results/churn-modelos-melhorados`.

Os IDs da rodada anterior foram renomeados para nomes descritivos (sem
"kaggle"/"leva1"); o mapa antigo → novo está em `results/RENOMEACOES.md`.

### Rodada 3 — Optuna com procedimento único (células no fim do notebook 06)

A rodada 2 não trouxe ganho mensurável (5 partições: tudo entre KS 0.528 e
0.535, desvio ≈ 0.015). A rodada 3 aplica o **mesmo procedimento** a MLP, GB,
XGBoost, STab, TabKAN e KAN: Optuna (TPE, mesma semente, mesmo nº de trials),
objetivo AUROC de validação, early stopping pelo mesmo critério nas redes,
teste usado uma vez. Saídas: importância de cada hiperparâmetro (fANOVA,
`reports/tables/optuna_importancia_*.csv` +
`reports/figures/optuna_importancia_hiperparametros.png`) e avaliação em 5
partições padrão vs. Optuna (`reports/tables/efeito_otimizacao_e_codificacao.csv`).

**Codificação por família:** GB/XGBoost usam `Preprocessor(modo="arvores")`
(numéricas na escala original + código ordinal por categórica, 19 colunas);
as redes seguem com z-score + one-hot (40 colunas). As rodadas 1–2 usaram
one-hot também nas árvores — a avaliação robusta mede esse efeito isolado.
TabPFN/Mitra ficam fora da busca (pré-treinados, sem hiperparâmetros de treino).

Num novo "Run All", o notebook pula as etapas já versionadas (rodada 2 e
buscas já salvas); os resultados vão para a branch `results/churn-otimizacao-optuna`.

### Rodada 4 — o que influenciou de verdade (células no fim do notebook 06)

A rodada 3 deu ganhos pequenos, e os trials variam pouco (TabKAN/KAN: ~0.003 de
AUROC), então a importância fANOVA sozinha pode estar repartindo ruído. A
rodada 4 mede cada efeito contra a régua certa:
`ruido_semente_vs_hiperparametros.csv/.png` (variação entre trials vs. só por
semente), TabPFN com/sem duplicatas em 5 partições e
`fatores_que_influenciam_ks.csv/.png` — todos os fatores testados nas rodadas,
com diferença de KS **pareada** por partição e IC 95% (t, 4 g.l.).

Resultados (branch `results/churn-fatores-influencia`):

- **Contexto do TabPFN sem as duplicatas do oversampling: +0.038 de KS**
  (IC 95% [+0.027, +0.050]) — o maior efeito do projeto. Nesse modo o TabPFN
  chega a KS 0.538 em 5 partições, o melhor modelo isolado.
- **GB com Optuna: +0.010** (IC [+0.002, +0.018]) — único ganho de busca de
  hiperparâmetros fora do ruído.
- Ajuste manual do TabKAN: −0.002 com IC estreito (treino quase determinístico),
  irrelevante na prática.
- Os outros 13 fatores (algoritmo, codificação, Optuna nas redes, bagging,
  ensemble) ficam **dentro do ruído**. Com 16 testes, ~1 falso positivo é esperado.
- Ruído de semente: nas árvores, a variação entre trials é 11–14× a de semente
  (os hiperparâmetros importam); em MLP/STab ~2× e no KAN 0.6× (a busca varia
  o resultado tanto quanto trocar a semente).

### Rodada 5 — limiar, stacking, especialistas e vazamento (células no fim do notebook 06)

Implementa os "Experimentos propostos" abaixo, nas mesmas 5 partições:
acurácia com o limiar escolhido na validação natural
(`acuracia_limiar_validacao_*.csv/.png`), stacking por regressão logística e
um XGBoost por tipo de contrato com teste pareado
(`fatores_rodada5_stacking_especialistas.csv/.png`), e a demonstração de
vazamento, com SMOTE/duplicação antes vs. depois da separação
(`demonstracao_vazamento_*.csv`, `demonstracao_vazamento.png`). As
probabilidades de todos os modelos ficam em `predicoes_5particoes_rodada5.csv`.

## Pontos de melhoria possíveis

Levantamento do que ainda pode ser feito, a partir dos resultados das rodadas
1–4 e de uma pesquisa na literatura. Os experimentos propostos no fim da
seção estão na rodada 5.

### Onde estamos (o teto atual)

- Todos os modelos convergem para **AUROC ≈ 0.84 e KS ≈ 0.53** (5 partições),
  de árvores a transformers e modelos pré-treinados (TabPFN, Mitra).
- **Acurácia no teste** (distribuição real, 73,4% de não-churn):
  - **73,4%** chutando sempre "não cancela";
  - **71–78%** no limiar 0.5 — baixo de propósito, porque o treino
    balanceado faz os modelos marcarem muito churn (recall ~75%);
  - **~80,5%** com o limiar escolhido na validação para maximizar acurácia
    (MLP Optuna, Mitra, TabKAN e ensembles empatados);
  - **80,6%** de teto escolhendo o limiar no próprio teste (só referência).
- Quinze configurações diferentes empatando no mesmo lugar indicam que o
  limite está na **informação das 19 variáveis**, não na escolha do modelo.
  Como o projeto se mantém no dataset do enunciado (ver "Dados externos"),
  as melhorias abaixo trabalham dentro desse limite.

### Cuidado com os 92–96% citados para este dataset

Resultados muito acima de ~82% de acurácia neste problema costumam ter
uma destas origens, e nenhuma é um modelo melhor no mesmo problema:

1. **Oversampling (SMOTE etc.) antes de separar o teste.** As linhas
   sintéticas são geradas a partir de linhas que vão parar no teste, e o
   modelo é avaliado em quase-cópias do que viu no treino. Em uma
   reanálise de um estudo médico, a acurácia caiu de ~99% para ~80% ao
   restringir o SMOTE ao treino
   ([PMC12842384](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12842384/));
   o efeito é reproduzido em experimentos controlados
   ([arXiv 2401.13796](https://arxiv.org/pdf/2401.13796)) e documentado em
   vários estudos publicados
   ([Princeton](https://reproducible.cs.princeton.edu/predicting-hits.html)).
   Trabalhos com 96% ([TEM Journal](https://temjournal.com/content/133/TEMJournalAugust2024_1927_1936.pdf))
   e 92,8% ([UMN](https://kc.umn.ac.id/id/eprint/47375/)) usam SMOTE sem
   deixar claro onde ele foi aplicado. Este pipeline aplica oversampling
   **só em treino/validação**, com o teste sempre na distribuição real.
2. **Versão expandida do dataset da IBM com colunas que vazam o alvo**
   ([11.1.3](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)):
   ela traz, entre outras, o **Churn Score** (saída de um modelo da IBM
   construído com os fatores de churn), Satisfaction Score, CLTV e Churn
   Reason (só preenchido para quem saiu). Com essas colunas o modelo
   praticamente recebe a resposta.
3. **Kaggle Playground (março/2026)**: vencida com um stack de 4 níveis e
   150 modelos ([NVIDIA](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)),
   com soluções simples em AUROC ~0.91
   ([Hugging Face](https://huggingface.co/NathanSJSU01/xgboost_churn_detection)).
   Mas as séries Playground usam dados **gerados sinteticamente** a partir
   do original, maiores e com outra distribuição — não comparável.

### Melhorias com ganho esperado (da maior para a menor evidência)

| Técnica | Ganho esperado | Observação |
|---|---|---|
| **Ajustar o limiar de decisão** (escolhido na validação) | **+3 pontos de acurácia** (77% → 80,5%) | Já medido nas predições salvas. Maior alavanca de acurácia disponível; não muda KS/AUROC. Basta reportar a acurácia neste limiar ao lado do KS |
| **Stacking** (meta-modelo treinado nas predições fora da amostra dos outros) | +0.002–0.005 de AUROC | É o que vence competições, mas exige modelos que erram em clientes diferentes. Os nossos são muito correlacionados: a média em ensemble não ganhou nada no teste pareado (+0.0001) |
| **Interações explícitas** (contrato × forma de pagamento, fibra × sem suporte técnico, etc.) | pequeno | Ajudam principalmente as redes; GB/XGBoost já aprendem interações sozinhos. As 3 features derivadas já testadas (`charges_per_tenure`, `n_servicos_adicionais`, `tenure_bucket`) não ajudaram |

### Ideias avaliadas, com baixa expectativa de ganho

- **Um modelo especializado em churn=Não e outro em churn=Sim, combinados.**
  Num problema binário, um classificador que separa Não de Sim já é bom nas
  duas classes; treinar cada classe separadamente (modelos generativos ou de
  uma classe só) costuma ser pior que o discriminativo direto. As variantes
  que fazem sentido são:
  - **especialistas por segmento** (ex.: um modelo por tipo de contrato — o
    churn é muito diferente entre mensal e bianual), mas árvores já dividem
    por contrato sozinhas;
  - **cascata** (o 1º modelo descarta os casos fáceis, o 2º decide os
    difíceis), que raramente supera um GBDT em dados deste tamanho — árvores
    continuam o estado da arte em dados tabulares médios (~10 mil linhas;
    [Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815v1)).
- **Dados artificiais** (SMOTE, CTGAN, TabDDPM, aplicados só no treino).
  Na literatura, dado sintético raramente supera o real, o SMOTE é um
  baseline difícil de bater ([TabDDPM](https://arxiv.org/pdf/2209.15421)) e
  o CTGAN gera dados de baixa qualidade em avaliação sistemática
  ([arXiv 2402.06806](https://arxiv.org/pdf/2402.06806)). Gerar linhas novas
  não cria informação que as variáveis não têm. O mini-projeto 2
  (`03_gan_augmentation`) já explorou GAN para aumento de dados.
- **Mais busca de hiperparâmetros.** A rodada 4 mostrou que os ganhos da
  busca são da ordem do ruído de semente (exceção: GB, +0.010 de KS).

### Dados externos — considerado e descartado

Os clientes são **fictícios** (a IBM descreve o dataset como uma
[empresa de telecom fictícia com 7.043 clientes na Califórnia](https://www.ibm.com/docs/SSEP7J_11.1.0/com.ibm.swg.ba.cognos.ig_smples.doc/c_telco_dm_sam.html)),
então não há base externa real com esses clientes. A única opção seria a
versão expandida da própria IBM (idade, indicações, uso de dados, CEP etc.,
descartando as colunas que vazam o alvo). **Decisão: não usar** — mudaria o
problema em relação ao dataset de 21 colunas do enunciado, e a comparação
com o resto do projeto deixaria de valer.

### Experimentos propostos (implementados na rodada 5)

1. Acurácia com limiar escolhido na validação, reportada para todos os modelos.
2. Stacking e especialistas por segmento, no procedimento pareado de 5 partições.
3. **Demonstração controlada do vazamento**: o mesmo modelo com SMOTE antes da
   separação (com vazamento) vs. depois (correto), mostrando de onde vêm os
   "95%".

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
python3 scripts/make_kaggle_notebook.py # gera notebooks/05_modelos_avancados_kaggle.ipynb (subir no Kaggle, não rodar aqui)
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
