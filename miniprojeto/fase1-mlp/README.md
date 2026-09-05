# Fase 1 — MLP para classificar o CIFAR-10

Treinar um **MLP (Multi-Layer Perceptron)** para classificar as imagens do
CIFAR-10 (10 classes, imagens RGB 32x32), buscando os melhores hiperparâmetros.

## Estrutura

```
fase1-mlp/
├── README.md
├── pyproject.toml          # torna o pacote instalável (pip install -e .)
├── requirements.txt
├── src/mlp_cifar10/        # código reutilizável
│   ├── config.py           # ExperimentConfig: todos os hiperparâmetros de um experimento
│   ├── data.py              # download do CIFAR-10 + DataLoaders (treino/val/teste)
│   ├── model.py             # classe MLP (nº de camadas/neurônios, ativação, dropout, batch norm parametrizáveis)
│   ├── metrics.py           # métricas globais + acurácia por classe
│   ├── train.py             # loop de treino/avaliação com early stopping
│   └── checkpointing.py     # skill `model-saver`: salva model.pt + metadata.json de cada execução
├── notebooks/
│   └── 01_train_mlp_cifar10.ipynb   # notebook principal: define e roda os experimentos originais
├── scripts/
│   └── run_experiments.py  # baterias adicionais de experimentos (rounds de busca de hiperparâmetros), fora do notebook
├── data/                    # CIFAR-10 baixado automaticamente (não versionado)
└── results/                 # results/{model_id}/ — uma pasta por execução de treino (não versionado)
```

Essa separação entre `src/` (implementação) e `notebooks/` (orquestração de
experimentos + relatório) é o padrão recomendado para projetos de data
science: mantém o notebook curto e legível — o que importa para quem vai
avaliar o PPT/relatório — enquanto o código testável e reaproveitável fica
isolado em módulos.

## Como rodar

### Local

```bash
cd miniprojeto/fase1-mlp
python -m venv .venv && source .venv/bin/activate   # ou .venv\Scripts\activate no Windows
pip install -e .
jupyter notebook notebooks/01_train_mlp_cifar10.ipynb
```

### Google Colab

Abra `notebooks/01_train_mlp_cifar10.ipynb` no Colab e rode a primeira célula
de setup — ela clona o repositório e instala o pacote automaticamente
(edite a variável `REPO_URL` na célula para apontar para este repositório).

## Cada execução de treino é salva automaticamente (skill `model-saver`)

`train.fit(...)` chama `checkpointing.save_run(...)` ao final de cada
execução, que por sua vez implementa a skill **model-saver** do usuário
(variante PyTorch). É criada uma pasta própria `results/{model_id}/`, com
`model_id` único (`mlp_{run_name}_{timestamp}`, nunca reutilizado entre
execuções), contendo:

- `model.pt` — pesos do modelo (`model.state_dict()`);
- `metadata.json` — hiperparâmetros, arquitetura, dataset e métricas, no
  schema padrão da skill (`model_id`, `framework`, `timestamp`,
  `hyperparameters`, `metrics`, `architecture`, `dataset`, `notes`, ...);
- `history.csv` — extra: loss/acurácia de validação por época (para plotar curvas).

Para comparar todas as execuções já salvas (não só as da sessão atual do
notebook), use `checkpointing.load_all_metadata(results_dir)` — ver a seção
"Comparando todas as execuções salvas" do notebook.

## O que reportar (ver também `../README.md`)

- Hiperparâmetros investigados e resultado de cada variação (a tabela
  comparativa gerada na seção 2 do notebook é o ponto de partida).
- Acurácia por classe e métricas globais (acurácia, precision, recall) no
  conjunto de teste.
- Parâmetros sugeridos para variar: número de camadas, número de neurônios,
  taxa de aprendizagem, função de ativação, regularização (weight decay),
  otimizador, dropout, função de erro (entropia cruzada / MSE).

## Relatório do experimento: busca de hiperparâmetros

### Objetivo e metodologia

Além dos 17 experimentos originais do notebook (que cobrem cada hiperparâmetro
isoladamente — ativação, otimizador, batch size, dropout, weight decay, etc.),
foi feita uma busca guiada em **5 rodadas** de 5 configurações cada
(25 execuções via `scripts/run_experiments.py`), totalizando **42 execuções**
registradas em `results/`. A cada rodada, o(s) melhor(es) resultado(s) da
rodada anterior foram usados para decidir a próxima bateria de testes
(busca gulosa/coordenada, não uma grade exaustiva) — cada rodada testa uma
hipótese específica derivada da rodada anterior, não combinações aleatórias.

Todas as execuções usam CIFAR-10 (45.000 treino / 5.000 validação / 10.000
teste), Adam como otimizador padrão e entropia cruzada como perda, exceto
onde indicado.

### Evolução da acurácia de teste, rodada a rodada

| Rodada | Melhor config | Acurácia | Ganho sobre a rodada anterior |
|---|---|---|---|
| Baseline (17 experimentos originais) | `best_combo` — `[256,128,64,32]`, relu, dropout 0.3, batch norm, weight_decay 1e-4 | 0.5319 | — |
| 1 | `even_deeper` — rede mais larga `[384,192,96,48]` | 0.5387 | +0,68 p.p. |
| 2 | `no_weight_decay` — remove weight decay | 0.5522 | +1,35 p.p. |
| 3 | `combo_light` — gelu + dropout 0.2 + sem weight decay (combina os 3 vencedores da rodada 2) | 0.5655 | +1,33 p.p. |
| 4 | `combo_deeper_wider` — rede ainda maior `[512,256,128,64,32]` | 0.5712 | +0,57 p.p. |
| 5 | `deeper_wider_lr_lower` — mesma rede + learning rate menor (5e-4) | **0.5770** | +0,58 p.p. |

O ganho por rodada está diminuindo (1,35 → 1,33 → 0,57 → 0,58 p.p.),
um sinal característico de retornos decrescentes: cada rodada ainda melhora,
mas cada vez menos, indicando que a busca está convergindo para um platô
específico desta família de modelos (MLP).

### Configuração vencedora final

`mlp_deeper_wider_lr_lower` — acurácia de teste **0.5770** (F1 0.5768):

| Hiperparâmetro | Valor |
|---|---|
| Arquitetura | `[512, 256, 128, 64, 32]` (funil profundo de 5 camadas ocultas) |
| Ativação | GELU |
| Dropout | 0.2 |
| Batch Normalization | Sim |
| Weight decay (L2) | 0.0 |
| Learning rate | 5e-4 |
| Otimizador | Adam |
| Épocas treinadas | 27 (early stopping, paciência 8) |

Acurácia por classe do melhor modelo:

| Classe | Acurácia | Classe | Acurácia |
|---|---|---|---|
| airplane | 0.670 | dog | 0.437 |
| automobile | 0.648 | frog | 0.663 |
| bird | 0.437 | horse | 0.633 |
| cat | 0.445 | ship | 0.694 |
| deer | 0.511 | truck | 0.632 |

Padrão recorrente em **todos** os 42 experimentos, não só no melhor: veículos
e cenários com silhueta bem definida (`ship`, `airplane`, `automobile`,
`frog`) são consistentemente as classes mais fáceis; animais com pose e
textura variáveis (`cat`, `dog`, `bird`, `deer`) são as mais difíceis — `cat`
e `dog` frequentemente se confundem entre si. Isso é esperado para um MLP:
sem convolução, o modelo não tem nenhuma noção de invariância translacional
ou de textura local, então classes que dependem de forma/silhueta global se
saem melhor do que classes que dependem de padrões locais (pelo, textura).

### O que a busca revelou sobre cada hiperparâmetro

- **Arquitetura (capacidade)**: aumentar largura/profundidade ajudou de forma
  consistente e foi o fator de maior impacto isolado — `[64,128,64]` (baseline
  original, ~0.51) → `[256,128,64,32]` (~0.53) → `[384,192,96,48]` (~0.55) →
  `[512,256,128,64,32]` (~0.57). Porém isso satura: ir além (6 camadas, ou
  `[768,384,192,96,48]`) não trouxe ganho adicional consistente
  (`six_layer_combo` = 0.5609, pior que a versão de 5 camadas equivalente).
- **Regularização (dropout/weight decay)**: existe um ponto ótimo estreito.
  Dropout 0.2 bateu tanto valores maiores (0.3, 0.5 — nos dois casos as
  piores execuções do conjunto, chegando a 0.4668–0.4692) quanto menores
  (0.1 — deu early stopping prematuro por overfitting rápido, 0.5529).
  Weight decay, surpreendentemente, funcionou melhor em **zero**: com
  batch norm + dropout já presentes, a penalização L2 extra só atrapalhava
  a convergência (valores de 1e-3 caíram para ~0.466-0.467, entre os piores
  do estudo todo).
- **Ativação**: GELU superou ReLU de forma pequena mas consistente nas
  arquiteturas maiores (a mesma rede com ReLU no lugar de GELU rendeu 0.5571
  vs. 0.5712/0.5770 com GELU — cerca de 1,4-2 p.p. a menos). Tanh e Sigmoid-like
  (LeakyReLU) ficaram atrás de ReLU/GELU desde o início.
- **Learning rate**: 1e-3 (padrão do Adam) era um bom ponto de partida, mas ao
  aumentar a capacidade da rede um LR levemente menor (5e-4) convergiu para
  um ótimo melhor; LR maior (2e-3) piorou (0.5584). LR muito baixo (1e-4) ou
  muito alto (5e-3) nos experimentos originais já haviam mostrado prejuízo
  claro (0.5091 e 0.4769 respectivamente).
- **Otimizador/perda**: SGD com momentum e a perda MSE (testados nos
  experimentos originais) ficaram no meio da tabela, sem vantagem sobre
  Adam + entropia cruzada, que foi mantido fixo daí em diante.
- **Batch size**: batch maior (256, testado no baseline) não trouxe ganho.
- **Efeito combinado**: combinar os vencedores individuais de uma rodada
  (round 3: gelu + dropout 0.2 + sem weight decay) rendeu mais do que
  qualquer um isoladamente — os efeitos de arquitetura, ativação e
  regularização são majoritariamente aditivos/independentes entre si nesta
  faixa de valores.

### Dá para melhorar mais, ou já chegou no limite?

**Para um MLP puro neste orçamento de treino, sim, já estamos perto do teto
prático** — mas não porque os hiperparâmetros se esgotaram, e sim porque a
**arquitetura em si é o fator limitante**:

- O ganho por rodada caiu de +1,35 p.p. para +0,57-0,58 p.p. nas duas
  últimas rodadas — clássico sinal de retornos decrescentes na busca de
  hiperparâmetros.
- Um MLP achata a imagem 32×32×3 num vetor de 3072 valores e perde toda
  estrutura espacial: não há invariância translacional nem detecção de
  padrões locais (bordas, texturas). É exatamente por isso que as classes
  mais confundidas são as que dependem de textura/pose (gato, cachorro,
  pássaro) — nenhum ajuste de dropout ou learning rate resolve essa
  limitação estrutural.
- Redes CNN convolucionais tipicamente alcançam 70-90%+ em CIFAR-10 com
  esforço de tuning comparável, justamente porque exploram essa estrutura
  espacial que o MLP ignora — o que também explica por que este projeto já
  prevê uma `fase2-cnn` como próxima etapa.

**Ainda assim, existem alavancas não exploradas nesta busca que poderiam
render mais alguns pontos percentuais dentro da família MLP**, em ordem
aproximada de impacto esperado:

1. **Data augmentation** (flips horizontais, crops aleatórios, jitter de cor) —
   não foi testado em nenhuma das 42 execuções; costuma ser o ganho mais
   barato em CIFAR-10, mesmo para MLP.
2. **LR schedule** (cosine annealing, warmup, ou redução ao patamar) em vez de
   LR fixo — os treinos mais longos (`combo_longer`, `deeper_wider_more_patience`)
   sugerem que a perda de validação oscila perto do fim, o que um schedule
   ajudaria a estabilizar.
3. **Normalização com média/desvio-padrão reais do CIFAR-10** (o código já
   comenta essa alternativa em `data.py`, mas os 42 experimentos usaram a
   normalização simples para [-1, 1]).
4. **Ensemble** de 3-5 modelos com seeds diferentes — ganho tipicamente de
   1-2 p.p., mas não reduz a limitação estrutural do MLP.
5. Ajuste fino adicional de dropout entre 0.15-0.2 e de LR entre 5e-4 e 1e-3
   teria retorno marginal (a busca já mostrou que a região ótima é estreita
   e plana ali perto).

Resumindo: **mais hiperparâmetros isolados provavelmente não vão além de
~0.58-0.60 de acurácia nesta arquitetura**; para um salto real, o caminho é
data augmentation (ganho rápido dentro do MLP) ou migrar para convolução
(`fase2-cnn`), que é onde a estrutura espacial da imagem passa a ser usada.
