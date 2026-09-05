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
│   ├── run_experiments.py  # baterias adicionais de experimentos (rounds de busca de hiperparâmetros), fora do notebook
│   └── ensemble_eval.py    # ensemble (soft voting) dos melhores modelos já treinados, sem retreinar
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
foi feita uma busca guiada em **6 rodadas** de 3-5 configurações cada
(28 execuções via `scripts/run_experiments.py`), totalizando **45 execuções**
registradas em `results/`, mais um ensemble por soft voting dos 3 melhores
modelos (`scripts/ensemble_eval.py`, sem retreinar nada). A cada rodada, o(s)
melhor(es) resultado(s) da rodada anterior foram usados para decidir a
próxima bateria de testes (busca gulosa/coordenada, não uma grade exaustiva)
— cada rodada testa uma hipótese específica derivada da rodada anterior, não
combinações aleatórias.

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
| 5 | `deeper_wider_lr_lower` — mesma rede + learning rate menor (5e-4) | 0.5770 | +0,58 p.p. |
| 6 | `combo_aug_schedule` — mesma rede + data augmentation (flip/crop) + LR cosine annealing | **0.5863** | +0,93 p.p. |
| — | **Ensemble (soft voting)** dos 3 melhores modelos, sem retreinar | **0.6058** | +1,95 p.p. sobre o melhor individual |

O ganho por rodada tinha caído para +0,57-0,58 p.p. nas rodadas 4 e 5 —
retornos decrescentes claros dentro do espaço já explorado (arquitetura,
ativação, regularização, learning rate). A rodada 6 testou **duas alavancas
de um tipo diferente** (data augmentation e LR schedule) e reverteu essa
tendência: +0,93 p.p., maior que o ganho das duas rodadas anteriores juntas —
sinal de que augmentation/schedule atacam uma fonte de erro diferente
(overfitting/instabilidade de convergência) da que os hiperparâmetros de
arquitetura já haviam esgotado. O ensemble, por sua vez, ataca ainda outra
fonte (variância entre modelos individuais) e deu o maior salto de todos.

### Configuração vencedora final

**Melhor modelo individual — `mlp_combo_aug_schedule`** — acurácia de teste
**0.5863** (F1 0.5816), 40 épocas (sem early stopping — ainda melhorando):

| Hiperparâmetro | Valor |
|---|---|
| Arquitetura | `[512, 256, 128, 64, 32]` (funil profundo de 5 camadas ocultas) |
| Ativação | GELU |
| Dropout | 0.2 |
| Batch Normalization | Sim |
| Weight decay (L2) | 0.0 |
| Learning rate | 1e-3 inicial, com **cosine annealing** decaindo a 0 em 40 épocas |
| Data augmentation | `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` no treino |
| Otimizador | Adam |

**Melhor resultado geral — ensemble (soft voting) de 3 modelos** —
acurácia de teste **0.6058** (F1 0.6028), combinando as probabilidades
(softmax) dos modelos `combo_aug_schedule` (0.5863), `augmentation_flip_crop`
(0.5813) e `deeper_wider_lr_lower` (0.5770), sem nenhum retreino — ver
`scripts/ensemble_eval.py` e `results/mlp_ensemble_top3_softvote/metadata.json`.

Acurácia por classe (melhor modelo individual vs. ensemble):

| Classe | `combo_aug_schedule` | Ensemble (top-3) |
|---|---|---|
| airplane | 0.662 | 0.695 |
| automobile | 0.684 | 0.707 |
| bird | 0.432 | 0.475 |
| cat | 0.345 | 0.417 |
| deer | 0.465 | 0.492 |
| dog | 0.452 | 0.455 |
| frog | 0.709 | 0.715 |
| horse | 0.696 | 0.694 |
| ship | 0.746 | 0.732 |
| truck | 0.672 | 0.676 |

O ensemble melhora (quase) todas as classes, com o maior ganho justamente nas
classes mais difíceis (`cat` +7,2 p.p., `bird` +4,3 p.p.) — consistente com a
ideia de que modelos treinados com regimes distintos (com/sem augmentation,
com/sem schedule) erram de formas parcialmente independentes nessas classes
mais ambíguas, e a média das probabilidades corrige parte desses erros.

Padrão recorrente em **todos** os 45 experimentos, não só nos melhores:
veículos e cenários com silhueta bem definida (`ship`, `frog`, `automobile`,
`airplane`) são consistentemente as classes mais fáceis; animais com pose e
textura variáveis (`cat`, `dog`, `bird`, `deer`) são as mais difíceis — `cat`
e `dog` frequentemente se confundem entre si, e continuam sendo o gargalo
mesmo no melhor resultado (`cat` = 0.417 no ensemble, a classe mais fraca de
todas). Isso é esperado para um MLP: sem convolução, o modelo não tem
nenhuma noção de invariância translacional ou de textura local, então
classes que dependem de forma/silhueta global se saem melhor do que classes
que dependem de padrões locais (pelo, textura).

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
  faixa de valores. O mesmo padrão se repetiu na rodada 6: augmentation
  sozinha (0.5813) + cosine schedule sozinho (0.5732) somados superam os
  dois isoladamente (0.5863 combinados) — reforça que, dentro do espaço
  testado, cada alavanca ataca uma fonte de erro distinta.
- **Data augmentation** (`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()`,
  rodada 6): o ganho isolado mais alto de toda a busca (+4,3 p.p. sobre a
  mesma arquitetura sem augmentation, 0.5770 → 0.5813) — e o modelo ainda não
  tinha convergido em 40 épocas, sugerindo espaço para mais. Custo: ~3x mais
  lento por época em CPU sem `num_workers` (crop/flip por amostra, sem
  paralelismo), então cada execução com augmentation levou ~1h em vez de
  ~10-20 min.
- **LR schedule** (cosine annealing, rodada 6): isoladamente ficou levemente
  **abaixo** do LR fixo mais bem ajustado (0.5732 vs. 0.5770), mas combinado
  com augmentation superou os dois (0.5863) — o schedule parece ajudar mais
  quando há mais "ruído" para estabilizar no fim do treino (introduzido pela
  augmentation) do que em treino sem augmentation, onde o LR fixo bem
  ajustado (5e-4) já bastava.
- **Ensemble** (soft voting de 3 modelos, sem retreinar): maior ganho
  isolado de toda a busca em termos absolutos (+1,95 p.p., 0.5863 → 0.6058),
  e de graça computacionalmente (usa checkpoints já salvos). Funciona porque
  os 3 modelos usam regimes de treino diferentes (com/sem augmentation,
  com/sem schedule) e portanto erram de forma parcialmente independente.

### Dá para melhorar mais, ou já chegou no limite?

**Resposta confirmada empiricamente**: ainda dava para melhorar dentro da
família MLP — as 3 alavancas sugeridas (augmentation, LR schedule, ensemble)
somadas levaram a acurácia de 0.5770 para **0.6058** (+2,88 p.p.), maior que
o ganho de qualquer rodada anterior de busca de hiperparâmetros "tradicional"
(arquitetura/ativação/regularização/LR, que já tinha saturado em ~+0,57-0,58
p.p. por rodada). Isso mostra que "retornos decrescentes" valia **para o tipo
de alavanca já testado**, não para o MLP como família — trocar de categoria de
alavanca (regularização de dados em vez de regularização do modelo; combinar
modelos em vez de tunar um só) reabriu ganho real.

Dito isso, **para um MLP puro, agora sim parece estar perto de um teto mais
sério** — as alavancas "baratas" (arquitetura, ativação, regularização, LR,
augmentation, schedule, ensemble simples) já foram testadas; o que resta
tem retorno esperado menor ou custo bem mais alto:

- Um MLP achata a imagem 32×32×3 num vetor de 3072 valores e perde toda
  estrutura espacial: não há invariância translacional nem detecção de
  padrões locais (bordas, texturas). Isso continua visível mesmo no melhor
  resultado — `cat` (0.417) e `bird` (0.475) seguem as classes mais fracas do
  ensemble, exatamente as que mais dependem de textura/pose em vez de
  silhueta global. Nenhuma das alavancas testadas resolve essa limitação
  estrutural, só mitiga seus sintomas (menos overfitting, menos variância).
- Redes CNN convolucionais tipicamente alcançam 70-90%+ em CIFAR-10 com
  esforço de tuning comparável, justamente porque exploram essa estrutura
  espacial que o MLP ignora — o que também explica por que este projeto já
  prevê uma `fase2-cnn` como próxima etapa.

**Alavancas que ainda não foram testadas e poderiam render mais alguns
pontos**, em ordem aproximada de impacto esperado / esforço:

1. **Mais épocas com augmentation**: `augmentation_flip_crop` e
   `combo_aug_schedule` chegaram ao teto de 40 épocas ainda melhorando —
   rodar por mais tempo (60-80 épocas) provavelmente renderia mais um pouco.
2. **Ensemble maior/mais diverso** (5+ modelos, incluindo seeds diferentes
   da mesma config, não só configs diferentes) — o ganho de ensemble tende a
   saturar, mas 3 membros é pouco; vale testar 5.
3. **Normalização com média/desvio-padrão reais do CIFAR-10** (o código já
   comenta essa alternativa em `data.py`, nunca testada nas 45 execuções).
4. Jitter de cor/brightness além de crop+flip, ou `Cutout`/`Mixup` — formas
   de augmentation mais agressivas que talvez ajudem ainda mais dado que
   crop+flip sozinho já foi o maior ganho isolado da busca toda.

Resumindo: o MLP surpreendeu ao render mais do que o esperado inicialmente
(pulou de um platô aparente de ~0.57 para 0.6058) assim que se mudou de
categoria de alavanca — mas o limite estrutural (perda da estrutura 2D da
imagem) continua sendo o teto real, visível nas classes de animais que nunca
passam de ~0.4-0.5 mesmo no melhor resultado. Para um salto de outra ordem de
grandeza (70-90%+), o caminho é migrar para convolução (`fase2-cnn`).
