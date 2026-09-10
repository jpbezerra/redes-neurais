# Fase 2 — CNN para classificar o CIFAR-10

**Status: em andamento.** Fundação do projeto criada (pacote `cnn_cifar10`,
notebook de treino, estrutura de resultados) — próximo passo é rodar os
experimentos. Continuidade direta da [Fase 1 (MLP)](../fase1-mlp/README.md),
que fechou em **0.6135 de acurácia** (ensemble de MLPs) e documentou a
limitação estrutural do MLP (perde a estrutura espacial da imagem) que a CNN
ataca diretamente via convolução.

## Objetivo (conforme o enunciado)

Repetir o experimento de classificação do CIFAR-10, agora com uma **CNN
(rede convolucional)**, como continuidade direta do material e do código da
Fase 1 (mesmo PPT/relatório, complementado; mesma linha de código, evoluída).

Referência de apoio: [neuralnetworksanddeeplearning.com/chap6.html](http://neuralnetworksanddeeplearning.com/chap6.html)

## O que entregar

- Complemento do PPT/relatório do MLP com os experimentos da CNN.
- Taxas de acerto por classe (acurácia) e métricas globais (acurácia,
  precision, recall).
- Parâmetros a variar: tamanho da rede, tamanho do filtro de convolução
  (kernel), stride, padding, dropout, janela de pooling, taxa de
  aprendizagem.
- Buscar ganhos reais de desempenho (não só documentar o efeito de cada
  hiperparâmetro).
- Apresentação de 15 minutos (grupos sorteados no dia) — feita junto com os
  resultados desta fase, não produzida agora.

## Estrutura do projeto

Espelha a Fase 1 de propósito, para que os dois relatórios fiquem
comparáveis e o notebook `02_results_report` possa ser reaproveitado quase
sem alterações:

```
fase2-cnn/
├── README.md
├── pyproject.toml / requirements.txt
├── src/cnn_cifar10/
│   ├── config.py          # ExperimentConfig (arquitetura conv + hiperparâmetros)
│   ├── data.py             # CIFAR-10 + DataLoaders (idêntico à Fase 1)
│   ├── model.py             # classe CNN parametrizável (blocos conv + cabeça densa)
│   ├── metrics.py           # accuracy/precision/recall/f1 (idêntico à Fase 1)
│   ├── train.py             # fit()/fit_or_load(), early stopping, LR scheduler
│   └── checkpointing.py     # skill "model-saver": salva model.pt + metadata.json
├── notebooks/
│   └── 01_train_cnn_cifar10.ipynb   # define/roda experimentos (treino)
├── scripts/                 # rodadas de busca guiada fora do notebook (a criar conforme necessário)
├── data/                    # CIFAR-10 baixado (não versionado)
└── results/                 # results/{model_id}/ — versionado (pesos + metadados)
```

## Arquitetura da CNN

`CNN` (`src/cnn_cifar10/model.py`) é parametrizável: um número arbitrário de
blocos convolucionais (`Conv2d -> [BatchNorm2d] -> ativação -> MaxPool2d ->
[Dropout2d]`) seguido de uma cabeça densa (`Linear -> ativação -> [Dropout]`
repetido) e uma camada de saída. O tamanho do vetor achatado entre conv e
cabeça densa é inferido automaticamente (forward "a seco" no `__init__`),
então qualquer combinação de `kernel_size`/`stride`/`padding`/`pool_size`
funciona sem recalcular a aritmética na mão.

`ExperimentConfig` (`src/cnn_cifar10/config.py`) cobre os hiperparâmetros
pedidos no enunciado — `conv_channels`, `kernel_size`, `stride`, `padding`,
`pool_size`, `fc_layers`, `dropout`, `learning_rate` — além dos herdados da
Fase 1 (ativação, otimizador, batch norm, weight decay, função de erro,
augmentation, LR schedule).

Baseline (notebook `01`, seção 1): 2 blocos conv (32 e 64 filtros, kernel
3x3, padding 1) + max pooling 2x2 + cabeça densa `120 -> 84 -> 10` — mesma
arquitetura do notebook de referência do professor (`temp/CIFAR10_with_CNNs.ipynb`,
adaptação do LeNet-5).

## Como rodar

**Google Colab (recomendado)** — a primeira célula do notebook detecta o
Colab, clona o repositório e instala o pacote automaticamente. Ativar GPU em
Ambiente de execução > Alterar tipo de ambiente de execução. Ver seção
"Como sincronizar com o Colab" abaixo.

**Local**:
```
cd miniprojeto/fase2-cnn
python -m venv .venv && .venv/Scripts/activate  # Windows
pip install -e .
jupyter notebook notebooks/01_train_cnn_cifar10.ipynb
```

## Como sincronizar com o Colab

1. **Commitar e dar push** das mudanças locais (o notebook do Colab clona o
   `main`/branch do GitHub, não lê o disco local).
2. No Colab, abrir `notebooks/01_train_cnn_cifar10.ipynb` — pela UI do Colab
   (Arquivo > Abrir notebook > GitHub > `jpbezerra/redes-neurais` > escolher
   o branch e o caminho do arquivo) ou fazendo upload manual do `.ipynb`.
3. Rodar a célula "Setup (Colab ou local)": ela clona o repo em
   `/content/redes-neurais`, faz `pip install -e` do pacote `cnn_cifar10` e
   ajusta `PROJECT_ROOT`/`DATA_DIR`/`RESULTS_DIR`. Se o repo já tiver sido
   clonado numa execução anterior da sessão, ela dá `git pull` em vez de
   clonar de novo.
4. Depois de treinar no Colab, os `results/{model_id}/` novos ficam dentro
   do clone em `/content/redes-neurais/miniprojeto/fase2-cnn/results/` —
   **isso é efêmero** (perdido quando a sessão do Colab reinicia). Para
   trazer de volta ao repositório local, duas opções:
   - **Mais simples**: baixar a pasta `results/` do Colab (zipar e usar
     `files.download`, ou montar o Google Drive) e copiar por cima da
     `results/` local, depois `git add`/`commit`/`push` daqui.
   - **Direto do Colab**: autenticar git no Colab (`git config` + token de
     acesso pessoal do GitHub) e commitar/dar push de lá mesmo. Mais rápido
     se for rodar várias rodadas de experimento só no Colab.
5. Se estiver iterando bastante no Colab antes de sincronizar, vale editar o
   notebook lá, testar, e só trazer o `.ipynb` final de volta pro repo local
   (`File > Download > Download .ipynb`, substituindo o arquivo aqui) junto
   com os `results/` novos.

## Estado da busca (levas 1-7)

Melhor modelo único até agora: **`vgg4_randaugment` = 0.9417** — VGG de 4
estágios (64/128/256/512, 2 convs por estágio), global average pooling,
SGD+momentum+cosine, `weight_decay=5e-4`, RandAugment, 120 épocas.

Alavancas que realmente moveram a agulha, em ordem de impacto: profundidade
real (2 convs por estágio de pooling, ~+1.5pp), SGD+momentum+cosine no lugar
do Adam (~+3pp em redes fundas), canais mais largos (~+0.7pp) e
`weight_decay=5e-4` (~+1.7pp sobre a família rasa). A leva 7 (regularização e
augmentation sobre o campeão) espalhou seis modelos numa faixa de 0.47pp —
dentro do ruído, ou seja, **o platô de ~0.94 não é de regularização**.

O erro restante é concentrado: veículos vão a 0.964, animais ficam em 0.918,
e `cat` (0.854) + `dog` (0.909) sozinhos respondem por cerca de um quarto de
todos os erros.

## Próximos passos

Três frentes já implementadas no notebook (seções 13-15), motivadas pela
análise acima:

- **Leva 8 — MixUp/CutMix** (`mixup_alpha`, `cutmix_alpha`, `mix_prob` no
  `ExperimentConfig`): o único regularizador forte ainda não testado, com 200
  épocas por ser mais lento a convergir.
- **Ensemble** (`cnn_cifar10.ensemble`): média das probabilidades dos modelos
  já salvos em `results/`, sem retreinar nada. Entra como comparação extra —
  o resultado principal do mini-projeto continua sendo o modelo único.
- **Hierárquico** (`cnn_cifar10.hierarchical`): um porteiro veículo vs. animal
  mais dois especialistas (4 e 6 classes), recompostos por
  `P(classe) = P(super) · P(classe | super)`. Testa se isolar o problema dos
  animais compensa cada especialista ver menos dados e herdar os erros do
  porteiro.

Ao final, comparar o melhor resultado da CNN diretamente com o ensemble MLP
(0.6135) no README.
