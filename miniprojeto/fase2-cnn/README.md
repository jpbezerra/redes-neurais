# Fase 2 — CNN para classificar o CIFAR-10

**Status: não iniciada.** Esta fase ainda não foi implementada — o foco atual
do repositório é a [Fase 1 (MLP)](../fase1-mlp/README.md).

## Objetivo (conforme o enunciado)

Repetir o experimento de classificação do CIFAR-10, agora com uma **CNN
(rede convolucional)**, como continuidade direta do material e do código da
Fase 1 (mesmo PPT/relatório, complementado; mesma linha de código, evoluída).

Referência de apoio: [neuralnetworksanddeeplearning.com/chap6.html](http://neuralnetworksanddeeplearning.com/chap6.html)

## O que entregar

- Complemento do PPT/relatório do MLP com os experimentos da CNN.
- Taxas de acerto por classe (acurácia) e métricas globais (acurácia,
  precision, recall).
- Parâmetros sugeridos para variar: tamanho da rede, tamanho do filtro de
  convolução, stride, padding, dropout, janela de pooling, taxa de
  aprendizagem.
- Apresentação de 15 minutos (grupos sorteados no dia).

## Estrutura planejada

Quando esta fase for iniciada, seguirá o mesmo padrão da Fase 1
(`src/<pacote>/`, `notebooks/`, `data/`, `results/`), reaproveitando o
que fizer sentido de `../fase1-mlp/src/mlp_cifar10/` (dados, métricas,
checkpointing) e adicionando um módulo de modelo próprio para a CNN.
