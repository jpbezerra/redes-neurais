# Mini-projeto 1: Classificação de Imagens com MLPs e CNNs

Baseado no material "Mini-projeto 1: Classificação de Imagens com MLPs e CNNs"
(Germano C. Vasconcelos, Centro de Informática - UFPE). Dataset: **CIFAR-10**
(60.000 imagens coloridas 32x32, 10 classes, 50.000 treino / 10.000 teste).

## Fases

| Fase | Pasta | Objetivo | Status |
|---|---|---|---|
| 1 | [`fase1-mlp/`](fase1-mlp/README.md) | Treinar uma rede **MLP** para classificar o CIFAR-10 | Em desenvolvimento |
| 2 | [`fase2-cnn/`](fase2-cnn/README.md) | Repetir o experimento com uma rede **CNN**, como continuação da Fase 1 | Não iniciada |

## Regras gerais da entrega (conforme enunciado)

- Grupos de 4 a 5 pessoas.
- Entrega: PPT (ou outro formato de apresentação/PDF) + notebook do Jupyter.
- Apenas uma pessoa do grupo precisa enviar, mas o material deve conter o nome
  de todos os integrantes.
- A apresentação da Fase 2 (CNN) é conjunta com a Fase 1 (MLP) — o material da
  CNN deve ser uma continuidade do material do MLP.
- Prazo total: 4 semanas (2 semanas para a Fase 1 + 2 semanas para a Fase 2).

## O que cada fase precisa reportar

Para ambas as fases (MLP e CNN):

- Descrição dos experimentos realizados: hiperparâmetros investigados,
  variações testadas e resultados (ganhos/perdas de desempenho).
- Métricas: acurácia por classe e métricas globais (acurácia, precision, recall).
- Busca por ganhos de desempenho ao longo dos experimentos.

Parâmetros sugeridos para variar:

- **MLP**: número de camadas, número de neurônios por camada, taxa de
  aprendizagem, função de ativação, regularização, algoritmo de aprendizagem
  (otimizador), dropout, função de erro (MSE / entropia cruzada).
- **CNN**: tamanho da rede, tamanho do filtro de convolução, stride, padding,
  dropout, janela de pooling, taxa de aprendizagem.
