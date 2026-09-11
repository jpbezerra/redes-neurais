# Mini-projeto 1: Classificação de Imagens com MLPs e CNNs

Baseado no material "Mini-projeto 1: Classificação de Imagens com MLPs e CNNs"
(Germano C. Vasconcelos, Centro de Informática - UFPE). Dataset: **CIFAR-10**
(60.000 imagens coloridas 32x32, 10 classes, 50.000 treino / 10.000 teste).

## Fases

| Fase | Pasta | Objetivo | Melhor resultado | Status |
|---|---|---|---|---|
| 1 | [`fase1-mlp/`](fase1-mlp/README.md) | Treinar uma rede **MLP** para classificar o CIFAR-10 | **0.6135** (ensemble) / 0.5998 (modelo único) | Concluída |
| 2 | [`fase2-cnn/`](fase2-cnn/README.md) | Repetir o experimento com uma rede **CNN**, como continuação da Fase 1 | **0.9554** (ensemble) / 0.9490 (modelo único) | Concluída |
| — | [`comparativo/`](comparativo/README.md) | Análise das duas fases lado a lado | — | Concluída |

**Resultado consolidado: a convolução valeu +34,2 pontos percentuais** sobre a
melhor configuração de MLP encontrada com esforço de busca equivalente — uma
redução de 88,5% na taxa de erro (de 38,6% para 4,5%). O ganho concentrou-se
nas classes de animais (+39,6 p.p. em média) muito mais que nas de veículos
(+26,0 p.p.), que é exatamente o que a teoria prevê: animais dependem de
textura local e pose, informação que o MLP destrói ao achatar a imagem num
vetor de 3.072 valores e que a convolução preserva.

## Metodologia comum às duas fases

As duas fases seguem deliberadamente o **mesmo protocolo**, o que torna os
resultados diretamente comparáveis e permite separar o que é lição sobre o
problema do que é lição sobre a arquitetura:

- Mesma divisão de dados: 45.000 treino / 5.000 validação / 10.000 teste,
  com a mesma seed (42), então o conjunto de validação é idêntico nas duas.
- **Busca gulosa por levas**, não grade exaustiva: cada leva testa hipóteses
  derivadas do que a anterior revelou, e o vencedor de uma leva vira a base da
  seguinte. Foram 7 rodadas na Fase 1 e 8 levas na Fase 2.
- Cada execução salva `model.pt`, `metadata.json` (hiperparâmetros,
  arquitetura, métricas, acurácia por classe) e `history.csv` em
  `results/{model_id}/`, seguindo a skill **model-saver** — 50 execuções na
  Fase 1 e 56 na Fase 2.
- As duas terminam com um **ensemble por soft voting** dos modelos já
  treinados, sem retreinar nada.

Cada fase é organizada como um mini-pacote Python (`src/<pacote>/`) consumido
por um notebook em `notebooks/`, em vez de um notebook monolítico.

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

Todos esses parâmetros foram cobertos, e os relatórios completos com a
evolução leva a leva, a configuração vencedora de cada fase, as métricas por
classe e a discussão de cada hiperparâmetro estão nos READMEs das respectivas
pastas.

## Onde está o quê, para a apresentação

| O que | Onde |
|---|---|
| Relatório completo da Fase 1 (MLP) | [`fase1-mlp/README.md`](fase1-mlp/README.md) |
| Relatório completo da Fase 2 (CNN) | [`fase2-cnn/README.md`](fase2-cnn/README.md) |
| Comparação MLP × CNN, com gráficos | [`comparativo/README.md`](comparativo/README.md) |
| Notebook de treino do MLP | `fase1-mlp/notebooks/01_train_mlp_cifar10.ipynb` |
| Notebook de relatório do MLP (só leitura) | `fase1-mlp/notebooks/02_results_report.ipynb` |
| Notebook de treino da CNN | `fase2-cnn/notebooks/01_train_cnn_cifar10.ipynb` |
| Notebook comparativo (só leitura, roda em segundos) | `comparativo/notebooks/01_comparativo_mlp_vs_cnn.ipynb` |
| Gráficos prontos para os slides | `fase1-mlp/results/plots/report/`, `fase2-cnn/results/plots/report/`, `comparativo/figures/` |

## Achados que valem destaque na apresentação

**A leva 3 da CNN rendeu exatamente zero.** Nenhuma das configurações que
esticavam o vencedor anterior o superou. Esse fracasso foi o que motivou
procurar um eixo diferente (weight decay, depois arquitetura VGG), e ilustra
bem o padrão que se repetiu nas duas fases: quando a busca satura, o ganho vem
de **trocar a categoria de alavanca**, não de ajustar melhor a mesma.

**O experimento hierárquico não funcionou — e o motivo é instrutivo.** Testamos
quebrar o problema em um "porteiro" veículo vs. animal (98,75% de acerto) mais
dois especialistas. A composição deu 0.9360, abaixo dos 0.9490 do modelo
achatado. O diagnóstico com porteiro perfeito (oráculo) chega a apenas 0.9466 —
ou seja, o gargalo não é o porteiro, são os especialistas, que viram menos
dados. As imagens de veículo ensinam features de baixo nível que continuam
úteis para classificar animais, e separar as tarefas joga isso fora.

**Três alavancas inverteram de sinal entre as fases:** weight decay (melhor em
zero no MLP, em 5e-4 na CNN), otimizador (Adam venceu no MLP, SGD+momentum
venceu por ~3 p.p. na CNN) e composição do ensemble (no MLP, poucos membros
diversos; na CNN, todos os 7). Já augmentation geométrica, color jitter e
normalização "real" se comportaram igual nas duas.

**O par `cat`/`dog` é o gargalo das duas fases.** No ensemble final da CNN, de
446 erros, `cat` (107) e `dog` (85) respondem por 43% — mais que o dobro da sua
participação proporcional. É o mesmo par que dominava os erros do MLP.
