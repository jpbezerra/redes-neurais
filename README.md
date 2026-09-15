# Redes Neurais

Repositório da disciplina de Redes Neurais (Centro de Informática - UFPE).

## Estrutura

```
.
├── miniprojeto/              # Mini-projetos da disciplina
│   ├── mp1-cifar10/          # Mini-projeto 1 — MLP e CNN no CIFAR-10 (concluído)
│   │   ├── fase1-mlp/         # Fase 1 — MLP   (melhor resultado: 0.6135)
│   │   ├── fase2-cnn/         # Fase 2 — CNN   (melhor resultado: 0.9554)
│   │   └── comparativo/       # Análise MLP × CNN lado a lado
│   └── mp2-lstm-bitcoin/     # Mini-projeto 2 — LSTM para prever o preço do Bitcoin
└── projeto/             # Projeto final da disciplina
```

Cada pasta de projeto/fase tem seu próprio `README.md` com instruções específicas
de execução, estrutura de código e status.

## Mini-projeto 1 (CIFAR-10) — resultado

As duas fases estão concluídas. Classificando o CIFAR-10 com a mesma
metodologia de busca (levas sucessivas de experimentos, cada uma informada
pela anterior, terminando num ensemble por soft voting):

| | Melhor modelo único | Melhor resultado (ensemble) |
|---|---|---|
| Fase 1 — MLP | 0.5998 | 0.6135 |
| Fase 2 — CNN | **0.9490** | **0.9554** |

A convolução valeu **+34,2 pontos percentuais**, uma redução de 88,5% na taxa
de erro. A análise lado a lado está em
[`miniprojeto/mp1-cifar10/comparativo/`](miniprojeto/mp1-cifar10/comparativo/).

## Mini-projeto 2 (Bitcoin) — em desenvolvimento

Previsão do preço do Bitcoin com **LSTM**, de dezembro de 2014 a maio de 2018,
com 80% dos registros para treino e 20% para teste. Ver
[`miniprojeto/mp2-lstm-bitcoin/`](miniprojeto/mp2-lstm-bitcoin/).

## Convenções gerais

- **Código**: cada fase é organizada como um mini-pacote Python (`src/<pacote>/`),
  com um notebook em `notebooks/` que consome esse pacote — em vez de um único
  notebook monolítico. Isso facilita reaproveitar código entre experimentos,
  testar funções isoladamente e manter o notebook focado em orquestrar
  experimentos e reportar resultados.
- **Experimentos/checkpoints**: cada execução de treino salva seus artefatos
  (pesos, hiperparâmetros, métricas, histórico) em `results/{model_id}/`
  dentro da fase correspondente, seguindo a skill **model-saver**. Esses
  artefatos **não são versionados no git** (ver `.gitignore`) — apenas o
  código que os gera.
- **Dados**: datasets baixados automaticamente (ex.: CIFAR-10 via `torchvision`)
  ficam em `data/` dentro de cada fase e também não são versionados.

## Requisitos gerais

- Python 3.10+
- Recomendado usar ambiente com GPU (Google Colab, por exemplo) para treinar os
  modelos mais rapidamente.
