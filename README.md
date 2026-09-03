# Redes Neurais

Repositório da disciplina de Redes Neurais (Centro de Informática - UFPE).

## Estrutura

```
.
├── miniprojeto/        # Mini-projeto 1: Classificação de imagens com MLPs e CNNs (CIFAR-10)
│   ├── fase1-mlp/       # Fase 1 — MLP
│   └── fase2-cnn/       # Fase 2 — CNN
└── projeto/             # Projeto final da disciplina
```

Cada pasta de projeto/fase tem seu próprio `README.md` com instruções específicas
de execução, estrutura de código e status.

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
