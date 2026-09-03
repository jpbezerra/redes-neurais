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
│   └── 01_train_mlp_cifar10.ipynb   # notebook principal: define e roda os experimentos
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
