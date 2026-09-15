# Mini-projetos — Redes Neurais (CIn/UFPE)

Pasta dos mini-projetos da disciplina, cada um com seu próprio pacote,
notebooks e relatório.

| Mini-projeto | Pasta | Tema | Status |
|---|---|---|---|
| 1 | [`mp1-cifar10/`](mp1-cifar10/README.md) | Classificação de imagens com **MLP** e **CNN** (CIFAR-10) | Concluído |
| 2 | [`mp2-lstm-bitcoin/`](mp2-lstm-bitcoin/README.md) | Previsão do preço do **Bitcoin** com **LSTM** | Em desenvolvimento |

## Mini-projeto 1 — CIFAR-10 (concluído)

Duas fases sobre o mesmo dataset, com a mesma metodologia de busca, para
isolar o efeito da arquitetura:

| | Melhor modelo único | Melhor resultado (ensemble) |
|---|---|---|
| Fase 1 — MLP | 0.5998 | 0.6135 |
| Fase 2 — CNN | **0.9490** | **0.9554** |

A convolução valeu **+34,2 pontos percentuais**, uma redução de 88,5% na taxa
de erro. A análise lado a lado está em
[`mp1-cifar10/comparativo/`](mp1-cifar10/comparativo/README.md).

## Mini-projeto 2 — Bitcoin (em desenvolvimento)

Prever o preço do Bitcoin de dezembro de 2014 a maio de 2018 com LSTM, usando
os primeiros 80% dos registros para treinar e os 20% finais para teste.

Base principal: o `btc.csv` do repositório creditado no enunciado (1.273 dias,
sem lacunas). Base secundária: a série de 1 minuto do Bitstamp no Kaggle,
reamostrada em diário, para testar generalização fora do período do enunciado.

O diagnóstico central já está documentado: as faixas de preço de treino
(US$ 120–2.698) e teste (US$ 3.617–19.650) **não se sobrepõem**, o que
inviabiliza prever o nível do preço e obriga a prever a variação. Detalhes no
[README do projeto](mp2-lstm-bitcoin/README.md).

---

## Convenções compartilhadas

Os dois mini-projetos seguem o mesmo padrão, o que facilita reaproveitar
código e comparar relatórios.

**Organização.** Cada projeto é um mini-pacote Python instalável
(`src/<pacote>/`) consumido por notebooks em `notebooks/`, em vez de um
notebook monolítico. Isso mantém o notebook curto e legível — o que importa
para quem avalia — enquanto o código testável fica isolado em módulos.

**Configuração.** Todos os hiperparâmetros de um experimento vivem num único
dataclass `ExperimentConfig`, serializável. É o que permite registrar,
comparar e reproduzir qualquer execução.

**Checkpointing (skill `model-saver`).** Cada treino cria
`results/{model_id}/` com `model.pt`, `metadata.json` (hiperparâmetros,
arquitetura, métricas) e `history.csv`. O `model_id` inclui timestamp e nunca
é reutilizado.

**Layout de pastas.** Todo projeto tem `src/` (código), `notebooks/`
(orquestração), `scripts/` (geradores), `data/` (datasets, não versionados),
`results/` (uma pasta por execução) e **`reports/figures/`** (os gráficos que
entram nos READMEs e nos slides).

**Idempotência.** `fit_or_load(...)` reaproveita um resultado já salvo em vez
de retreinar. Reexecutar o notebook do zero numa sessão nova do Kaggle não
custa horas de GPU.

**Busca por levas.** Cada leva testa hipóteses derivadas do que a anterior
revelou, e o vencedor vira a base da seguinte — busca gulosa, não grade
exaustiva. No Mini-projeto 2 isso é complementado por busca automática com
Optuna.

**Honestidade metodológica.** A seleção de modelo usa sempre o conjunto de
validação; o teste é tocado uma única vez, no fim. No Mini-projeto 2 há o
cuidado adicional da divisão temporal, já que embaralhar uma série faria o
modelo treinar com o futuro.

## Requisitos

Python 3.10+. O Mini-projeto 1 pede GPU (treinos de 2-4 h por configuração);
o Mini-projeto 2 roda confortavelmente em CPU, já que a série tem 1.273
pontos — GPU só compensa para o Optuna com muitos trials.
