"""Configuração de experimentos para a LSTM de previsão do preço do Bitcoin.

Mesma ideia dos `ExperimentConfig` do Mini-projeto 1: centralizar todos os
hiperparâmetros num único objeto serializável, para registrar, comparar e
reproduzir experimentos. O que muda aqui é a natureza do problema — série
temporal em vez de imagens — então aparecem campos novos (janela de entrada,
horizonte, features, modo de normalização) e somem os de convolução.

Um detalhe importante que este schema deixa explícito: a `task` pode ser
`regression` (prever o preço, que é o que o enunciado pede) ou `direction`
(prever se o preço sobe ou desce no dia seguinte). As duas usam exatamente o
mesmo pipeline de dados e a mesma arquitetura — só mudam a última camada e a
função de perda. Isso permite responder às duas perguntas com uma base de
código só.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Todos os hiperparâmetros de um experimento de treino da LSTM."""

    # Identificação
    run_name: str = "baseline"

    # ----- Dados -------------------------------------------------------
    dataset: str = "tutorial"  # tutorial (btc.csv do enunciado) | kaggle (1-min reamostrado em diário)
    features: tuple[str, ...] = ("Close",)  # colunas usadas como entrada
    target: str = "Close"  # coluna prevista (regressão) / base do sinal (direção)

    # Janela deslizante: quantos dias passados o modelo vê para prever o próximo.
    # É o hiperparâmetro mais característico de série temporal e não tem
    # equivalente no Mini-projeto 1.
    window: int = 60
    horizon: int = 1  # quantos dias à frente prever

    # Divisão temporal — NUNCA aleatória. Em série temporal, embaralhar deixaria
    # o modelo treinar com dados do futuro e prever o passado (data leakage).
    test_fraction: float = 0.2  # enunciado: últimos 20% para teste
    val_fraction: float = 0.1  # fatia final do treino, usada para early stopping

    # Normalização: ajustada SÓ no treino e aplicada aos demais conjuntos.
    scaler: str = "minmax"  # minmax | standard | none
    log_price: bool = False  # aplicar log no preço antes de escalar (estabiliza a variância)
    diff_target: bool = False  # prever a variação em vez do nível (remove a tendência)

    # ----- Arquitetura -------------------------------------------------
    model_type: str = "lstm"  # lstm | gru | rnn (para comparar as três famílias)
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False  # cuidado: só faz sentido se não houver vazamento do futuro
    dropout: float = 0.0  # aplicado entre camadas recorrentes (precisa de num_layers > 1)
    fc_layers: tuple[int, ...] = ()  # camadas densas extras antes da saída

    # ----- Tarefa ------------------------------------------------------
    task: str = "regression"  # regression (prever preço) | direction (prever alta/baixa)

    # ----- Otimização --------------------------------------------------
    optimizer: str = "adam"  # adam | sgd | rmsprop
    loss: str = "mse"  # regressão: mse | mae | huber | direção: bce
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # só SGD
    lr_schedule: str = "none"  # none | cosine | plateau
    grad_clip: float = 0.0  # corta a norma do gradiente; redes recorrentes são propensas a explodir

    # ----- Treinamento -------------------------------------------------
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15
    seed: int = 42

    # ----- Metadados livres --------------------------------------------
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["features"] = list(self.features)
        d["fc_layers"] = list(self.fc_layers)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        if "features" in d:
            d["features"] = tuple(d["features"])
        if "fc_layers" in d:
            d["fc_layers"] = tuple(d["fc_layers"])
        return cls(**d)
