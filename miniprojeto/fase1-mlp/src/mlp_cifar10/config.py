"""Configuração de experimentos para o MLP no CIFAR-10.

Centralizar os hiperparâmetros num único objeto (em vez de variáveis soltas
no notebook) facilita:
- registrar exatamente o que foi usado em cada execução (serializando o
  dataclass para JSON, ver `checkpointing.py`);
- comparar experimentos entre si;
- reaproveitar a mesma configuração em outros scripts/notebooks.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Todos os hiperparâmetros de um experimento de treino do MLP.

    Os campos cobrem os parâmetros sugeridos no enunciado do mini-projeto:
    número de camadas/neurônios, taxa de aprendizagem, função de ativação,
    regularização (weight decay / dropout), algoritmo de aprendizagem
    (otimizador) e função de erro.
    """

    # Identificação do experimento
    run_name: str = "baseline"

    # Arquitetura
    input_size: int = 32 * 32 * 3  # imagens RGB 32x32
    num_classes: int = 10
    hidden_layers: tuple[int, ...] = (64, 128, 64)
    activation: str = "relu"  # relu | tanh | sigmoid | leaky_relu | gelu
    dropout: float = 0.0
    batch_norm: bool = False

    # Otimização
    optimizer: str = "adam"  # adam | sgd | rmsprop
    loss: str = "cross_entropy"  # cross_entropy | mse
    learning_rate: float = 1e-3
    weight_decay: float = 0.0  # regularização L2
    momentum: float = 0.9  # usado apenas pelo SGD
    lr_schedule: str = "none"  # none | cosine (ver train.build_scheduler)

    # Treinamento
    batch_size: int = 64
    num_epochs: int = 100
    patience: int = 5  # early stopping (nº de épocas sem melhora)
    val_fraction: float = 0.1  # fração do treino usada como validação
    augment: bool = False  # data augmentation (flip + crop) no treino, ver data.get_dataloaders

    # Reprodutibilidade
    seed: int = 42

    # Metadados livres (ex.: motivação do experimento, o que mudou desde o
    # experimento anterior) — útil para o relatório/PPT.
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["hidden_layers"] = list(self.hidden_layers)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        if "hidden_layers" in d:
            d["hidden_layers"] = tuple(d["hidden_layers"])
        return cls(**d)
