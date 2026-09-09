"""Configuração de experimentos para a CNN no CIFAR-10.

Mesma ideia da Fase 1 (`mlp_cifar10.config.ExperimentConfig`): centralizar
todos os hiperparâmetros num único objeto serializável para registrar,
comparar e reproduzir experimentos. Os campos de arquitetura mudam (blocos
convolucionais em vez de camadas densas), mas otimização/treino/augmentation
seguem o mesmo schema da Fase 1 para manter os dois relatórios comparáveis.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Todos os hiperparâmetros de um experimento de treino da CNN.

    Cobre os parâmetros pedidos no enunciado da Fase 2: tamanho da rede
    (`conv_channels`/`fc_layers`), tamanho do filtro (`kernel_size`), stride,
    padding, dropout, janela de pooling (`pool_size`) e taxa de aprendizagem
    — além dos já herdados da Fase 1 (ativação, otimizador, regularização,
    função de erro, augmentation).
    """

    # Identificação do experimento
    run_name: str = "baseline"

    # Arquitetura — blocos convolucionais
    input_channels: int = 3  # imagens RGB
    input_size: int = 32  # imagens 32x32
    num_classes: int = 10
    conv_channels: tuple[int, ...] = (32, 64)  # nº de filtros de saída por bloco conv
    conv_layers_per_block: int = 1  # convs empilhadas por estágio de pooling (1 = original; 2 = estilo VGG)
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2  # MaxPool2d(kernel_size=pool_size, stride=pool_size) após cada bloco conv
    global_pool: bool = False  # AdaptiveAvgPool2d(1) antes da cabeça densa (corta params da cabeça, regulariza)

    # Arquitetura — cabeça densa (após achatar os mapas de features)
    fc_layers: tuple[int, ...] = (120, 84)
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
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 5  # early stopping (nº de épocas sem melhora)
    val_fraction: float = 0.1  # fração do treino usada como validação
    augment: bool = False  # data augmentation (flip + crop) no treino, ver data.get_dataloaders
    augment_strength: str = "light"  # light (crop+flip) | strong (+ color jitter), só usado se augment=True
    normalization: str = "default"  # default ([-1,1] simples) | real (média/desvio-padrão reais do CIFAR-10)

    # Reprodutibilidade
    seed: int = 42

    # Metadados livres (ex.: motivação do experimento, o que mudou desde o
    # experimento anterior) — útil para o relatório/PPT.
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["conv_channels"] = list(self.conv_channels)
        d["fc_layers"] = list(self.fc_layers)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        if "conv_channels" in d:
            d["conv_channels"] = tuple(d["conv_channels"])
        if "fc_layers" in d:
            d["fc_layers"] = tuple(d["fc_layers"])
        return cls(**d)
