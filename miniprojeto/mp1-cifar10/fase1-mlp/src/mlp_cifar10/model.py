"""Definição do MLP usado para classificar o CIFAR-10.

A arquitetura é parametrizável (número/tamanho de camadas ocultas, função de
ativação, dropout, batch norm) para permitir a busca de hiperparâmetros
pedida no enunciado sem precisar reescrever a classe a cada experimento.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


def get_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"Ativação '{name}' desconhecida. Opções: {list(_ACTIVATIONS)}"
        ) from exc


class MLP(nn.Module):
    """MLP totalmente conectado com número arbitrário de camadas ocultas.

    input_size -> hidden_layers[0] -> ... -> hidden_layers[-1] -> num_classes
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_layers: tuple[int, ...] = (64, 128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

    def architecture_summary(self, input_size: int, num_classes: int) -> dict:
        """Resumo da arquitetura no formato usado pelo `metadata.json` (skill model-saver)."""
        return {
            "layers": [type(layer).__name__ for layer in self.net],
            "input_shape": [input_size],
            "num_classes": num_classes,
        }
