"""Definição da CNN usada para classificar o CIFAR-10.

Arquitetura parametrizável (nº de blocos convolucionais, canais, kernel,
stride, padding, pooling, cabeça densa) para permitir a busca de
hiperparâmetros pedida no enunciado da Fase 2 sem reescrever a classe a cada
experimento — mesmo espírito do `mlp_cifar10.model.MLP` da Fase 1.

Cada bloco convolucional é: Conv2d -> [BatchNorm2d] -> ativação ->
MaxPool2d(pool_size) -> [Dropout2d]. O tamanho do vetor achatado antes da
cabeça densa depende de kernel/stride/padding/pool escolhidos, então é
inferido automaticamente com um forward "a seco" no `__init__` em vez de
calculado na mão (evita erro de aritmética a cada combinação nova testada).
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


class CNN(nn.Module):
    """CNN com nº arbitrário de blocos convolucionais + cabeça densa.

    input -> conv_channels[0] -> ... -> conv_channels[-1] -> flatten ->
    fc_layers[0] -> ... -> fc_layers[-1] -> num_classes
    """

    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 32,
        num_classes: int = 10,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
        fc_layers: tuple[int, ...] = (120, 84),
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        conv_blocks: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in conv_channels:
            conv_blocks.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            if batch_norm:
                conv_blocks.append(nn.BatchNorm2d(out_channels))
            conv_blocks.append(get_activation(activation))
            conv_blocks.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
            if dropout > 0:
                conv_blocks.append(nn.Dropout2d(dropout))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_blocks)

        # Infere o tamanho achatado com um forward "a seco" (evita calcular
        # na mão a fórmula de saída de conv/pool para cada combinação de
        # kernel/stride/padding/pool_size testada).
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            flat_size = self.conv(dummy).numel()

        fc_blocks: list[nn.Module] = []
        in_features = flat_size
        for out_features in fc_layers:
            fc_blocks.append(nn.Linear(in_features, out_features))
            fc_blocks.append(get_activation(activation))
            if dropout > 0:
                fc_blocks.append(nn.Dropout(dropout))
            in_features = out_features
        fc_blocks.append(nn.Linear(in_features, num_classes))
        self.fc = nn.Sequential(*fc_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def architecture_summary(self, input_size: int, num_classes: int) -> dict:
        """Resumo da arquitetura no formato usado pelo `metadata.json` (skill model-saver)."""
        return {
            "conv_layers": [type(layer).__name__ for layer in self.conv],
            "fc_layers": [type(layer).__name__ for layer in self.fc],
            "input_shape": [3, input_size, input_size],
            "num_classes": num_classes,
        }
