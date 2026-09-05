"""Carregamento do CIFAR-10 e construção dos DataLoaders (treino/val/teste)."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

# Normalização padrão (mesma do notebook original: centraliza em [-1, 1]).
# Alternativa mais "correta" estatisticamente seria usar média/desvio-padrão
# reais do CIFAR-10 (~ (0.4914, 0.4822, 0.4465) / (0.2470, 0.2435, 0.2616)),
# vale a pena testar como um dos experimentos de hiperparâmetros.
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)

# Média/desvio-padrão reais do CIFAR-10 (canal a canal), alternativa mais
# "correta" estatisticamente à normalização simples para [-1, 1] acima.
REAL_MEAN = (0.4914, 0.4822, 0.4465)
REAL_STD = (0.2470, 0.2435, 0.2616)


def get_normalization_stats(normalization: str = "default") -> tuple[tuple[float, ...], tuple[float, ...]]:
    if normalization == "default":
        return DEFAULT_MEAN, DEFAULT_STD
    if normalization == "real":
        return REAL_MEAN, REAL_STD
    raise ValueError(f"normalization '{normalization}' desconhecida. Opções: default, real")

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transform(mean=DEFAULT_MEAN, std=DEFAULT_STD) -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )


def build_augmented_transform(mean=DEFAULT_MEAN, std=DEFAULT_STD, strength: str = "light") -> transforms.Compose:
    """Transform de treino com data augmentation (flip horizontal + crop).

    Só deve ser usada no conjunto de treino — validação/teste usam sempre
    `build_transform` (sem augmentation), para que a métrica reportada meça o
    modelo em imagens "normais", não aumentadas.

    `strength="strong"` adiciona `ColorJitter` (brightness/contrast/saturation
    leves) sobre o crop+flip de `strength="light"` (padrão).
    """
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if strength == "strong":
        ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    elif strength != "light":
        raise ValueError(f"strength '{strength}' desconhecida. Opções: light, strong")
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def get_datasets(data_dir: str | Path = "./data", transform: transforms.Compose | None = None):
    """Baixa (se necessário) e retorna os datasets de treino e teste do CIFAR-10."""
    transform = transform or build_transform()
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def get_dataloaders(
    data_dir: str | Path = "./data",
    batch_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    augment: bool = False,
    normalization: str = "default",
    augment_strength: str = "light",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Retorna (train_loader, val_loader, test_loader).

    O conjunto de validação é separado do conjunto de treino original (o
    CIFAR-10 só tem treino/teste oficialmente) para permitir acompanhar o
    desempenho durante o treino sem "vazar" informação do teste.

    `augment=True` aplica `build_augmented_transform` (flip horizontal + crop
    aleatório) só no subconjunto de treino; validação e teste sempre usam a
    transform padrão sem augmentation. Para isso o dataset de treino é
    instanciado duas vezes (uma com cada transform) sobre os mesmos arquivos
    em disco (torchvision não baixa de novo) e a mesma seed de split é usada
    nas duas, garantindo que os índices de treino/validação continuem
    idênticos aos de `augment=False`.

    `num_workers=0` (padrão) evita o custo de multiprocessing do PyTorch, que
    no Windows exige rodar dentro de um bloco `if __name__ == "__main__":` em
    scripts (não é um problema em notebooks/Colab). Se estiver em
    Linux/macOS/Colab e quiser acelerar o carregamento de dados, pode chamar
    com `num_workers=2` (ou mais) manualmente.
    """
    mean, std = get_normalization_stats(normalization)
    plain_transform = build_transform(mean, std)
    train_transform = build_augmented_transform(mean, std, strength=augment_strength) if augment else plain_transform

    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    val_dataset = (
        torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=plain_transform)
        if augment
        else train_dataset
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=plain_transform
    )

    n_val = int(len(train_dataset) * val_fraction)
    n_train = len(train_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(n_train + n_val), [n_train, n_val], generator=generator)

    train_subset = Subset(train_dataset, train_indices.indices)
    val_subset = Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
