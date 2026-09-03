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

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transform(mean=DEFAULT_MEAN, std=DEFAULT_STD) -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )


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
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Retorna (train_loader, val_loader, test_loader).

    O conjunto de validação é separado do conjunto de treino original (o
    CIFAR-10 só tem treino/teste oficialmente) para permitir acompanhar o
    desempenho durante o treino sem "vazar" informação do teste.
    """
    train_dataset, test_dataset = get_datasets(data_dir)

    n_val = int(len(train_dataset) * val_fraction)
    n_train = len(train_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
