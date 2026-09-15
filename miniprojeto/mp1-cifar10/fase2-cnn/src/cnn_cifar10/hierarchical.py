"""Classificação hierárquica: um "porteiro" + dois especialistas.

Ideia do experimento
--------------------
A acurácia por classe dos melhores modelos mostra um padrão muito claro: os
quatro veículos do CIFAR-10 são fáceis (0.95-0.98) e os seis animais são o
gargalo (gato 0.84-0.88, cachorro 0.88-0.91, pássaro 0.91-0.93). Quase todo
o erro restante está *dentro* do grupo dos animais.

Isso motiva quebrar o problema em três redes:

1. **gate** (porteiro): 2 classes — veículo vs. animal. Vê todas as imagens.
2. **vehicle**: 4 classes — treinada só nas imagens de veículo.
3. **animal**: 6 classes — treinada só nas imagens de animal.

Na hora de prever, as probabilidades se compõem pela regra da cadeia:

    P(classe) = P(super-classe) · P(classe | super-classe)

Duas observações honestas antes de rodar:

- O CIFAR-10 tem **6 animais e 4 veículos** (não 8 + 2). O corte natural das
  10 classes é esse.
- A separação veículo/animal é quase trivial (espera-se ~99% no porteiro),
  então o ganho, se existir, virá do especialista em animais poder gastar
  toda a sua capacidade distinguindo gato/cachorro/pássaro em vez de também
  precisar separá-los de caminhões. O risco é o oposto: cada especialista vê
  menos dados (27k imagens de animal, 18k de veículo, contra 45k), e todo
  erro do porteiro é irrecuperável (se ele disser "veículo" para um gato,
  nenhum especialista conserta). O experimento serve justamente para medir
  qual dos dois efeitos vence.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split

from .data import CLASSES, build_augmented_transform, build_transform, get_normalization_stats
from .metrics import get_per_class_accuracy, get_scores

VEHICLES: tuple[str, ...] = ("airplane", "automobile", "ship", "truck")
ANIMALS: tuple[str, ...] = ("bird", "cat", "deer", "dog", "frog", "horse")
SUPER_CLASSES: tuple[str, ...] = ("vehicle", "animal")

# Índices globais (0-9, ordem de `data.CLASSES`) de cada grupo.
VEHICLE_IDS: tuple[int, ...] = tuple(CLASSES.index(c) for c in VEHICLES)
ANIMAL_IDS: tuple[int, ...] = tuple(CLASSES.index(c) for c in ANIMALS)
GROUP_IDS: tuple[tuple[int, ...], ...] = (VEHICLE_IDS, ANIMAL_IDS)

# global -> 0 (veículo) | 1 (animal)
SUPER_OF: dict[int, int] = {**{i: 0 for i in VEHICLE_IDS}, **{i: 1 for i in ANIMAL_IDS}}
# global -> índice local dentro do próprio grupo
LOCAL_OF: dict[int, int] = {
    **{g: k for k, g in enumerate(VEHICLE_IDS)},
    **{g: k for k, g in enumerate(ANIMAL_IDS)},
}


class RelabeledSubset(Dataset):
    """Subconjunto de um dataset com os rótulos remapeados.

    Usado para as três visões dos mesmos arquivos do CIFAR-10: rótulos
    binários (porteiro) sobre todas as imagens, e rótulos locais 0..3 / 0..5
    sobre apenas as imagens de cada grupo.
    """

    def __init__(self, dataset: Dataset, indices: list[int], label_map: dict[int, int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label = self.dataset[self.indices[i]]
        return image, self.label_map[int(label)]


def _split_indices(n_total: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Mesmo split treino/validação de `data.get_dataloaders` (mesma seed => mesmos índices)."""
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_train + n_val), [n_train, n_val], generator=generator)
    return list(train_idx.indices), list(val_idx.indices)


def get_hierarchical_dataloaders(
    data_dir: str | Path = "./data",
    batch_size: int = 32,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    augment: bool = True,
    augment_strength: str = "light",
    normalization: str = "default",
) -> dict[str, dict[str, DataLoader]]:
    """Monta os loaders das três tarefas: `gate`, `vehicle` e `animal`.

    Reaproveita exatamente o mesmo split treino/validação de
    `data.get_dataloaders` (mesma seed), então a validação continua sendo as
    mesmas 5000 imagens de sempre e as comparações com os modelos "achatados"
    (10 classes) são justas. O conjunto de teste é o oficial do CIFAR-10,
    intocado.
    """
    mean, std = get_normalization_stats(normalization)
    plain = build_transform(mean, std)
    train_tf = build_augmented_transform(mean, std, strength=augment_strength) if augment else plain

    train_base = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tf)
    val_base = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=plain)
    test_base = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=plain)

    train_idx, val_idx = _split_indices(len(train_base), val_fraction, seed)
    test_idx = list(range(len(test_base)))

    train_targets = list(train_base.targets)
    test_targets = list(test_base.targets)

    def keep(indices: list[int], targets: list[int], group: tuple[int, ...]) -> list[int]:
        group_set = set(group)
        return [i for i in indices if targets[i] in group_set]

    identity_super = SUPER_OF
    loaders: dict[str, dict[str, DataLoader]] = {}

    def make(dataset, indices, label_map, shuffle):
        return DataLoader(
            RelabeledSubset(dataset, indices, label_map),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    # Porteiro: todas as imagens, rótulo binário.
    loaders["gate"] = {
        "train": make(train_base, train_idx, identity_super, True),
        "val": make(val_base, val_idx, identity_super, False),
        "test": make(test_base, test_idx, identity_super, False),
    }

    # Especialistas: só as imagens do próprio grupo, rótulo local.
    for name, group in (("vehicle", VEHICLE_IDS), ("animal", ANIMAL_IDS)):
        loaders[name] = {
            "train": make(train_base, keep(train_idx, train_targets, group), LOCAL_OF, True),
            "val": make(val_base, keep(val_idx, train_targets, group), LOCAL_OF, False),
            "test": make(test_base, keep(test_idx, test_targets, group), LOCAL_OF, False),
        }

    return loaders


def flat_test_loader(
    data_dir: str | Path = "./data",
    batch_size: int = 256,
    normalization: str = "default",
    num_workers: int = 0,
) -> DataLoader:
    """Loader de teste com os rótulos originais de 0-9 (para avaliar a composição)."""
    mean, std = get_normalization_stats(normalization)
    test_base = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=build_transform(mean, std)
    )
    return DataLoader(test_base, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def hierarchical_predict(
    gate_model: torch.nn.Module,
    vehicle_model: torch.nn.Module,
    animal_model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compõe as três redes numa predição de 10 classes.

    Para cada imagem calcula P(super) pelo porteiro e P(classe | super) por
    cada especialista, e monta o vetor de 10 probabilidades pela regra da
    cadeia. Assim a decisão final continua sendo uma distribuição sobre as 10
    classes originais, comparável diretamente com os modelos achatados.
    """
    for m in (gate_model, vehicle_model, animal_model):
        m.eval().to(device)

    all_probs, all_targets, all_super_preds, all_super_targets = [], [], [], []

    for images, labels in test_loader:
        images = images.to(device)
        p_super = F.softmax(gate_model(images), dim=1)  # [B, 2]
        p_vehicle = F.softmax(vehicle_model(images), dim=1)  # [B, 4]
        p_animal = F.softmax(animal_model(images), dim=1)  # [B, 6]

        probs = torch.zeros(images.size(0), len(CLASSES), device=device)
        for local, global_id in enumerate(VEHICLE_IDS):
            probs[:, global_id] = p_super[:, 0] * p_vehicle[:, local]
        for local, global_id in enumerate(ANIMAL_IDS):
            probs[:, global_id] = p_super[:, 1] * p_animal[:, local]

        all_probs.append(probs.cpu())
        all_targets.append(labels)
        all_super_preds.append(p_super.argmax(1).cpu())
        all_super_targets.append(torch.tensor([SUPER_OF[int(y)] for y in labels]))

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets).numpy()
    preds = probs.argmax(1).numpy()
    super_preds = torch.cat(all_super_preds).numpy()
    super_targets = torch.cat(all_super_targets).numpy()

    return {
        "probs": probs,
        "predictions": preds,
        "targets": targets,
        "test_scores": get_scores(targets, preds),
        "per_class_accuracy": get_per_class_accuracy(targets, preds, list(CLASSES)),
        "gate_accuracy": float((super_preds == super_targets).mean()),
        "gate_scores": get_scores(super_targets, super_preds),
    }


def oracle_gate_accuracy(result: dict) -> float:
    """Acurácia que a composição teria se o porteiro fosse perfeito.

    Diagnóstico do experimento: separa quanto do erro final vem dos
    especialistas e quanto vem do porteiro ter mandado a imagem para o
    especialista errado. Recalcula o argmax restrito ao grupo verdadeiro de
    cada imagem.
    """
    probs, targets = result["probs"], result["targets"]
    correct = 0
    for i, y in enumerate(targets):
        group = GROUP_IDS[SUPER_OF[int(y)]]
        best = max(group, key=lambda g: probs[i, g].item())
        correct += int(best == y)
    return correct / len(targets)
