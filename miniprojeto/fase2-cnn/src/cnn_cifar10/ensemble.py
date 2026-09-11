"""Ensemble de modelos já treinados (sem retreinar nada).

Motivação: a leva 7 mostrou que os seis modelos VGG-4 estágios ficam todos
entre 0.937 e 0.9417 de acurácia, mas *erram imagens diferentes* — cada
regularizador (label smoothing, TrivialAugment, RandAugment, dropout,
weight decay) empurra o modelo para um mínimo distinto. Quando os erros são
parcialmente descorrelacionados, a média das probabilidades de vários
modelos acerta mais do que qualquer um deles isolado.

O ponto forte deste módulo é o custo: os pesos já estão em
`results/{model_id}/model.pt`, então o ensemble é só um forward pass a mais
no conjunto de teste — nenhum treino novo. É o melhor retorno por GPU-hora
que sobrou na busca.

Uso típico:

    from cnn_cifar10.ensemble import ensemble_runs

    res = ensemble_runs(
        run_names=["vgg4_randaugment", "vgg4_trivial_aug", "vgg4_label_smooth"],
        results_dir="../results",
        data_dir="./data",
        device=device,
    )
    print(res["test_scores"])
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import CLASSES, get_dataloaders
from .metrics import get_per_class_accuracy, get_scores
from .model import CNN

# Campos de arquitetura que o `metadata.json` guarda e que precisamos para
# reconstruir a `CNN` exatamente como ela foi treinada.
_ARCH_KEYS = (
    "conv_channels",
    "conv_layers_per_block",
    "kernel_size",
    "stride",
    "padding",
    "pool_size",
    "global_pool",
    "fc_layers",
    "activation",
    "dropout",
    "batch_norm",
)


def load_metadata(run_dir: str | Path) -> dict:
    with open(Path(run_dir) / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def find_run_dir(run_name: str, results_dir: str | Path = "../results") -> Path:
    """Pasta da execução mais recente salva para `run_name` (mesma regra do `find_existing_run`)."""
    results_dir = Path(results_dir)
    matches = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(f"cnn_{run_name}_")
    )
    if not matches:
        raise FileNotFoundError(f"Nenhuma execução salva encontrada para run_name='{run_name}' em {results_dir}")
    return matches[-1]


def build_model_from_metadata(metadata: dict, num_classes: int | None = None) -> CNN:
    """Reconstrói a `CNN` a partir dos hiperparâmetros salvos no `metadata.json`.

    Os campos de arquitetura foram todos persistidos pelo `checkpointing.save_run`,
    então dá para recriar a rede sem precisar do `ExperimentConfig` original.
    """
    hp = metadata["hyperparameters"]
    kwargs = {k: hp[k] for k in _ARCH_KEYS if k in hp}
    kwargs["conv_channels"] = tuple(kwargs.get("conv_channels", (32, 64)))
    kwargs["fc_layers"] = tuple(kwargs.get("fc_layers", (120, 84)))
    kwargs["num_classes"] = num_classes or metadata.get("architecture", {}).get("num_classes", 10)
    return CNN(**kwargs)


def load_run_model(run_dir: str | Path, device: torch.device, num_classes: int | None = None) -> tuple[CNN, dict]:
    """Carrega (modelo com pesos treinados, metadata) de uma pasta de execução."""
    run_dir = Path(run_dir)
    metadata = load_metadata(run_dir)
    model = build_model_from_metadata(metadata, num_classes=num_classes)
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, metadata


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna (probabilidades softmax [N, C], rótulos verdadeiros [N]) para um loader sem shuffle."""
    model.eval()
    probs, targets = [], []
    for images, labels in loader:
        out = model(images.to(device))
        probs.append(F.softmax(out, dim=1).cpu())
        targets.append(labels)
    return torch.cat(probs), torch.cat(targets)


def ensemble_runs(
    run_names: list[str],
    results_dir: str | Path = "../results",
    data_dir: str | Path = "./data",
    device: torch.device | None = None,
    batch_size: int = 256,
    weights: list[float] | None = None,
    num_workers: int = 0,
    class_names: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Faz a média das probabilidades de vários modelos já treinados no conjunto de teste.

    - `run_names`: nomes de experimento (`config.run_name`) já salvos em `results/`.
    - `weights`: pesos opcionais por modelo (default: média simples). Útil para
      dar mais peso aos membros mais fortes; a média simples costuma ser um
      baseline difícil de bater.

    Cada modelo é avaliado com **a normalização com que foi treinado**: os
    runs podem ter usado `normalization` diferente, então o loader de teste é
    construído uma vez por normalização distinta (o teste nunca embaralha, então
    a ordem das amostras é a mesma em todos os loaders e as probabilidades são
    alinháveis).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = class_names or list(CLASSES)

    if weights is None:
        weights = [1.0] * len(run_names)
    if len(weights) != len(run_names):
        raise ValueError("weights precisa ter o mesmo tamanho de run_names")

    loaders_by_norm: dict[str, object] = {}
    accumulated = None
    targets_ref = None
    members = []

    for run_name, weight in zip(run_names, weights):
        run_dir = find_run_dir(run_name, results_dir)
        model, metadata = load_run_model(run_dir, device)
        normalization = metadata["hyperparameters"].get("normalization", "default")

        if normalization not in loaders_by_norm:
            _, _, test_loader = get_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                normalization=normalization,
                num_workers=num_workers,
            )
            loaders_by_norm[normalization] = test_loader

        probs, targets = predict_probs(model, loaders_by_norm[normalization], device)
        if targets_ref is None:
            targets_ref = targets
        elif not torch.equal(targets_ref, targets):
            raise RuntimeError("Ordem das amostras de teste divergiu entre loaders — ensemble inválido.")

        solo_acc = (probs.argmax(1) == targets).float().mean().item()
        members.append(
            {
                "run_name": run_name,
                "model_id": metadata["model_id"],
                "weight": weight,
                "solo_test_accuracy": solo_acc,
            }
        )
        if verbose:
            print(f"[ensemble] {run_name:<24} solo={solo_acc:.4f} (peso {weight})")

        contribution = probs * weight
        accumulated = contribution if accumulated is None else accumulated + contribution

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_probs = accumulated / sum(weights)
    preds = mean_probs.argmax(1).numpy()
    targets_np = targets_ref.numpy()

    test_scores = get_scores(targets_np, preds)
    per_class_accuracy = get_per_class_accuracy(targets_np, preds, class_names)

    best_solo = max(m["solo_test_accuracy"] for m in members)
    if verbose:
        print(
            f"[ensemble] {len(members)} modelos | melhor isolado={best_solo:.4f} | "
            f"ensemble={test_scores['accuracy']:.4f} "
            f"({test_scores['accuracy'] - best_solo:+.4f})"
        )

    return {
        "members": members,
        "probs": mean_probs,
        "targets": targets_np,
        "predictions": preds,
        "test_scores": test_scores,
        "per_class_accuracy": per_class_accuracy,
        "best_solo_accuracy": best_solo,
    }


def save_ensemble(
    result: dict,
    run_name: str = "ensemble",
    results_dir: str | Path = "../results",
    notes: str = "",
) -> Path:
    """Registra o ensemble em `results/` no mesmo formato dos treinos.

    Não há `model.pt` (o "modelo" é a lista de membros), então salvamos só o
    `metadata.json` — o suficiente para o ensemble aparecer na tabela
    comparativa junto com os treinos individuais.
    """
    results_dir = Path(results_dir)
    model_id = f"cnn_{run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = results_dir / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_id": model_id,
        "framework": "PyTorch",
        "timestamp": datetime.now().isoformat(),
        "model_type": "Ensemble",
        "purpose": "Classificação de imagens CIFAR-10 (Mini-projeto 1 - Fase 2)",
        "run_name": run_name,
        "hyperparameters": {
            "members": [m["run_name"] for m in result["members"]],
            "weights": [m["weight"] for m in result["members"]],
            "combination": "média das probabilidades (softmax)",
        },
        "metrics": {
            "epochs_trained": 0,
            **{f"test_{k}": v for k, v in result["test_scores"].items()},
            "best_solo_accuracy": result["best_solo_accuracy"],
        },
        "per_class_accuracy": result["per_class_accuracy"],
        "members": result["members"],
        "notes": notes,
        "tags": ["ensemble"],
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[model-saver] Ensemble registrado em {save_dir}/")
    return save_dir
