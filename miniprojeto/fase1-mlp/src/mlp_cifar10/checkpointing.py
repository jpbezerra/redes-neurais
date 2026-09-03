"""Persistência dos artefatos de cada execução de treino.

Implementa a skill **model-saver** (variante PyTorch) para este projeto:
cada execução de treino gera uma pasta própria e rastreável em
`results/{model_id}/`, contendo o modelo e um `metadata.json` com
hiperparâmetros, arquitetura e métricas — nunca reaproveitando um
`model_id` entre execuções diferentes.

Estrutura gerada:

    results/
    └── {model_id}/
        ├── metadata.json
        ├── model.pt
        ├── history.csv        (extra: histórico de treino por época)
        └── plots/              (opcional: curvas de treino, etc.)
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from .config import ExperimentConfig
from .model import MLP


def make_model_id(run_name: str, timestamp: datetime | None = None) -> str:
    """Gera um `model_id` único a partir do nome do experimento + timestamp.

    Segue a convenção `{arquitetura}_{propósito}_{tag}` da skill model-saver,
    usando `run_name` como "propósito/tag" e prefixando com a arquitetura
    (mlp) — o timestamp garante que o `model_id` nunca seja reutilizado
    entre execuções, mesmo que `run_name` se repita.
    """
    ts = (timestamp or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"mlp_{run_name}_{ts}"


def save_model_pytorch(
    model: torch.nn.Module,
    model_id: str,
    hyperparams: dict,
    metrics: dict,
    results_dir: str | Path = "../results",
    extra_info: dict | None = None,
    history: list[dict] | None = None,
) -> Path:
    """Salva modelo + `metadata.json` em `results/{model_id}/` (skill model-saver, PyTorch).

    `extra_info` é mesclado no `metadata.json` (ex.: model_type, purpose,
    dataset, architecture, notes). `history`, quando fornecido, é salvo à
    parte em `history.csv` (não faz parte do schema padrão da skill, mas é
    útil para plotar curvas de treino depois).
    """
    save_dir = Path(results_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Salva o modelo
    torch.save(model.state_dict(), save_dir / "model.pt")

    # Monta e salva metadata.json
    metadata = {
        "model_id": model_id,
        "framework": "PyTorch",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": hyperparams,
        "metrics": metrics,
        **(extra_info or {}),
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Extra: histórico de treino por época (fora do schema padrão da skill)
    if history:
        with open(save_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print(f"[model-saver] Modelo salvo em {save_dir}/")
    return save_dir


def save_run(
    config: ExperimentConfig,
    model: MLP,
    history: list[dict],
    test_scores: dict | None = None,
    per_class_accuracy: dict | None = None,
    dataset_sizes: dict | None = None,
    results_dir: str | Path = "../results",
) -> Path:
    """Ponte entre o pipeline de treino (`train.py`) e a skill model-saver.

    Monta `hyperparameters`, `metrics` e `extra_info` a partir de
    `ExperimentConfig` + resultados do treino e delega para
    `save_model_pytorch`, que é a implementação canônica da skill.
    """
    model_id = make_model_id(config.run_name)

    hyperparams = {
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.num_epochs,
        "optimizer": config.optimizer,
        "loss": config.loss,
        "hidden_layers": list(config.hidden_layers),
        "activation": config.activation,
        "dropout": config.dropout,
        "batch_norm": config.batch_norm,
        "weight_decay": config.weight_decay,
        "momentum": config.momentum,
        "patience": config.patience,
        "seed": config.seed,
    }

    final_train_loss = history[-1]["train_loss"] if history else None
    final_val_loss = history[-1]["val_loss"] if history else None
    metrics = {
        "epochs_trained": len(history),
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        **{f"test_{k}": v for k, v in (test_scores or {}).items()},
    }

    extra_info = {
        "model_type": "MLP",
        "purpose": "Classificação de imagens CIFAR-10 (Mini-projeto 1 - Fase 1)",
        "dataset": {"name": "CIFAR-10", **(dataset_sizes or {})},
        "architecture": model.architecture_summary(config.input_size, config.num_classes),
        "per_class_accuracy": per_class_accuracy or {},
        "notes": config.notes,
        "tags": config.tags,
        "run_name": config.run_name,
    }

    return save_model_pytorch(
        model=model,
        model_id=model_id,
        hyperparams=hyperparams,
        metrics=metrics,
        results_dir=results_dir,
        extra_info=extra_info,
        history=history,
    )


def load_all_metadata(results_dir: str | Path = "../results"):
    """Carrega o `metadata.json` de todas as execuções para comparar runs.

    Retorna um `pandas.DataFrame` (uma linha por execução, colunas
    achatadas como `metrics.test_accuracy`, `hyperparameters.learning_rate`, ...).
    """
    import pandas as pd

    records = []
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return pd.DataFrame()

    for model_id in sorted(os.listdir(results_dir)):
        meta_path = results_dir / model_id / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                records.append(json.load(f))
    return pd.json_normalize(records)


def load_run(model_id: str, results_dir: str | Path = "../results") -> dict:
    """Recarrega o `metadata.json` de uma execução específica (não recarrega os pesos)."""
    run_dir = Path(results_dir) / model_id
    with open(run_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return {"metadata": metadata, "run_dir": run_dir}
