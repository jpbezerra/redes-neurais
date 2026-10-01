"""Salvamento de experimentos no padrao results/{model_id}/ (model-saver):
pesos/artefato do modelo + metadata.json (hiperparametros + metricas +
arquitetura) + history.csv (curva de treino, quando aplicavel).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def save_run(
    results_dir: str | Path,
    model_id: str,
    *,
    framework: str,
    hyperparameters: dict,
    metrics: dict,
    architecture: dict | None = None,
    dataset_info: dict | None = None,
    history: list[dict] | pd.DataFrame | None = None,
    notes: str = "",
    save_model_fn=None,
) -> Path:
    """Cria `results/{model_id}/`, grava `metadata.json`, `history.csv`
    (se houver) e chama `save_model_fn(save_dir)` para o artefato do
    modelo (cada framework salva o seu proprio formato).

    Nunca reaproveita um `model_id` ja existente — levanta erro se a pasta
    ja existir, para forcar nomes descritivos e unicos por experimento
    (mesma regra do skill model-saver)."""
    save_dir = Path(results_dir) / model_id
    if save_dir.exists():
        raise FileExistsError(
            f"results/{model_id}/ ja existe — use um model_id novo e descritivo "
            "em vez de sobrescrever um experimento anterior."
        )
    save_dir.mkdir(parents=True)

    if save_model_fn is not None:
        save_model_fn(save_dir)

    metadata = {
        "model_id": model_id,
        "framework": framework,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "architecture": architecture or {},
        "dataset": dataset_info or {},
        "notes": notes,
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if history is not None:
        hist_df = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
        hist_df.to_csv(save_dir / "history.csv", index=False)

    return save_dir


def load_all_metadata(results_dir: str | Path = "results") -> pd.DataFrame:
    """Varre results/*/metadata.json e devolve uma tabela comparativa —
    usado pelo notebook de relatorio para comparar experimentos sem
    retreinar nada."""
    records = []
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return pd.DataFrame()
    for model_dir in sorted(results_dir.iterdir()):
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                records.append(json.load(f))
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)
