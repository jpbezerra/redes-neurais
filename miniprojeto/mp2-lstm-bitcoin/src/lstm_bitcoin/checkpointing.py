"""Persistência dos artefatos de cada execução (skill model-saver, variante PyTorch).

Mesmo schema do Mini-projeto 1 — `results/{model_id}/` com `model.pt`,
`metadata.json` e `history.csv` — com duas adições específicas deste projeto:

- `metadata.json` guarda a **matriz de confusão** (contagem e percentual) nos
  experimentos de direção, para o relatório não depender de recalcular nada;
- há um helper `hyperparameter_table` que monta a tabela de hiperparâmetros
  variados entre execuções, mostrando só as colunas que de fato mudaram —
  é o que entra no relatório final.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .metrics import confusion_matrix_percent


def make_model_id(run_name: str, timestamp: datetime | None = None) -> str:
    ts = (timestamp or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"lstm_{run_name}_{ts}"


def save_run(
    config: ExperimentConfig,
    model: torch.nn.Module,
    history: list[dict],
    test_result: dict,
    val_scores: dict | None = None,
    split_sizes: dict | None = None,
    results_dir: str | Path = "../results",
) -> Path:
    """Salva modelo + metadata.json + history.csv em `results/{model_id}/`."""
    model_id = make_model_id(config.run_name)
    save_dir = Path(results_dir) / model_id
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "model.pt")

    metrics = {
        "epochs_trained": len(history),
        "train_loss": history[-1]["train_loss"] if history else None,
        "val_loss": history[-1]["val_loss"] if history else None,
        **{f"test_{k}": v for k, v in test_result["scores"].items()},
        **{f"val_{k}": v for k, v in (val_scores or {}).items()},
    }

    extra: dict = {
        "model_type": config.model_type.upper(),
        "purpose": "Previsão do preço do Bitcoin (Mini-projeto 2 — LSTM)",
        "task": config.task,
        "dataset": {"name": config.dataset, **(split_sizes or {})},
        "architecture": model.architecture_summary(),
        "notes": config.notes,
        "tags": config.tags,
        "run_name": config.run_name,
    }

    # Matriz de confusão, nas duas leituras pedidas para o relatório.
    if "confusion_matrix" in test_result:
        cm = np.asarray(test_result["confusion_matrix"])
        extra["confusion_matrix"] = {
            "labels": ["baixa", "alta"],
            "counts": cm.tolist(),
            "percent_by_true": np.round(confusion_matrix_percent(cm, "true"), 2).tolist(),
            "percent_by_pred": np.round(confusion_matrix_percent(cm, "pred"), 2).tolist(),
            "percent_of_total": np.round(confusion_matrix_percent(cm, "all"), 2).tolist(),
        }

    metadata = {
        "model_id": model_id,
        "framework": "PyTorch",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": config.to_dict(),
        "metrics": metrics,
        **extra,
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if history:
        with open(save_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print(f"[model-saver] Modelo salvo em {save_dir}/")
    return save_dir


def load_all_metadata(results_dir: str | Path = "../results"):
    """DataFrame com uma linha por execução (colunas achatadas)."""
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


def hyperparameter_table(results_dir: str | Path = "../results", only_varied: bool = True):
    """Tabela dos hiperparâmetros variados entre execuções + métricas de teste.

    É a tabela pedida no relatório final. Com `only_varied=True` (padrão),
    colunas que têm o mesmo valor em todas as execuções são omitidas — o que
    sobra é exatamente o que foi de fato investigado, sem poluir a tabela com
    dezenas de constantes.
    """
    import pandas as pd

    df = load_all_metadata(results_dir)
    if df.empty:
        return df

    hp_cols = [c for c in df.columns if c.startswith("hyperparameters.")]
    metric_cols = [c for c in df.columns if c.startswith("metrics.test_")]

    table = df[["run_name", *hp_cols, *metric_cols]].copy()
    table.columns = [c.replace("hyperparameters.", "").replace("metrics.test_", "test_")
                     for c in table.columns]

    # `run_name` aparece tanto na raiz do metadata quanto dentro de
    # `hyperparameters`, então após o rename existem duas colunas com o mesmo
    # nome — e `table[col]` devolveria um DataFrame, não uma Series.
    table = table.loc[:, ~table.columns.duplicated()]

    if only_varied:
        keep = ["run_name"]
        for col in table.columns[1:]:
            if col.startswith("test_"):
                keep.append(col)
                continue
            if table[col].map(str).nunique(dropna=False) > 1:
                keep.append(col)
        table = table[keep]

    sort_col = "test_rmse" if "test_rmse" in table.columns else (
        "test_accuracy" if "test_accuracy" in table.columns else None)
    if sort_col:
        table = table.sort_values(sort_col, ascending=(sort_col == "test_rmse"))
    return table.reset_index(drop=True)


def find_existing_run(run_name: str, results_dir: str | Path = "../results") -> dict | None:
    """Procura a execução mais recente salva para `run_name` (idempotência do notebook)."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return None

    matches = sorted(d for d in results_dir.iterdir()
                     if d.is_dir() and d.name.startswith(f"lstm_{run_name}_"))
    if not matches:
        return None

    run_dir = matches[-1]
    with open(run_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    history = []
    history_path = run_dir / "history.csv"
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                history.append({k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()})

    metrics = metadata.get("metrics", {})
    return {
        "model": None,
        "history": history,
        "test_scores": {k[len("test_"):]: v for k, v in metrics.items() if k.startswith("test_")},
        "val_scores": {k[len("val_"):]: v for k, v in metrics.items() if k.startswith("val_")},
        "test_result": {"confusion_matrix": metadata.get("confusion_matrix", {}).get("counts")},
        "run_dir": run_dir,
        "loaded_from_disk": True,
    }
