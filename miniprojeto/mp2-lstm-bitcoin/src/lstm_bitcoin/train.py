"""Loop de treino/avaliação da LSTM, com early stopping e checkpoint automático.

Mesma estrutura do Mini-projeto 1 (`fit` treina e salva, `fit_or_load`
reaproveita resultados já em disco para o notebook ser idempotente), adaptada
para série temporal:

- os DataLoaders vêm de arrays NumPy já janelados, não de um dataset de imagens;
- a avaliação desfaz a normalização antes de calcular o erro, para reportar em
  dólares;
- toda avaliação de regressão também reporta o **acerto direcional** e o
  **baseline ingênuo**, porque só o RMSE esconde o caso em que o modelo apenas
  copia o último preço.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .checkpointing import find_existing_run, save_run
from .config import ExperimentConfig
from .metrics import (
    classification_scores,
    confusion_matrix,
    directional_accuracy,
    naive_baseline_scores,
    regression_scores,
)
from .model import RecurrentForecaster

_OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}


def build_model(config: ExperimentConfig, input_size: int) -> RecurrentForecaster:
    return RecurrentForecaster(
        input_size=input_size,
        model_type=config.model_type,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        dropout=config.dropout,
        fc_layers=config.fc_layers,
    )


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer not in _OPTIMIZERS:
        raise ValueError(f"Otimizador '{config.optimizer}' desconhecido. Opções: {list(_OPTIMIZERS)}")
    kwargs = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer == "sgd":
        kwargs["momentum"] = config.momentum
    return _OPTIMIZERS[config.optimizer](model.parameters(), **kwargs)


def build_scheduler(optimizer, config: ExperimentConfig):
    if config.lr_schedule == "none":
        return None
    if config.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    if config.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(config.patience // 3, 2))
    raise ValueError(f"lr_schedule '{config.lr_schedule}' desconhecido. Opções: none, cosine, plateau")


def build_loss(config: ExperimentConfig) -> nn.Module:
    if config.task == "direction":
        return nn.BCEWithLogitsLoss()
    losses = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss()}
    if config.loss not in losses:
        raise ValueError(f"loss '{config.loss}' desconhecida. Opções: {list(losses)} (ou task=direction)")
    return losses[config.loss]


def make_loaders(splits: dict, config: ExperimentConfig) -> dict[str, DataLoader]:
    """DataLoaders a partir dos arrays janelados.

    `shuffle=True` só no treino. Embaralhar as JANELAS é legítimo e ajuda a
    convergência — cada janela já é uma amostra completa e independente, com
    seu passado inteiro dentro dela. O que não se pode fazer é embaralhar
    antes de dividir treino/teste, o que já foi evitado em `data.prepare_splits`.
    """
    loaders = {}
    for name in ("train", "val", "test"):
        ds = TensorDataset(
            torch.from_numpy(splits[f"X_{name}"]),
            torch.from_numpy(splits[f"y_{name}"]),
        )
        loaders[name] = DataLoader(ds, batch_size=config.batch_size, shuffle=(name == "train"))
    return loaders


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    splits: dict,
    config: ExperimentConfig,
    naive: np.ndarray | None = None,
) -> dict:
    """Avalia um conjunto e devolve métricas já em escala original."""
    raw_pred, raw_true = predict(model, loader, device)

    if config.task == "direction":
        prob = 1 / (1 + np.exp(-raw_pred))
        pred_cls = (prob >= 0.5).astype(int)
        scores = classification_scores(raw_true, pred_cls)
        return {
            "scores": scores,
            "confusion_matrix": confusion_matrix(raw_true, pred_cls).tolist(),
            "y_true": raw_true, "y_pred": pred_cls, "y_prob": prob,
        }

    # Regressão: desfaz a normalização para reportar em dólares.
    scaler, idx = splits["scaler"], splits["target_index"]
    y_pred = scaler.inverse_target(raw_pred, idx)
    y_true = scaler.inverse_target(raw_true, idx)

    anchor = splits.get("test_anchor")
    use_anchor = config.diff_target and anchor is not None and len(anchor) == len(y_true)

    if use_anchor:
        # O modelo previu a VARIAÇÃO; o preço é o último observado mais ela.
        # Em escala de log, somar a variação equivale a multiplicar o preço.
        y_pred = anchor + y_pred
        y_true = anchor + y_true
        last = anchor
        if config.log_price:
            y_pred, y_true, last = np.exp(y_pred), np.exp(y_true), np.exp(last)
    else:
        if config.log_price:
            y_pred, y_true = np.exp(y_pred), np.exp(y_true)
        last = None
        if naive is not None:
            last = scaler.inverse_target(naive, idx)
            if config.log_price:
                last = np.exp(last)

    scores = regression_scores(y_true, y_pred)

    if last is not None and len(last) == len(y_true):
        scores["directional_accuracy"] = directional_accuracy(y_true, y_pred, last)
        scores["naive_rmse"] = naive_baseline_scores(y_true, last)["rmse"]
        scores["beats_naive"] = bool(scores["rmse"] < scores["naive_rmse"])

    return {"scores": scores, "y_true": y_true, "y_pred": y_pred, "anchor": last}


def fit(
    config: ExperimentConfig,
    splits: dict,
    device: torch.device,
    results_dir: str | Path = "../results",
    save_checkpoint: bool = True,
    verbose: bool = True,
):
    """Treina conforme `config`, avalia no teste e (opcionalmente) salva a execução."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    loaders = make_loaders(splits, config)
    model = build_model(config, splits["n_features"]).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    loss_fn = build_loss(config)

    best_val, patience_counter, best_state = float("inf"), 0, None
    history: list[dict] = []

    iterator = range(config.num_epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"treinando '{config.run_name}'")

    for epoch in iterator:
        model.train()
        train_loss = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            # Redes recorrentes são propensas a gradientes explosivos: o mesmo
            # peso é reaplicado a cada passo de tempo, então o gradiente pode
            # crescer geometricamente ao longo da janela.
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(loaders["train"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item() * x.size(0)
        val_loss /= len(loaders["val"].dataset)

        if scheduler is not None:
            scheduler.step(val_loss) if config.lr_schedule == "plateau" else scheduler.step()

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val, patience_counter = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    tqdm.write("Early stopping acionado.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate_split(model, loaders["test"], device, splits, config, splits.get("naive_test"))
    val = evaluate_split(model, loaders["val"], device, splits, config)

    if verbose:
        print("Métricas no teste:", {k: round(v, 4) if isinstance(v, float) else v
                                     for k, v in test["scores"].items()})

    run_dir = None
    if save_checkpoint:
        run_dir = save_run(
            config=config, model=model, history=history,
            test_result=test, val_scores=val["scores"],
            split_sizes=splits["split_sizes"], results_dir=results_dir,
        )

    return {
        "model": model, "history": history,
        "test_scores": test["scores"], "test_result": test,
        "val_scores": val["scores"], "run_dir": run_dir,
        "loaded_from_disk": False,
    }


def fit_or_load(
    config: ExperimentConfig,
    splits: dict,
    device: torch.device,
    results_dir: str | Path = "../results",
    force_retrain: bool = False,
):
    """Como `fit()`, mas reaproveita um resultado já salvo para `config.run_name`."""
    if not force_retrain:
        existing = find_existing_run(config.run_name, results_dir)
        if existing is not None:
            print(f"[fit_or_load] Reaproveitando '{config.run_name}' de {existing['run_dir']}.")
            return existing
    return fit(config=config, splits=splits, device=device, results_dir=results_dir)
