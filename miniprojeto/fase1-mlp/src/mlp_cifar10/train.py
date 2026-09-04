"""Loop de treino/avaliação do MLP, com early stopping e checkpoint automático."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpointing import find_existing_run, save_run
from .config import ExperimentConfig
from .metrics import get_per_class_accuracy, get_scores
from .model import MLP

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


def build_model(config: ExperimentConfig) -> MLP:
    return MLP(
        input_size=config.input_size,
        num_classes=config.num_classes,
        hidden_layers=config.hidden_layers,
        activation=config.activation,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
    )


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer not in _OPTIMIZERS:
        raise ValueError(f"Otimizador '{config.optimizer}' desconhecido. Opções: {list(_OPTIMIZERS)}")
    kwargs = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer == "sgd":
        kwargs["momentum"] = config.momentum
    return _OPTIMIZERS[config.optimizer](model.parameters(), **kwargs)


def build_loss(config: ExperimentConfig) -> nn.Module:
    if config.loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    if config.loss == "mse":
        # MSE precisa do alvo em one-hot; ver `_mse_loss_wrapper` abaixo.
        return nn.MSELoss()
    raise ValueError(f"Função de erro '{config.loss}' desconhecida. Opções: cross_entropy, mse")


def _compute_loss(loss_fn: nn.Module, outputs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if isinstance(loss_fn, nn.MSELoss):
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, labels.unsqueeze(1), 1.0)
        return loss_fn(outputs, targets_one_hot)
    return loss_fn(outputs, labels)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, num_classes: int):
    model.eval()
    total_loss = 0.0
    predictions, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = _compute_loss(loss_fn, outputs, labels, num_classes)
        total_loss += loss.item() * images.size(0)
        predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, targets, predictions


def fit(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    results_dir: str | Path = "../results",
    save_checkpoint: bool = True,
):
    """Treina o MLP conforme `config`, avalia no teste e (opcionalmente) salva a execução.

    Retorna um dict com: model, history (lista por época), test_scores,
    per_class_accuracy e run_dir (se `save_checkpoint=True`).
    """
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    loss_fn = build_loss(config)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history: list[dict] = []

    for epoch in tqdm(range(config.num_epochs), desc=f"treinando '{config.run_name}'"):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = _compute_loss(loss_fn, outputs, labels, config.num_classes)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, val_targets, val_preds = evaluate(model, val_loader, loss_fn, device, config.num_classes)
        val_accuracy = get_scores(val_targets, val_preds)["accuracy"]

        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy}
        )
        tqdm.write(
            f"Época {epoch + 1}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}"
        )

        # Early stopping com base na perda de validação
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                tqdm.write("Early stopping acionado.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_targets, test_preds = evaluate(model, test_loader, loss_fn, device, config.num_classes)
    test_scores = get_scores(test_targets, test_preds)
    per_class_accuracy = get_per_class_accuracy(test_targets, test_preds, class_names)

    print("Métricas no conjunto de teste:", test_scores)
    print("Acurácia por classe:", per_class_accuracy)

    run_dir = None
    if save_checkpoint:
        # Skill `model-saver`: salva model.pt + metadata.json em results/{model_id}/
        dataset_sizes = {
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
        }
        run_dir = save_run(
            config=config,
            model=model,
            history=history,
            test_scores=test_scores,
            per_class_accuracy=per_class_accuracy,
            dataset_sizes=dataset_sizes,
            results_dir=results_dir,
        )

    return {
        "model": model,
        "history": history,
        "test_scores": test_scores,
        "per_class_accuracy": per_class_accuracy,
        "run_dir": run_dir,
        "loaded_from_disk": False,
    }


def fit_or_load(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    results_dir: str | Path = "../results",
    force_retrain: bool = False,
):
    """Como `fit()`, mas reaproveita um resultado já salvo em `results/` para `config.run_name`.

    Torna o notebook idempotente: reexecutar uma célula de experimento não
    retreina do zero um modelo cujo resultado já existe em disco — só chama
    `fit()` de verdade se não houver run salvo (ou se `force_retrain=True`).
    Quando reaproveitado, `result["model"]` vem `None` (os pesos não são
    recarregados) e `result["loaded_from_disk"]` vem `True`.
    """
    if not force_retrain:
        existing = find_existing_run(config.run_name, results_dir)
        if existing is not None:
            print(
                f"[fit_or_load] Reaproveitando execução existente de "
                f"'{config.run_name}' em {existing['run_dir']} (não retreinado)."
            )
            return existing

    return fit(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        results_dir=results_dir,
    )
