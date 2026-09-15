"""Utilidades: seed, device e os gráficos padrão do relatório."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

# Paleta consistente em todos os gráficos do projeto.
C_REAL, C_PRED, C_NAIVE, C_GREY = "#2F6F9F", "#B5651D", "#8A8A8A", "#3E7C59"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": .25,
                         "axes.spines.top": False, "axes.spines.right": False})


def plot_forecast(dates, y_true, y_pred, naive=None, title="Previsão vs. real", save_path=None):
    """Série real contra previsão no conjunto de teste.

    Quando `naive` é passado, a linha do baseline ingênuo aparece junto — e é
    aí que fica visível o problema clássico deste tipo de modelo: se a curva
    prevista for apenas a curva real deslocada um dia à direita, o modelo está
    copiando o último preço, não prevendo.
    """
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, y_true, color=C_REAL, lw=1.6, label="real")
    ax.plot(dates, y_pred, color=C_PRED, lw=1.4, label="previsto")
    if naive is not None:
        ax.plot(dates, naive, color=C_NAIVE, lw=1, ls="--", alpha=.8, label="ingênuo (repete ontem)")
    ax.set_ylabel("Preço (USD)"); ax.legend(frameon=False, ncol=3)
    ax.set_title(title, fontsize=11, loc="left")
    fig.autofmt_xdate(); fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_training_curves(history, title="Curvas de treino", save_path=None):
    import matplotlib.pyplot as plt
    _style()
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(epochs, [h["train_loss"] for h in history], color=C_REAL, label="treino")
    ax.plot(epochs, [h["val_loss"] for h in history], color=C_PRED, label="validação")
    ax.set_xlabel("Época"); ax.set_ylabel("Perda"); ax.legend(frameon=False)
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion(cm, labels=("baixa", "alta"), title="Matriz de confusão", save_path=None):
    """Matriz de confusão com contagem E percentual na mesma figura.

    Cada célula mostra o número absoluto em cima e o percentual da linha
    embaixo — as duas leituras que o relatório pede, sem precisar de dois
    gráficos separados.
    """
    import matplotlib.pyplot as plt
    from .metrics import confusion_matrix_percent
    _style()

    cm = np.asarray(cm)
    pct = confusion_matrix_percent(cm, "true")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if pct[i, j] > 55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1f}%", ha="center", va="center",
                    color=color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=.046, label="% da linha")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_residuals(y_true, y_pred, title="Resíduos", save_path=None):
    """Erro ao longo do tempo e sua distribuição.

    Resíduo com estrutura visível (tendência, períodos de erro sistemático)
    indica que sobrou sinal que o modelo não capturou.
    """
    import matplotlib.pyplot as plt
    _style()
    resid = np.asarray(y_pred).ravel() - np.asarray(y_true).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(resid, color=C_REAL, lw=.9)
    axes[0].axhline(0, color=C_GREY, lw=1)
    axes[0].set_xlabel("Dia do teste"); axes[0].set_ylabel("Erro (USD)")
    axes[0].set_title("Erro ao longo do tempo", fontsize=10, loc="left")
    axes[1].hist(resid, bins=40, color=C_REAL, alpha=.85)
    axes[1].axvline(0, color=C_GREY, lw=1)
    axes[1].set_xlabel("Erro (USD)"); axes[1].set_ylabel("Frequência")
    axes[1].set_title("Distribuição do erro", fontsize=10, loc="left")
    fig.suptitle(title, fontsize=11, x=.02, ha="left")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
