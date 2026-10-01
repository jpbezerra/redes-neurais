"""MLP baseline — scikit-learn MLPClassifier.

Decisao de ambiente (nao do enunciado): a instalacao de PyTorch neste
ambiente trava por causa do peso das dependencias CUDA que o pacote
generico do PyPI traz mesmo sem GPU, e o indice oficial CPU-only do
PyTorch (download.pytorch.org) esta fora da allowlist de rede disponivel
aqui. Em vez de bloquear os baselines por causa disso, o MLP usa
`sklearn.neural_network.MLPClassifier`, que cobre os mesmos
hiperparametros pedidos (camadas/unidades, ativacao, otimizador,
regularizacao L2, early stopping por patience). Se mais controle de
arquitetura for necessario depois (dropout de verdade, por exemplo),
reavaliar PyTorch rodando no Kaggle, onde o ambiente ja vem pronto.

Treino manual epoca-a-epoca (via `partial_fit`) em vez de so chamar
`.fit()` — necessario para registrar a curva de treino em `history.csv`
(train/val loss por epoca) e implementar o criterio de parada por
patience exigido no enunciado (nao o `n_iter_no_change` padrao do
sklearn, que so olha a ultima metrica, e sim: se nao houver melhora da
loss de validacao por `patience` epocas seguidas, para).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neural_network import MLPClassifier

from ..metrics import cross_entropy, mse


@dataclass
class MLPHyperparams:
    hidden_layer_sizes: tuple[int, ...] = (10,)   # navalha de Occam: comeca pequeno
    activation: str = "relu"                       # 'relu' | 'tanh' | 'logistic'
    solver: str = "adam"                            # 'adam' | 'sgd'
    learning_rate_init: float = 1e-3
    alpha: float = 1e-4                              # regularizacao L2
    batch_size: int = 64
    max_epochs: int = 10_000
    patience: int = 10                               # sobe para 20 se parar cedo demais
    loss_metric: str = "cross_entropy"               # 'cross_entropy' | 'mse' — comparado na baselines
    seed: int = 0


def treinar_mlp(X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 hp: MLPHyperparams, verbose: bool = False) -> tuple[MLPClassifier, list[dict]]:
    """Treina epoca a epoca com early stopping manual por patience.

    Retorna o melhor modelo (pesos no ponto de menor loss de validacao, nao
    necessariamente a ultima epoca) e o historico completo por epoca.
    """
    modelo = MLPClassifier(
        hidden_layer_sizes=hp.hidden_layer_sizes,
        activation=hp.activation,
        solver=hp.solver,
        learning_rate_init=hp.learning_rate_init,
        alpha=hp.alpha,
        batch_size=hp.batch_size,
        max_iter=1,            # controlamos as epocas manualmente
        warm_start=True,       # preserva pesos entre chamadas de fit()
        random_state=hp.seed,
    )

    loss_fn = cross_entropy if hp.loss_metric == "cross_entropy" else mse

    historico = []
    melhor_val_loss = np.inf
    melhor_estado = None
    falhas_seguidas = 0
    classes = np.unique(y_train)

    for epoca in range(hp.max_epochs):
        modelo.partial_fit(X_train, y_train, classes=classes)

        p_train = modelo.predict_proba(X_train)[:, 1]
        p_val = modelo.predict_proba(X_val)[:, 1]
        train_loss = loss_fn(y_train, p_train)
        val_loss = loss_fn(y_val, p_val)

        historico.append({"epoch": epoca, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < melhor_val_loss - 1e-6:
            melhor_val_loss = val_loss
            melhor_estado = {
                "coefs_": [c.copy() for c in modelo.coefs_],
                "intercepts_": [b.copy() for b in modelo.intercepts_],
            }
            falhas_seguidas = 0
        else:
            falhas_seguidas += 1

        if verbose and epoca % max(hp.max_epochs // 20, 1) == 0:
            print(f"  epoca {epoca:>5} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"falhas={falhas_seguidas}/{hp.patience}")

        if falhas_seguidas >= hp.patience:
            if verbose:
                print(f"  parada antecipada na epoca {epoca} (patience={hp.patience} esgotada)")
            break

    # Restaura os pesos do melhor ponto de validacao (nao a ultima epoca).
    if melhor_estado is not None:
        modelo.coefs_ = melhor_estado["coefs_"]
        modelo.intercepts_ = melhor_estado["intercepts_"]

    return modelo, historico
