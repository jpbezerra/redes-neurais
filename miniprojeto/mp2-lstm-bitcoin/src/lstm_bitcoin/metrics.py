"""Métricas de regressão e de classificação de direção.

A separação entre as duas famílias é proposital. Previsão de preço é
regressão e pede RMSE/MAE/MAPE; previsão de direção é classificação e pede
acurácia, precision, recall, F1 e **matriz de confusão**. Ter as duas no
mesmo relatório é o que permite responder a pergunta que realmente importa:
o modelo prevê bem o *número*, mas acerta o *movimento*?

Um modelo pode ter RMSE excelente e acertar a direção em 50% dos casos — o
que significa que ele é inútil para decidir qualquer coisa, porque só está
copiando o preço de ontem. É por isso que este módulo calcula
`directional_accuracy` mesmo nos experimentos de regressão.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Regressão
# ---------------------------------------------------------------------------

def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, MAPE e R². Espera valores já em escala original (dólares)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true

    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())

    # MAPE ignora pontos com alvo zero para não estourar.
    nonzero = y_true != 0
    mape = float(np.abs(err[nonzero] / y_true[nonzero]).mean() * 100) if nonzero.any() else float("nan")

    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "mape": mape,
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, last_observed: np.ndarray) -> float:
    """Fração de dias em que o modelo acertou o SINAL do movimento.

    `last_observed` é o último valor da janela de entrada — o ponto de partida
    a partir do qual medimos se o preço subiu ou desceu. Sem ele não dá para
    definir "direção".

    Esta é a métrica mais reveladora num relatório de previsão de preço:
    um RMSE baixo com 50% de acerto direcional significa que o modelo apenas
    copia o último valor, e não tem poder preditivo nenhum.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    last = np.asarray(last_observed).ravel()
    return float(((y_pred > last) == (y_true > last)).mean())


def naive_baseline_scores(y_true: np.ndarray, last_observed: np.ndarray) -> dict:
    """Desempenho do palpite 'amanhã é igual a hoje'.

    É o baseline obrigatório: qualquer modelo precisa bater isso para ter
    valor. O acerto direcional dele é indefinido (ele nunca prevê movimento),
    então reportamos só as métricas de erro.
    """
    return regression_scores(y_true, last_observed)


# ---------------------------------------------------------------------------
# Classificação (direção)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Matriz de confusão em CONTAGEM absoluta. Linhas = real, colunas = previsto."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def confusion_matrix_percent(cm: np.ndarray, normalize: str = "true") -> np.ndarray:
    """Converte a matriz de contagem em percentual.

    - `normalize="true"` (padrão): cada LINHA soma 100%. Responde "das vezes
      que o preço realmente subiu, em quantas o modelo disse que subiria?" —
      é a leitura de recall, e a mais útil para diagnosticar viés do modelo.
    - `normalize="pred"`: cada COLUNA soma 100%. Responde "das vezes que o
      modelo disse alta, em quantas ele acertou?" — leitura de precision.
    - `normalize="all"`: a matriz inteira soma 100%.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        denom = cm.sum()
    else:
        raise ValueError(f"normalize '{normalize}' desconhecido. Opções: true, pred, all")
    return np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0) * 100


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Acurácia, precision, recall, F1 e a taxa da classe majoritária.

    A `majority_rate` é o baseline de classificação: se 55% dos dias foram de
    alta, um modelo que responde "alta" sempre acerta 55%. Acurácia abaixo
    disso é pior que não fazer nada.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / max(cm.sum(), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    counts = np.bincount(y_true, minlength=2)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "majority_rate": float(counts.max() / max(counts.sum(), 1)),
    }


def format_confusion(cm: np.ndarray, labels: tuple[str, ...] = ("baixa", "alta")) -> str:
    """Formata as duas matrizes (contagem e percentual por linha) lado a lado."""
    pct = confusion_matrix_percent(cm, normalize="true")
    width = max(len(l) for l in labels) + 2
    header = " " * (width + 8) + "".join(f"{l:>12}" for l in labels)
    lines = ["CONTAGEM  (linha = real, coluna = previsto)", header]
    for i, label in enumerate(labels):
        lines.append(f"  real {label:<{width}}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(labels))))
    lines += ["", "PERCENTUAL POR LINHA  (cada linha soma 100%)", header]
    for i, label in enumerate(labels):
        lines.append(f"  real {label:<{width}}" + "".join(f"{pct[i, j]:>11.1f}%" for j in range(len(labels))))
    return "\n".join(lines)
