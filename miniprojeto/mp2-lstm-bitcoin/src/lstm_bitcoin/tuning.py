"""Busca de hiperparâmetros com Optuna.

Diferença em relação ao Mini-projeto 1: lá a busca foi **gulosa e manual** —
cada leva testava hipóteses derivadas da anterior, e a decisão de o que testar
era humana. Isso deu um relatório muito rico em "por quê", mas gastou 8 levas
para percorrer o espaço.

Aqui a busca é **automática e amostral**. O Optuna propõe combinações, observa
o resultado e concentra as próximas propostas nas regiões promissoras (é o que
o amostrador TPE faz: modela a distribuição dos bons e dos maus resultados e
sorteia onde a razão entre elas é alta). Em troca da riqueza narrativa, você
ganha cobertura de um espaço bem maior.

As duas abordagens são complementares, e o relatório fica mais forte usando as
duas: **levas manuais** para entender o efeito de cada alavanca isoladamente, e
**Optuna** para achar a melhor combinação dentro da região que as levas
revelaram promissora.

Dois cuidados implementados aqui:

- **A seleção usa validação, nunca teste.** O Optuna otimiza a métrica de
  validação; o teste só é tocado uma vez, no final, para o modelo escolhido.
  Otimizar direto no teste seria escolher o modelo que teve sorte com aquele
  conjunto específico.
- **Pruning.** Execuções que vão claramente mal são interrompidas cedo, o que
  multiplica o número de configurações testáveis no mesmo tempo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import torch

from .config import ExperimentConfig
from .data import prepare_splits
from .train import build_loss, build_model, build_optimizer, make_loaders

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Espaço de busca padrão. Cada entrada é (tipo, argumentos) e é lida por
# `suggest_config` — deixar isso como dado em vez de código permite ajustar o
# espaço no notebook sem tocar na função.
DEFAULT_SPACE = {
    "window": ("categorical", [15, 30, 45, 60, 90]),
    "hidden_size": ("categorical", [16, 32, 64, 128, 256]),
    "num_layers": ("int", 1, 3),
    "dropout": ("float", 0.0, 0.5),
    "learning_rate": ("loguniform", 1e-4, 1e-2),
    "batch_size": ("categorical", [16, 32, 64, 128]),
    "weight_decay": ("loguniform", 1e-6, 1e-2),
    "model_type": ("categorical", ["lstm", "gru"]),
    "scaler": ("categorical", ["minmax", "standard"]),
    "grad_clip": ("categorical", [0.0, 1.0, 5.0]),
}


def suggest_config(trial: optuna.Trial, base: ExperimentConfig, space: dict | None = None) -> ExperimentConfig:
    """Monta um `ExperimentConfig` a partir das sugestões do trial."""
    space = space or DEFAULT_SPACE
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "loguniform":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        else:
            raise ValueError(f"tipo de espaço '{kind}' desconhecido")

    merged = {**base.to_dict(), **params, "run_name": f"{base.run_name}_trial{trial.number}"}
    return ExperimentConfig.from_dict(merged)


def _train_for_tuning(config: ExperimentConfig, splits: dict, device: torch.device,
                      trial: optuna.Trial | None = None) -> float:
    """Treino enxuto que devolve a melhor perda de validação (sem salvar em disco)."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    loaders = make_loaders(splits, config)
    model = build_model(config, splits["n_features"]).to(device)
    optimizer = build_optimizer(model, config)
    loss_fn = build_loss(config)

    best_val, patience_counter = float("inf"), 0

    for epoch in range(config.num_epochs):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item() * x.size(0)
        val_loss /= len(loaders["val"].dataset)

        if val_loss < best_val:
            best_val, patience_counter = val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

        # Pruning: avisa o Optuna do progresso para ele abortar trials ruins.
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val


def run_study(
    df,
    base_config: ExperimentConfig,
    device: torch.device,
    n_trials: int = 40,
    space: dict | None = None,
    study_name: str = "lstm_btc",
    storage: str | None = None,
    timeout: int | None = None,
) -> optuna.Study:
    """Roda o estudo completo.

    `df` é o DataFrame bruto: as janelas são refeitas a cada trial, porque
    `window` e `scaler` fazem parte do espaço de busca e mudam o pré-processamento.

    Passe `storage="sqlite:///optuna.db"` para que o estudo sobreviva à queda
    da sessão do Kaggle/Colab e possa ser retomado de onde parou.
    """
    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial, base_config, space)
        splits = prepare_splits(df, config)
        return _train_for_tuning(config, splits, device, trial)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        sampler=optuna.samplers.TPESampler(seed=base_config.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study


def best_config(study: optuna.Study, base_config: ExperimentConfig, run_name: str = "optuna_best") -> ExperimentConfig:
    """Converte o melhor trial de volta num `ExperimentConfig` treinável."""
    merged = {**base_config.to_dict(), **study.best_params, "run_name": run_name}
    return ExperimentConfig.from_dict(merged)


def study_dataframe(study: optuna.Study):
    """Tabela de todos os trials — entra no relatório como evidência da busca."""
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    df.columns = [c.replace("params_", "") for c in df.columns]
    return df.sort_values("value")


def importance_table(study: optuna.Study):
    """Importância de cada hiperparâmetro segundo o Optuna (fANOVA).

    Responde "qual hiperparâmetro mais explicou a variação de desempenho?" —
    é o análogo automático da análise "o que a busca revelou sobre cada
    hiperparâmetro" que o Mini-projeto 1 fez à mão.
    """
    import pandas as pd

    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as exc:  # poucos trials completos ainda
        print(f"[optuna] importância indisponível: {exc}")
        return pd.DataFrame()
    return pd.DataFrame({"hiperparametro": list(imp), "importancia": list(imp.values())})
