"""Script para rodar baterias de experimentos do MLP fora do notebook.

Uso:
    .venv/Scripts/python.exe scripts/run_experiments.py round1
    .venv/Scripts/python.exe scripts/run_experiments.py round2
    ...

Cada "round" é uma lista de ExperimentConfig definida abaixo. Reaproveita
fit_or_load (idempotente): reexecutar não retreina configs já salvas em results/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from mlp_cifar10.config import ExperimentConfig
from mlp_cifar10.data import get_dataloaders, CLASSES
from mlp_cifar10.train import fit_or_load

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

ROUNDS: dict[str, list[ExperimentConfig]] = {
    # Round 1: partindo do padrão comum aos melhores modelos (arquitetura funil
    # mais profunda [256,128,64,32] + batch_norm + dropout leve + weight_decay
    # leve + adam/relu/lr=1e-3), explora ao redor desse ponto ótimo.
    "round1": [
        ExperimentConfig(
            run_name="more_patience",
            hidden_layers=(256, 128, 64, 32),
            activation="relu", dropout=0.3, batch_norm=True,
            weight_decay=1e-4, learning_rate=1e-3, optimizer="adam",
            num_epochs=15, patience=4,
            notes="best_combo treinou todas as 30 epocas sem early stopping; testa mais paciencia/epocas.",
        ),
        ExperimentConfig(
            run_name="even_deeper",
            hidden_layers=(384, 192, 96, 48),
            activation="relu", dropout=0.3, batch_norm=True,
            weight_decay=1e-4, learning_rate=1e-3, optimizer="adam",
            num_epochs=15, patience=4,
            notes="rede um pouco mais larga que best_combo, mesma receita de regularizacao.",
        ),
        ExperimentConfig(
            run_name="dropout_higher",
            hidden_layers=(256, 128, 64, 32),
            activation="relu", dropout=0.5, batch_norm=True,
            weight_decay=1e-4, learning_rate=1e-3, optimizer="adam",
            num_epochs=15, patience=4,
            notes="mesma arquitetura do best_combo, dropout mais agressivo.",
        ),
        ExperimentConfig(
            run_name="gelu_combo",
            hidden_layers=(256, 128, 64, 32),
            activation="gelu", dropout=0.3, batch_norm=True,
            weight_decay=1e-4, learning_rate=1e-3, optimizer="adam",
            num_epochs=15, patience=4,
            notes="gelu foi a 2a melhor ativacao isolada; combina com a arquitetura vencedora.",
        ),
        ExperimentConfig(
            run_name="wd_higher",
            hidden_layers=(256, 128, 64, 32),
            activation="relu", dropout=0.3, batch_norm=True,
            weight_decay=1e-3, learning_rate=1e-3, optimizer="adam",
            num_epochs=15, patience=4,
            notes="mesma arquitetura do best_combo, weight_decay 10x maior.",
        ),
    ],
    # Round 2: even_deeper ([384,192,96,48]) foi o novo melhor (0.5387),
    # superando best_combo mesmo com metade das epocas; nenhum experimento
    # deu early stop. gelu ficou proximo do relu; dropout/weight_decay mais
    # agressivos pioraram bastante. Explora ao redor de even_deeper.
    "round2": [
        ExperimentConfig(
            run_name="even_deeper_more_epochs",
            hidden_layers=(384, 192, 96, 48),
            activation="relu", dropout=0.3, batch_norm=True,
            weight_decay=1e-4, learning_rate=1e-3, optimizer="adam",
            num_epochs=25, patience=6,
            notes="even_deeper nao convergiu em 15 epocas; da mais tempo/paciencia.",
        ),
        ExperimentConfig(
            run_name="gelu_even_deeper",
            hidden_layers=(384, 192, 96, 48),
            activation="gelu", dropout=0.3, batch_norm=True,
            weight_decay=1e-4, learning_rate=1e-3, optimizer="adam",
            num_epochs=20, patience=5,
            notes="gelu (2o melhor isolado) combinado com a arquitetura vencedora do round1.",
        ),
        ExperimentConfig(
            run_name="wider_still",
            hidden_layers=(512, 256, 128, 64),
            activation="relu", dropout=0.3, batch_norm=True,
            weight_decay=1e-4, learning_rate=1e-3, optimizer="adam",
            num_epochs=20, patience=5,
            notes="testa se aumentar ainda mais a largura continua ajudando.",
        ),
        ExperimentConfig(
            run_name="dropout_lighter",
            hidden_layers=(384, 192, 96, 48),
            activation="relu", dropout=0.2, batch_norm=True,
            weight_decay=1e-4, learning_rate=1e-3, optimizer="adam",
            num_epochs=20, patience=5,
            notes="dropout 0.5 piorou muito e 0.3 foi bom; testa um ponto intermediario mais leve (0.2).",
        ),
        ExperimentConfig(
            run_name="no_weight_decay",
            hidden_layers=(384, 192, 96, 48),
            activation="relu", dropout=0.3, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=20, patience=5,
            notes="testa se o weight_decay leve (1e-4) esta realmente ajudando nesta arquitetura maior ou se batch_norm+dropout ja bastam.",
        ),
    ],
    # Round 3: round2 empatou por volta de 0.55 (no_weight_decay=0.5522 foi o
    # melhor, dropout_lighter=0.2 e gelu ficaram bem proximos). Combina os
    # vencedores individuais e testa mais capacidade/tempo de treino.
    "round3": [
        ExperimentConfig(
            run_name="combo_light",
            hidden_layers=(384, 192, 96, 48),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=25, patience=6,
            notes="combina os 3 vencedores individuais do round2: gelu + dropout 0.2 + sem weight_decay.",
        ),
        ExperimentConfig(
            run_name="deeper_wider2",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="relu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=20, patience=5,
            notes="mais uma camada e mais largura que wider_still, sem weight_decay e dropout leve.",
        ),
        ExperimentConfig(
            run_name="longer_training",
            hidden_layers=(384, 192, 96, 48),
            activation="relu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=35, patience=8,
            notes="repete no_weight_decay (melhor do round2) com mais epocas/paciencia, ja que nenhum run deu early stop.",
        ),
        ExperimentConfig(
            run_name="lr_higher",
            hidden_layers=(384, 192, 96, 48),
            activation="relu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=2e-3, optimizer="adam",
            num_epochs=20, patience=5,
            notes="testa se lr maior converge mais rapido/melhor com essa capacidade maior.",
        ),
        ExperimentConfig(
            run_name="five_layer",
            hidden_layers=(256, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=25, patience=6,
            notes="variante de 5 camadas com bloco inicial mais largo, gelu + regularizacao leve.",
        ),
    ],
    # Round 4: combo_light (gelu + dropout 0.2 + sem weight_decay,
    # [384,192,96,48]) foi o melhor (0.5655). lr maior piorou. Testa mais
    # epocas/paciencia nesse combo, arquiteturas mais profundas com o mesmo
    # combo, e lr menor.
    "round4": [
        ExperimentConfig(
            run_name="combo_longer",
            hidden_layers=(384, 192, 96, 48),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=35, patience=8,
            notes="combo_light (melhor ate agora) com mais epocas/paciencia.",
        ),
        ExperimentConfig(
            run_name="combo_deeper_wider",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=25, patience=6,
            notes="aplica o combo vencedor (gelu+dropout0.2+sem wd) na arquitetura mais profunda/larga do round3.",
        ),
        ExperimentConfig(
            run_name="combo_lr_lower",
            hidden_layers=(384, 192, 96, 48),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=5e-4, optimizer="adam",
            num_epochs=30, patience=7,
            notes="lr maior (2e-3) piorou no round3; testa lr menor que o padrao com mais epocas para compensar.",
        ),
        ExperimentConfig(
            run_name="combo_dropout_01",
            hidden_layers=(384, 192, 96, 48),
            activation="gelu", dropout=0.1, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=25, patience=6,
            notes="dropout 0.2 bateu 0.3 e 0.5; testa se um dropout ainda mais leve (0.1) continua a tendencia.",
        ),
        ExperimentConfig(
            run_name="six_layer_combo",
            hidden_layers=(512, 384, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=25, patience=6,
            notes="rede ainda mais profunda (6 camadas) com o combo vencedor, para ver se capacidade extra continua ajudando.",
        ),
    ],
    # Round 5 (final): combo_deeper_wider ([512,256,128,64,32] + gelu +
    # dropout 0.2 + sem weight_decay) foi o melhor (0.5712). dropout 0.1 piorou
    # (overfit rapido), 6 camadas piorou (capacidade demais). Combina os
    # fatores vencedores e da o maximo de espaco para convergir.
    "round5": [
        ExperimentConfig(
            run_name="deeper_wider_lr_lower",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=5e-4, optimizer="adam",
            num_epochs=35, patience=8,
            notes="combina a melhor arquitetura (deeper_wider) com lr menor, que tambem ajudou isoladamente.",
        ),
        ExperimentConfig(
            run_name="deeper_wider_more_patience",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=40, patience=10,
            notes="repete a melhor config ate agora com muito mais paciencia/epocas para ver o teto real.",
        ),
        ExperimentConfig(
            run_name="dropout_015",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.15, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=30, patience=7,
            notes="dropout 0.1 piorou e 0.2 foi o melhor; testa um ponto intermediario.",
        ),
        ExperimentConfig(
            run_name="even_wider",
            hidden_layers=(768, 384, 192, 96, 48),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=25, patience=6,
            notes="testa se aumentar ainda mais a largura (mantendo profundidade) continua ajudando.",
        ),
        ExperimentConfig(
            run_name="relu_deeper_wider",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="relu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=25, patience=6,
            notes="confirma se gelu ainda vence relu nesta arquitetura mais profunda (nas rodadas anteriores a diferenca era pequena).",
        ),
    ],
    # Round 6: testa 3 alavancas ainda nao exploradas nas rodadas 1-5, sobre a
    # melhor config ate agora (deeper_wider_lr_lower, 0.5770): data
    # augmentation (flip + crop), LR schedule (cosine annealing) e a
    # combinacao das duas. Ensemble dos melhores modelos ja treinados fica
    # em scripts/ensemble_eval.py (nao precisa retreinar).
    "round6": [
        ExperimentConfig(
            run_name="augmentation_flip_crop",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=5e-4, optimizer="adam",
            num_epochs=40, patience=10, augment=True,
            notes="melhor config (deeper_wider_lr_lower) + data augmentation (RandomCrop+HorizontalFlip) no treino. Mais epocas/paciencia pois augmentation converge mais devagar.",
        ),
        ExperimentConfig(
            run_name="lr_cosine_schedule",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=30, patience=30, lr_schedule="cosine",
            notes="mesma arquitetura vencedora, mas com LR fixo trocado por cosine annealing (LR inicial 1e-3 decaindo a 0 em 30 epocas). patience=30 (~sem early stop) para deixar o schedule completar.",
        ),
        ExperimentConfig(
            run_name="combo_aug_schedule",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=40, patience=40, augment=True, lr_schedule="cosine",
            notes="combina data augmentation + cosine annealing, para ver se os efeitos se somam como aconteceu com gelu+dropout+weight_decay na rodada 3.",
        ),
    ],
    # Round 7: combo_aug_schedule (rodada 6) foi o melhor individual (0.5863)
    # e ainda nao tinha convergido em 40 epocas; o ensemble top-3 chegou a
    # 0.6058. Testa: mais epocas para deixar convergir de verdade,
    # normalizacao real do CIFAR-10 (nunca testada), e augmentation mais forte
    # (color jitter). O ensemble final (top-5) fica para depois desta rodada
    # rodar, via `scripts/ensemble_eval.py top5 <ids...>`.
    "round7": [
        ExperimentConfig(
            run_name="augmentation_more_epochs",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=5e-4, optimizer="adam",
            num_epochs=55, patience=12, augment=True,
            notes="augmentation_flip_crop nao tinha convergido em 40 epocas (val_acc ainda subindo); da mais espaco (55 ep, patience 12).",
        ),
        ExperimentConfig(
            run_name="combo_aug_schedule_more_epochs",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=55, patience=55, augment=True, lr_schedule="cosine",
            notes="combo_aug_schedule (melhor individual, 0.5863) tambem nao tinha convergido em 40 epocas; repete com 55 epocas e cosine T_max=55.",
        ),
        ExperimentConfig(
            run_name="real_normalization",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=40, patience=40, augment=True, lr_schedule="cosine",
            normalization="real",
            notes="repete combo_aug_schedule trocando a normalizacao simples ([-1,1]) pela media/desvio-padrao reais do CIFAR-10 (nunca testado nas rodadas 1-6).",
        ),
        ExperimentConfig(
            run_name="augmentation_color_jitter",
            hidden_layers=(512, 256, 128, 64, 32),
            activation="gelu", dropout=0.2, batch_norm=True,
            weight_decay=0.0, learning_rate=1e-3, optimizer="adam",
            num_epochs=40, patience=40, augment=True, lr_schedule="cosine",
            augment_strength="strong",
            notes="repete combo_aug_schedule adicionando ColorJitter (brightness/contrast/saturation leves) ao crop+flip, para ver se augmentation mais forte ajuda ainda mais.",
        ),
    ],
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ROUNDS:
        print(f"Uso: python run_experiments.py <{'|'.join(ROUNDS)}>")
        sys.exit(1)

    round_name = sys.argv[1]
    configs = ROUNDS[round_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader_cache: dict[tuple, tuple] = {}

    def loaders_for(cfg: ExperimentConfig):
        key = (cfg.augment, cfg.normalization, cfg.augment_strength)
        if key not in loader_cache:
            loader_cache[key] = get_dataloaders(
                data_dir=DATA_DIR, batch_size=64, val_fraction=0.1, seed=42, num_workers=0,
                augment=cfg.augment, normalization=cfg.normalization, augment_strength=cfg.augment_strength,
            )
        return loader_cache[key]

    for cfg in configs:
        print(f"\n{'=' * 60}\n{round_name} :: {cfg.run_name}\n{'=' * 60}")
        t0 = time.time()
        train_loader, val_loader, test_loader = loaders_for(cfg)
        result = fit_or_load(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            class_names=list(CLASSES),
            results_dir=RESULTS_DIR,
        )
        dt = time.time() - t0
        status = "carregado do disco" if result["loaded_from_disk"] else f"treinado em {dt:.1f}s"
        print(f"[{cfg.run_name}] {status} -> {result['run_dir']}")
        if not result["loaded_from_disk"]:
            print(f"  test_scores: {result['test_scores']}")


if __name__ == "__main__":
    main()
