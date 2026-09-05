"""Ensemble (soft voting) dos melhores modelos já treinados, sem retreinar.

Carrega os pesos (`model.pt`) de alguns dos melhores `results/{model_id}/`,
reconstrói cada arquitetura a partir do `metadata.json`, faz a média das
probabilidades (softmax) de cada um no conjunto de teste e reporta a acurácia
do ensemble comparada à do melhor modelo individual.

Uso:
    .venv/Scripts/python.exe scripts/ensemble_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from mlp_cifar10.data import get_dataloaders, CLASSES
from mlp_cifar10.metrics import get_per_class_accuracy, get_scores
from mlp_cifar10.model import MLP

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Os N melhores modelos (arquiteturas distintas) dentre os 45 já treinados.
# Escolhidos por acurácia de teste em results/ (ver README.md para a tabela completa).
MEMBER_MODEL_IDS = [
    "mlp_combo_aug_schedule_20260905-090100",      # 0.5863 - melhor individual
    "mlp_augmentation_flip_crop_20260904-225516",  # 0.5813
    "mlp_deeper_wider_lr_lower_20260904-190116",   # 0.5770
]


def load_member(model_id: str, device: torch.device):
    run_dir = RESULTS_DIR / model_id
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    h = meta["hyperparameters"]
    model = MLP(
        input_size=32 * 32 * 3,
        num_classes=10,
        hidden_layers=tuple(h["hidden_layers"]),
        activation=h["activation"],
        dropout=h["dropout"],
        batch_norm=h["batch_norm"],
    )
    state_dict = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, meta


@torch.no_grad()
def collect_probs(model, loader, device):
    all_probs = []
    all_targets = []
    for images, labels in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1).cpu()
        all_probs.append(probs)
        all_targets.append(labels)
    return torch.cat(all_probs), torch.cat(all_targets)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = get_dataloaders(data_dir=DATA_DIR, batch_size=256, num_workers=0)

    member_probs = []
    targets = None
    print(f"\nMembros do ensemble ({len(MEMBER_MODEL_IDS)}):")
    for model_id in MEMBER_MODEL_IDS:
        model, meta = load_member(model_id, device)
        probs, t = collect_probs(model, test_loader, device)
        acc_individual = meta["metrics"]["test_accuracy"]
        print(f"  - {model_id} (acc individual salva: {acc_individual:.4f})")
        member_probs.append(probs)
        targets = t

    avg_probs = torch.stack(member_probs).mean(dim=0)
    preds = avg_probs.argmax(dim=1)

    scores = get_scores(targets.tolist(), preds.tolist())
    per_class = get_per_class_accuracy(targets.tolist(), preds.tolist(), list(CLASSES))

    print("\nMétricas do ensemble (soft voting, média das probabilidades):")
    print(scores)
    print("\nAcurácia por classe do ensemble:")
    print(per_class)

    best_individual = max(
        json.loads((RESULTS_DIR / mid / "metadata.json").read_text(encoding="utf-8"))["metrics"]["test_accuracy"]
        for mid in MEMBER_MODEL_IDS
    )
    print(f"\nMelhor individual entre os membros: {best_individual:.4f}")
    print(f"Ensemble: {scores['accuracy']:.4f}")
    print(f"Ganho do ensemble sobre o melhor individual: {scores['accuracy'] - best_individual:+.4f}")

    # Registra o resultado do ensemble em results/ para rastreabilidade
    # (não é um "modelo" com pesos próprios, então sem model.pt — só metadata).
    ensemble_dir = RESULTS_DIR / "mlp_ensemble_top3_softvote"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": "mlp_ensemble_top3_softvote",
        "framework": "PyTorch",
        "model_type": "Ensemble (soft voting) de MLPs já treinados",
        "purpose": "Classificação de imagens CIFAR-10 (Mini-projeto 1 - Fase 1)",
        "members": MEMBER_MODEL_IDS,
        "metrics": {f"test_{k}": v for k, v in scores.items()},
        "per_class_accuracy": per_class,
        "notes": "Média das probabilidades (softmax) dos 3 melhores modelos individuais já treinados, sem retreinar nada.",
    }
    (ensemble_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[ensemble] metadata salvo em {ensemble_dir}/")


if __name__ == "__main__":
    main()
