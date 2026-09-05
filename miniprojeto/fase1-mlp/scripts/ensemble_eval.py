"""Ensemble (soft voting) dos melhores modelos já treinados, sem retreinar.

Carrega os pesos (`model.pt`) de alguns dos melhores `results/{model_id}/`,
reconstrói cada arquitetura a partir do `metadata.json`, faz a média das
probabilidades (softmax) de cada um no conjunto de teste e reporta a acurácia
do ensemble comparada à do melhor modelo individual.

Cada membro é avaliado com a normalização (`default` ou `real`) com que foi
treinado (lida do seu próprio `metadata.json`), então o ensemble funciona
mesmo misturando membros treinados com normalizações diferentes.

Uso:
    .venv/Scripts/python.exe scripts/ensemble_eval.py [ensemble_id member_id1 member_id2 ...]

Sem argumentos, roda o ensemble padrão "top3" (ver ENSEMBLES abaixo).
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

# Ensembles pré-definidos: nome -> lista de model_ids membros.
# Escolhidos por acurácia de teste em results/ (ver README.md para a tabela completa).
ENSEMBLES: dict[str, list[str]] = {
    "top3": [
        "mlp_combo_aug_schedule_20260905-090100",      # 0.5863 - melhor individual
        "mlp_augmentation_flip_crop_20260904-225516",  # 0.5813
        "mlp_deeper_wider_lr_lower_20260904-190116",   # 0.5770
    ],
}


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


def run_ensemble(ensemble_id: str, member_model_ids: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_loader_cache: dict[str, torch.utils.data.DataLoader] = {}
    member_probs = []
    targets = None
    print(f"\nMembros do ensemble '{ensemble_id}' ({len(member_model_ids)}):")
    for model_id in member_model_ids:
        model, meta = load_member(model_id, device)
        normalization = meta["hyperparameters"].get("normalization", "default")
        if normalization not in test_loader_cache:
            _, _, test_loader_cache[normalization] = get_dataloaders(
                data_dir=DATA_DIR, batch_size=256, num_workers=0, normalization=normalization
            )
        probs, t = collect_probs(model, test_loader_cache[normalization], device)
        acc_individual = meta["metrics"]["test_accuracy"]
        print(f"  - {model_id} (acc individual salva: {acc_individual:.4f}, normalization={normalization})")
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
        for mid in member_model_ids
    )
    print(f"\nMelhor individual entre os membros: {best_individual:.4f}")
    print(f"Ensemble: {scores['accuracy']:.4f}")
    print(f"Ganho do ensemble sobre o melhor individual: {scores['accuracy'] - best_individual:+.4f}")

    # Registra o resultado do ensemble em results/ para rastreabilidade
    # (não é um "modelo" com pesos próprios, então sem model.pt — só metadata).
    ensemble_dir = RESULTS_DIR / f"mlp_ensemble_{ensemble_id}_softvote"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": f"mlp_ensemble_{ensemble_id}_softvote",
        "framework": "PyTorch",
        "model_type": "Ensemble (soft voting) de MLPs já treinados",
        "purpose": "Classificação de imagens CIFAR-10 (Mini-projeto 1 - Fase 1)",
        "members": member_model_ids,
        "metrics": {f"test_{k}": v for k, v in scores.items()},
        "per_class_accuracy": per_class,
        "notes": f"Média das probabilidades (softmax) de {len(member_model_ids)} modelos já treinados, sem retreinar nada.",
    }
    (ensemble_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[ensemble] metadata salvo em {ensemble_dir}/")


def main():
    if len(sys.argv) == 1:
        ensemble_id = "top3"
        member_model_ids = ENSEMBLES["top3"]
    elif len(sys.argv) >= 3:
        ensemble_id = sys.argv[1]
        member_model_ids = sys.argv[2:]
    else:
        print("Uso: ensemble_eval.py [ensemble_id member_id1 member_id2 ...]")
        sys.exit(1)

    run_ensemble(ensemble_id, member_model_ids)


if __name__ == "__main__":
    main()
