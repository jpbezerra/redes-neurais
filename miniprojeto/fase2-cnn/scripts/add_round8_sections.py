"""Insere no notebook as seções da leva 8, do ensemble e do experimento hierárquico.

Editar um .ipynb de 270 KB à mão é frágil (JSON gigante, `outputs` embutidos),
então as seções novas são geradas por script — mesmo padrão do
`make_notebook.py` usado para criar o notebook original. O script é
idempotente: se as seções já existirem, ele não duplica nada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_train_cnn_cifar10.ipynb"

MARKER = "## 12. O que a leva 7 mostrou"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


NEW_CELLS = [
    md(
        """
## 12. O que a leva 7 mostrou — o platô de 0.94

| Config | Acurácia | vs. campeão da leva 6 |
|---|---|---|
| `vgg4_randaugment` | **0.9417** | +0.47pp |
| `vgg4_trivial_aug` | 0.9411 | +0.41pp |
| `vgg4_reg_combo` | 0.9407 | +0.37pp |
| `vgg4_label_smooth` | 0.9389 | +0.19pp |
| `vgg4_dropout` | 0.9370 | 0.00pp |
| `vgg4_wd_higher` | 0.9370 | 0.00pp |

**Leitura honesta: a leva 7 não produziu um vencedor claro.** Os seis
resultados cabem numa faixa de 0.47pp, e a acurácia de validação das últimas
épocas fica em 0.940-0.946 em *todos* eles. Numa amostra de teste de 10.000
imagens, o erro padrão de uma acurácia de ~0.94 é ~0.24pp, ou seja, a
diferença entre o "melhor" e o "pior" da leva está a ~2 desvios — perto do
ruído. Trocar o regularizador não move mais a agulha.

O que a leva 7 *sim* confirmou: augmentation forte (`randaugment`,
`trivial`) é levemente melhor que regularização por peso/dropout, e o
label smoothing deixa a rede bem menos superconfiante (`train_loss` 0.506
contra 0.006 do campeão) **sem** ganhar acurácia proporcional — o problema
não era só calibração.

### Onde o erro realmente está

Acurácia por classe do melhor modelo (`vgg4_randaugment`):

| Grupo | Classes | Acurácia média |
|---|---|---|
| Veículos | airplane, automobile, ship, truck | **0.964** |
| Animais | bird, cat, deer, dog, frog, horse | **0.918** |

E dentro dos animais: `cat` 0.854, `dog` 0.909, `bird` 0.927. **Gato e
cachorro sozinhos respondem por ~24% de todos os erros do modelo.** Esse
padrão se repete em todos os sete modelos VGG treinados.

Isso define as três frentes das próximas seções, em ordem de custo:

1. **Leva 8 — MixUp/CutMix** (seção 13): o único regularizador forte ainda
   não testado, e o que ataca justamente pares de classes confundíveis.
2. **Ensemble** (seção 14): custo zero de treino, só um forward a mais.
   Os sete modelos erram imagens diferentes; a média deve capturar isso.
3. **Hierárquico** (seção 15): porteiro veículo/animal + dois especialistas,
   motivado diretamente pela tabela acima.
"""
    ),
    md(
        """
## 13. Leva 8 — MixUp e CutMix

`MixUp` interpola duas imagens pixel a pixel (e seus rótulos na mesma
proporção); `CutMix` cola um recorte retangular de uma imagem sobre a outra,
com o rótulo ponderado pela área. Os dois forçam a rede a produzir
probabilidades intermediárias em vez de decorar exemplos, e são conhecidos
por ajudar exatamente onde o CIFAR-10 dói (pares visualmente próximos como
gato/cachorro).

Duas diferenças importantes em relação às levas anteriores:

- **Mais épocas (200 em vez de 120).** Com mistura, o modelo converge mais
  devagar — rodar 120 épocas subestimaria o método.
- **`patience` alta (40).** A perda de validação com mistura é mais ruidosa;
  early stopping agressivo cortaria o treino cedo demais.

Isso torna esta leva a mais cara do projeto (~2x o custo de uma leva normal
por config), então são só 4 configs em vez de 6.
"""
    ),
    code(
        """
# Leva 8 - base = mesmo _CHAMP das levas 6/7, agora com mistura de amostras.
# Usa 3 campos novos do ExperimentConfig: mixup_alpha, cutmix_alpha, mix_prob
# (implementados em train.apply_mix; 0.0 desliga e o loop degenera no treino normal).

_CHAMP_LONG = {**_CHAMP, "num_epochs": 200, "patience": 40}

round8_configs = [
    ExperimentConfig(run_name="vgg4_mixup", **{**_CHAMP_LONG, "mixup_alpha": 0.2, "mix_prob": 0.5},
        notes="Campeao + MixUp alpha=0.2 em 50% dos batches. Alpha baixo = mistura suave, o default mais seguro para CIFAR."),
    ExperimentConfig(run_name="vgg4_cutmix", **{**_CHAMP_LONG, "cutmix_alpha": 1.0, "mix_prob": 0.5},
        notes="Campeao + CutMix alpha=1.0. Recorte espacial em vez de interpolacao - preserva texturas locais, costuma bater o MixUp em imagens pequenas."),
    ExperimentConfig(run_name="vgg4_mixcut_both", **{**_CHAMP_LONG, "mixup_alpha": 0.2, "cutmix_alpha": 1.0, "mix_prob": 0.5},
        notes="Alterna MixUp e CutMix por batch (receita padrao de treinos modernos de CIFAR/ImageNet)."),
    ExperimentConfig(run_name="vgg4_mixcut_aug_ls", **{**_CHAMP_LONG, "mixup_alpha": 0.2, "cutmix_alpha": 1.0,
                                                       "mix_prob": 0.5, "augment_strength": "randaugment", "label_smoothing": 0.1, "nesterov": True},
        notes="Combo maximo: mistura + RandAugment (melhor da leva 7) + label smoothing + Nesterov. Se o teto de 0.94 for de regularizacao, este quebra."),
]

for config in round8_configs:
    if config.run_name in experiment_results:
        continue
    set_seed(config.seed)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
        num_workers=NUM_WORKERS,
        augment=config.augment,
        normalization=config.normalization,
        augment_strength=config.augment_strength,
    )
    experiment_results[config.run_name] = fit_or_load(
        config, train_loader, val_loader, test_loader, device,
        class_names=list(CLASSES), results_dir=RESULTS_DIR,
    )
"""
    ),
    code(
        """
#@title Tabela comparativa - levas 1..8
comparison8 = pd.DataFrame(
    {name: r["test_scores"] for name, r in experiment_results.items()}
).T.sort_values("accuracy", ascending=False)
comparison8
"""
    ),
    md(
        """
## 14. Ensemble dos modelos já treinados (custo zero de treino)

Sete redes VGG de 4 estágios já estão salvas em `results/`, cada uma treinada
com um regularizador diferente. Elas têm acurácia parecida (0.937-0.9417) mas
**erram imagens diferentes** — cada regularização leva a um mínimo distinto do
espaço de pesos. Quando os erros são parcialmente descorrelacionados, a média
das probabilidades acerta mais do que qualquer membro isolado.

O custo é só um forward pass a mais no teste: nada é retreinado. Por isso
esta é a melhor relação ganho/GPU-hora que sobrou na busca.

Ressalva metodológica: o objetivo declarado do mini-projeto é o **modelo
único**, então o ensemble entra como comparação extra — não substitui o
campeão individual no relatório.
"""
    ),
    code(
        """
#@title Ensemble por media das probabilidades (soft voting)
from cnn_cifar10.ensemble import ensemble_runs, save_ensemble

# Todos os VGG de 4 estagios ja treinados (levas 6 e 7).
vgg4_runs = [
    "vgg_4stage_gap_sgd", "vgg4_label_smooth", "vgg4_trivial_aug",
    "vgg4_randaugment", "vgg4_dropout", "vgg4_wd_higher", "vgg4_reg_combo",
]

ensemble_all = ensemble_runs(
    run_names=vgg4_runs, results_dir=RESULTS_DIR, data_dir=DATA_DIR,
    device=device, num_workers=NUM_WORKERS,
)
save_ensemble(ensemble_all, run_name="ensemble_vgg4_all", results_dir=RESULTS_DIR,
              notes="Media simples das probabilidades dos 7 VGG-4 estagios das levas 6-7.")

# Variante enxuta: so os 3 melhores. Membros fracos podem puxar a media para
# baixo, entao vale medir se menos modelos rendem mais.
ensemble_top3 = ensemble_runs(
    run_names=["vgg4_randaugment", "vgg4_trivial_aug", "vgg4_reg_combo"],
    results_dir=RESULTS_DIR, data_dir=DATA_DIR, device=device, num_workers=NUM_WORKERS,
)
save_ensemble(ensemble_top3, run_name="ensemble_vgg4_top3", results_dir=RESULTS_DIR,
              notes="Media simples dos 3 melhores VGG-4 estagios.")

pd.DataFrame({
    "ensemble_7_modelos": ensemble_all["test_scores"],
    "ensemble_top3": ensemble_top3["test_scores"],
    "melhor_isolado": {"accuracy": ensemble_all["best_solo_accuracy"]},
}).T
"""
    ),
    md(
        """
## 15. Experimento: classificação hierárquica (porteiro + especialistas)

A tabela da seção 12 mostra que veículos vão a 0.964 e animais ficam em
0.918. A hipótese: se uma rede não precisasse gastar capacidade separando
gato de caminhão, ela poderia se dedicar inteiramente a separar gato de
cachorro.

Três redes:

1. **`gate`** — 2 classes (veículo vs. animal), treinada em todas as 45.000
   imagens de treino.
2. **`vehicle`** — 4 classes, treinada só nas ~18.000 imagens de veículo.
3. **`animal`** — 6 classes, treinada só nas ~27.000 imagens de animal.

A predição final recompõe as 10 classes pela regra da cadeia,
`P(classe) = P(super) · P(classe | super)`, então o resultado é diretamente
comparável com os modelos achatados.

**Nota sobre o recorte:** o CIFAR-10 tem 6 animais e 4 veículos (não 8 + 2)
— `bird, cat, deer, dog, frog, horse` contra `airplane, automobile, ship,
truck`.

**As duas forças em disputa**, que é o que o experimento vai medir:

- *A favor:* cada especialista resolve um problema mais fácil, com fronteiras
  de decisão menos disputadas.
- *Contra:* cada especialista vê **bem menos dados** (27k / 18k contra 45k), e
  todo erro do porteiro é **irrecuperável** — se ele mandar um gato para o
  especialista em veículos, nenhuma rede depois conserta. O custo também
  triplica: são três treinos em vez de um.

A célula de diagnóstico no fim calcula a acurácia que a composição teria com
um **porteiro perfeito** (oráculo), separando quanto do erro é dos
especialistas e quanto é do porteiro.
"""
    ),
    code(
        """
#@title Treina as tres redes do experimento hierarquico
from cnn_cifar10.hierarchical import (
    ANIMALS, VEHICLES, SUPER_CLASSES,
    get_hierarchical_dataloaders, flat_test_loader,
    hierarchical_predict, oracle_gate_accuracy,
)
from cnn_cifar10.ensemble import find_run_dir, load_run_model

# Mesma arquitetura/otimizacao do campeao - so muda num_classes. Isso mantem a
# comparacao justa: se a hierarquia ganhar, o ganho vem da decomposicao da
# tarefa, nao de uma rede diferente.
_HIER_BASE = {**_CHAMP, "augment_strength": "randaugment"}  # melhor augmentation da leva 7

hier_configs = {
    "gate":    ExperimentConfig(run_name="hier_gate",    num_classes=2, **_HIER_BASE,
                                notes="Porteiro veiculo vs animal (2 classes), todas as 45k imagens."),
    "vehicle": ExperimentConfig(run_name="hier_vehicle", num_classes=4, **_HIER_BASE,
                                notes="Especialista em veiculos (4 classes), ~18k imagens."),
    "animal":  ExperimentConfig(run_name="hier_animal",  num_classes=6, **_HIER_BASE,
                                notes="Especialista em animais (6 classes), ~27k imagens - onde esta quase todo o erro do modelo achatado."),
}

hier_class_names = {
    "gate": list(SUPER_CLASSES),
    "vehicle": list(VEHICLES),
    "animal": list(ANIMALS),
}

hier_results = {}
for task, config in hier_configs.items():
    set_seed(config.seed)
    loaders = get_hierarchical_dataloaders(
        data_dir=DATA_DIR, batch_size=config.batch_size, val_fraction=config.val_fraction,
        seed=config.seed, num_workers=NUM_WORKERS, augment=config.augment,
        augment_strength=config.augment_strength, normalization=config.normalization,
    )[task]
    print(f"\\n=== {task}: {len(loaders['train'].dataset)} treino / "
          f"{len(loaders['val'].dataset)} val / {len(loaders['test'].dataset)} teste ===")
    hier_results[task] = fit_or_load(
        config, loaders["train"], loaders["val"], loaders["test"], device,
        class_names=hier_class_names[task], results_dir=RESULTS_DIR,
    )

pd.DataFrame({t: r["test_scores"] for t, r in hier_results.items()}).T
"""
    ),
    code(
        """
#@title Compoe as tres redes e compara com o modelo achatado
# fit_or_load pode ter reaproveitado runs do disco (model=None), entao os pesos
# sao recarregados a partir de results/ - assim a celula funciona tanto logo
# apos treinar quanto numa sessao nova.
_NUM_CLASSES = {"gate": 2, "vehicle": 4, "animal": 6}
hier_models = {
    task: load_run_model(find_run_dir(f"hier_{task}", RESULTS_DIR), device, num_classes=n)[0]
    for task, n in _NUM_CLASSES.items()
}

hier_combined = hierarchical_predict(
    hier_models["gate"], hier_models["vehicle"], hier_models["animal"],
    flat_test_loader(data_dir=DATA_DIR, batch_size=256, num_workers=NUM_WORKERS),
    device,
)

flat_best = max(
    (r["test_scores"]["accuracy"] for r in experiment_results.values()),
    default=float("nan"),
)

print(f"Porteiro (veiculo vs animal): {hier_combined['gate_accuracy']:.4f}")
print(f"Hierarquico composto (10 classes): {hier_combined['test_scores']['accuracy']:.4f}")
print(f"Melhor modelo achatado ate agora:  {flat_best:.4f}")
print(f"Com porteiro perfeito (oraculo):   {oracle_gate_accuracy(hier_combined):.4f}")

pd.DataFrame({
    "hierarquico": hier_combined["per_class_accuracy"],
    "melhor_achatado": experiment_results[
        max(experiment_results, key=lambda k: experiment_results[k]["test_scores"]["accuracy"])
    ]["per_class_accuracy"],
})
"""
    ),
]

FINAL_MD = md(
    """
## 16. Próximas rodadas

O que fazer depois depende de qual das três frentes acima moveu a agulha:

- **Se MixUp/CutMix (leva 8) ganhou**, o teto era mesmo de regularização:
  vale esticar para 300 épocas e testar `mixup_alpha=0.4`.
- **Se o ensemble ganhou bem mais que qualquer modelo isolado**, os erros são
  descorrelacionados e a diversidade compensa: treinar 3 modelos com seeds
  diferentes (não regularizadores diferentes) e ensemblar costuma render mais.
- **Se o hierárquico ganhou**, o caminho é aprofundar a decomposição
  (por exemplo, um terceiro nível só para `cat` vs `dog`).
- **Se nada ganhou**, o platô de ~0.94 é da família VGG simples. Passar disso
  pede conexões residuais (ResNet) ou treino muito mais longo com
  augmentation pesada — mudança de arquitetura, não de hiperparâmetro.

Alavancas de arquitetura ainda não exploradas: `conv_layers_per_block=3`,
bloco final `1x1` para misturar canais, ou `pool_size` só nos primeiros
estágios. Se as células ficarem grandes/lentas demais, mova para um
`scripts/run_experiments.py` (ver `../fase1-mlp/scripts/run_experiments.py`).
"""
)


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    if any(MARKER in "".join(c["source"]) for c in cells):
        print("Seções já presentes — nada a fazer.")
        return 0

    # Ponto de inserção: logo após a tabela comparativa da leva 7.
    insert_at = next(
        i for i, c in enumerate(cells)
        if c["cell_type"] == "code" and "comparison7" in "".join(c["source"])
    ) + 1

    # A antiga seção "12. Próximas rodadas" é substituída pela nova seção 16.
    old_next_steps = next(
        (i for i, c in enumerate(cells)
         if c["cell_type"] == "markdown" and "## 12. Próximas rodadas" in "".join(c["source"])),
        None,
    )
    if old_next_steps is not None:
        cells[old_next_steps] = FINAL_MD

    cells[insert_at:insert_at] = NEW_CELLS

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Inseridas {len(NEW_CELLS)} células em {insert_at}; notebook agora tem {len(cells)} células.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
