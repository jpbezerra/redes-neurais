"""Fecha o notebook da Fase 2: substitui a seção "Próximas rodadas" pelas
conclusões finais, agora que todas as levas (1-8), o ensemble e o experimento
hierárquico já rodaram.

Mesmo padrão de `add_round8_sections.py`: editar um .ipynb de ~350 KB à mão é
frágil, então a mudança é feita por script e é idempotente.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_train_cnn_cifar10.ipynb"
MARKER = "## 16. Conclusão da Fase 2"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


CONCLUSION = md(
    """
## 16. Conclusão da Fase 2

**A busca está encerrada.** Oito levas, 53 execuções de 10 classes, mais o
ensemble e o experimento hierárquico. Resultado final:

| | Acurácia | F1 |
|---|---|---|
| Melhor modelo único — `vgg4_mixcut_aug_ls` | **0.9490** | 0.9490 |
| Melhor resultado geral — `ensemble_vgg4_all` (7 modelos, sem retreino) | **0.9554** | 0.9553 |
| *(referência: melhor resultado do MLP na Fase 1)* | *0.6135* | *0.6102* |

### O que cada uma das três frentes finais mostrou

**Leva 8 (MixUp/CutMix) — funcionou, e foi o que quebrou o platô.** MixUp
sozinho deu 0.9443 e CutMix 0.9429, ambos já acima de toda a leva 7;
alternados chegaram a 0.9469; e o combo completo (mistura + RandAugment +
label smoothing + Nesterov) a **0.9490**. Confirma o diagnóstico da seção 12:
o teto de 0.94 não era de regularização do *modelo*, era de variedade nos
*dados* de treino.

**Ensemble — o maior ganho absoluto, a custo zero de treino.** Os 7 modelos
VGG-4 das levas 6-7 promediados chegaram a **0.9554**, +1,37 p.p. sobre o
melhor membro isolado. Curiosamente, usar todos os 7 (incluindo os três
empatados em 0.9370) bateu usar só os 3 melhores (0.9519) — o oposto do que
tinha acontecido no MLP da Fase 1, porque aqui cada membro foi treinado com
um regularizador diferente e a diversidade já estava garantida por construção.

**Hierárquico — a hipótese era boa, o resultado foi negativo.** O porteiro
veículo/animal acerta 98,75%, mas a composição chegou a apenas 0.9360, abaixo
dos 0.9490 do modelo achatado. O diagnóstico do oráculo é o achado
interessante: mesmo com **porteiro perfeito** a composição daria 0.9466 —
ainda pior. O gargalo não é o porteiro, são os especialistas, que viram menos
dados (27k e 18k contra 45k). As imagens de veículo ensinam features de baixo
nível (bordas, texturas) que continuam úteis para classificar animais, e
separar as tarefas joga esse aproveitamento fora.

### O erro que sobrou

De 446 erros do ensemble no teste, `cat` (107) e `dog` (85) respondem por
**43%** — mais que o dobro da participação proporcional. Veículos chegaram a
0.9730 de média, animais a 0.9437. A diferença encolheu de 3,7 p.p. (leva 6)
para 2,9 p.p., mas nunca fechou. É o mesmo par de classes que dominava os
erros do MLP na Fase 1.

### Para ir além daqui

As alavancas de hiperparâmetro estão esgotadas — os ganhos das últimas três
frentes foram todos abaixo de 1 p.p., contra +14 p.p. da leva 2. O que ainda
renderia, em ordem: **conexões residuais (ResNet)**, a única mudança
estrutural de peso não testada; **ensemble de seeds** da configuração campeã
(em vez de recombinar modelos de acurácia menor); treino mais longo
(300-400 épocas); uma **rede de tronco compartilhado com duas cabeças**, que
é a forma correta de testar a intuição hierárquica sem fragmentar os dados; e
**test-time augmentation**, quase de graça.

O relatório completo está no [`README.md`](../README.md) da fase, e a
comparação com o MLP em [`../../comparativo/`](../../comparativo/).
"""
)


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    if any(MARKER in "".join(c["source"]) for c in cells):
        print("Notebook já finalizado — nada a fazer.")
        return 0

    idx = next(
        (i for i, c in enumerate(cells)
         if c["cell_type"] == "markdown" and "## 16. Próximas rodadas" in "".join(c["source"])),
        None,
    )
    if idx is None:
        print("Seção '16. Próximas rodadas' não encontrada.", file=sys.stderr)
        return 1

    cells[idx] = CONCLUSION
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Seção 16 substituída pelas conclusões finais (célula {idx}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
