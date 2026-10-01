"""Carregamento bruto e particionamento dos dados de churn.

EDA: `load_raw` le o CSV exatamente como veio, sem nenhuma transformacao —
usado so para a EDA.

pipeline de dados: `split_three_stage` implementa o procedimento de particionamento
exigido pelo enunciado da disciplina:

1. Separar o dataset em duas classes (Churn=Yes / Churn=No).
2. Dividir CADA classe em 50% treino / 25% validacao / 25% teste.
3. Reamostrar (com repeticao) a classe minoritaria DENTRO de cada particao
   de treino e validacao, para igualar o tamanho da classe majoritaria
   naquela particao — o teste NUNCA e reamostrado, permanece com a
   distribuicao real (ver nota de design abaixo).
4. Recombinar as duas classes de cada particao e embaralhar.

Nota de design (decisao explicita, o enunciado nao detalha isso): a
reamostragem por repeticao so e aplicada em treino e validacao. Se o teste
fosse reamostrado, a mesma linha apareceria (via copia) em treino E em
teste, o que e uma forma de vazamento — infla recall/F1 artificialmente.
O conjunto de teste reflete sempre a proporcao real de classes (~73/27).

Como alternativa auditavel, `split_three_stage` tambem devolve os indices
originais de cada particao ANTES da reamostragem — isso permite reconstruir
facilmente a versao "sem oversampling" (so com `class_weight`/`scale_pos_weight`)
para comparar as duas estrategias lado a lado, como esperado na baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV_NAME = "telco_customer_churn.csv"
# Mirror publico do IBM Telco Customer Churn — mesmo schema do dataset
# Kaggle "customers-churned-in-telecom-services" citado no enunciado (ver
# EDA, secao 3). Usado so como fallback de download quando o CSV nao esta
# em `data_dir` (ex.: clone fresco do repo no Kaggle, onde `data/` nao e
# versionado) — mesmo padrao ja usado no mini-projeto 2 (`load_tutorial`).
RAW_CSV_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw(data_dir: str | Path, force_download: bool = False) -> pd.DataFrame:
    """Le o CSV bruto exatamente como veio (sem dtype forcado, sem dropna).

    Se o arquivo nao existir em `data_dir` (ex.: clone fresco do repo no
    Kaggle, onde `data/` nao e versionado), baixa automaticamente de
    `RAW_CSV_URL` antes de ler — mesmo padrao do `load_tutorial` do
    mini-projeto 2. `force_download=True` baixa de novo mesmo se ja existir.

    `TotalCharges` propositalmente NAO e convertido para float aqui — a
    conversao e responsabilidade do pipeline de pre-processamento (pipeline de dados),
    que deve decidir e documentar como tratar os valores em branco
    conhecidos (ver EDA em reports/tables/eda_*.csv).
    """
    data_dir = Path(data_dir)
    path = data_dir / RAW_CSV_NAME
    if force_download or not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"Baixando CSV de {RAW_CSV_URL} ...")
        try:
            urllib.request.urlretrieve(RAW_CSV_URL, path)
        except Exception as exc:
            raise FileNotFoundError(
                f"CSV bruto nao encontrado em {path} e o download automatico de "
                f"{RAW_CSV_URL} falhou ({exc}). Baixe manualmente o dataset do "
                "Kaggle 'customers-churned-in-telecom-services' (ou o mirror "
                "publico do IBM Telco Customer Churn, mesmo schema) e salve "
                "nesse caminho."
            ) from exc
        print(f"CSV salvo em {path}")
    return pd.read_csv(path)


@dataclass
class SplitResult:
    """Indices (posicionais, relativos ao df original) de cada particao.

    `*_no_oversample` guarda os indices ANTES da reamostragem — usados tanto
    para o teste de vazamento quanto para montar o baseline sem oversampling.
    `*_idx` guarda os indices finais, JA com a reamostragem aplicada (treino
    e validacao tem indices repetidos de proposito; teste nunca repete).
    """
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_no_oversample_idx: np.ndarray
    val_no_oversample_idx: np.ndarray
    test_no_oversample_idx: np.ndarray
    seed: int


def split_three_stage(df: pd.DataFrame, seed: int = 0,
                       train_frac: float = 0.5, val_frac: float = 0.25) -> SplitResult:
    """Implementa o split de 3 etapas por classe, com reamostragem em
    treino/validacao (ver docstring do modulo).

    Retorna `SplitResult` com arrays de indices POSICIONAIS (`.iloc`),
    nunca rotulos de index do pandas — evita ambiguidade se `df` tiver
    index nao-sequencial.
    """
    assert abs(train_frac + 2 * val_frac - 1.0) < 1e-9, "train_frac + 2*val_frac deve somar 1"
    rng = np.random.default_rng(seed)

    pos_by_class = {
        cls: np.where(df[TARGET_COL].values == cls)[0]
        for cls in df[TARGET_COL].unique()
    }
    # Classe minoritaria = menos linhas no dataset original.
    classe_minoritaria = min(pos_by_class, key=lambda c: len(pos_by_class[c]))
    classe_majoritaria = [c for c in pos_by_class if c != classe_minoritaria][0]

    def split_classe(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = indices.copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train = idx[:n_train]
        val = idx[n_train:n_train + n_val]
        test = idx[n_train + n_val:]
        return train, val, test

    maj_train, maj_val, maj_test = split_classe(pos_by_class[classe_majoritaria])
    min_train, min_val, min_test = split_classe(pos_by_class[classe_minoritaria])

    def reamostra(min_part: np.ndarray, alvo_n: int) -> np.ndarray:
        """Reamostra com repeticao a particao minoritaria para ter `alvo_n`
        linhas, sorteando SOMENTE dentro desta particao (nunca de um pool
        maior) — e assim que a reamostragem de cada particao fica
        independente das outras, sem vazar repeticoes entre treino/val."""
        return rng.choice(min_part, size=alvo_n, replace=True)

    min_train_rs = reamostra(min_train, len(maj_train))
    min_val_rs = reamostra(min_val, len(maj_val))
    # Teste: SEM reamostragem, distribuicao real preservada.

    def recombina_embaralha(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        combinado = np.concatenate([a, b])
        rng.shuffle(combinado)
        return combinado

    train_idx = recombina_embaralha(maj_train, min_train_rs)
    val_idx = recombina_embaralha(maj_val, min_val_rs)
    test_idx = recombina_embaralha(maj_test, min_test)

    train_no_os = recombina_embaralha(maj_train, min_train)
    val_no_os = recombina_embaralha(maj_val, min_val)
    test_no_os = test_idx  # teste e identico nas duas estrategias

    return SplitResult(
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        train_no_oversample_idx=train_no_os, val_no_oversample_idx=val_no_os,
        test_no_oversample_idx=test_no_os, seed=seed,
    )


def assert_sem_vazamento(split: SplitResult) -> None:
    """Confere que nenhuma linha ORIGINAL (unica) aparece em mais de uma
    particao. A reamostragem cria COPIAS da mesma linha dentro da mesma
    particao (treino ou validacao) — isso e esperado e nao e vazamento.
    O que NAO pode acontecer e uma linha original aparecer em treino E
    teste, ou validacao E teste, etc.
    """
    conjuntos = {
        "train": set(split.train_no_oversample_idx.tolist()),
        "val": set(split.val_no_oversample_idx.tolist()),
        "test": set(split.test_no_oversample_idx.tolist()),
    }
    nomes = list(conjuntos)
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            a, b = nomes[i], nomes[j]
            intersecao = conjuntos[a] & conjuntos[b]
            assert not intersecao, (
                f"Vazamento detectado: {len(intersecao)} linha(s) originais "
                f"aparecem em '{a}' E '{b}' ao mesmo tempo."
            )
    # Garante tambem que a uniao das 3 particoes (sem reamostrar) cobre
    # exatamente o dataset original, sem linha perdida ou duplicada entre
    # particoes diferentes.
    todas = conjuntos["train"] | conjuntos["val"] | conjuntos["test"]
    soma_tamanhos = sum(len(c) for c in conjuntos.values())
    assert len(todas) == soma_tamanhos, "Alguma linha original ficou em mais de uma particao."
