"""Pre-processamento: imputacao, encoding e normalizacao.

Regra inegociavel (checada pelos testes deste modulo): todo objeto que
aprende alguma estatistica dos dados (scaler, lista de categorias do
one-hot, mediana de imputacao) e ajustado (`fit`) SOMENTE no conjunto de
treino, e so depois aplicado (`transform`) em treino/validacao/teste.

Nenhuma variavel e removida aqui, exceto o identificador `customerID`
(regra explicita do enunciado: nao eliminar variaveis no primeiro modelo).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import ID_COL, TARGET_COL

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
# Categoricas binarias (Yes/No ou Female/Male) -> 0/1 direto.
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
# Categoricas com 3+ niveis -> one-hot.
MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# engenharia de features — features derivadas (opcionais, testadas isoladamente e so
# combinadas se melhorarem a metrica na validacao; ver scripts/engenharia_features.py).
SERVICOS_ADICIONAIS_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
TENURE_BUCKETS = [(0, 12), (12, 24), (24, 48), (48, 73)]  # meses; limite superior exclusivo
FEATURES_DERIVADAS_DISPONIVEIS = ("charges_per_tenure", "n_servicos_adicionais", "tenure_bucket")


def _to_total_charges_float(series: pd.Series) -> pd.Series:
    """Converte TotalCharges (string, com brancos nos clientes tenure=0)
    para float. Os brancos viram NaN aqui; a imputacao por 0 acontece em
    `Preprocessor.transform`, usando a mesma logica para treino/val/teste
    (nao depende de nenhuma estatistica ajustada, entao e segura de aplicar
    igual nas 3 particoes sem vazamento)."""
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")



def _computa_features_derivadas(base: pd.DataFrame) -> pd.DataFrame:
    """Calcula as colunas derivadas candidatas (engenharia de features). Nao depende de
    nenhuma estatistica ajustada no treino (exceto a normalizacao de
    `charges_per_tenure`, feita separadamente no Preprocessor) — e seguro
    chamar em qualquer particao."""
    out = pd.DataFrame(index=base.index)
    out["charges_per_tenure"] = base["TotalCharges"] / (base["tenure"] + 1)
    out["n_servicos_adicionais"] = sum((base[c] == "Yes").astype(int) for c in SERVICOS_ADICIONAIS_COLS)
    bucket_labels = []
    for lo, hi in TENURE_BUCKETS:
        bucket_labels.append(f"{lo}-{hi}")
    def _bucket(t):
        for (lo, hi), label in zip(TENURE_BUCKETS, bucket_labels):
            if lo <= t < hi:
                return label
        return bucket_labels[-1]
    out["tenure_bucket"] = base["tenure"].apply(_bucket)
    return out

@dataclass
class Preprocessor:
    """Scaler (media/desvio) ajustado so no treino; listas de categorias
    dos one-hots tambem fixadas no treino (categoria nova em val/teste cai
    em todas as colunas = 0, nunca quebra o pipeline nem cria coluna nova)."""
    numeric_mean_: pd.Series | None = None
    numeric_std_: pd.Series | None = None
    cat_levels_: dict[str, list[str]] = field(default_factory=dict)
    binary_maps_: dict[str, dict[str, int]] = field(default_factory=dict)
    extra_features: tuple[str, ...] = ()  # subconjunto de FEATURES_DERIVADAS_DISPONIVEIS
    charges_per_tenure_mean_: float | None = None
    charges_per_tenure_std_: float | None = None
    fitted: bool = False

    def _prepare_base(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.drop(columns=[ID_COL], errors="ignore").copy()
        out["TotalCharges"] = _to_total_charges_float(out["TotalCharges"])
        # Imputacao: tenure=0 -> TotalCharges=0 (nunca foi cobrado, valor
        # logicamente correto, nao uma estimativa). Nao depende de
        # estatistica de treino, entao e seguro aplicar em qualquer particao.
        out.loc[out["tenure"] == 0, "TotalCharges"] = out.loc[out["tenure"] == 0, "TotalCharges"].fillna(0.0)
        return out

    def fit(self, df_train: pd.DataFrame) -> "Preprocessor":
        base = self._prepare_base(df_train)
        self.numeric_mean_ = base[NUMERIC_COLS].mean()
        self.numeric_std_ = base[NUMERIC_COLS].std().replace(0, 1.0)
        for col in BINARY_COLS:
            niveis = sorted(base[col].dropna().unique().tolist())
            # Mapeia o PRIMEIRO nivel (ordem alfabetica) para 0 e o segundo para 1
            # — deterministico, nao depende da ordem de aparicao nos dados.
            self.binary_maps_[col] = {v: i for i, v in enumerate(niveis)}
        for col in MULTI_CAT_COLS:
            self.cat_levels_[col] = sorted(base[col].dropna().unique().tolist())

        if "charges_per_tenure" in self.extra_features:
            derivadas = _computa_features_derivadas(base)
            self.charges_per_tenure_mean_ = float(derivadas["charges_per_tenure"].mean())
            self.charges_per_tenure_std_ = float(derivadas["charges_per_tenure"].std() or 1.0)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if not self.fitted:
            raise RuntimeError("Preprocessor.fit() precisa ser chamado antes de transform().")
        base = self._prepare_base(df)

        numeric = (base[NUMERIC_COLS] - self.numeric_mean_) / self.numeric_std_
        colunas = list(NUMERIC_COLS)
        blocos = [numeric.to_numpy(dtype=np.float64)]

        for col in BINARY_COLS:
            mapa = self.binary_maps_[col]
            # Categoria nao vista no treino (nao deveria acontecer em dados
            # binarios, mas defensivo): cai em NaN -> preenchido com 0.
            valores = base[col].map(mapa).fillna(0).to_numpy(dtype=np.float64)
            blocos.append(valores.reshape(-1, 1))
            colunas.append(f"{col}__bin")

        for col in MULTI_CAT_COLS:
            niveis = self.cat_levels_[col]
            for nivel in niveis:
                blocos.append((base[col] == nivel).to_numpy(dtype=np.float64).reshape(-1, 1))
                colunas.append(f"{col}__{nivel}")

        if self.extra_features:
            derivadas = _computa_features_derivadas(base)
            if "charges_per_tenure" in self.extra_features:
                valores = (derivadas["charges_per_tenure"] - self.charges_per_tenure_mean_) / self.charges_per_tenure_std_
                blocos.append(valores.to_numpy(dtype=np.float64).reshape(-1, 1))
                colunas.append("charges_per_tenure__num")
            if "n_servicos_adicionais" in self.extra_features:
                blocos.append(derivadas["n_servicos_adicionais"].to_numpy(dtype=np.float64).reshape(-1, 1))
                colunas.append("n_servicos_adicionais__num")
            if "tenure_bucket" in self.extra_features:
                bucket_labels = [f"{lo}-{hi}" for lo, hi in TENURE_BUCKETS]
                for label in bucket_labels:
                    blocos.append((derivadas["tenure_bucket"] == label).to_numpy(dtype=np.float64).reshape(-1, 1))
                    colunas.append(f"tenure_bucket__{label}")

        X = np.concatenate(blocos, axis=1)
        return X, colunas

    def transform_target(self, df: pd.DataFrame) -> np.ndarray:
        """Churn: 'Yes' -> 1, 'No' -> 0. Fixo (nao depende de fit) — o
        enunciado trata 'Yes' (churn) como a classe positiva/minoritaria."""
        return (df[TARGET_COL].values == "Yes").astype(np.float64)
