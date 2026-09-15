"""Carregamento do preço do Bitcoin e construção das janelas deslizantes.

Este módulo concentra as três decisões que mais afetam a honestidade do
resultado numa série temporal, e todas elas são fáceis de errar:

1. **A divisão é temporal, nunca aleatória.** Se você embaralhar antes de
   dividir, o modelo treina com dias de 2018 e é testado em dias de 2015 —
   ou seja, aprende o futuro e prevê o passado. O resultado fica
   espetacular e completamente falso. Aqui o treino é sempre o começo da
   série e o teste é sempre o fim dela.

2. **O scaler é ajustado só no treino.** Ajustar no conjunto inteiro faria
   o mínimo e o máximo do teste vazarem para o treino. Como o Bitcoin sobe
   de ~300 para ~20.000 dólares na série, esse vazamento é enorme.

3. **As janelas são recortadas dentro de cada partição, não antes.** Se
   você recortar antes de dividir, a última janela do treino contém dias
   que pertencem ao teste.

Duas bases são suportadas: `tutorial`, o `btc.csv` do repositório citado no
enunciado (dez/2014 a mai/2018, diário, 1.273 linhas), e `kaggle`, a base de
1 minuto do Bitstamp reamostrada em diário — muito mais longa e atual, boa
para verificar se o que funciona no período do enunciado continua valendo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Base usada no enunciado (Brian Mwangi). Colunas:
# Date, Symbol, Open, High, Low, Close, Volume From, Volume To
TUTORIAL_URL = (
    "https://raw.githubusercontent.com/brynmwangy/"
    "predicting-bitcoin-prices-using-LSTM/master/btc.csv"
)

# Nomes canônicos usados internamente, independente da base de origem.
CANONICAL = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_tutorial(data_dir: str | Path = "./data", force_download: bool = False) -> pd.DataFrame:
    """Carrega o `btc.csv` do enunciado, ordenado do mais antigo para o mais recente.

    O arquivo original vem em ordem decrescente de data (a primeira linha é
    26/05/2018) — se não ordenarmos, a janela deslizante ficaria invertida no
    tempo e o modelo aprenderia a prever o passado.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "btc_tutorial.csv"

    if force_download or not path.exists():
        pd.read_csv(TUTORIAL_URL).to_csv(path, index=False)

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.rename(columns={"Volume From": "Volume", "Volume To": "VolumeQuote"})
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", *CANONICAL, "VolumeQuote"]]


def load_kaggle(csv_path: str | Path, resample: str = "1D") -> pd.DataFrame:
    """Carrega a base de 1 minuto do Kaggle (mczielinski/bitcoin-historical-data) em diário.

    Baixe o CSV pela interface do Kaggle ou com
    `kaggle datasets download -d mczielinski/bitcoin-historical-data` e aponte
    `csv_path` para o arquivo extraído.

    A reamostragem para diário segue a convenção OHLCV: abertura do primeiro
    minuto do dia, máxima do dia, mínima do dia, fechamento do último minuto e
    soma do volume.
    """
    df = pd.read_csv(csv_path)

    ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "date", "datetime")), None)
    if ts_col is None:
        raise ValueError(f"Nenhuma coluna de data reconhecida em {list(df.columns)}")

    if np.issubdtype(df[ts_col].dtype, np.number):
        df["Date"] = pd.to_datetime(df[ts_col], unit="s")
    else:
        df["Date"] = pd.to_datetime(df[ts_col])

    rename = {c: c.capitalize() for c in df.columns if c.lower() in
              ("open", "high", "low", "close", "volume")}
    df = df.rename(columns=rename)

    daily = (
        df.set_index("Date")
        .resample(resample)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
        .reset_index()
    )
    return daily


def load_dataset(name: str = "tutorial", data_dir: str | Path = "./data", **kwargs) -> pd.DataFrame:
    if name == "tutorial":
        return load_tutorial(data_dir, **kwargs)
    if name == "kaggle":
        csv_path = kwargs.pop("csv_path", Path(data_dir) / "btcusd_1-min_data.csv")
        return load_kaggle(csv_path, **kwargs)
    raise ValueError(f"dataset '{name}' desconhecido. Opções: tutorial, kaggle")


# ---------------------------------------------------------------------------
# Escalonamento
# ---------------------------------------------------------------------------

class Scaler:
    """Normalizador simples, ajustado só no treino.

    Guardamos os parâmetros explicitamente (em vez de usar o sklearn direto)
    porque precisamos **desfazer** a transformação para reportar erros em
    dólares, não em unidades normalizadas — um RMSE de 0.02 não diz nada a
    ninguém; um RMSE de 300 dólares diz.
    """

    def __init__(self, kind: str = "minmax") -> None:
        self.kind = kind
        self.a_: np.ndarray | None = None  # mínimo (minmax) ou média (standard)
        self.b_: np.ndarray | None = None  # amplitude (minmax) ou desvio (standard)

    def fit(self, x: np.ndarray) -> "Scaler":
        if self.kind == "none":
            self.a_, self.b_ = np.zeros(x.shape[1]), np.ones(x.shape[1])
        elif self.kind == "minmax":
            self.a_ = x.min(axis=0)
            self.b_ = np.where(x.max(axis=0) - self.a_ == 0, 1.0, x.max(axis=0) - self.a_)
        elif self.kind == "standard":
            self.a_ = x.mean(axis=0)
            self.b_ = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
        else:
            raise ValueError(f"scaler '{self.kind}' desconhecido. Opções: minmax, standard, none")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.a_) / self.b_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.b_ + self.a_

    def inverse_target(self, y: np.ndarray, target_index: int) -> np.ndarray:
        """Desfaz a escala só da coluna alvo (para reportar em dólares)."""
        return y * self.b_[target_index] + self.a_[target_index]


# ---------------------------------------------------------------------------
# Janelas deslizantes
# ---------------------------------------------------------------------------

def make_windows(
    array: np.ndarray,
    target_col: int,
    window: int,
    horizon: int = 1,
    task: str = "regression",
) -> tuple[np.ndarray, np.ndarray]:
    """Converte uma série [T, F] em pares (X, y) de janelas deslizantes.

    Para `window=60` e `horizon=1`, cada X tem forma [60, F] (os 60 dias
    anteriores, com todas as features) e cada y é o valor do alvo no dia
    seguinte.

    Com `task="direction"`, o alvo vira 1 se o preço subiu em relação ao
    último dia da janela e 0 se caiu — a mesma janela, outra pergunta.
    """
    xs, ys = [], []
    for i in range(len(array) - window - horizon + 1):
        xs.append(array[i : i + window])
        future = array[i + window + horizon - 1, target_col]
        if task == "direction":
            last = array[i + window - 1, target_col]
            ys.append(1.0 if future > last else 0.0)
        else:
            ys.append(future)
    if not xs:
        raise ValueError(
            f"Série curta demais: {len(array)} pontos para window={window}, horizon={horizon}"
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def prepare_splits(df: pd.DataFrame, config) -> dict:
    """Pipeline completo: divide no tempo, escala com o treino, recorta janelas.

    Retorna um dicionário com X/y de cada partição, o scaler ajustado, as datas
    correspondentes a cada previsão do teste (úteis para plotar) e o
    `naive_test`, que é o baseline de referência.

    Sobre o **baseline ingênuo**: em previsão de preço, o palpite "amanhã vale
    o mesmo que hoje" é surpreendentemente difícil de bater, porque o preço é
    quase um passeio aleatório. Qualquer modelo que não supere esse baseline
    não aprendeu nada útil — ele só aprendeu a copiar o último valor com um
    dia de atraso. Reportar isso lado a lado é o que separa um relatório
    honesto de um gráfico bonito e vazio.
    """
    feats = list(config.features)
    if config.target not in feats:
        raise ValueError(f"target '{config.target}' precisa estar em features={feats}")
    target_idx = feats.index(config.target)

    values = df[feats].to_numpy(dtype=np.float64).copy()
    dates = df["Date"].to_numpy()

    if config.log_price:
        # O log só faz sentido para PREÇO. O volume pode ser zero (36 dias na
        # base do enunciado) e, mais importante, volume não é uma quantidade
        # cujo crescimento seja multiplicativo do mesmo jeito — aplicar log nele
        # junto seria uma escolha diferente, não uma consequência de log_price.
        #
        # Então: log nas colunas de preço, `log1p` nas demais (que aceita zero e
        # se comporta como log para valores grandes). A coluna alvo é sempre
        # tratada como preço, já que é ela que será reconstruída na avaliação.
        price_like = {"Open", "High", "Low", "Close", "VolumeQuote"}
        for j, name in enumerate(feats):
            col = values[:, j]
            if name in price_like or j == target_idx:
                if (col <= 0).any():
                    raise ValueError(
                        f"log_price=True exige valores positivos na coluna de preço '{name}' "
                        f"(encontrado mínimo {col.min()})"
                    )
                values[:, j] = np.log(col)
            else:
                if (col < 0).any():
                    raise ValueError(f"coluna '{name}' tem valores negativos; log1p não se aplica")
                values[:, j] = np.log1p(col)

    # `diff_target` troca o nível pela VARIAÇÃO entre dias consecutivos.
    #
    # Isto é o ajuste mais importante deste projeto, e o motivo é concreto: no
    # recorte do enunciado o treino vê o preço entre US$ 120 e US$ 2.698,
    # enquanto o teste vive entre US$ 3.617 e US$ 19.650. Prevendo o NÍVEL, o
    # modelo é obrigado a extrapolar para uma faixa que nunca viu, e nenhuma
    # escolha de normalização conserta isso — a saída de uma rede saturada em
    # tanh/sigmoid simplesmente não alcança aqueles valores.
    #
    # ATENÇÃO — combine SEMPRE com `log_price=True`. A escolha entre os dois
    # modos de diferença decide se o alvo é de fato estacionário, e medindo na
    # série do enunciado a diferença é gritante:
    #
    #   diff_target sozinho (variação em US$):   desvio treino 26,2 / teste 608,2  -> 23,2x
    #   diff_target + log_price (log-retorno):   desvio treino 0,044 / teste 0,055 ->  1,23x
    #
    # A variação em dólares NÃO é estacionária: um movimento de 3% valia US$ 20
    # em 2015 e US$ 500 em 2018. Como o scaler é ajustado só no treino, o
    # modelo aprende numa escala 23x menor que a do teste — e o resultado é um
    # RMSE que apenas empata com o baseline ingênuo, porque a rede aprende a
    # prever "aproximadamente zero" o tempo todo.
    #
    # O log-retorno resolve: subir 3% em 2016 e subir 3% em 2018 é o mesmo
    # número. Guardamos o nível (já em log, se aplicável) em `levels` para
    # reconstruir o preço depois — somar log-retorno equivale a multiplicar o
    # preço, e a exponencial é desfeita na avaliação.
    levels = values[:, target_idx].copy()
    if config.diff_target:
        values = np.diff(values, axis=0)
        levels = levels[1:]          # alinha: values[i] é a variação que CHEGA em levels[i]
        dates = dates[1:]

    # 1) Corte temporal. Tudo antes do corte é passado; tudo depois é futuro.
    n = len(values)
    n_test = int(n * config.test_fraction)
    n_trainval = n - n_test
    n_val = int(n_trainval * config.val_fraction)
    n_train = n_trainval - n_val

    # 2) Scaler ajustado SÓ no treino.
    scaler = Scaler(config.scaler).fit(values[:n_train])
    scaled = scaler.transform(values)

    # 3) Janelas recortadas dentro de cada partição.
    #    As partições de validação e teste recebem `window` pontos de contexto
    #    anteriores para que a primeira previsão de cada uma tenha história
    #    suficiente — esses pontos entram apenas como ENTRADA, nunca como alvo,
    #    então não há vazamento de rótulo.
    w, h = config.window, config.horizon
    parts, offsets = {}, {}
    for name, start, end in [
        ("train", 0, n_train),
        ("val", max(0, n_train - w - h + 1), n_trainval),
        ("test", max(0, n_trainval - w - h + 1), n),
    ]:
        x, y = make_windows(scaled[start:end], target_idx, w, h, config.task)
        parts[name] = (x, y)
        offsets[name] = start + w + h - 1  # índice, na série original, do 1º alvo

    test_dates = dates[offsets["test"] : offsets["test"] + len(parts["test"][1])]

    # `test_anchor` é o preço observado no ÚLTIMO dia de cada janela de teste —
    # o ponto de partida do movimento que o modelo tenta prever. Serve para
    # reconstruir o preço quando `diff_target=True` e para definir "direção".
    anchor_idx = offsets["test"] - config.horizon
    test_anchor = levels[anchor_idx : anchor_idx + len(parts["test"][1])]

    if config.diff_target:
        # Prevendo variação, o palpite ingênuo é "amanhã varia zero", ou seja,
        # o preço de amanhã é o de hoje — exatamente o mesmo baseline de sempre.
        naive_test = np.zeros(len(parts["test"][1]), dtype=np.float32)
    else:
        naive_test = parts["test"][0][:, -1, target_idx]

    return {
        "X_train": parts["train"][0], "y_train": parts["train"][1],
        "X_val": parts["val"][0], "y_val": parts["val"][1],
        "X_test": parts["test"][0], "y_test": parts["test"][1],
        "scaler": scaler,
        "target_index": target_idx,
        "test_dates": test_dates,
        "test_anchor": test_anchor,
        "naive_test": naive_test,
        "diff_target": config.diff_target,
        "log_price": config.log_price,
        "n_features": len(feats),
        "split_sizes": {"train": n_train, "val": n_val, "test": n_test},
    }
