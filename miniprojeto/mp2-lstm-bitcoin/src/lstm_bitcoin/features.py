"""Features derivadas de OHLCV para a tarefa de direção.

Por que este módulo existe
--------------------------

O experimento mais produtivo do projeto foi trocar `Close` por
`(Open, High, Low, Close)` na classificação de direção: a acurácia saltou de
~60,6% para ~68,1%, e o colapso na classe majoritária desapareceu.

Isso levanta uma hipótese testável: o ganho veio da **faixa do dia** e da
**posição do fechamento dentro dela** — informação que o preço de fechamento
sozinho destrói. Se for esse o mecanismo, calcular essas quantidades de forma
explícita deve ajudar ainda mais, porque a rede não precisa mais inferi-las a
partir de quatro séries correlacionadas.

Regra inviolável: nenhuma feature aqui pode usar informação do dia que se quer
prever. Toda janela móvel usa apenas passado (`shift` antes de agregar, nunca
`center=True`), e o teste `assert_sem_vazamento` verifica isso empiricamente.

Grupos disponíveis
------------------

- `basico`   : OHLC cru (o que já foi testado, para comparação justa)
- `faixa`    : amplitude do dia e posição do fechamento — a hipótese direta
- `retorno`  : log-retornos defasados (a autocorrelação de −0,20 no lag 1)
- `vol`      : volatilidade realizada em janelas móveis
- `completo` : todos acima

O alvo continua sendo montado em `data.make_windows`; este módulo só acrescenta
COLUNAS ao DataFrame, sem tocar no janelamento nem na divisão temporal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _log_retorno(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))


def adicionar_features(df: pd.DataFrame, grupos: tuple[str, ...] = ("faixa",)) -> tuple[pd.DataFrame, list[str]]:
    """Devolve (df com colunas novas, lista dos nomes criados).

    As linhas iniciais que ficam com NaN (por causa dos `shift` e das janelas
    móveis) são removidas — é por isso que a função devolve o DataFrame e não
    só as colunas: o corte precisa ser aplicado a tudo junto.
    """
    out = df.copy()
    criadas: list[str] = []

    def add(nome: str, serie: pd.Series) -> None:
        out[nome] = serie
        criadas.append(nome)

    tem_ohlc = {"Open", "High", "Low"}.issubset(out.columns)

    if "faixa" in grupos or "completo" in grupos:
        if not tem_ohlc:
            raise ValueError("grupo 'faixa' exige as colunas Open/High/Low no DataFrame")
        faixa = out["High"] - out["Low"]
        # Amplitude relativa: o quanto o preço oscilou no dia, em % do nível.
        # Normalizar pelo Close é o que torna a feature comparável ao longo do
        # tempo — 500 dólares de oscilação significam coisas diferentes com o
        # BTC a 10k ou a 60k.
        add("amplitude", faixa / (out["Close"] + EPS))
        # Posição do fechamento dentro da faixa do dia: 1 = fechou na máxima,
        # 0 = fechou na mínima. É a feature que Close sozinho NÃO consegue
        # expressar, e a candidata mais direta a explicar o ganho do OHLC.
        add("pos_close", (out["Close"] - out["Low"]) / (faixa + EPS))
        # Corpo do candle com sinal: fechou acima ou abaixo da abertura?
        add("corpo", (out["Close"] - out["Open"]) / (faixa + EPS))
        # Sombras: rejeição de máximas/mínimas, leitura clássica de candle.
        add("sombra_sup", (out["High"] - out[["Close", "Open"]].max(axis=1)) / (faixa + EPS))
        add("sombra_inf", (out[["Close", "Open"]].min(axis=1) - out["Low"]) / (faixa + EPS))

    if "retorno" in grupos or "completo" in grupos:
        r = _log_retorno(out["Close"])
        add("ret_1", r)
        # Defasagens: a autocorrelação medida foi −0,199 no lag 1 e ≤0,13 nos
        # demais. Dar os lags explicitamente poupa a rede de descobri-los.
        for k in (2, 3, 5):
            add(f"ret_{k}", r.shift(k - 1))
        # Retorno acumulado em janelas — só com passado (shift antes da soma).
        for j in (5, 10):
            add(f"ret_acum_{j}", r.rolling(j).sum().shift(1))

    if "vol" in grupos or "completo" in grupos:
        r = _log_retorno(out["Close"])
        for j in (5, 10, 20):
            # shift(1) ANTES da janela garante que o dia corrente não entra.
            add(f"vol_{j}", r.shift(1).rolling(j).std())
        # Razão entre volatilidade curta e longa: regime de calmaria ou stress?
        add("vol_razao", out["vol_5"] / (out["vol_20"] + EPS))

    if "volume" in grupos or "completo" in grupos:
        if "Volume" not in out.columns:
            raise ValueError("grupo 'volume' exige a coluna Volume")
        v = out["Volume"]
        add("vol_rel", v / (v.shift(1).rolling(20).mean() + EPS))

    antes = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    removidas = antes - len(out)
    if removidas:
        print(f"[features] {removidas} linha(s) iniciais removidas (NaN das janelas moveis)")

    return out, criadas


def assert_sem_vazamento(
    df: pd.DataFrame,
    colunas: list[str],
    grupos: tuple[str, ...] = ("completo",),
    n_teste: int = 40,
) -> None:
    """Verifica empiricamente que nenhuma feature usa informacao do futuro.

    O teste altera o `Close` de um unico dia D e verifica a invariante: uma
    feature honesta pode mudar em D ou depois, **nunca antes de D**. Se alguma
    linha anterior a D mudar, aquela feature olha para frente.

    A comparacao e feita pela coluna `Date`, nao pelo indice posicional. Isso
    importa porque `adicionar_features` descarta as linhas iniciais com NaN, e
    quantas ela descarta depende das features pedidas — alinhar por posicao
    produziria falsos positivos (e, pior, falsos negativos).

    Este teste existe porque vazamento temporal e o erro que gera resultados
    espetaculares e sem valor, e e silencioso: nada falha, o numero so fica bom
    demais.
    """
    idx = len(df) - n_teste
    data_alvo = df["Date"].to_numpy()[idx]

    base, _ = adicionar_features(df, grupos)
    mexido = df.copy()
    mexido.iloc[idx, mexido.columns.get_loc("Close")] *= 1.5
    depois, _ = adicionar_features(mexido, grupos)

    # Alinha os dois DataFrames pelas datas que sobreviveram em AMBOS.
    comuns = np.intersect1d(base["Date"].to_numpy(), depois["Date"].to_numpy())
    b = base[base["Date"].isin(comuns)].sort_values("Date").reset_index(drop=True)
    d = depois[depois["Date"].isin(comuns)].sort_values("Date").reset_index(drop=True)
    pos_alvo = int(np.searchsorted(b["Date"].to_numpy(), data_alvo))

    for c in colunas:
        if c not in b.columns:
            continue
        dif = np.where(~np.isclose(b[c].to_numpy(), d[c].to_numpy(), equal_nan=True))[0]
        if len(dif) and dif.min() < pos_alvo:
            raise AssertionError(
                f"VAZAMENTO: '{c}' mudou na posicao {dif.min()} ao alterar a posicao "
                f"{pos_alvo} ({data_alvo}) — a feature depende do futuro."
            )

    print(f"[features] sem vazamento: {len(colunas)} feature(s) verificada(s)")
