"""Rede recorrente parametrizável para previsão de séries temporais.

A mesma classe cobre LSTM, GRU e RNN simples, trocando só `model_type`. Isso
permite a comparação das três famílias sem reescrever nada — e essa comparação
é interessante no relatório, porque a LSTM foi inventada exatamente para
resolver o problema que a RNN simples tem (esquecer dependências longas), e a
GRU é uma simplificação da LSTM com menos parâmetros.

Como uma rede recorrente processa a sequência: ela lê a janela dia a dia,
mantendo um "estado" interno que resume tudo que viu até ali. Ao final da
janela, esse estado é a memória comprimida dos 60 dias, e é ele que alimenta
a camada densa que produz a previsão.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_RECURRENT = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}


class RecurrentForecaster(nn.Module):
    """Rede recorrente + cabeça densa, para regressão ou classificação de direção.

    entrada [B, window, F] -> recorrente -> último estado -> densa -> saída [B]

    A saída é um único número por amostra: o preço previsto (regressão) ou o
    logit da probabilidade de alta (direção). Nos dois casos a forma é a mesma,
    o que muda é a função de perda aplicada em cima.
    """

    def __init__(
        self,
        input_size: int = 1,
        model_type: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.0,
        fc_layers: tuple[int, ...] = (),
    ) -> None:
        super().__init__()

        if model_type not in _RECURRENT:
            raise ValueError(f"model_type '{model_type}' desconhecido. Opções: {list(_RECURRENT)}")
        self.model_type = model_type

        # O PyTorch só aplica dropout ENTRE camadas recorrentes, então ele é
        # silenciosamente ignorado quando num_layers=1. Zeramos explicitamente
        # para o metadata.json não registrar um valor que não teve efeito.
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.recurrent = _RECURRENT[model_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=effective_dropout,
        )

        out_size = hidden_size * (2 if bidirectional else 1)
        head: list[nn.Module] = []
        in_features = out_size
        for units in fc_layers:
            head += [nn.Linear(in_features, units), nn.ReLU()]
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            in_features = units
        head.append(nn.Linear(in_features, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)      # [B, window, hidden*dirs]
        last = output[:, -1, :]            # estado ao fim da janela
        return self.head(last).squeeze(-1) # [B]

    def architecture_summary(self) -> dict:
        return {
            "model_type": self.model_type,
            "recurrent": repr(self.recurrent),
            "head": [type(layer).__name__ for layer in self.head],
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }
