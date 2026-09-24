"""GAN recorrente para gerar dias sintéticos de OHLC (aumento de dados).

Por que isto é um teste de expectativa negativa
-------------------------------------------------

Um GAN aprende a **imitar a distribuição estatística dos dados de treino** —
ele não inventa informação que não está lá. Se a série real, com estas
features, está estatisticamente perto de um passeio aleatório (que é a
conclusão já estabelecida no relatório: quatro famílias de modelo convergem
para o acaso), o gerador vai aprender a produzir **mais passeios aleatórios
parecidos**. Isso pode ajudar um modelo que sofre de escassez de dados a
generalizar melhor a mesma distribuição — mas não pode criar um sinal
preditivo que os dados reais não continham.

O ponto de testar mesmo assim: é uma pergunta legítima (o professor pede para
"buscar ganhos de desempenho"), e comparar honestamente contra um baseline
sem aumento é a única forma de saber se a intuição acima se confirma ou se o
aumento de dados ajuda por algum outro mecanismo (regularização, por
exemplo) mesmo sem criar sinal novo.

O cuidado contra vazamento
---------------------------

O gerador só pode ver os dias do **split de treino**. Nunca validação, nunca
teste — do contrário o GAN "vaza" estrutura do futuro para dentro dos dados
sintéticos que depois treinam o classificador, inflando o resultado do mesmo
jeito que embaralhar a divisão temporal inflaria.

A arquitetura
--------------

Um GAN recorrente simples (variante do RGAN/RCGAN da literatura, sem o
estágio de "embedder/supervisor" do TimeGAN completo — simplificado para caber
no orçamento de CPU deste projeto, mas com a mesma ideia central: gerador e
discriminador recorrentes operando sobre sequências multivariadas):

- **Gerador**: recebe ruído gaussiano em cada passo de tempo, processa com uma
  LSTM e projeta cada estado oculto para o espaço de features (OHLC, já em
  log-retorno e escalados pelo MESMO scaler ajustado no treino real).
- **Discriminador**: LSTM que lê a sequência inteira (real ou sintética) e
  produz um único logit — "essa sequência parece real?".

Treino adversarial padrão (BCE), com `label smoothing` leve no discriminador
para estabilizar — GANs recorrentes pequenos, com poucos dados, são notórios
por colapso de modo, e o objetivo aqui não é fidelidade artística, é gerar
sequências plausíveis o suficiente para testar a hipótese de aumento de dados.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GeradorRecorrente(nn.Module):
    """Ruído -> sequência sintética [B, comprimento, n_features]."""

    def __init__(self, n_features: int, ruido_dim: int = 16, hidden_size: int = 32):
        super().__init__()
        self.ruido_dim = ruido_dim
        self.rnn = nn.LSTM(input_size=ruido_dim, hidden_size=hidden_size, batch_first=True)
        self.saida = nn.Linear(hidden_size, n_features)

    def forward(self, batch_size: int, comprimento: int, device: torch.device) -> torch.Tensor:
        ruido = torch.randn(batch_size, comprimento, self.ruido_dim, device=device)
        h, _ = self.rnn(ruido)
        return self.saida(h)  # [B, comprimento, n_features]


class DiscriminadorRecorrente(nn.Module):
    """Sequência [B, comprimento, n_features] -> logit real/sintético."""

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.rnn = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.saida = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.rnn(x)
        return self.saida(h_n[-1]).squeeze(-1)  # [B]


def treinar_gan(
    sequencias_reais: np.ndarray,
    n_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 2e-4,
    hidden_size: int = 32,
    ruido_dim: int = 16,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[GeradorRecorrente, list[dict]]:
    """Treina o GAN recorrente em `sequencias_reais` [N, comprimento, n_features].

    `sequencias_reais` já deve estar na escala usada pelo classificador (saída
    do `Scaler.transform` ajustado SÓ no treino) — o gerador aprende a imitar
    essa escala diretamente, sem transformação adicional.
    """
    device = device or torch.device("cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    n, comprimento, n_features = sequencias_reais.shape
    dados = torch.from_numpy(sequencias_reais.astype(np.float32))

    ger = GeradorRecorrente(n_features, ruido_dim, hidden_size).to(device)
    disc = DiscriminadorRecorrente(n_features, hidden_size).to(device)
    opt_g = torch.optim.Adam(ger.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    historico = []
    n_batches = max(n // batch_size, 1)
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        perda_d_ep, perda_g_ep = 0.0, 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            if len(idx) == 0:
                continue
            reais = dados[idx].to(device)
            bs = reais.size(0)

            # --- discriminador ---
            with torch.no_grad():
                falsas = ger(bs, comprimento, device)
            opt_d.zero_grad()
            logit_real = disc(reais)
            logit_falso = disc(falsas)
            # Label smoothing leve (0.9 em vez de 1.0) para nao deixar o
            # discriminador confiante demais e estagnar o gradiente do gerador.
            alvo_real = torch.full((bs,), 0.9, device=device)
            alvo_falso = torch.zeros(bs, device=device)
            perda_d = bce(logit_real, alvo_real) + bce(logit_falso, alvo_falso)
            perda_d.backward()
            opt_d.step()

            # --- gerador ---
            opt_g.zero_grad()
            falsas = ger(bs, comprimento, device)
            logit_falso = disc(falsas)
            perda_g = bce(logit_falso, torch.ones(bs, device=device))
            perda_g.backward()
            opt_g.step()

            perda_d_ep += perda_d.item(); perda_g_ep += perda_g.item()

        historico.append({"epoch": epoch, "loss_d": perda_d_ep / n_batches, "loss_g": perda_g_ep / n_batches})
        if verbose and epoch % max(n_epochs // 10, 1) == 0:
            print(f"  epoch {epoch:>4} | loss_D={historico[-1]['loss_d']:.4f} "
                  f"loss_G={historico[-1]['loss_g']:.4f}")

    return ger, historico


def gerar_sequencias(ger: GeradorRecorrente, n: int, comprimento: int,
                      device: torch.device | None = None, seed: int = 0) -> np.ndarray:
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    ger.eval()
    with torch.no_grad():
        seqs = ger(n, comprimento, device).cpu().numpy()
    return seqs
