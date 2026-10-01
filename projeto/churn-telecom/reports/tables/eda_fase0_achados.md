# Fase 0 — achados da EDA (dataset real, confirmados empiricamente)

Fonte: mirror público do IBM Telco Customer Churn (mesmo schema do dataset
Kaggle `customers-churned-in-telecom-services` citado no enunciado) —
`data/telco_customer_churn.csv`.

## Confirmações contra a especificação do enunciado

- **Shape: 7.043 linhas × 21 colunas** — bate com os ~7 mil registros e 19
  variáveis independentes citados no enunciado (21 = 19 + `customerID` + `Churn`).
- **`customerID` existe e é único** (0 nulos, 0 duplicatas) — é o identificador
  a remover antes do primeiro modelo, conforme regra do enunciado de não
  descartar variáveis no primeiro modelo exceto identificadores. Nenhuma
  outra coluna deve ser removida nesta fase.
- **`TotalCharges` é `object` (string), não numérica** — confirma a pegadinha
  antecipada. **11 linhas** têm `TotalCharges` em branco (`" "`), todas
  com `tenure=0` e `Churn=No` — clientes novíssimos, sem cobrança acumulada
  ainda. Nenhum `NaN` explícito do pandas em nenhuma coluna (os "ausentes"
  são strings em branco, não `NaN` — então `.isna()` sozinho não pega).
- **Proporção do alvo: 73.46% `No` / 26.54% `Yes`** (5.174 / 1.869) — confirma
  o desbalanceamento moderado (~2,8:1), motivando a estratégia de
  reamostragem exigida pelo enunciado.
- **0 linhas duplicadas.**

## Achados adicionais (relevantes para o pré-processamento)

- **Cardinalidade das categóricas**: a maioria é binária (`gender`, `Partner`,
  `Dependents`, `PhoneService`, `PaperlessBilling`) ou ternária com o padrão
  "No / Yes / No internet(ou phone) service" (`MultipleLines`,
  `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
  `StreamingTV`, `StreamingMovies`). `Contract` tem 3 categorias, `PaymentMethod`
  tem 4. Nenhuma categórica de alta cardinalidade (bom para one-hot direto,
  sem necessidade de target encoding).
- **Consistência `TotalCharges ≈ MonthlyCharges × tenure`**: desvio absoluto
  médio de 45.09 (mediano 28.65), e **33% das linhas válidas desviam mais de
  50** do valor esperado pela multiplicação simples. Isso é esperado (planos
  mudam ao longo do tempo, promoções, cobrança pro-rata) e **não é um
  indício de erro no dado** — mas significa que `TotalCharges` carrega
  informação própria (histórico real de cobrança), não é redundante com
  `MonthlyCharges × tenure`. Vale manter as duas colunas e considerar a razão
  `TotalCharges / (tenure + 1)` como feature derivada mais adiante, mas não
  substituir uma pela outra.
- **`tenure`**: varia de 0 a 72 meses (não há valor negativo ou fora de
  faixa), média 32.37, mediana 29 — sem outlier óbvio.
- **`MonthlyCharges`**: 18.25 a 118.75, sem valor negativo ou implausível.
- **`SeniorCitizen`** já vem como 0/1 numérico (não precisa de encoding).

## Decisão de pré-processamento recomendada para a Fase 1 (a confirmar ao implementar)

- Tratar as 11 linhas de `TotalCharges` em branco como **imputação por 0**
  (cliente com `tenure=0` nunca foi cobrado — é o valor logicamente correto,
  não uma estimativa), documentando como etapa de dados ausentes com
  checkpoint de desempenho antes/depois.
- Converter `TotalCharges` para `float64` nesse mesmo passo.
- Nenhuma remoção de outlier parece necessária nesta primeira passada — não
  há valores fora de faixa plausível em nenhuma coluna numérica.

Tabelas completas: `eda_fase0_resumo_colunas.csv` (dtype/nulos/cardinalidade
por coluna) e `eda_fase0_proporcao_churn.csv` (contagem e proporção do alvo).
