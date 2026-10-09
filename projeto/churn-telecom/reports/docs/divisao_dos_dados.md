# Divisão e preparação dos dados

Este documento descreve como o dataset de churn será dividido e preparado
antes do treinamento de qualquer modelo. O objetivo é permitir a conferência,
antes de começar os experimentos, de que não haverá vazamento de informação
entre treino, validação e teste.

Os números abaixo vêm da divisão principal (semente aleatória 0). Eles
dependem apenas dos dados e do procedimento de divisão, não de nenhum modelo.

## 1. Dataset de partida

| Item | Valor |
|---|---|
| Linhas (clientes) | 7.043 |
| Colunas | 21 (customerID, 19 variáveis explicativas e Churn) |
| customerID repetido? | Não, cada cliente aparece uma única vez |
| Churn = No | 5.174 (73,5%) |
| Churn = Yes | 1.869 (26,5%), classe minoritária |

A coluna customerID é descartada antes da modelagem. Nenhuma outra variável
é removida.

## 2. Procedimento de divisão

A divisão é feita em quatro passos.

**Passo 1. Separar por classe.** O dataset é dividido em dois grupos:
clientes que saíram (Churn = Yes) e clientes que ficaram (Churn = No).

**Passo 2. Dividir cada classe em 50% / 25% / 25%.** Cada grupo é embaralhado
e cortado em treino, validação e teste. Assim, a proporção de churn é a mesma
nas três partições.

| Classe | Total | Treino (50%) | Validação (25%) | Teste (25%) |
|---|---:|---:|---:|---:|
| Churn = No | 5.174 | 2.587 | 1.294 | 1.293 |
| Churn = Yes | 1.869 | 934 | 467 | 468 |
| Total | 7.043 | 3.521 | 1.761 | 1.761 |

**Passo 3. Balancear treino e validação por reamostragem com repetição.**
Dentro do treino, os clientes Yes são sorteados com repetição até igualar o
número de clientes No do treino (934 → 2.587). O mesmo é feito, de forma
independente, dentro da validação (467 → 1.294).

- O sorteio do treino usa apenas clientes do treino.
- O sorteio da validação usa apenas clientes da validação.
- Nenhuma cópia passa de uma partição para outra.

**O teste não é balanceado.** Ele mantém a proporção real de churn (26,6%),
e cada cliente aparece nele uma única vez. Se o teste fosse balanceado por
cópia, as métricas ficariam infladas (principalmente recall e F1) e não
representariam a população real.

**Passo 4. Recombinar e embaralhar.** As duas classes de cada partição são
juntadas e embaralhadas.

## 3. Tamanho final de cada partição

| Partição | Linhas | Clientes distintos | Churn = No | Churn = Yes | % churn |
|---|---:|---:|---:|---:|---:|
| Treino (balanceado) | 5.174 | 3.461 | 2.587 | 2.587 | 50,0% |
| Validação (balanceada) | 2.588 | 1.726 | 1.294 | 1.294 | 50,0% |
| Teste | 1.761 | 1.761 | 1.293 | 468 | 26,6% |

Leitura da tabela:

- Treino e validação têm mais linhas do que clientes distintos porque os
  clientes Yes foram copiados. Como o sorteio é com repetição, alguns
  aparecem várias vezes e outros nenhuma: 874 dos 934 clientes Yes do treino
  foram sorteados pelo menos uma vez, e 432 dos 467 na validação.
- No teste, o número de linhas é igual ao de clientes distintos (1.761). Não
  há nenhuma cópia.
- Antes do balanceamento, treino, validação e teste somam 3.521 + 1.761 +
  1.761 = 7.043, exatamente o dataset inteiro.

Também são guardadas as versões de treino e validação antes do passo 3 (sem
cópias, com a proporção real de churn). Elas servem para os modelos que
tratam o desbalanceamento com peso de classe, em vez de cópias, e para
modelos que aprendem diretamente a partir dos exemplos de treino (como o
TabPFN), nos quais linhas duplicadas atrapalham.

## 4. Checagens de vazamento

Toda vez que os dados forem divididos, antes de qualquer treinamento, o
pipeline confere automaticamente:

| Checagem | Resultado esperado |
|---|---|
| Algum cliente aparece em treino e validação ao mesmo tempo? | Não |
| Algum cliente aparece em treino e teste ao mesmo tempo? | Não |
| Algum cliente aparece em validação e teste ao mesmo tempo? | Não |
| As três partições juntas cobrem o dataset inteiro, sem sobreposição? | Sim (7.043 clientes) |

As checagens comparam os clientes originais, antes das cópias do passo 3. Se
qualquer uma falhar, a execução é interrompida com erro.

## 5. Pré-processamento

Regra geral: tudo o que depende de alguma estatística dos dados é calculado
somente no treino e depois aplicado, sem recalcular, na validação e no teste.

| Etapa | O que faz | Usa estatística dos dados? | Calculado em |
|---|---|---|---|
| Remoção do customerID | Descarta o identificador | Não | Não se aplica |
| Conversão de TotalCharges | Texto para número | Não | Não se aplica |
| Preenchimento de TotalCharges | Os 11 valores em branco são todos de clientes com tenure = 0 e recebem 0, porque esses clientes ainda não foram cobrados | Não (regra fixa) | Não se aplica |
| Codificação binária | gender, Partner, Dependents, PhoneService e PaperlessBilling viram 0/1 | Sim (quais valores existem) | Só no treino |
| One-hot | 10 variáveis categóricas com 3 ou mais categorias | Sim (lista de categorias) | Só no treino |
| Padronização (z-score) | tenure, MonthlyCharges, TotalCharges e SeniorCitizen | Sim (média e desvio-padrão) | Só no treino |
| Codificação do alvo | Churn: Yes = 1, No = 0 | Não | Não se aplica |

Detalhes:

- Se aparecer na validação ou no teste uma categoria que não existe no
  treino, ela é codificada como zero em todas as colunas daquela variável.
  Nenhuma coluna nova é criada a partir de validação ou teste.
- Após a codificação, a entrada dos modelos tem 40 colunas, iguais nas três
  partições.
- Qualquer variável derivada que venha a ser testada seguirá a mesma regra:
  se precisar de alguma estatística (por exemplo, para padronizar), ela será
  calculada só no treino.

## 6. Como cada partição será usada

| Partição | Será usada para | Não será usada para |
|---|---|---|
| Treino | Ajustar os parâmetros dos modelos e o pré-processamento | Escolher hiperparâmetros ou reportar resultados |
| Validação | Parada antecipada (early stopping), busca de hiperparâmetros, escolha de variáveis derivadas e escolha de modelos para combinar | Ajustar parâmetros dos modelos ou o pré-processamento |
| Teste | Calcular as métricas finais, uma única vez por modelo, depois de todas as escolhas feitas | Qualquer decisão de modelagem |

Como o teste não participa de nenhuma escolha, é esperado que alguns ganhos
observados na validação não se repitam no teste. Isso será reportado como
resultado, e o teste não será usado para refazer escolhas.

## 7. Repetição com várias divisões

O teste terá cerca de 470 clientes com churn. Com esse tamanho, a métrica de
uma única divisão varia bastante de acordo com quais clientes caíram em cada
partição. Por isso, o procedimento completo será repetido com 5 sementes
diferentes (0 a 4). Para cada semente:

1. Nova divisão do zero, seguindo as seções 2 e 3.
2. As checagens de vazamento da seção 4.
3. Pré-processamento recalculado do zero, só no treino daquela divisão.
4. Treino, validação e teste usados como na seção 6.

Nada é reaproveitado de uma semente para outra: nem pré-processamento, nem
modelo, nem escolha de hiperparâmetros. O resultado final de cada modelo
será a média e o desvio-padrão entre as 5 divisões.

## 8. Pontos de atenção

Nenhum destes pontos é vazamento entre partições. Eles estão registrados
para facilitar a revisão.

**Validação também é balanceada por cópia.** A validação tem clientes Yes
repetidos (sempre cópias de clientes da própria validação). Na prática, cada
cliente Yes da validação pesa cerca de 2,8 vezes mais na parada antecipada e
nas métricas de validação. Isso muda a escala da validação, mas não traz
nenhuma informação do teste.

**Clientes diferentes com dados idênticos.** Há 73 clientes, todos com
tenure = 1, cujas 19 variáveis explicativas são idênticas às de pelo menos
outro cliente, formando 33 grupos. São pessoas diferentes (customerID
distintos), tipicamente clientes recém-chegados com plano mínimo. Pela
divisão aleatória, 17 desses grupos têm clientes em partições diferentes, e 8
deles incluem o teste. Como são clientes distintos, não é vazamento, e o
efeito possível é muito pequeno (poucas linhas entre as 1.761 do teste).

**Análise exploratória no dataset inteiro.** A análise exploratória foi
feita com as 7.043 linhas, antes da divisão, de forma apenas descritiva.
Ela pode sugerir variáveis derivadas, mas a decisão de usar ou não cada uma
será tomada só com base na validação.

**Preenchimento de TotalCharges.** Os 11 valores em branco foram preenchidos
com 0 por uma regra lógica (o cliente ainda não foi cobrado), sem usar média,
mediana ou qualquer outra estatística de nenhuma partição.
