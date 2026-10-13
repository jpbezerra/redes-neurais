# Renomeação dos IDs de experimento

Os IDs abaixo foram renomeados (via `git mv`, histórico preservado) para
descrever o **modelo em si** em vez do ambiente ("kaggle") ou da ordem de
execução ("leva1"). Nenhuma métrica, peso ou predição foi alterada — só o
nome da pasta, o campo `model_id`, a lista `membros` dos ensembles e os
textos de `notes`/`framework` que citavam "Kaggle"/"leva 1". Esta tabela é o
único registro do nome antigo.

Os notebooks já executados (`03_hyperparam_search`, `04_feature_engineering`,
`05_modelos_avancados_kaggle`) e as tabelas que eles geraram em
`reports/tables/` **não foram editados** — são o registro de auditoria do
que rodou, e por isso ainda mostram os IDs originais. Use esta tabela para
cruzar os dois.

| ID original | ID descritivo |
|---|---|
| `gb_churn_kaggle` | `gb_churn_baseline_oversample_com_predicoes` |
| `xgb_churn_kaggle` | `xgb_churn_baseline_oversample_com_predicoes` |
| `mlp_churn_kaggle` | `mlp_churn_baseline_oversample_com_predicoes` |
| `gb_churn_optuna_leva1` | `gb_churn_optuna_20trials_ks_validacao` |
| `mlp_churn_optuna_leva1` | `mlp_churn_optuna_20trials_ks_validacao` |
| `stab_churn` | `stab_churn_dim32_depth2_heads4_oversample` |
| `kan_churn` | `kan_churn_largura8_grid3_k3_oversample` |
| `tabkan_churn` | `tabkan_churn_chebyshev_largura10_grau3_oversample` |
| `tabpfn_churn` | `tabpfn_v2_churn_contexto_oversample` |
| `tabpfn_churn_sem_duplicatas` | `tabpfn_v2_churn_contexto_original_prior_corrigido` |
| `mitra_churn` | `mitra_churn_contexto_oversample` |
| `mitra_churn_sem_duplicatas` | `mitra_churn_contexto_original_prior_corrigido` |
| `ensemble_todos_churn` | `ensemble_media_prob_10modelos_churn` |
| `ensemble_top3_val_churn` | `ensemble_media_prob_top3_validacao_churn` |

Os `*_com_predicoes` são os mesmos baselines de `*_baseline_oversample`
(métricas idênticas), re-treinados junto dos modelos avançados só para
gravar `predicoes.npz` na mesma partição e entrar nos ensembles.
