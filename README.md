# Temporal Graph Attention Networks for Financial Fraud Detection with Explainable AI: A Time-Series Approach

Code and results for my MSc Data Science dissertation at London South Bank University, supervised by Prof. Daqing Chen.

The project models card-not-present fraud as **temporal edge classification**: cards and addresses are nodes, transactions are timestamped edges. A Temporal Graph Attention Network (TGAT) is compared against matched non-graph baselines, and both SHAP and GNNExplainer are applied to the TGAT and validated for faithfulness and stability.

---

## Project Architecture

The diagram below illustrates the complete pipeline, from raw transaction data through temporal graph construction to explanation validation:

![Architecture diagram](docs/architecture.png)

---

## Headline result

TGAT did **not** outperform the tabular baselines, and a controlled time ablation showed the temporal component contributed nothing measurable on this graph. This is reported as a negative result with a diagnosis, not as a failed implementation.

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| TGAT (temporal) | 0.135 ± 0.012 | 0.738 ± 0.009 | 0.201 ± 0.016 |
| GNN, time-ablated | 0.139 ± 0.008 | 0.739 ± 0.006 | 0.211 ± 0.010 |
| MLP (same 31 features) | 0.204 ± 0.008 | 0.782 ± 0.007 | 0.225 ± 0.019 |
| XGBoost, matched features | 0.209 ± 0.002 | 0.770 ± 0.003 | 0.247 ± 0.004 |
| XGBoost, rich features | 0.244 ± 0.004 | 0.848 ± 0.002 | 0.271 ± 0.006 |

Mean ± standard deviation over five seeds (0–4). PR-AUC is the primary metric given a test-period fraud rate of 2.28%. Model selection used validation PR-AUC; the decision threshold was frozen on validation and applied once to test.

**Why the graph does not help here.** The card side is too sparse to support temporal attention (median card node: 4 transactions), while the address side is not a meaningful entity at all — `addr1` takes only 332 distinct values across 524,834 edges, so each address node mixes thousands of unrelated users. Cold start is *not* the explanation: only 1.22% of test edges touch a node unseen during training.

---

## Repository layout

```
.
├── src/                     model, training, baselines, explainers
├── notebooks/               end-to-end analysis notebook
├── results/                 saved JSON outputs behind every reported number
├── models/                  trained TGAT weights
└── data/                    dataset goes here (not tracked)
```

| File | Purpose |
|---|---|
| `src/tgat_model.py` | TGAT: Bochner time encoder, causal neighbour finder, attention layer, edge classifier. Includes `use_time` for the ablation and the masked forward pass used by GNNExplainer. |
| `src/train.py` | Training loop, evaluation, multi-seed runner. |
| `src/baseline_xgb.py` | XGBoost baselines (matched and rich feature sets) on the identical complete-case split. |
| `src/baseline_mlp.py` | Feed-forward baseline on the same 31 edge features, no graph, no time. |
| `src/xai_shap_tgat.py` | SHAP (KernelExplainer) feature attribution on the TGAT. |
| `src/xai_gnnexplainer.py` | GNNExplainer adapted to temporal edge classification. |
| `notebooks/Temporal_Graph_Attention_Networks.ipynb` | Data preparation, graph construction, experiments, XAI, validation. |
| `models/tgat_trained.pt` | Trained TGAT weights used for all XAI results. |
| `results/*.json` | Saved outputs backing every number in the dissertation. |

---

## Data

The IEEE-CIS Fraud Detection dataset is **not included** in this repository, as its use is governed by the Kaggle competition terms.

1. Download `train_transaction.csv` from https://www.kaggle.com/c/ieee-fraud-detection/data
2. Drop it into the `data/` folder. No code changes are needed — the first cell of the
   notebook resolves every path automatically.
3. To keep the data elsewhere, set `FRAUD_DATA_DIR` instead. See `data/README.md`.

Preprocessing retains complete cases only (rows with both `card1` and `addr1` present), sorted by `TransactionDT`, then split chronologically 70/15/15. This yields 524,834 edges over 12,398 nodes with a 31-dimensional edge feature vector.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch numpy pandas scikit-learn xgboost shap matplotlib
```

A CUDA-capable GPU is assumed (`device='cuda'` throughout). Pass `device='cpu'` to the training and explanation functions to run on CPU, considerably more slowly.

---

## Reproducing the results

Run the notebook top to bottom (start with the configuration cell, which puts
`src/` on the import path), or use the modules directly from the repository root:

```python
# TGAT, five seeds
from train import run_seeds
rows = run_seeds(edge_index=edge_index, edge_time=edge_time, edge_attr=edge_attr,
                 edge_label=edge_label, train_mask=train_mask, val_mask=val_mask,
                 test_mask=test_mask, num_nodes=num_nodes, k=20)

# Time ablation: identical architecture and parameter count, temporal signal removed
rows_static = run_seeds(..., use_time=False)
```

```bash
python src/baseline_xgb.py      # both XGBoost baselines
```

Class imbalance is handled by a weighted loss (`pos_weight = 38.74`) rather than SMOTE, so that the feature distribution seen by SHAP remains the real one.

---

## Explainability

Both explainers are applied to the TGAT itself, addressing different questions:

- **SHAP** (KernelExplainer, since TGAT is not a tree) attributes a prediction to the target transaction's own 31 features. `ProductCD_W`, `ProductCD_H`, `C13` and `TransactionAmt` together account for roughly half the total mean|SHAP| value.
- **GNNExplainer** learns sparse masks over the target edge's temporal neighbourhood, identifying *which prior transactions* drove the score — something tabular SHAP cannot provide.

### Validation

- **Correctness.** The masked forward pass reproduces the unmasked logit to six decimal places (`base` vs `allkeep` in `xai_validation.json`), confirming `explain_forward` is faithful to the standard forward pass.
- **Comprehensiveness against a random baseline.** Removing GNNExplainer's top-5 neighbours moves the logit 1.1–5.6× further than removing five random neighbours (mean ≈ 2.7×), so the explainer identifies genuinely influential neighbours.
- **Stability.** Jaccard overlap across repeated runs is 1.0, but this is exact *by construction*: deterministic most-recent-k neighbour sampling combined with zero-initialised mask logits leaves no stochasticity in the procedure. The figure therefore verifies determinism, not robustness to sampling noise, and should be read as such.
- **A caveat on scope.** Ablating the entire neighbourhood *increases* the fraud logit in three of five explained edges, and all five remain strongly fraud-positive under every ablation. The neighbourhood is not what drives these predictions; the transaction's own features are, acting through the classification head. This is consistent with both the time ablation and the SHAP attributions.

---

## Known limitations

- The neighbour finder is constructed over all edges. Sampling is strictly causal (an event at time *t* never sees anything at time ≥ *t*), so no future information leaks, but a test-period transaction may attend to earlier test-period transactions. This is transductive and realistic for streaming deployment, though it is not an inductive evaluation.
- `explain_edge_tgat` sets `requires_grad_(False)` on all model parameters and does not restore them. Training after running the explainer in the same session will silently have no effect; restart the kernel or reload the checkpoint first.
- SHAP on a neural graph model is approximate and computationally heavy, unlike TreeExplainer's exact attributions on XGBoost. That cost–precision gap is itself part of the explainability comparison.

---

## Author

Muhib Ul Aziz — MSc Data Science, London South Bank University, 2026.

Code released for academic assessment and review. The IEEE-CIS dataset remains subject to its original terms.
