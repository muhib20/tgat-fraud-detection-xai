# Data

The IEEE-CIS Fraud Detection dataset is **not** included in this repository: its
use is governed by the Kaggle competition terms, and the files are too large to
distribute here.

## What to download

From https://www.kaggle.com/c/ieee-fraud-detection/data:

| File | Needed for |
|---|---|
| `train_transaction.csv` | everything — model, baselines, XAI |
| `train_identity.csv` | the exploratory node-candidate cells only |

Place them in this folder. Nothing else needs changing: the notebook and
`src/baseline_xgb.py` both resolve `data/` automatically.

To keep the data elsewhere, set an environment variable instead:

```bash
export FRAUD_DATA_DIR=/path/to/your/data     # Windows: set FRAUD_DATA_DIR=...
```

## Preprocessing applied downstream

Complete cases only (rows with both `card1` and `addr1` present), sorted by
`TransactionDT` with `TransactionID` as tie-breaker, then split chronologically
70/15/15. This yields 524,834 edges over 12,398 nodes with a 31-dimensional
edge feature vector and a test-period fraud rate of 2.28%.
