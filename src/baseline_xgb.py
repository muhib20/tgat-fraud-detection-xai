"""
XGBoost baseline for fraud detection — Day 6
Muhib Ul Aziz | MSc Dissertation | LSBU

Purpose: give the TGAT number (PR-AUC 0.135 +/- 0.012) something to mean.

Two comparisons, both evaluated on the SAME complete-case test period as TGAT:

  1. XGB-matched : the SAME 31 edge features TGAT used, with NO graph structure.
                   -> isolates the value added by the graph + temporal attention.
                   If TGAT > XGB-matched, the graph modelling is doing real work.

  2. XGB-rich    : a fuller tabular feature set (what a practitioner would
                   actually build). -> the practical strong baseline.

A SEPARATE all-rows variant (retaining the fraud-dense missing-addr1 rows plus
an addr1_missing flag) is Day 7 — that is the informative-missingness finding,
kept apart so it does not contaminate the fair head-to-head here.

Class imbalance handled with scale_pos_weight (XGBoost's equivalent of our
class-weighted loss) — consistent with the 'no SMOTE' decision.
"""

import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_recall_curve)


def _metrics(y, p, threshold=None):
    pr = average_precision_score(y, p)
    roc = roc_auc_score(y, p)
    if threshold is None:
        prec, rec, thr = precision_recall_curve(y, p)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        threshold = float(thr[np.nanargmax(f1s[:-1])]) if len(thr) else 0.5
    f1 = f1_score(y, (p >= threshold).astype(int), zero_division=0)
    return {'pr_auc': pr, 'roc_auc': roc, 'f1': f1, 'threshold': threshold}


def load_complete_case(path):
    """Rebuild the EXACT complete-case, time-sorted frame TGAT used.

    Same filter (drop missing card1/addr1), same sort key -> same rows,
    same chronological 70/15/15 split. This is what makes the comparison fair.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=['card1', 'addr1'])
    # deterministic order even when TransactionDT has ties
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    split = np.array(['train'] * n, dtype=object)
    split[i_tr:i_va] = 'val'
    split[i_va:] = 'test'
    df['_split'] = split
    return df


def _matched_features(df):
    """The same 31-d edge feature space TGAT consumed (pre-scaling; trees don't
    need scaling). log1p on amount; one-hot ProductCD/card4/card6; C1..C14."""
    cols_c = [f'C{i}' for i in range(1, 15)]
    X = df[['TransactionAmt'] + cols_c].copy()
    X['TransactionAmt'] = np.log1p(X['TransactionAmt'])
    for c in cols_c:
        X[c] = X[c].fillna(-1)
    oh = pd.get_dummies(df[['ProductCD', 'card4', 'card6']],
                        columns=['ProductCD', 'card4', 'card6'], dummy_na=True)
    return pd.concat([X, oh], axis=1)


def _rich_features(df):
    """A fuller, still-honest tabular set: amounts, C/D counters, a handful of
    categoricals. NO target leakage, NO aggregate stats over the full timeline."""
    num = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
           'dist1', 'dist2'] + [f'C{i}' for i in range(1, 15)] \
        + [f'D{i}' for i in range(1, 16)]
    num = [c for c in num if c in df.columns]
    X = df[num].copy()
    X['TransactionAmt'] = np.log1p(X['TransactionAmt'])
    X = X.fillna(-1)
    cat = [c for c in ['ProductCD', 'card4', 'card6',
                       'P_emaildomain', 'R_emaildomain'] if c in df.columns]
    oh = pd.get_dummies(df[cat], columns=cat, dummy_na=True)
    return pd.concat([X, oh], axis=1)


def run_xgb(df, feat_fn, name, seeds=(0, 1, 2, 3, 4), pos_weight=38.74):
    tr = df['_split'] == 'train'
    va = df['_split'] == 'val'
    te = df['_split'] == 'test'
    X = feat_fn(df).astype(np.float32).values
    y = df['isFraud'].values

    rows = []
    for s in seeds:
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,            # imbalance handling
            eval_metric='aucpr', tree_method='hist',
            random_state=s, n_jobs=-1)
        clf.fit(X[tr], y[tr],
                eval_set=[(X[va], y[va])], verbose=False)
        p_va = clf.predict_proba(X[va])[:, 1]
        p_te = clf.predict_proba(X[te])[:, 1]
        thr = _metrics(y[va], p_va)['threshold']   # threshold frozen on val
        m = _metrics(y[te], p_te, threshold=thr)
        rows.append(m)
        print(f"  [{name}] seed {s}: PR-AUC {m['pr_auc']:.4f} | "
              f"ROC {m['roc_auc']:.4f} | F1 {m['f1']:.4f}")

    def ms(key):
        v = np.array([r[key] for r in rows]); return v.mean(), v.std()
    out = {k: ms(k) for k in ('pr_auc', 'roc_auc', 'f1')}
    print(f"  [{name}] MEAN: PR-AUC {out['pr_auc'][0]:.4f}±{out['pr_auc'][1]:.4f} | "
          f"ROC {out['roc_auc'][0]:.4f} | F1 {out['f1'][0]:.4f}")
    return out, rows


if __name__ == '__main__':
    # Point FRAUD_DATA_DIR at the folder holding train_transaction.csv,
    # or drop the CSV into ./data. See README for the download link.
    DATA_DIR = os.environ.get('FRAUD_DATA_DIR', 'data')
    PATH = os.path.join(DATA_DIR, 'train_transaction.csv')
    df = load_complete_case(PATH)
    print(f"complete-case rows: {len(df)} | "
          f"test fraud rate: {df[df._split=='test']['isFraud'].mean()*100:.2f}%\n")
    print("XGB-matched (same 31 features as TGAT, no graph):")
    run_xgb(df, _matched_features, 'matched')
    print("\nXGB-rich (fuller tabular features):")
    run_xgb(df, _rich_features, 'rich')
