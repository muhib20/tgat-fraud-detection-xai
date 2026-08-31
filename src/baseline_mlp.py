"""
MLP baseline for fraud detection — Day 7
Muhib Ul Aziz | MSc Dissertation | LSBU

A plain feed-forward net on the SAME 31 edge features, NO graph, NO time.
Same protocol as everything else: class-weighted BCE (pos_weight=38.74),
model selection on validation PR-AUC, threshold frozen on val, multi-seed.

Purpose in the comparison table: separates "neural vs trees" from
"graph vs no-graph". If MLP ~ XGBoost, the story is simply that this data
suits any strong tabular learner. If MLP << XGBoost, trees specifically win.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_recall_curve)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _metrics(y, p, threshold=None):
    pr = average_precision_score(y, p)
    roc = roc_auc_score(y, p)
    if threshold is None:
        prec, rec, thr = precision_recall_curve(y, p)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        threshold = float(thr[np.nanargmax(f1s[:-1])]) if len(thr) else 0.5
    f1 = f1_score(y, (p >= threshold).astype(int), zero_division=0)
    return {'pr_auc': pr, 'roc_auc': roc, 'f1': f1, 'threshold': threshold}


def train_mlp_one(X, y, train_mask, val_mask, test_mask,
                  pos_weight=38.74, lr=1e-3, weight_decay=1e-5,
                  epochs=60, batch_size=2048, patience=8, seed=0,
                  device='cuda', verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    tr, va, te = (torch.where(m)[0] for m in (train_mask, val_mask, test_mask))

    model = MLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    @torch.no_grad()
    def ev(idx, threshold=None):
        model.eval()
        p = torch.sigmoid(model(X[idx].to(device))).cpu().numpy()
        return _metrics(y[idx].numpy(), p, threshold)

    best, best_state, best_thr, wait = -1, None, 0.5, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            loss = loss_fn(model(X[b].to(device)), y[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        m = ev(va)
        if verbose:
            print(f"  ep {ep:2d} val PR-AUC {m['pr_auc']:.4f}")
        if m['pr_auc'] > best:
            best, best_thr = m['pr_auc'], m['threshold']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return ev(te, threshold=best_thr)


def run_mlp(X, y, train_mask, val_mask, test_mask,
            seeds=(0, 1, 2, 3, 4), **kw):
    rows = []
    for s in seeds:
        m = train_mlp_one(X, y, train_mask, val_mask, test_mask, seed=s, **kw)
        rows.append(m)
        print(f"  [MLP] seed {s}: PR-AUC {m['pr_auc']:.4f} | "
              f"ROC {m['roc_auc']:.4f} | F1 {m['f1']:.4f}")
    def ms(key):
        v = np.array([r[key] for r in rows]); return v.mean(), v.std()
    out = {k: ms(k) for k in ('pr_auc', 'roc_auc', 'f1')}
    print(f"  [MLP] MEAN: PR-AUC {out['pr_auc'][0]:.4f}±{out['pr_auc'][1]:.4f} | "
          f"ROC {out['roc_auc'][0]:.4f} | F1 {out['f1'][0]:.4f}")
    return out, rows
