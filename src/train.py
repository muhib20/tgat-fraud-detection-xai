"""
Training + evaluation for TGAT temporal edge classification — Day 5
Muhib Ul Aziz | MSc Dissertation | LSBU

Locked decisions honoured here:
  * class-weighted loss (pos_weight ~= 38.74), NO SMOTE
  * PR-AUC = primary metric; AUC-ROC + F1 supporting
  * model selection on VALIDATION PR-AUC (never on test)
  * threshold chosen on validation, then applied ONCE to test
  * multi-seed runner -> report mean +/- std
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_recall_curve)
from tgat_model import TGAT, NeighborFinder


# ----------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, idx, edge_index, edge_time, edge_attr, edge_label,
             nf, k, batch_size=1024, device='cuda', threshold=None):
    """Return metrics dict + raw probabilities over the edges in `idx`."""
    model.eval()
    probs = []
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        logit = model(edge_index[0][b], edge_index[1][b], edge_time[b],
                      edge_attr[b], nf, edge_attr, k=k)
        probs.append(torch.sigmoid(logit).cpu())
    probs = torch.cat(probs).numpy()
    y = edge_label[idx].cpu().numpy()

    pr_auc = average_precision_score(y, probs)      # PRIMARY (threshold-free)
    roc = roc_auc_score(y, probs)                   # threshold-free

    # F1 needs a threshold. If none supplied, pick the best-F1 point on THIS set
    # (used on validation). On test we pass the frozen validation threshold.
    if threshold is None:
        prec, rec, thr = precision_recall_curve(y, probs)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best = np.nanargmax(f1s[:-1]) if len(thr) else 0
        threshold = float(thr[best]) if len(thr) else 0.5
    f1 = f1_score(y, (probs >= threshold).astype(int), zero_division=0)

    return {'pr_auc': pr_auc, 'roc_auc': roc, 'f1': f1,
            'threshold': threshold}, probs


# ----------------------------------------------------------------------
# ONE TRAINING RUN (single seed)
# ----------------------------------------------------------------------
def train_one(edge_index, edge_time, edge_attr, edge_label,
              train_mask, val_mask, test_mask, num_nodes,
              pos_weight=38.74, k=20, node_dim=64, time_dim=32,
              n_layers=2, n_heads=2, lr=1e-3, weight_decay=1e-5,
              epochs=25, batch_size=512, patience=5, seed=0,
              device='cuda', verbose=True):

    torch.manual_seed(seed); np.random.seed(seed)

    nf = NeighborFinder(edge_index, edge_time, num_nodes)
    model = TGAT(num_nodes, edge_attr.shape[1], node_dim, time_dim,
                 n_layers, n_heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))

    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask)[0]

    best_val, best_state, best_thr, wait = -1.0, None, 0.5, 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]   # shuffle order only
        total = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            logit = model(edge_index[0][b], edge_index[1][b], edge_time[b],
                          edge_attr[b], nf, edge_attr, k=k)
            loss = loss_fn(logit, edge_label[b].float().to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        train_loss = total / len(perm)

        val_m, _ = evaluate(model, val_idx, edge_index, edge_time, edge_attr,
                            edge_label, nf, k, device=device)
        history.append({'epoch': ep, 'train_loss': train_loss, **val_m})
        if verbose:
            print(f"  ep {ep:2d} | loss {train_loss:.4f} | "
                  f"val PR-AUC {val_m['pr_auc']:.4f} | "
                  f"val ROC {val_m['roc_auc']:.4f} | val F1 {val_m['f1']:.4f}")

        # model selection on validation PR-AUC (primary metric)
        if val_m['pr_auc'] > best_val:
            best_val = val_m['pr_auc']
            best_thr = val_m['threshold']
            best_state = {kk: v.detach().cpu().clone()
                          for kk, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep} (best val PR-AUC {best_val:.4f})")
                break

    # restore best, evaluate ONCE on test with the frozen validation threshold
    model.load_state_dict(best_state)
    test_m, test_probs = evaluate(model, test_idx, edge_index, edge_time,
                                  edge_attr, edge_label, nf, k,
                                  device=device, threshold=best_thr)
    return {'seed': seed, 'best_val_pr_auc': best_val, 'test': test_m,
            'history': history}, model, test_probs


# ----------------------------------------------------------------------
# MULTI-SEED RUNNER  -> mean +/- std  (the RME statistical-rigour fix)
# ----------------------------------------------------------------------
def run_seeds(seeds=(0, 1, 2, 3, 4), **kw):
    rows = []
    for s in seeds:
        print(f"\n=== seed {s} ===")
        res, _, _ = train_one(seed=s, **kw)
        t = res['test']
        rows.append(t)
        print(f"  -> TEST PR-AUC {t['pr_auc']:.4f} | ROC {t['roc_auc']:.4f} "
              f"| F1 {t['f1']:.4f}")

    def ms(key):
        v = np.array([r[key] for r in rows])
        return v.mean(), v.std()

    print("\n" + "=" * 46)
    print(f"{'metric':10} {'mean':>10} {'std':>10}")
    for key in ('pr_auc', 'roc_auc', 'f1'):
        m, s = ms(key)
        print(f"{key:10} {m:10.4f} {s:10.4f}")
    return rows
