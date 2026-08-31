"""
SHAP on TGAT (the MAIN model) — feature attribution
Muhib Ul Aziz | MSc Dissertation | LSBU

Per the proposal/ethics form: BOTH SHAP and GNNExplainer are applied to TGAT.
  * SHAP (here)        -> WHICH of the 31 transaction FEATURES drove the score.
  * GNNExplainer (other)-> WHICH prior transactions/entities drove the score.

TGAT is not a tree, so TreeExplainer does not apply. We use the model-agnostic
KernelExplainer (Lundberg & Lee, 2017), which approximates Shapley values by
perturbing the input feature vector and observing the output.

Why this is exact-in-scope: a target edge's own 31 features enter the model
ONLY through the classification head (a node embedding is built from its
NEIGHBOURS' features, never the target edge's own features). So varying the
target edge's feature vector captures the full dependence of the prediction on
those features — nothing is missed by holding the neighbourhood fixed.

NOTE for the write-up: KernelExplainer on a neural graph model is heavier and
APPROXIMATE, versus TreeExplainer's exactness on XGBoost. That cost/precision
gap is itself part of the explainability comparison.
"""

import numpy as np
import torch
import shap


def _predict_fn(model, src, dst, t, nf, edge_attr, k, device):
    """Return f(F): [n,De] feature rows -> [n] fraud probabilities, for a FIXED
    target edge (src,dst,t) and fixed neighbourhood. Only the head-input varies."""
    def f(F):
        F = torch.as_tensor(F, dtype=torch.float32, device=device)
        n = F.shape[0]
        s = torch.full((n,), int(src), dtype=torch.long)
        d = torch.full((n,), int(dst), dtype=torch.long)
        tt = torch.full((n,), float(t), dtype=torch.float)
        with torch.no_grad():
            logit = model(s, d, tt, F, nf, edge_attr, k=k)
        return torch.sigmoid(logit).cpu().numpy()
    return f


def explain_tgat_shap(model, target_edges, edge_index, edge_time, edge_attr,
                      nf, feature_names, background, k=20, nsamples=100,
                      device='cuda'):
    """SHAP feature attribution for a list of target edge indices.

    target_edges : iterable of edge ids (rows) to explain
    background   : [n_bg, De] representative feature rows (e.g. train medians/sample)
    returns per-edge SHAP values + aggregated global importance.
    """
    model.eval()
    bg = np.asarray(background, dtype=np.float32)
    all_sv, explained = [], []

    for ti in target_edges:
        ti = int(ti)
        src = int(edge_index[0][ti]); dst = int(edge_index[1][ti])
        t = float(edge_time[ti])
        x = edge_attr[ti].detach().cpu().numpy().reshape(1, -1).astype(np.float32)

        f = _predict_fn(model, src, dst, t, nf, edge_attr, k, device)
        explainer = shap.KernelExplainer(f, bg)
        sv = explainer.shap_values(x, nsamples=nsamples, silent=True)
        sv = sv[0] if isinstance(sv, list) else sv
        all_sv.append(np.asarray(sv).reshape(-1))
        explained.append(ti)

    all_sv = np.vstack(all_sv)                       # [n_edges, De]
    mean_abs = np.abs(all_sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    global_imp = [(feature_names[i], float(mean_abs[i])) for i in order]

    return {'shap_values': all_sv, 'explained_edges': explained,
            'global_importance': global_imp, 'feature_names': feature_names}


def make_background(edge_attr, train_mask, n=50, seed=0):
    """A small representative background set from TRAINING edges only."""
    rng = np.random.default_rng(seed)
    idx = torch.where(train_mask)[0].cpu().numpy()
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    return edge_attr[pick].detach().cpu().numpy().astype(np.float32)


def top_features(result, k=15):
    print(f"{'rank':>4}  {'feature':<28}{'mean|SHAP|':>12}")
    for r, (name, val) in enumerate(result['global_importance'][:k], 1):
        print(f"{r:>4}  {name:<28}{val:>12.5f}")
