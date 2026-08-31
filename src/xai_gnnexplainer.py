"""
GNNExplainer for TGAT — Day 9
Muhib Ul Aziz | MSc Dissertation | LSBU

Faithful to Ying, Bourgeois, You, Zitnik & Leskovec (2019),
"GNNExplainer: Generating Explanations for Graph Neural Networks", NeurIPS.

Core idea: for ONE target prediction, learn soft masks that keep the model's
output while being SPARSE. The surviving neighbours/features are the explanation.

We learn:
  * w_src, w_dst : masks over the target edge's temporal neighbours (src & dst)
                   -> the RELATIONAL / TEMPORAL explanation (which prior
                      transactions made THIS one look fraudulent). This is the
                      part SHAP on a tabular model cannot give.
  * feat_mask    : mask over the target edge's own 31 features
                   -> feature attribution, directly comparable to SHAP.

Objective (per GNNExplainer): keep the predicted logit high for the predicted
class, plus L1 (sparsity) and entropy (push masks toward 0/1) penalties.
"""

import numpy as np
import torch


def explain_edge_tgat(model, src, dst, t, edge_feat, nf, edge_attr, k=20,
                      epochs=200, lr=0.05, l1=0.05, ent=0.5,
                      device='cuda', verbose=False):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)      # freeze model; optimise only the masks

    # target's neighbours (stable because sampling is deterministic)
    nbr_s, eid_s, dt_s, m_s = model.outer_neighbours(int(src), float(t), nf, k)
    nbr_d, eid_d, dt_d, m_d = model.outer_neighbours(int(dst), float(t), nf, k)
    Ks, Kd = len(nbr_s), len(nbr_d)
    De = edge_feat.shape[-1]

    # reference prediction (unmasked)
    with torch.no_grad():
        base_logit = model(torch.tensor([src]), torch.tensor([dst]),
                           torch.tensor([float(t)]), edge_feat.unsqueeze(0),
                           nf, edge_attr, k=k).item()
    target = 1.0 if base_logit > 0 else 0.0     # explain the predicted class

    # learnable mask logits
    ws = torch.zeros(Ks, device=device, requires_grad=True)
    wd = torch.zeros(Kd, device=device, requires_grad=True)
    wf = torch.zeros(De, device=device, requires_grad=True)
    opt = torch.optim.Adam([ws, wd, wf], lr=lr)

    valid_s = torch.from_numpy(m_s).to(device)
    valid_d = torch.from_numpy(m_d).to(device)
    ef = edge_feat.to(device)

    for ep in range(epochs):
        opt.zero_grad()
        msk_s = torch.sigmoid(ws) * valid_s.float()
        msk_d = torch.sigmoid(wd) * valid_d.float()
        msk_f = torch.sigmoid(wf)

        logit = model.explain_forward(int(src), int(dst), float(t), nf,
                                      edge_attr, k, msk_s, msk_d, ef, msk_f)
        p = torch.sigmoid(logit)
        # keep the predicted class (BCE toward the original decision)
        pred_loss = -(target * torch.log(p + 1e-9)
                      + (1 - target) * torch.log(1 - p + 1e-9))

        masks = torch.cat([msk_s[valid_s], msk_d[valid_d], msk_f])
        l1_loss = l1 * masks.mean()
        ent_loss = ent * (-(masks * torch.log(masks + 1e-9)
                            + (1 - masks) * torch.log(1 - masks + 1e-9))).mean()

        loss = pred_loss + l1_loss + ent_loss
        loss.backward()
        opt.step()
        if verbose and ep % 50 == 0:
            print(f"  ep {ep:3d} loss {loss.item():.4f} p {p.item():.3f}")

    with torch.no_grad():
        msk_s = (torch.sigmoid(ws) * valid_s.float()).cpu().numpy()
        msk_d = (torch.sigmoid(wd) * valid_d.float()).cpu().numpy()
        msk_f = torch.sigmoid(wf).cpu().numpy()

    def top_nbrs(nbr, eid, dt, msk, valid):
        rows = [(int(nbr[i]), int(eid[i]), float(dt[i]), float(msk[i]))
                for i in range(len(nbr)) if valid[i]]
        return sorted(rows, key=lambda r: r[3], reverse=True)

    return {
        'base_logit': base_logit,
        'predicted_class': int(target),
        'src_neighbours': top_nbrs(nbr_s, eid_s, dt_s, msk_s, m_s),
        'dst_neighbours': top_nbrs(nbr_d, eid_d, dt_d, msk_d, m_d),
        'feature_mask': msk_f,          # [De] importance in [0,1]
    }


def print_explanation(res, feature_names=None, k=5):
    cls = 'FRAUD' if res['predicted_class'] == 1 else 'legit'
    print(f"target predicted {cls} (logit {res['base_logit']:+.3f})\n")
    print(f"most influential CARD-side prior transactions (node, edge_id, dt, mask):")
    for r in res['src_neighbours'][:k]:
        print(f"  node {r[0]:6d} | edge {r[1]:7d} | dt {r[2]:.0f} | mask {r[3]:.3f}")
    print(f"most influential ADDRESS-side prior transactions:")
    for r in res['dst_neighbours'][:k]:
        print(f"  node {r[0]:6d} | edge {r[1]:7d} | dt {r[2]:.0f} | mask {r[3]:.3f}")
    fm = res['feature_mask']
    order = np.argsort(fm)[::-1][:k]
    print(f"most influential TARGET-EDGE features:")
    for i in order:
        nm = feature_names[i] if feature_names else f'feat_{i}'
        print(f"  {nm:<24} mask {fm[i]:.3f}")
