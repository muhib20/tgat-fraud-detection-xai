"""
TGAT for temporal edge classification (fraud detection) — Day 4
Muhib Ul Aziz | MSc Dissertation | LSBU

Graph design (Option A): cards & addresses are NODES, transactions are
TIMESTAMPED EDGES. Task = temporal edge classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. TIME ENCODER  (Bochner / Fourier time encoding)
# ----------------------------------------------------------------------
class TimeEncoder(nn.Module):
    """Maps a scalar time-difference dt -> a d-dimensional vector.

    phi(dt) = cos(w * dt + b), with w and b LEARNABLE.
    Frequencies are initialised across many orders of magnitude so the model
    can represent both 'seconds ago' and 'weeks ago'.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        # geometric init: 1, 1/10, 1/100, ... -> multi-scale temporal resolution
        self.w.weight = nn.Parameter(
            torch.from_numpy(1.0 / 10 ** np.linspace(0, 9, dim))
            .float().reshape(dim, 1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, dt):
        # dt: [...]  ->  [..., dim]
        return torch.cos(self.w(dt.unsqueeze(-1)))


# ----------------------------------------------------------------------
# 2. TEMPORAL NEIGHBOUR FINDER
# ----------------------------------------------------------------------
class NeighborFinder:
    """For node u at time t, return its K most recent neighbours with time < t.

    STRICTLY causal: an event at time t never sees anything at time >= t.
    This is what keeps the model honest AND keeps memory bounded.
    """

    def __init__(self, edge_index, edge_time, num_nodes):
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        t = edge_time.numpy()
        eid = np.arange(len(t))

        # graph is undirected for message passing: store both directions
        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])
        e = np.concatenate([eid, eid])
        tt = np.concatenate([t, t])

        order = np.lexsort((tt, u))          # sort by node, then by time
        self.u = u[order]
        self.nbr = v[order]
        self.eid = e[order]
        self.t = tt[order]

        # CSR-style offsets: adjacency of node i lives in [off[i], off[i+1])
        self.off = np.searchsorted(self.u, np.arange(num_nodes + 1))
        self.num_nodes = num_nodes

        # --- composite key enabling ONE global vectorised searchsorted ---
        # key = node * M + time, with M > max(time). Because rows are sorted by
        # (node, time), `key` is globally sorted -> we can binary-search the whole
        # batch at once instead of looping node by node.
        self.ti = np.rint(self.t).astype(np.int64)
        assert np.allclose(self.t, self.ti), \
            "Timestamps must be integers for the vectorised sampler."
        self.M = int(self.ti.max()) + 2
        self.key = self.u.astype(np.int64) * self.M + self.ti

    def sample(self, nodes, times, k):
        """Vectorised. nodes:[B] int64, times:[B] float -> 4 arrays of [B,k]."""
        nodes = np.asarray(nodes, dtype=np.int64)
        t_q = np.asarray(times, dtype=np.float64)
        ti_q = np.rint(t_q).astype(np.int64)

        # first position with (node, time) >= (node, t_q)  -> everything before is
        # strictly in the past for THIS node
        cut = np.searchsorted(self.key, nodes * self.M + ti_q, side='left')
        lo = np.maximum(self.off[nodes], cut - k)      # keep the K most recent
        cnt = cut - lo                                 # how many are valid, 0..k

        ar = np.arange(k)
        mask = ar[None, :] < cnt[:, None]              # [B,k]
        idx = np.where(mask, lo[:, None] + ar[None, :], 0)

        nbr = np.where(mask, self.nbr[idx], 0).astype(np.int64)
        eid = np.where(mask, self.eid[idx], 0).astype(np.int64)
        dt = np.where(mask, t_q[:, None] - self.t[idx], 0.0).astype(np.float32)

        return (torch.from_numpy(nbr), torch.from_numpy(eid),
                torch.from_numpy(dt), torch.from_numpy(mask))

    def sample_slow(self, nodes, times, k):
        """Original loop version — kept ONLY to verify the fast path is identical."""
        B = len(nodes)
        nbr = np.zeros((B, k), dtype=np.int64)
        eid = np.zeros((B, k), dtype=np.int64)
        dt = np.zeros((B, k), dtype=np.float32)
        mask = np.zeros((B, k), dtype=bool)

        for i in range(B):
            n, ti = nodes[i], times[i]
            s, e_ = self.off[n], self.off[n + 1]
            if s == e_:
                continue
            # index of first event with time >= ti  -> everything before is valid
            cut = s + np.searchsorted(self.t[s:e_], ti, side='left')
            if cut == s:
                continue
            lo = max(s, cut - k)                 # take the K MOST RECENT
            m = cut - lo
            nbr[i, :m] = self.nbr[lo:cut]
            eid[i, :m] = self.eid[lo:cut]
            dt[i, :m] = ti - self.t[lo:cut]      # time GAP, not raw timestamp
            mask[i, :m] = True

        return (torch.from_numpy(nbr), torch.from_numpy(eid),
                torch.from_numpy(dt), torch.from_numpy(mask))


# ----------------------------------------------------------------------
# 3. TGAT ATTENTION LAYER
# ----------------------------------------------------------------------
class TGATLayer(nn.Module):
    """Multi-head attention over a node's temporal neighbourhood.

    Query  = the target node itself (+ time 0)
    Key/Val= [neighbour embedding || edge features || time encoding]
    """

    def __init__(self, node_dim, edge_dim, time_dim, out_dim, n_heads=2, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.out_dim = out_dim

        q_in = node_dim + time_dim
        k_in = node_dim + edge_dim + time_dim

        self.q_proj = nn.Linear(q_in, out_dim)
        self.k_proj = nn.Linear(k_in, out_dim)
        self.v_proj = nn.Linear(k_in, out_dim)
        self.merge = nn.Linear(out_dim + node_dim, out_dim)   # residual merge
        self.drop = nn.Dropout(dropout)

    def forward(self, h_self, h_nbr, e_feat, t_enc_self, t_enc_nbr, mask):
        # h_self [B,Dn] | h_nbr [B,K,Dn] | e_feat [B,K,De]
        # t_enc_self [B,Dt] | t_enc_nbr [B,K,Dt] | mask [B,K]
        B, K, _ = h_nbr.shape
        H, dk = self.n_heads, self.d_k

        q = self.q_proj(torch.cat([h_self, t_enc_self], dim=-1))      # [B,Do]
        kv_in = torch.cat([h_nbr, e_feat, t_enc_nbr], dim=-1)         # [B,K,*]
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        q = q.view(B, H, 1, dk)
        k = k.view(B, K, H, dk).transpose(1, 2)                       # [B,H,K,dk]
        v = v.view(B, K, H, dk).transpose(1, 2)

        scores = (q * k).sum(-1) / (dk ** 0.5)                        # [B,H,K]
        scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # nodes with NO valid history: all -inf -> softmax would be NaN
        empty = ~mask.any(dim=1)                                      # [B]
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)

        out = (attn.unsqueeze(-1) * v).sum(dim=2)                     # [B,H,dk]
        out = out.reshape(B, self.out_dim)
        out = out * (~empty).float().unsqueeze(-1)                    # zero if no history

        out = F.relu(self.merge(torch.cat([out, h_self], dim=-1)))    # residual
        return out, attn


# ----------------------------------------------------------------------
# 4. FULL TGAT MODEL
# ----------------------------------------------------------------------
class TGAT(nn.Module):
    def __init__(self, num_nodes, edge_dim, node_dim=64, time_dim=32,
                 n_layers=2, n_heads=2, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        # LEARNABLE node embeddings (Day 3 decision: no leaky aggregate features)
        self.node_emb = nn.Embedding(num_nodes, node_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.time_enc = TimeEncoder(time_dim)
        self.layers = nn.ModuleList([
            TGATLayer(node_dim, edge_dim, time_dim, node_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # edge classifier: [h_card || h_addr || edge features] -> fraud logit
        self.head = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _embed(self, nodes, times, nf, edge_attr, k, layer):
        """Recursive temporal embedding. layer=0 -> raw embedding lookup.

        `k` may be an int (same fanout everywhere) or a list/tuple indexed by
        layer, e.g. k=[10, 20] -> 20 neighbours at the outer hop, 10 at the
        inner hop. Shrinking the inner fanout is the cheapest way to cut cost,
        because work grows multiplicatively across hops.
        """
        device = self.node_emb.weight.device
        h = self.node_emb(nodes.to(device))
        if layer == 0:
            return h

        k_l = k[layer - 1] if isinstance(k, (list, tuple)) else k
        nbr, eid, dt, mask = nf.sample(nodes.cpu().numpy(), times.cpu().numpy(), k_l)
        B, K = nbr.shape

        nbr_times = times.cpu().unsqueeze(1) - dt                      # absolute time of neighbour event
        h_nbr = self._embed(nbr.reshape(-1), nbr_times.reshape(-1), nf,
                            edge_attr, k, layer - 1).view(B, K, -1)

        # index on-device: edge_attr should already live on the GPU
        e_feat = edge_attr[eid.reshape(-1).to(edge_attr.device)].view(B, K, -1).to(device)
        t_nbr = self.time_enc(dt.to(device))
        t_self = self.time_enc(torch.zeros(B, device=device))

        h, attn = self.layers[layer - 1](h, h_nbr, e_feat, t_self, t_nbr,
                                         mask.to(device))
        self.last_attn = attn                                          # kept for GNNExplainer later
        return h

    def forward(self, src, dst, times, edge_feat, nf, edge_attr, k=20):
        device = self.node_emb.weight.device
        h_src = self._embed(src, times, nf, edge_attr, k, self.n_layers)
        h_dst = self._embed(dst, times, nf, edge_attr, k, self.n_layers)
        z = torch.cat([h_src, h_dst, edge_feat.to(device)], dim=-1)
        return self.head(z).squeeze(-1)                                 # raw logits
