"""Exact LightGBM prediction on the GPU (same trees, same decision rule), for scoring ~100M test pairs when the
CPU is the bottleneck. Mirrors LightGBM's NumericalDecision:
  NaN with missing_type != NaN  -> treated as 0.0
  missing_type Zero and |x|<=1e-35, or missing_type NaN and x is NaN -> default direction
  otherwise x <= threshold -> left
Features are float32, so each double threshold t is replaced by the largest float32 <= t: for float32 x,
x <= t32  <=>  x <= t  exactly. Leaf values are summed in float64, then the sigmoid, like Booster.predict."""
import numpy as np
import torch

K_ZERO = 1e-35
MT = {"None": 0, "Zero": 1, "NaN": 2}


def _floor_f32(t):
    t32 = np.float32(t)
    if float(t32) > t:
        t32 = np.nextafter(t32, np.float32(-np.inf))
    return t32


class GpuForest:
    def __init__(self, booster, device):
        dump = booster.dump_model()
        feat, thr, child, flags, leaves, roots = [], [], [], [], [], []

        def add(node):
            if "split_index" not in node:            # leaf (also a whole single-leaf tree)
                leaves.append(node["leaf_value"])
                return -len(leaves)                  # leaf k is encoded as -(k+1)
            if node["decision_type"] != "<=":
                raise ValueError(f"unsupported decision type {node['decision_type']}")
            i = len(feat)
            feat.append(node["split_feature"]); thr.append(_floor_f32(node["threshold"]))
            flags.append(int(bool(node["default_left"])) | (MT[node["missing_type"]] << 1))
            child.extend([0, 0])
            child[2 * i] = add(node["left_child"])
            child[2 * i + 1] = add(node["right_child"])
            return i

        for t in dump["tree_info"]:
            roots.append(add(t["tree_structure"]))
        d = device
        self.feat = torch.tensor(feat, dtype=torch.int32, device=d).long()
        self.thr = torch.tensor(np.array(thr, dtype=np.float32), device=d)
        self.child = torch.tensor(child, dtype=torch.int32, device=d)
        self.flags = torch.tensor(flags, dtype=torch.int8, device=d)
        self.leaf = torch.tensor(leaves, dtype=torch.float64, device=d)
        self.roots = torch.tensor(roots, dtype=torch.int32, device=d)
        self.n_trees = len(roots)

    @torch.no_grad()
    def raw(self, X):
        """X: (B, F) float32 tensor on device -> raw scores (B,) float64."""
        B = X.shape[0]
        node = self.roots.unsqueeze(0).expand(B, -1).contiguous()     # (B, T) int32
        while True:
            act = node >= 0
            if not bool(act.any()):
                break
            n = node.clamp(min=0).long()
            x = torch.gather(X, 1, self.feat[n])
            fl = self.flags[n]
            mt = fl >> 1
            nan = torch.isnan(x)
            x = torch.where(nan & (mt != 2), 0.0, x)
            use_default = ((mt == 1) & (x.abs() <= K_ZERO)) | ((mt == 2) & nan)
            go_right = torch.where(use_default, (fl & 1) == 0, x > self.thr[n])
            nxt = self.child[2 * n + go_right.long()]
            node = torch.where(act, nxt, node)
        return self.leaf[(-node - 1).long()].sum(1)

    def predict(self, X_np, batch=None):
        batch = batch or max(1024, int(6e7 // max(self.n_trees, 1)))
        out = np.empty(len(X_np), dtype=np.float64)
        dev = self.feat.device
        for a in range(0, len(X_np), batch):
            X = torch.from_numpy(np.ascontiguousarray(X_np[a:a + batch], dtype=np.float32)).to(dev)
            out[a:a + batch] = torch.sigmoid(self.raw(X)).cpu().numpy()
        return out
