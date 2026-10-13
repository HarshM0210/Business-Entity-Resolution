"""Load pre-computed embeddings. side = 's1' or 'pool' (pool = S2 rows then S3 rows).

  kind "shards"  e5-large: <dir>/source{1,2,3}/part-XXXXX-<from>-<to>.npy + manifest.json (row order = parquet)
  kind "s1pool"  bge-m3:   <dir>/{s1,pool}.npy + {s1,pool}.ids.txt
  kind "matrix"  e5-small: <dir>/<split>_{s1,pool}/matrix.npy + ids.txt + meta.json
"""
import json
import re
from pathlib import Path

import numpy as np

from config import EMB


class Emb:
    """Row-addressable view over one embedding matrix (single .npy or ordered shard files)."""

    def __init__(self, parts):
        self.parts = parts                      # list of memmapped arrays
        self.offsets = np.cumsum([0] + [len(p) for p in parts])
        self.dim = parts[0].shape[1]

    def __len__(self):
        return int(self.offsets[-1])

    def take(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), self.dim), dtype=np.float16)
        which = np.searchsorted(self.offsets, rows, side="right") - 1
        for k, p in enumerate(self.parts):
            m = which == k
            if m.any():
                out[m] = p[rows[m] - self.offsets[k]]
        return out

    def all(self):
        out = np.empty((len(self), self.dim), dtype=np.float16)
        for k, p in enumerate(self.parts):
            out[self.offsets[k]:self.offsets[k + 1]] = p
        return out


def _shard_parts(d):
    d = Path(d)
    man = json.loads((d / "manifest.json").read_text())
    files = sorted(d.glob("part-*.npy"))
    spans = [tuple(int(x) for x in re.match(r"part-\d+-(\d+)-(\d+)\.npy", f.name).groups()) for f in files]
    exp = 0
    for (a, b) in spans:
        if a != exp:
            raise RuntimeError(f"{d}: shard gap/overlap at row {a} (expected {exp})")
        exp = b
    if exp != man["rows"]:
        raise RuntimeError(f"{d}: shards cover {exp} rows, manifest says {man['rows']}")
    return [np.load(f, mmap_mode="r") for f in files], man


def _dir(model, split, side):
    spec = EMB[model]
    d = Path(spec[split])
    if spec["kind"] == "matrix":
        return d / f"{split}_{side}"
    return d


def load_emb(model, split, side):
    spec = EMB[model]
    d = _dir(model, split, side)
    if spec["kind"] == "s1pool":
        return Emb([np.load(d / f"{side}.npy", mmap_mode="r")])
    if spec["kind"] == "matrix":
        return Emb([np.load(d / "matrix.npy", mmap_mode="r")])
    if side == "s1":
        parts, _ = _shard_parts(d / "source1")
    else:
        p2, _ = _shard_parts(d / "source2")
        p3, _ = _shard_parts(d / "source3")
        parts = p2 + p3
    return Emb(parts)


def emb_ids(model, split, side):
    """Row ids if the source ships them (bge, e5-small), else None (e5-large guarantees parquet row order in its
    manifest; its alignment is checked semantically by the manifest stage)."""
    spec = EMB[model]
    d = _dir(model, split, side)
    if spec["kind"] == "s1pool":
        return (d / f"{side}.ids.txt").read_text().split()
    if spec["kind"] == "matrix":
        return (d / "ids.txt").read_text().split()
    return None
