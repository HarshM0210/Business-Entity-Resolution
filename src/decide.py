"""Calibration, exact expected-F0.5 selection per S1 (optionally with a missed-match term), one-owner
arbitration, and the macro-F0.5 metric."""
import math

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# ----------------------------------------------------------------------------------------------- metric

MULT = np.int64(1 << 32)
_GT_CACHE = {}


def in_sorted(x, s):
    """Membership of x in sorted array s by binary search (no per-call hashing of s)."""
    if len(s) == 0:
        return np.zeros(len(x), dtype=bool)
    pos = np.clip(np.searchsorted(s, x), 0, len(s) - 1)
    return s[pos] == x


def gt_keys(gt_q, gt_c):
    k = (id(gt_q), len(gt_q))
    if k not in _GT_CACHE:
        _GT_CACHE[k] = np.sort(gt_q.astype(np.int64) * MULT + gt_c)
    return _GT_CACHE[k]


def macro_f05(pred, q_eval, n_true, gt_q, gt_c):
    """pred: DataFrame(q, c) of predicted matches. Every q in q_eval counts, singletons included."""
    q_eval = np.asarray(q_eval)
    pred = pred[in_sorted(pred.q.values.astype(np.int64), np.sort(q_eval).astype(np.int64))]
    k = pred.q.values.astype(np.int64) * MULT + pred.c.values
    tp = pd.Series(in_sorted(k, gt_keys(gt_q, gt_c)).astype(np.int64)).groupby(pred.q.values).sum()
    npred = pred.groupby("q").size()
    s = pd.DataFrame({"q": q_eval})
    s["tp"] = s.q.map(tp).fillna(0).values
    s["np"] = s.q.map(npred).fillna(0).values
    s["nt"] = n_true[q_eval]
    p = np.where(s.np > 0, s.tp / s.np.clip(lower=1), 0)
    r = np.where(s.nt > 0, s.tp / s.nt.clip(lower=1), 0)
    f = np.where(s.tp > 0, 1.25 * p * r / (0.25 * p + r + 1e-12), 0.0)
    f = np.where((s.np == 0) & (s.nt == 0), 1.0, f)
    s["f"] = f
    return float(f.mean()), s


def f05_selftest():
    """The problem statement's worked example: predicted 3, true 2, both true predicted -> 0.714."""
    pred = pd.DataFrame({"q": [0, 0, 0], "c": [1, 2, 3]})
    f, _ = macro_f05(pred, [0], np.array([2]), np.array([0, 0]), np.array([1, 3]))
    assert abs(f - 0.7142857) < 1e-4, f
    f, _ = macro_f05(pred.iloc[:0], [0], np.array([0]), np.array([], np.int64), np.array([], np.int64))
    assert f == 1.0


# ----------------------------------------------------------------------------------------------- calibration

def fit_isotonic(score, y):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(score, y)
    return ir


def fold_of(q, n, salt=0):
    """Deterministic group fold in [0, n) from a splitmix64 mix of (q, salt). Different salts give independent
    partitions (a linear hash would correlate them: e.g. parity would be shared)."""
    with np.errstate(over="ignore"):
        x = np.asarray(q).astype(np.uint64) + np.uint64(salt) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(n)).astype(np.int64)


def crossfit_isotonic(score, y, q, n_folds=5):
    out = np.empty_like(score, dtype=np.float64)
    fold = fold_of(q, n_folds, salt=0)
    for k in range(n_folds):
        m = fold == k
        out[m] = fit_isotonic(score[~m], y[~m]).predict(score[m])
    return out


# ----------------------------------------------------------------------------------------------- expected F0.5

def _poisson(lam, n=8):
    if lam <= 0:
        return np.array([1.0])
    v = np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(n)])
    v[-1] += max(0.0, 1 - v.sum())
    return v


def best_k(p, miss=None):
    """p sorted descending. Exact E[F0.5] of taking the top-k for every k via Poisson-binomial DPs. `miss` is the
    distribution of the number of true matches that are NOT among the candidates (None = assume none)."""
    n = len(p)
    miss = np.array([1.0]) if miss is None else miss
    if n == 0:
        return 0, float(miss[0])
    suf = [None] * (n + 1)
    d = miss.copy()                          # false negatives = unselected true candidates + missed matches
    suf[n] = d
    for i in range(n - 1, -1, -1):
        d = np.convolve(d, [1 - p[i], p[i]])
        suf[i] = d
    best, best_v = 0, float(suf[0][0])       # k = 0 scores 1 only if there is no true match at all
    tp = np.array([1.0])
    for k in range(1, n + 1):
        tp = np.convolve(tp, [1 - p[k - 1], p[k - 1]])
        fn = suf[k]
        t = np.arange(k + 1)[:, None]
        f = np.arange(len(fn))[None, :]
        F = 1.25 * t / (0.25 * (t + f) + k)
        v = float((tp[:, None] * fn[None, :] * F).sum())
        if v > best_v:
            best, best_v = k, v
    return best, best_v


def expected_f05(p_sel, p_rest, miss=None):
    """E[F0.5] of predicting exactly the candidates with probabilities p_sel, when the other candidates have p_rest
    and `miss` is the distribution of true matches outside the candidate set (label-free self-assessment)."""
    tp = np.array([1.0])
    for x in p_sel:
        tp = np.convolve(tp, [1 - x, x])
    fn = np.array([1.0]) if miss is None else np.asarray(miss, dtype=np.float64)
    for x in p_rest:
        fn = np.convolve(fn, [1 - x, x])
    k = len(p_sel)
    if k == 0:
        return float(fn[0])
    t = np.arange(k + 1)[:, None]
    f = np.arange(len(fn))[None, :]
    return float((tp[:, None] * fn[None, :] * (1.25 * t / (0.25 * (t + f) + k))).sum())


def _choose_group(args):
    qs, probs_list, family, thr = args
    miss = _poisson(thr) if family == "expf05" and thr else None
    out = []
    for q, p in zip(qs, probs_list):
        if family == "threshold":
            k = int((p >= thr).sum())
        else:
            k, _ = best_k(p, miss)
        out.append(k)
    return out


def choose(df, family, thr=None, pool=None, banned=None):
    """df(q, c, p) -> selected rows. Candidates sorted by p desc inside each q. For family "expf05", thr is the
    Poisson rate of missed matches per S1 (0/None = none)."""
    d = df.sort_values(["q", "p"], ascending=[True, False], kind="stable")
    if banned is not None and len(banned):
        d = d[~in_sorted(d.q.values.astype(np.int64) * MULT + d.c.values, banned)]
    d = d[d.p >= 1e-4]
    if d.empty:
        return d
    qv = d.q.values
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    ends = np.r_[starts[1:], len(d)]
    pv = d.p.values
    qs = qv[starts]
    probs = [pv[a:e] for a, e in zip(starts, ends)]
    step = 20_000
    args = [(qs[i:i + step], probs[i:i + step], family, thr) for i in range(0, len(qs), step)]
    ks = [k for part in (pool.map(_choose_group, args) if pool else map(_choose_group, args)) for k in part]
    take = np.zeros(len(d), dtype=bool)
    for a, k in zip(starts, ks):
        take[a:a + k] = True
    return d[take]


def decide(df, family, thr=None, max_iter=30, pool=None):
    """Per-S1 choice, then one-owner arbitration iterated to a fixpoint: the higher-probability claim keeps
    a contested S2/S3 record (ties broken by the smaller S1 index, deterministically) and the loser re-decides
    without it. Asserts one owner per record on exit."""
    banned = np.empty(0, dtype=np.int64)
    sel = choose(df, family, thr, pool)
    for _ in range(max_iter):
        dup = sel[sel.duplicated("c", keep=False)]
        if dup.empty:
            break
        win = dup.sort_values(["c", "p", "q"], ascending=[True, False, True], kind="stable").drop_duplicates("c")
        lose = dup.merge(win[["c", "q"]], on="c", suffixes=("", "_w"))
        lose = lose[lose.q != lose.q_w]
        banned = np.union1d(banned, lose.q.values.astype(np.int64) * MULT + lose.c.values)
        redo = np.unique(lose.q.values)
        keep = sel[~np.isin(sel.q.values, redo)]
        again = choose(df[np.isin(df.q.values, redo)], family, thr, pool, banned)
        sel = pd.concat([keep, again], ignore_index=True)
    dup = sel[sel.duplicated("c", keep=False)]
    if not dup.empty:                                  # not converged: drop the weaker claims outright
        sel = sel.sort_values(["c", "p", "q"], ascending=[True, False, True], kind="stable").drop_duplicates("c")
    assert not sel.c.duplicated().any(), "a record has two owners after arbitration"
    return sel.sort_values(["q", "p"], ascending=[True, False]).reset_index(drop=True)
