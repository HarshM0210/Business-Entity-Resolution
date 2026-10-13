"""Listwise cross-encoder (README §5.4): one transformer pass per S1 over the S1 record and all its candidates.

  [CLS] s1 name | s1 address [SEP] [CAND] c1 name | c1 address [SEP] [CAND] c2 ... [SEP]

Every candidate attends to the S1 and to every other candidate (decoys with a different house number, sibling
copies of one business), which no pointwise model can do. Heads: a match logit per [CAND] (2-layer MLP) and a
"no listed candidate matches" logit on [CLS]. Each [CAND] embedding gets learned source (S2/S3) and rank-bucket
biases. Backbone: the XLM-R-large reranker fine-tuned by the ce_b stage (Apache-2.0, 8k positions).

Trained on ce + lce entities (disjoint from stack/val), so its stack/val/test scores need no out-of-fold scheme.
Training shuffles candidate order every step; inference uses p1 order (lce_passes=2 also averages the reversed
order)."""
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import Progress, drain, get_logger, gpu_patient, save_json, save_npz, save_parquet, spawn_gpu_workers
from config import CFG, PATHS, WORK
from cross_encoder import Prefetch, ce_dir, s2_file, texts
from prep import load_prep, load_splits

log = get_logger("lce")
os.environ.setdefault("HF_HOME", str(PATHS["hf_home"]))
D = WORK / "lce"
N_RANK_BUCKETS = 8


def rank_bucket(r):
    return np.minimum(np.floor(np.log2(np.asarray(r, dtype=np.float64) + 1) * 2), N_RANK_BUCKETS - 1).astype(np.int64)


class LCE(nn.Module):
    def __init__(self, backbone, cand_id):
        super().__init__()
        self.enc = backbone
        h = backbone.config.hidden_size
        self.cand_id = cand_id
        self.src_bias = nn.Embedding(4, h)
        self.rank_bias = nn.Embedding(N_RANK_BUCKETS, h)
        nn.init.zeros_(self.src_bias.weight); nn.init.zeros_(self.rank_bias.weight)
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.cls_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, ids, mask, cand_pos, cand_src, cand_rank):
        """ids/mask: (B, L); cand_pos: (B, C) token index of each [CAND] (-1 = none)."""
        emb = self.enc.embeddings.word_embeddings(ids)
        valid = cand_pos >= 0
        b_idx = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(cand_pos)
        bias = (self.src_bias(cand_src) + self.rank_bias(cand_rank)).to(emb.dtype)
        emb = emb.index_put((b_idx[valid], cand_pos[valid]), bias[valid], accumulate=True)
        pad = self.enc.config.pad_token_id
        pos = (torch.cumsum(mask, 1) * mask + pad).long()
        hs = self.enc(inputs_embeds=emb, attention_mask=mask, position_ids=pos).last_hidden_state
        hc = hs[b_idx, cand_pos.clamp(min=0)]
        logit = self.head(hc).squeeze(-1).masked_fill(~valid, 0.0)
        return logit, self.cls_head(hs[:, 0]).squeeze(-1)


def _backbone():
    from transformers import AutoModel, AutoTokenizer
    src = ce_dir("b") / "final"
    tok = AutoTokenizer.from_pretrained(src)
    tok.add_special_tokens({"additional_special_tokens": ["[CAND]"]})
    enc = AutoModel.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa", add_pooling_layer=False)
    need = CFG["lce_q_tokens"] + CFG["lce_max_cand"] * (CFG["lce_cand_tokens"] + 2) + 4
    assert enc.config.max_position_embeddings > need, "backbone positions too short for a full candidate list"
    old = enc.get_input_embeddings().weight.shape[0]
    enc.resize_token_embeddings(len(tok))
    cand_id = tok.convert_tokens_to_ids("[CAND]")
    with torch.no_grad():
        W = enc.get_input_embeddings().weight
        if cand_id >= old:
            W[cand_id] = W[tok.sep_token_id]            # [CAND] starts as a copy of [SEP]
    return tok, enc, cand_id


class Builder:
    """Assembles token sequences for lists of candidates (text tokenised without special tokens, truncated)."""

    def __init__(self, tok, cand_id, ta, tb, src_pool):
        self.tok, self.cand_id, self.ta, self.tb, self.src = tok, cand_id, ta, tb, src_pool

    def __call__(self, lists):
        """lists: [(q, [c...], [rank...])] -> tensors."""
        tok = self.tok
        qt = tok([self.ta[q] for q, _, _ in lists], add_special_tokens=False, truncation=True,
                 max_length=CFG["lce_q_tokens"])["input_ids"]
        allc = [c for _, cs, _ in lists for c in cs]
        ct = tok([self.tb[c] for c in allc], add_special_tokens=False, truncation=True,
                 max_length=CFG["lce_cand_tokens"])["input_ids"] if allc else []
        seqs, cpos, k = [], [], 0
        C = max(1, max(len(cs) for _, cs, _ in lists))
        for (q, cs, rs), qi in zip(lists, qt):
            s = [tok.cls_token_id] + qi + [tok.sep_token_id]
            pos = []
            for _ in cs:
                pos.append(len(s))
                s += [self.cand_id] + ct[k] + [tok.sep_token_id]
                k += 1
            seqs.append(s)
            cpos.append(pos + [-1] * (C - len(pos)))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        src = np.zeros((len(lists), C), dtype=np.int64)
        rk = np.zeros((len(lists), C), dtype=np.int64)
        for i, (_, cs, rs) in enumerate(lists):
            src[i, :len(cs)] = self.src[np.asarray(cs, dtype=np.int64)]
            rk[i, :len(cs)] = rank_bucket(rs)
        return ids, mask, torch.tensor(cpos), torch.from_numpy(src), torch.from_numpy(rk)


def _token_len(n_cand):
    return CFG["lce_q_tokens"] // 2 + 2 + n_cand * (CFG["lce_cand_tokens"] // 2 + 2)


def _plan(lens, seed, budget):
    """Batches of lists with similar estimated length under a token budget; shuffled batch order."""
    rng = np.random.default_rng(seed)
    order = np.argsort(lens + rng.random(len(lens)), kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lens[i])
        if cur and m * (len(cur) + 1) > budget:
            batches.append(np.array(cur)); cur, cur_max = [], 0
            m = lens[i]
        cur.append(i); cur_max = m
    if cur:
        batches.append(np.array(cur))
    rng.shuffle(batches)
    return batches


# ----------------------------------------------------------------------------------------------- training lists

def train_lists():
    path = D / "train_lists.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sp = load_splits()
    groups = np.sort(np.concatenate([sp["g_ce"], sp["g_lce"]]))
    s1 = pd.read_parquet(WORK / "stage1" / "train.parquet", columns=["q", "c", "y", "p1"])
    s1 = s1[np.isin(s1.q.values, groups)]
    s1 = s1.sort_values(["q", "p1"], ascending=[True, False], kind="stable")
    s1["rank"] = s1.groupby("q").cumcount()
    top = s1[(s1["rank"] < CFG["lce_max_cand"]) & (s1.p1 >= CFG["s2_min_prob"])][["q", "c", "y", "rank"]]
    # hard positives: true matches outside the list replace the lowest-ranked negatives with probability 0.5
    rng = np.random.default_rng(CFG["seed"])
    gm = np.isin(sp["gt_q"], groups)
    gt = pd.DataFrame({"q": sp["gt_q"][gm], "c": sp["gt_c"][gm]})
    key = lambda d: d.q.values.astype(np.int64) * (1 << 32) + d.c.values
    outside = gt[~np.isin(key(gt), key(top))]
    outside = outside[rng.random(len(outside)) < CFG["lce_hardpos_prob"]]
    add = outside.assign(y=1, rank=CFG["lce_max_cand"])
    lists = pd.concat([top, add], ignore_index=True).sort_values(["q", "rank", "y"], ascending=[True, True, False])
    # keep at most lce_max_cand per list: drop the lowest-ranked negatives first
    lists["neg_order"] = np.where(lists.y == 1, -1, lists["rank"])
    lists = lists.sort_values(["q", "neg_order"]).groupby("q").head(CFG["lce_max_cand"])
    lists = lists.sort_values(["q", "rank"]).drop(columns="neg_order")
    qs = lists.q.unique()
    if len(qs) > CFG["lce_max_lists"]:
        keep = rng.choice(qs, CFG["lce_max_lists"], replace=False)
        lists = lists[np.isin(lists.q.values, keep)]
    save_parquet(lists.reset_index(drop=True), path)
    log.info(f"lce train lists: {lists.q.nunique()} lists, {len(lists)} rows, pos {int(lists.y.sum())}")
    return lists


def run_lce_train():
    final = D / "final.pt"
    if final.exists():
        return
    D.mkdir(parents=True, exist_ok=True)
    lists = train_lists()
    tok, enc, cand_id = _backbone()
    ta, tb = texts("train", "s1", "b"), texts("train", "pool", "b")
    src_pool = load_prep("train", "pool", ["src"]).src.values
    build = Builder(tok, cand_id, ta, tb, src_pool)
    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc.config.use_cache = False
    model = gpu_patient(lambda: LCE(enc, cand_id).cuda(), log, "load lce")
    g = lists.groupby("q", sort=True)
    Lq = g.q.first().values
    Lc = [x.values for _, x in g.c]
    Ly = [x.values for _, x in g.y]
    Lr = [x.values for _, x in g["rank"]]
    lens = np.array([_token_len(len(c)) for c in Lc])
    epochs = CFG["lce_epochs"]
    plans = [_plan(lens, CFG["seed"] + e, CFG["lce_batch_tokens"]) for e in range(epochs)]
    total = sum(len(p) for p in plans)
    if CFG["lce_max_steps"]:
        total = min(total, CFG["lce_max_steps"])
    flat = [(e, b) for e in range(epochs) for b in range(len(plans[e]))][:total]
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lce_lr"], weight_decay=0.01)
    warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total - s) / max(1, total - warm)))
    ck = D / "ckpt.pt"
    step = 0
    if ck.exists():
        st = torch.load(ck, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = st["step"]
        log.info(f"resumed lce from step {step}/{total}")
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def make(s):
        e, b = flat[s]
        idx = plans[e][b]
        rng = np.random.default_rng(CFG["seed"] * 1000 + s)
        batch, ys = [], []
        for i in idx:
            perm = rng.permutation(len(Lc[i]))              # order-invariance: shuffle every step
            batch.append((Lq[i], Lc[i][perm].tolist(), Lr[i][perm].tolist()))
            ys.append(Ly[i][perm])
        t = build(batch)
        C = t[2].shape[1]
        y = np.zeros((len(ys), C), dtype=np.float32)
        for i, v in enumerate(ys):
            y[i, :len(v)] = v
        ycls = (y.sum(1) == 0).astype(np.float32)
        return t, torch.from_numpy(y), torch.from_numpy(ycls)

    model.train()
    t0, run = time.time(), []
    it = iter(Prefetch(make, range(step, total)))
    pending = None
    n_micro = 1                                       # grows on OOM: same effective batch, less memory
    oom_since = None                                  # continuous OOM for > 2 h fails the stage (retry may pick another GPU)

    def fwd_bwd(t, y, ycls, rows, n_valid, n_rows):
        ids, mask, cpos, csrc, crank = [x[rows].cuda() for x in t]
        L = int(mask.sum(1).max())                    # trim padding of this micro-batch
        ids, mask = ids[:, :L], mask[:, :L]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, cls = model(ids, mask, cpos, csrc, crank)
        valid = (cpos >= 0).float()
        loss = (bce(logit.float(), y[rows].cuda()) * valid).sum() / n_valid + \
            CFG["lce_cls_weight"] * bce(cls.float(), ycls[rows].cuda()).sum() / n_rows
        loss.backward()
        return loss.item()

    while step < total:
        s, (t, y, ycls) = pending if pending else next(it)
        pending = None
        B = t[0].shape[0]
        n_valid = float(max(1, int((t[2] >= 0).sum())))
        try:
            lv = 0.0
            for rows in np.array_split(np.arange(B), min(n_micro, B)):
                lv += fwd_bwd(t, y, ycls, torch.from_numpy(rows), n_valid, float(B))
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if n_micro < 16:
                n_micro *= 2
                log.warning(f"lce OOM at step {step}: now {n_micro} micro-batches per step")
            else:
                oom_since = oom_since or time.time()
                if time.time() - oom_since > 2 * 3600:
                    raise RuntimeError("lce: GPU out of memory for 2 h even with 16 micro-batches")
                log.warning(f"lce OOM at step {step} with {n_micro} micro-batches; waiting 60s for the shared GPU")
                time.sleep(60)
            pending = (s, (t, y, ycls))
            continue
        oom_since = None
        loss = torch.tensor(lv)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            raise RuntimeError(f"lce: non-finite gradient at step {step}")
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        run.append(loss.item())
        if step % 100 == 0:
            log.info(f"lce step {step}/{total} loss {np.mean(run[-100:]):.4f} {(time.time() - t0) / 60:.1f}m")
        if step % CFG["lce_ckpt_every"] == 0 or step == total:
            tmp = ck.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step}, tmp)
            os.replace(tmp, ck)
    tmp = final.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "cand_id": cand_id, "n_tok": len(tok)}, tmp)
    tok.save_pretrained(D / "tokenizer")
    os.replace(tmp, final)
    ck.unlink(missing_ok=True)
    save_json({"steps": total, "lists": int(len(Lq)), "minutes": (time.time() - t0) / 60},
              WORK / "reports" / "lce_train.json")


# ----------------------------------------------------------------------------------------------- inference

def lce_rows(split):
    """Stage-2 rows the listwise model scores: everything except pairs that only the ename channel found. The model
    was trained on stage-1 lists (no ename-only pairs), and dropping them halves the list length; those rows get NaN
    listwise columns, which LightGBM treats as missing."""
    s2 = pd.read_parquet(s2_file(split), columns=["q", "c", "p1"])
    xp = WORK / "stage1x" / f"{split}.parquet"
    if xp.exists():
        x = pd.read_parquet(xp, columns=["q", "c"])
        k = lambda d: d.q.values.astype(np.int64) * (1 << 32) + d.c.values
        from decide import in_sorted
        s2 = s2[~in_sorted(k(s2), np.sort(k(x)))]
    return s2.reset_index(drop=True)


def lce_needed(split):
    """Test-side listwise inference only when the stack gate adopted the listwise model (README v2 §3.5)."""
    if split != "test":
        return True
    b = WORK / "models" / "stage2_bundle.json"
    return b.exists() and json.loads(b.read_text())["config"].startswith("ce_lce")


def _q_shards(split):
    qs = np.unique(lce_rows(split).q.values)
    n = max(1, math.ceil(len(qs) / CFG["lce_shard_q"]))
    return qs, [WORK / "lce_scores" / split / f"shard_{i:05d}.npz" for i in range(n)]


def run_lce_infer_worker(split):
    qs, paths = _q_shards(split)
    if all(p.exists() for p in paths):
        return
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(D / "tokenizer")
    st = torch.load(D / "final.pt", map_location="cpu", weights_only=False)
    enc = AutoModel.from_pretrained(ce_dir("b") / "final", dtype=torch.float32, attn_implementation="sdpa",
                                    add_pooling_layer=False)
    enc.resize_token_embeddings(st["n_tok"])
    model = LCE(enc, st["cand_id"])
    model.load_state_dict(st["model"])
    if CFG["infer_bf16"]:                  # bf16 weights at inference (see cross_encoder.load_infer_model)
        model = model.to(torch.bfloat16)
    model = gpu_patient(lambda: model.cuda().eval(), log, "load lce")
    s2 = lce_rows(split).sort_values(["q", "p1"], ascending=[True, False])
    ta, tb = texts(split, "s1", "b"), texts(split, "pool", "b")
    build = Builder(tok, st["cand_id"], ta, tb, load_prep(split, "pool", ["src"]).src.values)
    qv = s2.q.values
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    ends = np.r_[starts[1:], len(qv)]
    where = dict(zip(qv[starts], zip(starts, ends)))
    S = CFG["lce_shard_q"]

    @torch.no_grad()
    def job(i):
        my = qs[i * S:(i + 1) * S]
        spans = [where[q] for q in my]
        cs = [s2.c.values[a:e][:CFG["lce_max_cand"]] for a, e in spans]
        lens = np.array([_token_len(len(c)) for c in cs])
        out_logit = [np.zeros(len(c), dtype=np.float32) for c in cs]
        out_cls = np.zeros(len(my), dtype=np.float32)
        budget = CFG["lce_batch_tokens"] * 4          # no activations kept: larger batches than training
        for p in range(CFG["lce_passes"]):
            batches = _plan(lens, 0, budget)
            done = 0
            while done < len(batches):
                idx = batches[done]
                lists = []
                for j in idx:
                    c = cs[j] if p == 0 else cs[j][::-1]
                    r = np.arange(len(c)) if p == 0 else np.arange(len(c))[::-1]
                    lists.append((my[j], c.tolist(), r.tolist()))
                try:
                    ids, mask, cpos, csrc, crank = [x.cuda() for x in build(lists)]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg, cl = model(ids, mask, cpos, csrc, crank)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    budget = max(2000, budget // 2)
                    batches = batches[:done] + [b for x in batches[done:] for b in np.array_split(x, 2) if len(b)]
                    log.warning(f"lce infer OOM -> smaller batches")
                    continue
                lg, cl = lg.float().cpu().numpy(), cl.float().cpu().numpy()
                for k, j in enumerate(idx):
                    v = lg[k, :len(cs[j])]
                    out_logit[j] += (v if p == 0 else v[::-1]) / CFG["lce_passes"]
                    out_cls[j] += cl[k] / CFG["lce_passes"]
                done += 1
        q_rep = np.concatenate([np.full(len(c), q) for q, c in zip(my, cs)]) if cs else np.empty(0)
        save_npz(paths[i], q=q_rep.astype(np.int32), c=np.concatenate(cs).astype(np.int32),
                 lce=np.concatenate(out_logit), cls_q=np.asarray(my, dtype=np.int32), cls=out_cls)
    drain(paths, job, log, f"lce infer {split} (GPU {os.environ.get('CUDA_VISIBLE_DEVICES')})")


def run_lce_infer(split):
    if not lce_needed(split):
        log.info(f"lce infer {split}: skipped - the stack gate did not adopt the listwise model")
        return
    spawn_gpu_workers(["lce_infer_worker", split], log, CFG["gpu_min_free_mb"]["lce_infer"])
    qs, paths = _q_shards(split)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"missing lce shards {missing[:5]}")


def load_lce_scores(split):
    """(q, c, lce, lce_cls); candidates beyond the list cap get NaN (LightGBM handles missing)."""
    qs, paths = _q_shards(split)
    zs = [np.load(p) for p in paths]
    d = pd.DataFrame({"q": np.concatenate([z["q"] for z in zs]), "c": np.concatenate([z["c"] for z in zs]),
                      "lce": np.concatenate([z["lce"] for z in zs])})
    cls = pd.Series(np.concatenate([z["cls"] for z in zs]), index=np.concatenate([z["cls_q"] for z in zs]))
    d["lce_cls"] = cls.reindex(d.q.values).values.astype(np.float32)
    return d
