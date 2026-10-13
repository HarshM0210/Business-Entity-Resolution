"""Pairwise cross-encoders, fine-tuned on the ce split (disjoint from stack/val/test, so their scores are
leak-free features downstream):
  m = microsoft/mdeberta-v3-base (MIT, 278M)      text "name | address"
  b = BAAI/bge-reranker-v2-m3   (Apache-2.0, 568M) text "name: .. | addr: .. | country: .."
Resumable training (exact step, same data order); inference shards are claimed dynamically by one worker per
GPU, so the two shared cards drain one queue at whatever speed each has."""
import math
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import torch

from common import (Progress, drain, get_logger, gpu_patient, load_json, save_json, save_npy, save_parquet,
                    shard_path, spawn_gpu_workers)
from config import CES, CFG, PATHS, WORK
from prep import load_prep, load_splits
from retrieve_dense import n_shards

log = get_logger("ce")
os.environ.setdefault("HF_HOME", str(PATHS["hf_home"]))


def ce_dir(key):
    return WORK / f"ce_{key}"


def texts(split, side, key):
    d = load_prep(split, side, ["raw_name", "raw_addr", "raw_country"])
    if CES[key]["text"] == "tagged":
        return ("name: " + d.raw_name + " | addr: " + d.raw_addr + " | country: " + d.raw_country).values
    return (d.raw_name + " | " + d.raw_addr).values


def _hf():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return AutoTokenizer, AutoModelForSequenceClassification


def build_pairs():
    """Positives + hard (top RRF rank) + random negatives of ce-split entities. Shared by both cross-encoders."""
    path = WORK / "ce_pairs.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sp = load_splits()
    K = load_json(WORK / "reports" / "k_choice.json")["K"]
    parts = []
    for i in range(n_shards("train")):
        d = pd.read_parquet(shard_path("block", "train", i), columns=["q", "c", "y", "rrf_rank"])
        parts.append(d[np.isin(d.q.values, sp["g_ce"]) & (d.rrf_rank < K)])
    d = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(CFG["seed"])
    # every true pair of the ce split, including those blocking missed
    ce_set = np.isin(sp["gt_q"], sp["g_ce"])
    pos = pd.DataFrame({"q": sp["gt_q"][ce_set], "c": sp["gt_c"][ce_set], "y": 1})
    neg = d[d.y == 0].sort_values(["q", "rrf_rank"])
    hard = neg.groupby("q").head(CFG["ce_neg_hard"])
    rest = neg.drop(hard.index)
    rnd = rest.assign(r=rng.random(len(rest))).sort_values(["q", "r"]).groupby("q").head(CFG["ce_neg_rand"])
    pairs = pd.concat([pos, hard[["q", "c", "y"]], rnd[["q", "c", "y"]]], ignore_index=True)
    pairs["y"] = pairs.y.astype(np.int8)
    pairs = pairs.sample(frac=1.0, random_state=CFG["seed"]).reset_index(drop=True)
    from decide import fold_of
    pairs["dev"] = fold_of(pairs.q.values, 30, salt=3) == 0
    save_parquet(pairs, path)
    log.info(f"ce pairs: {len(pairs)} (pos {int(pairs.y.sum())}), dev {int(pairs.dev.sum())}")
    return pairs


def _lr_lambda(total, warm):
    return lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total - s) / max(1, total - warm))


class Prefetch:
    """Tokenise upcoming batches on a background thread (HF fast tokenisers release the GIL)."""

    def __init__(self, make, steps, depth=4):
        self.q = queue.Queue(depth)
        self.t = threading.Thread(target=self._run, args=(make, steps), daemon=True)
        self.t.start()

    def _run(self, make, steps):
        for s in steps:
            try:
                self.q.put((s, make(s)))
            except Exception as e:                  # surface tokeniser errors in the main thread
                self.q.put((s, e))
                return
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item[1], Exception):
                raise item[1]
            yield item


def run_ce_train(key):
    spec = CES[key]
    D = ce_dir(key)
    final = D / "final"
    if (final / "config.json").exists():
        return
    AutoTokenizer, AutoModel = _hf()
    pairs = build_pairs()
    tr, dv = pairs[~pairs.dev].reset_index(drop=True), pairs[pairs.dev].reset_index(drop=True)
    if len(tr) > spec["max_pairs"]:
        tr = tr.head(spec["max_pairs"])            # pairs are pre-shuffled: a uniform sample, pos/neg mix kept
    ta, tb = texts("train", "s1", key), texts("train", "pool", key)
    tok = AutoTokenizer.from_pretrained(spec["model"])
    # transformers>=5 loads the stored dtype (fp16 for mDeBERTa); fp16 master weights + AdamW -> NaN. fp32 master
    # weights + bf16 autocast.
    model = AutoModel.from_pretrained(spec["model"], num_labels=1, dtype=torch.float32)
    model = gpu_patient(lambda: model.cuda(), log, "load cross-encoder")
    B = spec["batch"]
    spe = math.ceil(len(tr) / B)
    total = spe * spec["epochs"]
    if CFG["ce_max_steps"]:
        total = min(total, CFG["ce_max_steps"])
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, int(0.05 * total)))
    ck = D / "ckpt.pt"
    step = 0
    if ck.exists():
        st = torch.load(ck, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = st["step"]
        log.info(f"[{key}] resumed from step {step}/{total}")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    perms = {}

    def make(s):
        e, b = divmod(s, spe)
        if e not in perms:
            perms.clear()
            perms[e] = np.random.default_rng(CFG["seed"] + e).permutation(len(tr))
        idx = perms[e][b * B:(b + 1) * B]
        q, c, y = tr.q.values[idx], tr.c.values[idx], tr.y.values[idx]
        swap = ((q.astype(np.int64) + c + e) % 2) == 1          # both text orders
        a, bb = ta[q], tb[c]
        x1, x2 = np.where(swap, bb, a).tolist(), np.where(swap, a, bb).tolist()
        enc = tok(x1, x2, truncation=True, max_length=spec["max_len"], padding=True, return_tensors="pt")
        return enc, torch.tensor(y, dtype=torch.float32)

    t0, run = time.time(), []
    D.mkdir(parents=True, exist_ok=True)
    it = iter(Prefetch(make, range(step, total)))
    pending = None
    n_micro = 1                                 # grows on OOM: same effective batch, less memory
    oom_since = None                            # continuous OOM for > 2 h fails the stage (retry may pick another GPU)
    while step < total:
        s, (enc, y) = pending if pending else next(it)
        pending = None
        assert s == step
        try:
            lv = 0.0
            n = len(y)
            for rows in np.array_split(np.arange(n), min(n_micro, n)):
                part = {k: v[rows].to("cuda") for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**part).logits.squeeze(-1)
                loss = lossf(logits.float(), y[rows].cuda()) * (len(rows) / n)
                loss.backward()
                lv += loss.item()
        except torch.cuda.OutOfMemoryError:     # shared GPU filled up: smaller micro-batches, then wait
            opt.zero_grad(set_to_none=True)
            logits = loss = part = None
            torch.cuda.empty_cache()
            if n_micro < 16:
                n_micro *= 2
                log.warning(f"[{key}] OOM at step {step}: now {n_micro} micro-batches per step")
            else:
                oom_since = oom_since or time.time()
                if time.time() - oom_since > 2 * 3600:
                    raise RuntimeError(f"[{key}] GPU out of memory for 2 h even with 16 micro-batches")
                log.warning(f"[{key}] OOM at step {step}; waiting 60s for the shared GPU")
                time.sleep(60)
            pending = (s, (enc, y))
            continue
        oom_since = None
        loss = torch.tensor(lv)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            raise RuntimeError(f"[{key}] non-finite gradient at step {step}; refusing to save a corrupted checkpoint")
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        run.append(loss.item())
        if step % 100 == 0:
            log.info(f"[{key}] step {step}/{total} loss {np.mean(run[-100:]):.4f} {(time.time() - t0) / 60:.1f}m")
        if step % CFG["ce_ckpt_every"] == 0 or step == total:
            tmp = ck.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step}, tmp)
            os.replace(tmp, ck)
    model.eval()
    p = score(model, tok, ta[dv.q.values], tb[dv.c.values], spec)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(dv.y.values, p) if dv.y.nunique() > 1 else float("nan")
    tmpd = D / "final.tmp"
    model.save_pretrained(tmpd); tok.save_pretrained(tmpd)
    os.replace(tmpd, final)
    save_json({"dev_auc": auc, "steps": total, "n_train": len(tr), "model": spec["model"]},
              WORK / "reports" / f"ce_train_{key}.json")
    ck.unlink(missing_ok=True)
    log.info(f"[{key}] cross-encoder dev AUC {auc:.5f}")


@torch.no_grad()
def score(model, tok, a, b, spec, bs=None):
    """Length-sorted batches (minimal padding), tokenised one batch ahead on a background thread."""
    bs = bs or spec["infer_batch"]
    n = len(a)
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    waited = 0
    while True:
        batches = [order[i:i + bs] for i in range(0, n, bs)]

        def make(k):
            idx = batches[k]
            return tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=spec["max_len"],
                       padding=True, return_tensors="pt")
        try:
            for k, enc in Prefetch(make, range(len(batches))):
                enc = enc.to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out[batches[k]] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            return out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 16:
                waited += 60
                if waited > 2 * 3600:
                    raise RuntimeError("cross-encoder inference: GPU out of memory for 2 h at the minimum batch")
                log.warning("OOM at the minimum batch; waiting 60s for the shared GPU")
                time.sleep(60)
            bs = max(16, bs // 2)
            log.warning(f"OOM -> inference batch {bs} (restarting this shard)")


def s2_file(split):
    return WORK / "s2" / f"{split}.parquet"


# Scores are cached by (q, c): when the stage-2 set changes (e.g. the ename union), only pairs without a score are
# sent to the GPUs. Work units are shards of the "need" list, named by a hash of that list, so a changed list can
# never pick up shards of an older one.

def cache_path(key, split):
    return WORK / "ce_cache" / f"{key}_{split}.parquet"


def _read_cache(key, split):
    p = cache_path(key, split)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame({"q": np.empty(0, np.int32), "c": np.empty(0, np.int32),
                                                               "ce": np.empty(0, np.float32)})


def _write_cache(key, split, add):
    cur = _read_cache(key, split)
    new = pd.concat([cur, add[["q", "c", "ce"]]], ignore_index=True).drop_duplicates(["q", "c"], keep="last")
    save_parquet(new.reset_index(drop=True), cache_path(key, split))


def cache_scores(key, split, old_s2):
    """Move legacy row-position shards (scores of an older stage-2 set, in its row order) into the keyed cache, then
    rename their folder so they can never be matched against a different set."""
    n = len(old_s2)
    paths = [shard_path(f"ce_scores_{key}", split, i, "npy") for i in range(max(1, math.ceil(n / CFG["ce_shard_pairs"])))]
    if n and all(p.exists() for p in paths):
        ce = np.concatenate([np.load(p) for p in paths])
        if len(ce) == n:
            _write_cache(key, split, old_s2.assign(ce=ce))
            d = paths[0].parent
            os.replace(d, d.with_name(f"{split}_legacy_{int(time.time())}"))
            log.info(f"ce_{key}/{split}: cached {n} legacy scores by (q, c)")


def _need(key, split):
    s2 = pd.read_parquet(s2_file(split), columns=["q", "c"])
    cache = _read_cache(key, split)
    k2 = s2.q.values.astype(np.int64) * (1 << 32) + s2.c.values
    kc = np.sort(cache.q.values.astype(np.int64) * (1 << 32) + cache.c.values)
    from decide import in_sorted
    need = s2[~in_sorted(k2, kc)].reset_index(drop=True)
    import hashlib
    h = hashlib.sha1((need.q.values.astype(np.int64) * (1 << 32) + need.c.values).tobytes()).hexdigest()[:10]
    return need, h


def _need_file(key, split, h):
    return WORK / "ce_need" / f"{key}_{split}_{h}.parquet"


def _need_paths(key, split, h, n):
    return [shard_path(f"ce_scores_{key}", split, i, "npy", sub=h) for i in range(max(1, math.ceil(n / CFG["ce_shard_pairs"])))]


def load_infer_model(key):
    """Inference copy: bf16 weights (autocast already runs every matmul in bf16; bf16 weights drop the per-op casts
    and halve memory -> larger batches) and fused SDPA attention where the architecture supports it."""
    AutoTokenizer, AutoModel = _hf()
    d = ce_dir(key) / "final"
    tok = AutoTokenizer.from_pretrained(d)
    kw = {"attn_implementation": "sdpa"} if "xlm" in (d / "config.json").read_text().lower() else {}
    dt = torch.bfloat16 if CFG["infer_bf16"] else torch.float32
    model = AutoModel.from_pretrained(d, dtype=dt, **kw)
    model = gpu_patient(lambda: model.cuda().eval(), log, "load cross-encoder")
    return tok, model


def _text_shard(key, split, h, i):
    return WORK / "ce_need" / f"{key}_{split}_{h}" / f"shard_{i:05d}.parquet"


def _write_text_shards(key, split, h, need, paths):
    """The stage process materialises each unfinished shard's text pairs once, so every GPU worker holds only the
    model and one shard in RAM (loading the texts of all 10M pool records per worker made them the largest processes
    on the box - and the kernel's OOM killer's first choice when other users fill the RAM)."""
    todo = [i for i, p in enumerate(paths) if not p.exists() and not _text_shard(key, split, h, i).exists()]
    if not todo:
        return
    ta, tb = texts(split, "s1", key), texts(split, "pool", key)
    S = CFG["ce_shard_pairs"]
    for i in todo:
        q, c = need.q.values[i * S:(i + 1) * S], need.c.values[i * S:(i + 1) * S]
        save_parquet(pd.DataFrame({"a": ta[q], "b": tb[c]}), _text_shard(key, split, h, i))
    log.info(f"ce_{key}/{split}: wrote text for {len(todo)} shard(s)")


def run_ce_infer_worker(key, split, h):
    need = pd.read_parquet(_need_file(key, split, h), columns=["q"])
    paths = _need_paths(key, split, h, len(need))
    del need
    if all(p.exists() for p in paths):
        return
    spec = CES[key]
    tok, model = load_infer_model(key)

    def job(i):
        t = pd.read_parquet(_text_shard(key, split, h, i))
        save_npy(paths[i], score(model, tok, t.a.values, t.b.values, spec))
    drain(paths, job, log, f"ce_{key} infer {split} (GPU {os.environ.get('CUDA_VISIBLE_DEVICES')})")


def run_ce_infer(key, split):
    need, h = _need(key, split)
    log.info(f"ce_{key}/{split}: {len(need)} pairs need scores (list {h})")
    if len(need):
        f = _need_file(key, split, h)
        if not f.exists():
            save_parquet(need, f)
        paths = _need_paths(key, split, h, len(need))
        _write_text_shards(key, split, h, need, paths)
        import gc
        gc.collect()
        spawn_gpu_workers(["ce_infer_worker", key, split, h], log, CES[key]["mem_infer"])
        while True:                      # helpers on other GPUs may still hold claims on the last shards
            missing = [p for p in paths if not p.exists()]
            if not missing:
                break
            if not any(os.path.exists(str(p) + ".claim") for p in missing):
                raise RuntimeError(f"missing ce_{key} shards {[str(p) for p in missing[:5]]}")
            time.sleep(30)
        _write_cache(key, split, need.assign(ce=np.concatenate([np.load(p) for p in paths])[:len(need)]))
    load_ce_scores(key, split)            # asserts full coverage


def load_ce_scores(key, split):
    s2 = pd.read_parquet(s2_file(split), columns=["q", "c"])
    d = s2.merge(_read_cache(key, split), on=["q", "c"], how="left")
    miss = int(d.ce.isna().sum())
    if miss:
        raise RuntimeError(f"ce_{key}/{split}: {miss} stage-2 pairs have no score")
    return d.rename(columns={"ce": f"ce_{key}"})
