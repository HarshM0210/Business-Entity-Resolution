"""Remote (Colab) lanes for cross-encoder inference, exchanged through a PRIVATE Hugging Face dataset repo.

  python remote.py export --key b --split test --per-lane 20 [--lanes A,B]   claim + package + upload
  python remote.py import --key b --split test                                pull results (runs until done)
  python remote.py status --key b --split test

export: takes the highest-numbered shards of the current need list that are neither finished nor claimed, and
writes a server-side claim "remote-<lane>:0" for each (local GPU workers skip them; common.claim treats a remote
claim as stale after REMOTE_STALE_S without progress, so a dead Colab session never strands work). Each shard is
packaged with its exact text pairs; a parity sample (pairs the server already scored with the same model) lets the
notebook verify it reproduces the server's scores before it uploads anything. The fine-tuned model is uploaded once.
import: downloads each finished shard, checks length and finiteness, writes it atomically where the pipeline's own
workers would have, removes the claim, and refreshes the lane's remaining claims while that lane makes progress.
The token is read from $HF_TOKEN or ~/.hf_m5_token and never written anywhere else."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import get_logger, save_npy, save_parquet, shard_path
from config import CES, CFG, WORK

log = get_logger("remote")
REPO_NAME = os.environ.get("M5_REMOTE_REPO_NAME", "model5-ce-offload")


def _token():
    t = os.environ.get("HF_TOKEN")
    p = Path.home() / ".hf_m5_token"
    if not t and p.exists():
        t = p.read_text().strip()
    if not t:
        raise SystemExit("no HF token: put a WRITE token in ~/.hf_m5_token (chmod 600) or $HF_TOKEN")
    return t


def _api():
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    repo = os.environ.get("M5_REMOTE_REPO") or f"{api.whoami()['name']}/{REPO_NAME}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    return api, repo


def _current(key, split):
    from cross_encoder import _need, _need_file, _need_paths
    need, h = _need(key, split)
    f = _need_file(key, split, h)
    if not f.exists():
        raise SystemExit(f"need list {h} for ce_{key}/{split} not written yet (is ce_infer_{split}_{key} running?)")
    need = pd.read_parquet(f)
    return need, h, _need_paths(key, split, h, len(need))


def _claim_file(p):
    return Path(str(p) + ".claim")


def export(key, split, lanes, per_lane):
    from cross_encoder import texts
    api, repo = _api()
    need, h, paths = _current(key, split)
    tag = f"ce_{key}_{split}_{h}"
    S = CFG["ce_shard_pairs"]
    free = [i for i in range(len(paths)) if not paths[i].exists() and not _claim_file(paths[i]).exists()]
    take = sorted(free, reverse=True)[:per_lane * len(lanes)]
    if not take:
        raise SystemExit("no unclaimed shards left to offload")
    assign = {ln: sorted(take[j::len(lanes)]) for j, ln in enumerate(lanes)}
    for ln, ids in assign.items():                         # claim first, so local workers stop picking them up
        for i in ids:
            _claim_file(paths[i]).write_text(f"remote-{ln}:0")
    log.info(f"{tag}: claimed for remote lanes {assign}")
    out = WORK / "remote" / tag
    out.mkdir(parents=True, exist_ok=True)
    from cross_encoder import _text_shard, _write_text_shards
    _write_text_shards(key, split, h, need, paths)
    import shutil
    for i in take:
        f = out / f"shard_{i:05d}.parquet"
        if not f.exists():
            shutil.copy2(_text_shard(key, split, h, i), f)
    # parity sample: 2,000 pairs the server already scored with this model (train side, same text template)
    from cross_encoder import _read_cache, texts as _t
    cache = _read_cache(key, "train").sample(2000, random_state=0)
    pa, pb = _t("train", "s1", key), _t("train", "pool", key)
    save_parquet(pd.DataFrame({"a": pa[cache.q.values], "b": pb[cache.c.values], "server_logit": cache.ce.values}),
                 out / "parity.parquet")
    spec = CES[key]
    manifest = {"tag": tag, "key": key, "split": split, "list": h, "lanes": assign, "max_len": spec["max_len"],
                "infer_batch": spec["infer_batch"], "model_dir": f"models/ce_{key}", "created": time.time(),
                "scoring": "AutoModelForSequenceClassification fp32 weights, torch.autocast(bfloat16), "
                           "tokenizer(a, b, truncation=True, max_length, padding=True), logits.squeeze(-1), "
                           "batches of infer_batch pairs in ascending len(a)+len(b) order",
                "parity_tolerance_logit": 0.1}     # p99; max <= 20x (cross-GPU bf16 kernels differ slightly)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    have = set(api.list_repo_files(repo, repo_type="dataset"))
    if f"models/ce_{key}/config.json" not in have:
        log.info(f"uploading the fine-tuned model to {repo} (about 2.3 GB, once)")
        api.upload_folder(folder_path=str(WORK / f"ce_{key}" / "final"), path_in_repo=f"models/ce_{key}",
                          repo_id=repo, repo_type="dataset")
    api.upload_folder(folder_path=str(out), path_in_repo=f"jobs/{tag}", repo_id=repo, repo_type="dataset")
    log.info(f"exported {len(take)} shards to {repo}/jobs/{tag}; lanes {list(assign)}")
    print(json.dumps({"repo": repo, "tag": tag, "lanes": assign}, indent=1))


def _results(api, repo, tag):
    fs = api.list_repo_files(repo, repo_type="dataset")
    pre = f"results/{tag}/shard_"
    return sorted(int(f[len(pre):len(pre) + 5]) for f in fs if f.startswith(pre) and f.endswith(".npy"))


def do_import(key, split, poll=60):
    from huggingface_hub import hf_hub_download
    api, repo = _api()
    need, h, paths = _current(key, split)
    tag = f"ce_{key}_{split}_{h}"
    man = json.loads((WORK / "remote" / tag / "manifest.json").read_text())
    lane_of = {i: ln for ln, ids in man["lanes"].items() for i in ids}
    S = CFG["ce_shard_pairs"]
    while True:
        pending = [i for i in lane_of if not paths[i].exists()]
        if not pending:
            log.info(f"{tag}: all remote shards are in")
            return
        progressed = set()
        for i in _results(api, repo, tag):
            if i not in lane_of or paths[i].exists():
                continue
            f = hf_hub_download(repo, f"results/{tag}/shard_{i:05d}.npy", repo_type="dataset",
                                local_dir=str(WORK / "remote" / tag / "dl"), token=_token())
            v = np.load(f)
            n = len(need.q.values[i * S:(i + 1) * S])
            if len(v) != n or not np.isfinite(v).all():
                log.error(f"{tag} shard {i}: bad result (len {len(v)} vs {n}); leaving it to local workers")
            else:
                save_npy(paths[i], v.astype(np.float32))
                log.info(f"{tag} shard {i}: imported from lane {lane_of[i]}")
                progressed.add(lane_of[i])
            _claim_file(paths[i]).unlink(missing_ok=True)
        for i in pending:                                   # a lane that is making progress keeps its claims alive
            if lane_of[i] in progressed and _claim_file(paths[i]).exists():
                os.utime(_claim_file(paths[i]))
        time.sleep(poll)


def status(key, split):
    api, repo = _api()
    need, h, paths = _current(key, split)
    tag = f"ce_{key}_{split}_{h}"
    done_local = sum(p.exists() for p in paths)
    remote = _results(api, repo, tag)
    claims = {p.name: _claim_file(p).read_text() for p in paths if _claim_file(p).exists()}
    print(json.dumps({"repo": repo, "tag": tag, "shards": len(paths), "done": done_local,
                      "remote_results": remote, "claims": claims}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import", "status"])
    ap.add_argument("--key", default="b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lanes", default="A,B")
    ap.add_argument("--per-lane", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.key, a.split, a.lanes.split(","), a.per_lane)
    elif a.cmd == "import":
        do_import(a.key, a.split)
    else:
        status(a.key, a.split)
