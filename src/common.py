"""Logging, atomic writes, stage state, GPU discovery, and a crash-safe shard work queue."""
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG, LOGS, STATE, WORK


def get_logger(name):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%m-%d %H:%M:%S")
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        log.addHandler(h)
        log.propagate = False
    return log


def _tmp(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp{os.getpid()}")


def save_parquet(df, path):
    tmp = _tmp(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def save_npy(path, arr):
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def save_json(obj, path):
    tmp = _tmp(path)
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def stage_done(name):
    return (STATE / f"{name}.done").exists()


def mark_done(name, info=None):
    save_json({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}, STATE / f"{name}.done")


class Progress:
    def __init__(self, log, total, what, every=60):
        self.log, self.total, self.what, self.every = log, total, what, every
        self.t0 = self.last = time.time()
        self.done = 0

    def step(self, n=1):
        self.done += n
        now = time.time()
        if now - self.last >= self.every or self.done >= self.total:
            self.last = now
            rate = self.done / max(now - self.t0, 1e-9)
            eta = (self.total - self.done) / rate if rate > 0 else float("nan")
            self.log.info(f"{self.what}: {self.done}/{self.total} ({100 * self.done / max(self.total, 1):.1f}%) "
                          f"elapsed {(now - self.t0) / 60:.1f}m eta {eta / 60:.1f}m")


def gpu_status():
    """[(index, free_mb, util_pct)] for the GPUs this pipeline may use (M5_GPUS restricts)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    allowed = os.environ.get("M5_GPUS")
    allowed = {int(x) for x in allowed.split(",") if x != ""} if allowed else None
    res = []
    for line in out.strip().splitlines():
        try:
            i, free, util = [int(x.strip()) for x in line.split(",")]
        except ValueError:
            continue
        if allowed is None or i in allowed:
            res.append((i, free, util))
    return res


def gpu_patient(fn, log, what, wait_s=60, max_wait_s=3 * 3600):
    """Run fn(); on CUDA OOM (other users filling the shared GPU) free cache, wait, retry."""
    import torch
    t0 = time.time()
    while True:
        try:
            return fn()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if time.time() - t0 > max_wait_s:
                raise
            log.warning(f"{what}: GPU out of memory (shared card busy), retrying in {wait_s}s")
            time.sleep(wait_s)


def n_workers():
    """Worker processes: the configured cap, reduced when other users already load the (shared) CPUs."""
    ncpu = os.cpu_count() or 8
    try:
        busy = os.getloadavg()[0]
    except OSError:
        busy = 0
    return int(max(8, min(CFG["n_workers"], ncpu - busy - 4)))


def shard_path(kind, split, i, ext="parquet", sub=None):
    base = WORK / kind / split
    if sub:
        base = base / sub
    return base / f"shard_{i:05d}.{ext}"


def read_shards(kind, split, n, columns=None, sub=None, ext="parquet"):
    return pd.concat([pd.read_parquet(shard_path(kind, split, i, ext, sub), columns=columns) for i in range(n)],
                     ignore_index=True)


# ----------------------------------------------------------------------------------------------- work queue

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


REMOTE_STALE_S = 2.5 * 3600      # a remote (Colab) claim with no refresh for this long goes back to local workers


def claim(out_path):
    """Claim the job that produces out_path. Returns True if this process owns it now. A claim is a file created
    with O_EXCL holding host:pid; claims of dead processes (crash, kill, OOM) are broken and re-taken, so any
    number of workers on any GPUs can drain one shard list without double work or lost shards."""
    out_path = Path(out_path)
    if out_path.exists():
        return False
    c = out_path.with_name(out_path.name + ".claim")
    c.parent.mkdir(parents=True, exist_ok=True)
    me = f"{socket.gethostname()}:{os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            if out_path.exists():            # finished between our check and our claim
                c.unlink(missing_ok=True)
                return False
            return True
        except FileExistsError:
            try:
                host, pid = c.read_text().split(":")
                if host.startswith("remote"):         # a Colab lane (src/remote.py): stale after REMOTE_STALE_S
                    stale = time.time() - c.stat().st_mtime > REMOTE_STALE_S
                else:
                    stale = host == socket.gethostname() and not _pid_alive(int(pid))
            except (ValueError, FileNotFoundError):
                stale = True
            if not stale:
                return False
            c.unlink(missing_ok=True)
    return False


def release(out_path):
    Path(str(out_path) + ".claim").unlink(missing_ok=True)


def drain(paths, fn, log, what):
    """Run fn(i) for every index whose output does not exist yet, claiming each one first. Safe to run from
    several processes at once. Returns the number of jobs this process completed."""
    todo = [i for i, p in enumerate(paths) if not Path(p).exists()]
    prog = Progress(log, len(todo), what)
    n = 0
    for i in todo:
        if not claim(paths[i]):
            continue
        try:
            fn(i)
            n += 1
        finally:
            release(paths[i])
        prog.step()
    return n


def spawn_gpu_workers(stage_args, log, min_free_mb):
    """Start one worker process per assigned GPU (python stages.py <stage_args...>), wait for all, fail if any
    failed. Workers share the job list through claim()."""
    gpus = [g for g in os.environ.get("M5_ASSIGNED_GPUS", "0").split(",") if g != ""]
    src = Path(__file__).resolve().parent
    procs = []
    for g in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, M5_ASSIGNED_GPUS=g)
        # the trailing owner tag is ignored by stages.py; it labels the process for anyone listing GPU jobs on the
        # shared account (other sessions have been killing unrecognised GPU processes)
        procs.append(subprocess.Popen([sys.executable, "-u", str(src / "stages.py"), *stage_args,
                                       "--owner=model-5-pipeline-DO-NOT-KILL"], env=env, cwd=str(src)))
        log.info(f"worker {stage_args} on GPU {g} pid={procs[-1].pid}")
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"GPU workers {stage_args} failed: {codes}")


def disk_free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9
