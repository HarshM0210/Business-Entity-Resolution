"""Resumable DAG runner. Each stage runs in its own process group; finished stages are skipped, failed stages are
retried with back-off, sharded stages resume from their last finished shard. GPU stages go to the card with the
most free memory (several stages may share a card when memory allows); multi-GPU stages get every card with
room and their workers share one shard queue. A watchdog kills and retries stages whose log stops growing.

  python run_pipeline.py            run / resume everything
  python run_pipeline.py --status   show stage states
  python run_pipeline.py --reset S  forget stage S (and everything after it) so it reruns
  python run_pipeline.py --only A,B run only these stages (their dependencies must be done)
"""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import disk_free_gb, get_logger, gpu_status, stage_done
from config import CES, CFG, LCE_ON, LOGS, STATE, WORK

SRC = Path(__file__).resolve().parent
EXIT_LOCKED = 3


def build_dag():
    """name: (deps, resource, gpu-memory key). resource: cpu | light | gpu | gpu_multi."""
    D = {"fetch": ([], "light", None),
         "manifest": (["fetch"], "cpu", None),
         "prep": (["manifest"], "cpu", None),
         "test_tsv": (["prep"], "light", None),
         "splits": (["prep"], "light", None)}
    for sp in ("train", "test"):
        for m in ("e5", "bge", "e5s"):
            D[f"dense_{sp}_{m}"] = (["splits"], "gpu", "dense")
        D[f"lex_{sp}"] = (["splits"], "cpu", None)
    D["hop_train"] = (["dense_train_e5"], "gpu", "hop")
    D["block_train"] = ([f"dense_train_{m}" for m in ("e5", "bge", "e5s")] + ["lex_train", "hop_train"], "cpu", None)
    D["hop_test"] = (["dense_test_e5", "block_train"], "gpu", "hop")
    D["block_test"] = ([f"dense_test_{m}" for m in ("e5", "bge", "e5s")] + ["lex_test", "hop_test", "block_train"],
                       "cpu", None)
    for sp in ("train", "test"):
        for m in ("e5", "bge", "e5s"):
            D[f"cos_{sp}_{m}"] = ([f"block_{sp}"], "gpu", "cos")
        D[f"feat_{sp}"] = ([f"cos_{sp}_{m}" for m in ("e5", "bge", "e5s")], "cpu", None)
    D["f1_select"] = (["feat_train"], "cpu", None)
    D["stage1_train"] = (["f1_select"], "cpu", None)
    D["stage1_test"] = (["feat_test", "stage1_train"], "cpu", None)      # CPU LightGBM: ~6x the GPU evaluator here
    # README v2: ename channel as a guaranteed union after the K cut, with train-side rows (stack/val)
    for sp in ("train", "test"):
        D[f"ename_{sp}"] = ([f"block_{sp}"], "cpu", None)
        D[f"featx_{sp}"] = ([f"feat_{sp}", f"ename_{sp}"], "cpu", None)
        D[f"stage1x_{sp}"] = ([f"featx_{sp}", "stage1_train"], "light", None)
    D["s2set_train"] = (["stage1_train", "stage1x_train"], "light", None)
    D["s2set_test"] = (["stage1_test", "stage1x_test"], "light", None)
    for sp in ("train", "test"):
        D[f"rev_{sp}"] = ([f"s2set_{sp}"], "gpu", "rev")               # whole-table reverse search (R3 / C3)
    for k in CES:
        D[f"ce_train_{k}"] = (["block_train"], "gpu", f"ce_train_{k}")
        D[f"ce_infer_train_{k}"] = ([f"ce_train_{k}", "s2set_train"], "gpu_multi", f"ce_infer_{k}")
        D[f"ce_infer_test_{k}"] = ([f"ce_train_{k}", "s2set_test"], "gpu_multi", f"ce_infer_{k}")
    prelim = [f"ce_infer_train_{k}" for k in CES] + ["rev_train"]
    s2_train_deps = ["stage2_prelim"]
    s2_test_deps = ["stage2_train", "test_tsv", "rev_test"] + [f"ce_infer_test_{k}" for k in CES]
    if LCE_ON:
        D["lce_train"] = (["ce_train_b", "stage1_train"], "gpu", "lce_train")
        D["lce_infer_train"] = (["lce_train", "s2set_train"], "gpu_multi", "lce_infer")
        # test-side listwise inference waits for the stack gate in stage2_train (skipped if not adopted)
        D["lce_infer_test"] = (["lce_train", "s2set_test", "stage2_train"], "gpu_multi", "lce_infer")
        s2_train_deps.append("lce_infer_train")
        s2_test_deps.append("lce_infer_test")
    D["stage2_prelim"] = (prelim, "cpu", None)
    D["stage2_train"] = (s2_train_deps, "cpu", None)
    D["stage2_test"] = (s2_test_deps, "cpu", None)
    D["loco"] = (["stage2_prelim"], "cpu", None)
    D["package"] = (["stage2_test"], "light", None)
    return D


DAG = build_dag()
# critical path first: the long GPU chain (reranker -> listwise model) and the test-side CPU chain
PRIORITY = ["ce_train_b", "lce_train", "ce_train_m", "hop_train", "dense_train_e5", "feat_test", "stage1_test",
            "ename_test", "featx_test", "stage1x_test", "s2set_test", "lce_infer_test", "ce_infer_test_b",
            "ce_infer_test_m", "stage2_prelim",
            "lce_infer_test"]
ORDER = sorted(DAG, key=lambda s: (PRIORITY.index(s) if s in PRIORITY else len(PRIORITY), list(DAG).index(s)))
OPTIONAL = {"loco"}                       # a dead optional stage never blocks the submission


def descendants(name):
    out, frontier = set(), {name}
    while frontier:
        nxt = {s for s, (d, _, _) in DAG.items() if set(d) & frontier} - out
        out |= nxt
        frontier = nxt
    return out


def status():
    for s in DAG:
        f = STATE / f"{s}.fail"
        tries = int(f.read_text()) if f.exists() else 0
        run = (STATE / "running" / s).exists()
        st = "done" if stage_done(s) else ("running" if run else ("FAILED x%d" % tries if tries else "pending"))
        print(f"{s:20s} {st}")


def _kill_group(pgid, sig=signal.SIGTERM):
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def reap_leftovers(log):
    """Stage process groups left behind by a runner that died (they would duplicate work): stop them. All their
    output is atomic/claimed, so the rerun resumes cleanly."""
    d = STATE / "running"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        try:
            pgid = int(f.read_text().split()[0])
        except (ValueError, IndexError):
            pgid = None
        if pgid and _kill_group(pgid):
            log.warning(f"stopped leftover stage {f.name} (pgid {pgid}) from a previous runner")
            time.sleep(2)
            _kill_group(pgid, signal.SIGKILL)
        f.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset")
    ap.add_argument("--only", help="comma-separated stages to run (deps must be done)")
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if a.status:
        return status()
    if a.reset:
        for s in {a.reset} | descendants(a.reset):
            for suf in (".done", ".fail"):
                (STATE / f"{s}{suf}").unlink(missing_ok=True)
        print(f"reset {a.reset} and dependents (shard files are kept; delete them to recompute)")
        return
    lock = open(STATE / "pipeline.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another pipeline run holds the lock")
        sys.exit(EXIT_LOCKED)
    (STATE / "pipeline.pid").write_text(f"{os.getpid()} {os.getpgid(0)}\n")
    log = get_logger("pipeline")

    def _term(signum, frame):              # SIGTERM/SIGINT: stop our stage groups before exiting
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    reap_leftovers(log)
    only = set(a.only.split(",")) if a.only else None
    for s in DAG:                       # a fresh start forgives earlier failures
        (STATE / f"{s}.fail").unlink(missing_ok=True)
    running, fails, retry_at, last_wait_log = {}, {}, {}, 0
    log.info(f"pipeline start: work={WORK} smoke={os.environ.get('M5_SMOKE', '0')} lce={LCE_ON} "
             f"gpus={gpu_status()}")

    def stop_stage(s, why):
        p, gl, t0 = running[s]
        log.error(f"[kill] {s}: {why}")
        _kill_group(p.pid)
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            _kill_group(p.pid, signal.SIGKILL)
            p.wait()

    try:
        while True:
            now = time.time()
            for s, (p, gl, t0) in list(running.items()):
                rc = p.poll()
                if rc is None:                   # watchdog: no log output for too long -> hung
                    lf = LOGS / f"{s}.log"
                    idle = now - max(lf.stat().st_mtime if lf.exists() else t0, t0)
                    if idle > CFG["stage_hang_min"] * 60:
                        stop_stage(s, f"no log output for {idle / 60:.0f} min")
                        rc = -999
                    else:
                        continue
                del running[s]
                (STATE / "running" / s).unlink(missing_ok=True)
                if rc == 0 and stage_done(s):
                    log.info(f"[done] {s} ({(now - t0) / 60:.1f} min) gpus={gl}")
                else:
                    fails[s] = fails.get(s, 0) + 1
                    (STATE / f"{s}.fail").write_text(str(fails[s]))
                    log.error(f"[fail] {s} rc={rc} attempt {fails[s]}/{CFG['max_tries']} -> see {LOGS / (s + '.log')}")
                    retry_at[s] = now + 120 * fails[s]
            dead = {s for s, n in fails.items() if n >= CFG["max_tries"]}
            blocked = set().union(*[descendants(s) for s in dead]) if dead else set()
            pending = [s for s in ORDER if not stage_done(s) and s not in running and s not in dead
                       and s not in blocked and (only is None or s in only)]
            if not pending and not running:
                break
            if disk_free_gb(WORK) < CFG["min_free_disk_gb"]:
                if now - last_wait_log > 600:
                    log.warning(f"only {disk_free_gb(WORK):.0f} GB free on {WORK}; not starting new stages")
                    last_wait_log = now
                time.sleep(60)
                continue
            # stages started < 5 min ago may not have allocated yet: reserve their memory on top of nvidia-smi
            reserved = {}
            for s, (_, gl, t0) in running.items():
                if gl and now - t0 < 300:
                    for g in gl:
                        reserved[g] = reserved.get(g, 0) + CFG["gpu_min_free_mb"][DAG[s][2]]
            n_cpu = sum(1 for s in running if DAG[s][1] == "cpu")
            gstat = None
            for s in pending:
                deps, res, memkey = DAG[s]
                if not all(stage_done(d) for d in deps) or now < retry_at.get(s, 0):
                    continue
                gl = []
                if res == "cpu":
                    if n_cpu >= CFG["max_cpu_stages"]:
                        continue
                    n_cpu += 1
                elif res.startswith("gpu"):
                    need = CFG["gpu_min_free_mb"][memkey]
                    gstat = gstat if gstat is not None else gpu_status()
                    avail = [(g, fr - reserved.get(g, 0)) for g, fr, _ in gstat]
                    free = sorted([x for x in avail if x[1] >= need], key=lambda x: -x[1])
                    if not free:
                        if now - last_wait_log > 600:
                            log.info(f"waiting for a GPU with {need} MB free for {s}: {gstat}")
                            last_wait_log = now
                        continue
                    gl = [str(g[0]) for g in (free if res == "gpu_multi" else free[:1])]
                    for g in gl:
                        reserved[int(g)] = reserved.get(int(g), 0) + need
                env = dict(os.environ, PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES=gl[0] if gl else "",
                           M5_ASSIGNED_GPUS=",".join(gl), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
                out = open(LOGS / f"{s}.log", "a")
                p = subprocess.Popen([sys.executable, "-u", str(SRC / "stages.py"), s], stdout=out,
                                     stderr=subprocess.STDOUT, env=env, cwd=str(SRC), start_new_session=True)
                out.close()
                running[s] = (p, [int(g) for g in gl], now)
                (STATE / "running" / s).write_text(f"{p.pid}\n")
                log.info(f"[start] {s} pid={p.pid} gpus={gl or '-'}")
            time.sleep(15)
    finally:
        for s in list(running):
            stop_stage(s, "runner exiting")
            (STATE / "running" / s).unlink(missing_ok=True)
    bad = [s for s in DAG if not stage_done(s) and (only is None or s in only) and s not in OPTIONAL]
    if bad:
        log.error(f"pipeline stopped with unfinished stages: {bad}")
        sys.exit(1)
    log.info("pipeline complete")


if __name__ == "__main__":
    main()
