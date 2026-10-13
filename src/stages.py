"""Run one pipeline stage: python stages.py <stage>. Writes state/<stage>.done on success.
Worker entry points (started by multi-GPU stages): ce_infer_worker <m|b> <split> <list>, lce_infer_worker <split>,
stage1_test_worker <threads>."""
import sys
import time
import traceback

from common import get_logger, mark_done, stage_done
from config import CES, LCE_ON


def _registry():
    import blocking, cross_encoder as ce, ename, features, package, prep, retrieve_dense as rd, retrieve_lex as rl
    import stacker, stage2
    R = {
        "fetch": prep.run_fetch,
        "manifest": prep.run_manifest,
        "prep": prep.run_prep,
        "test_tsv": prep.run_test_tsv,
        "splits": prep.run_splits,
        "f1_select": stacker.run_f1_select,
        "stage1_train": stacker.run_stage1_train,
        "stage1_test": stacker.run_stage1_test,
        "stage2_prelim": stage2.run_stage2_prelim,
        "stage2_train": stage2.run_stage2_train,
        "stage2_test": stage2.run_stage2_test,
        "loco": stage2.run_loco,
        "package": package.run_package,
    }
    for split in ("train", "test"):
        R[f"lex_{split}"] = (lambda s: lambda: rl.run_lex(s))(split)
        R[f"block_{split}"] = (lambda s: lambda: blocking.run_block(s))(split)
        R[f"hop_{split}"] = (lambda s: lambda: rd.run_hop(s))(split)
        R[f"feat_{split}"] = (lambda s: lambda: features.run_features(s))(split)
        R[f"s2set_{split}"] = (lambda s: lambda: stacker.run_s2set(s))(split)
        R[f"ename_{split}"] = (lambda s: lambda: ename.run_ename(s))(split)
        R[f"featx_{split}"] = (lambda s: lambda: features.run_featx(s))(split)
        R[f"stage1x_{split}"] = (lambda s: lambda: stacker.run_stage1x(s))(split)
        R[f"rev_{split}"] = (lambda s: lambda: rd.run_rev(s))(split)
        for m in ("e5", "bge", "e5s"):
            R[f"dense_{split}_{m}"] = (lambda s, mm: lambda: rd.run_dense(mm, s))(split, m)
            R[f"cos_{split}_{m}"] = (lambda s, mm: lambda: rd.run_cos(mm, s))(split, m)
        for k in CES:
            R[f"ce_infer_{split}_{k}"] = (lambda s, kk: lambda: ce.run_ce_infer(kk, s))(split, k)
    for k in CES:
        R[f"ce_train_{k}"] = (lambda kk: lambda: ce.run_ce_train(kk))(k)
    if LCE_ON:
        import lce
        R["lce_train"] = lce.run_lce_train
        R["lce_infer_train"] = lambda: lce.run_lce_infer("train")
        R["lce_infer_test"] = lambda: lce.run_lce_infer("test")
    return R


def main():
    sys.argv = [a for a in sys.argv if not a.startswith("--owner=")]
    name = sys.argv[1]
    log = get_logger("stage")
    if name == "ce_infer_worker":
        import cross_encoder
        return cross_encoder.run_ce_infer_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    if name == "lce_infer_worker":
        import lce
        return lce.run_lce_infer_worker(sys.argv[2])
    if name == "stage1_test_worker":
        import stacker
        return stacker.run_stage1_test_worker(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    if stage_done(name):
        log.info(f"{name}: already done")
        return
    t0 = time.time()
    log.info(f"{name}: start")
    try:
        _registry()[name]()
    except Exception:
        log.error(f"{name}: FAILED\n{traceback.format_exc()}")
        sys.exit(1)
    mark_done(name, {"minutes": round((time.time() - t0) / 60, 2)})
    log.info(f"{name}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
