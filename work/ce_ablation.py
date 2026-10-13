"""One-off ablation (not part of the DAG): stage-2 `ce` config with mDeBERTa only vs both cross-encoders.
Same folds, seeds, cross-fitted calibration, decision families and competitor-claims scoring as the ladder."""
import json, pickle, sys
sys.path.insert(0, "/scratch/user4/dai/model-5/src")
from multiprocessing import Pool
import stage2 as S
from common import n_workers, save_json
from config import WORK
from stacker import MODELS

sp, s2, stack, val = S._load_train()
lam = S._miss_rate(sp, stack)
fams = [("expf05", 0.0), ("expf05", lam)]
full = S.cols(s2, "ce")
monly = [c for c in full if not (c.startswith("ce_b") or c == "ce_mean_qz")]
res = {}
with Pool(max(8, n_workers() // 2)) as pool:
    oof, vp = S._fit("ce_monly", stack, val, monly)
    res["ce_monly"] = S._score_cfg("ce_monly", oof, vp, stack, val, sp, fams, pool)
prelim = pickle.loads((MODELS / "s2_prelim.pkl").read_bytes())
res["ce_both"] = {k: v for k, v in prelim["ce"].items() if k not in ("oof", "vp", "feats")}
res["bge_reranker_gain_stack"] = res["ce_both"]["stack"] - res["ce_monly"]["stack"]
res["bge_reranker_gain_val"] = res["ce_both"]["val_claims"] - res["ce_monly"]["val_claims"]
res["dropped_columns"] = sorted(set(full) - set(monly))
save_json(res, WORK / "reports" / "ce_ablation.json")
print(json.dumps({k: res[k] for k in ("bge_reranker_gain_stack", "bge_reranker_gain_val", "dropped_columns")}, indent=1))
