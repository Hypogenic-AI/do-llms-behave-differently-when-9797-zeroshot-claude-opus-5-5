"""Write-up analysis (no model calls): src/11_eval_dir_check.py logic over the saved activations and directions,
for the two models whose directions were saved (Qwen, Llama). Adds the random-direction null for |cos|."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
out = []
for key in ["llama", "qwen"]:
    D = np.load(ROOT / f"results/{key}/directions.npz"); L = int(D["best"])
    layers = list(np.load(ROOT / f"data/acts/{key}/layers.npy")); j = layers.index(L)
    X = np.load(ROOT / f"data/acts/{key}/mean.npy", mmap_mode="r")[:, j].astype(np.float32)
    idx = pd.read_parquet(ROOT / f"data/acts/{key}/index.parquet")
    u = lambda v: v / np.linalg.norm(v)
    ev, vo, rg = u(D[f"eval_{L}"]), u(D[f"voice_{L}"]), u(D[f"reg_{L}"])
    rnd = np.random.RandomState(0).randn(2000, len(ev)); null = np.abs((rnd / np.linalg.norm(rnd, axis=1, keepdims=True)) @ ev)
    pe, pv = X @ ev, X @ vo
    o = (idx["style"] == "orig").values; bench = (idx.task != "wild").values
    w = idx.assign(pe=pe).pivot_table(index="id", columns="style", values="pe")
    dv = ((w.llm - w.human_elab) + (w.llm_terse - w.human)) / 2
    out.append(dict(model=key, layer=L, d_model=len(ev), cos_voice_eval=float(vo @ ev), cos_reg_eval=float(rg @ ev),
                    cos_voice_reg=float(vo @ rg), null_abs_cos_95=float(np.percentile(null, 95)),
                    auc_evaldir_bench_vs_wild=roc_auc_score(bench[o], pe[o]), auc_voicedir_bench_vs_wild=roc_auc_score(bench[o], pv[o]),
                    evaldir_voice_shift_sd=float(dv.mean() / pe.std())))
df = pd.DataFrame(out); print(df.round(3).to_string()); df.to_csv(ROOT / "results/paper/eval_direction_check.csv", index=False)
