"""Relation between the authorship representation and evaluation awareness, per model.

- cosine(voice dir, eval dir) at the chosen layer vs the null distribution of |cos| with random directions
- validity of the eval direction: does projecting user-span activations onto it separate original benchmark
  prompts (JBB/XSTest/TruthfulQA/model-written sycophancy) from real WildChat prompts? (AUC)
- does the same projection move with the style manipulation (LLM-voice minus human-voice, paired)?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import DATA, RESULTS

out = []
for key in ["gemma", "llama", "qwen"]:
    D = np.load(RESULTS / key / "directions.npz")
    L = int(D["best"])
    layers = list(np.load(DATA / "acts" / key / "layers.npy"))
    j = layers.index(L)
    X = np.load(DATA / "acts" / key / "mean.npy", mmap_mode="r")[:, j].astype(np.float32)
    idx = pd.read_parquet(DATA / "acts" / key / "index.parquet")
    u = lambda v: v / np.linalg.norm(v)
    ev, vo, rg = u(D[f"eval_{L}"]), u(D[f"voice_{L}"]), u(D[f"reg_{L}"])
    rnd = np.random.RandomState(0).randn(2000, len(ev))
    null = np.abs((rnd / np.linalg.norm(rnd, axis=1, keepdims=True)) @ ev)
    proj_e, proj_v = X @ ev, X @ vo
    o = (idx["style"] == "orig").values
    bench = (idx.task != "wild").values
    auc_eval = roc_auc_score(bench[o], proj_e[o])
    auc_voice_bench = roc_auc_score(bench[o], proj_v[o])
    idx = idx.assign(pe=proj_e, pv=proj_v)
    w = idx.pivot_table(index="id", columns="style", values="pe")
    dvoice_e = (((w.llm - w.human_elab) + (w.llm_terse - w.human)) / 2)
    sd_e = proj_e.std()
    out.append(dict(model=key, layer=L, cos_voice_eval=float(vo @ ev), cos_reg_eval=float(rg @ ev),
                    null_abs_cos_95=float(np.percentile(null, 95)), auc_evaldir_bench_vs_wild=auc_eval,
                    auc_voicedir_bench_vs_wild=auc_voice_bench,
                    evaldir_voice_shift_sd=float(dvoice_e.mean() / sd_e)))
df = pd.DataFrame(out)
print(df.round(3).to_string())
df.to_csv(RESULTS / "analysis" / "eval_direction_check.csv", index=False)
