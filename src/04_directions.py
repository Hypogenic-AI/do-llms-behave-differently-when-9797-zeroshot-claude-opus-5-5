"""Stage 2 for one model: linear 'LLM-authorship' representation.

Training data: the WildChat items only (never the behavioural items). Each WildChat prompt appears in the
2x2 (voice x register) design: human, human_elab (human voice) vs llm_terse, llm (LLM voice). Because each
voice contains a short and a long variant, a voice classifier cannot rely on length alone.

Outputs results/<model>/directions.npz and probe_layers.csv, probe_scores.parquet.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import DATA, RESULTS, surface_features, load_jsonl
from lm import LM
from prompts import EVAL_CAA

key = sys.argv[1]
ad = DATA / "acts" / key
out = RESULTS / key
idx = pd.read_parquet(ad / "index.parquet")
mean = np.load(ad / "mean.npy").astype(np.float32)
layers = np.load(ad / "layers.npy")
texts = {(it["id"], s): t for it in load_jsonl(DATA / "rewrites.jsonl") for s, t in it["texts"].items()}
idx["logw"] = [np.log(max(len(texts[(i, s)].split()), 1)) for i, s in zip(idx.id, idx["style"])]

VOICE = {"human": 0, "human_elab": 0, "llm": 1, "llm_terse": 1}
REG = {"human": 0, "llm_terse": 0, "llm": 1, "human_elab": 1}
wild_ids = sorted(idx[idx.task == "wild"].id.unique())
rng = np.random.RandomState(0)
rng.shuffle(wild_ids)
train_ids = set(wild_ids[: int(0.6 * len(wild_ids))])
idx["split"] = np.where(idx.task != "wild", "behav", np.where(idx.id.isin(train_ids), "train", "test"))

tr = (idx.split == "train") & idx["style"].isin(list(VOICE))
te = (idx.split == "test") & idx["style"].isin(list(VOICE))
y_tr = idx[tr]["style"].map(VOICE).values
y_te = idx[te]["style"].map(VOICE).values

rows = []
dirs = {}
for j, L in enumerate(layers):
    X = mean[:, j]
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-3
    Z = (X - mu) / sd
    # (1) diff-in-means directions on train
    d_voice = X[tr][y_tr == 1].mean(0) - X[tr][y_tr == 0].mean(0)
    r_tr = idx[tr]["style"].map(REG).values
    d_reg = X[tr][r_tr == 1].mean(0) - X[tr][r_tr == 0].mean(0)
    # (2) logistic probe on standardised activations
    clf = LogisticRegression(C=0.01, max_iter=3000).fit(Z[tr], y_tr)
    s = clf.decision_function(Z)
    dm = X @ (d_voice / np.linalg.norm(d_voice))
    te_df = idx[te].assign(s=s[te], dm=dm[te])

    def pair_auc(a, b, col):
        x = te_df[te_df["style"].isin([a, b])]
        return roc_auc_score((x["style"] == b).astype(int), x[col])

    def orig_auc(sc):
        m = (idx.split == "test") & idx["style"].isin(["orig", "llm"])
        return roc_auc_score((idx[m]["style"] == "llm").astype(int), sc[m.values])

    # length-only baseline on the same split
    lclf = LogisticRegression().fit(idx[tr][["logw"]], y_tr)
    rows.append(dict(layer=int(L), auc_probe=roc_auc_score(y_te, s[te]), auc_dm=roc_auc_score(y_te, dm[te]),
                     auc_len=roc_auc_score(y_te, lclf.decision_function(idx[te][["logw"]])),
                     # length-matched contrasts: same register, different voice
                     auc_short=pair_auc("human", "llm_terse", "s"), auc_long=pair_auc("human_elab", "llm", "s"),
                     # anti-length contrast: long human vs short LLM
                     auc_cross=pair_auc("human_elab", "llm_terse", "s"),
                     # genuine human (WildChat original) vs LLM rewrite
                     auc_orig=orig_auc(s),
                     cos_voice_reg=float(d_voice @ d_reg / np.linalg.norm(d_voice) / np.linalg.norm(d_reg)),
                     norm_dvoice=float(np.linalg.norm(d_voice)), norm_mean=float(np.linalg.norm(X[tr], axis=1).mean())))
    dirs[int(L)] = dict(voice=d_voice, reg=d_reg, probe_w=clf.coef_[0] / sd, probe_mu=mu, probe_sd=sd,
                        probe_coef=clf.coef_[0], probe_b=clf.intercept_[0])
    idx[f"probe_{L}"] = s
res = pd.DataFrame(rows)
print(res.round(3).to_string())
res.to_csv(out / "probe_layers.csv", index=False)

# probe/steering layer: fixed at ~40% depth (AUCs are flat across layers, so AUC-based choice would be noise)
nL = int(layers.max()) + 2
best = int(layers[np.argmin(np.abs(layers - 0.4 * nL))])
print("chosen layer", best)
idx["probe"] = idx[f"probe_{best}"]
idx[["id", "task", "source", "style", "split", "logw", "probe"] + [f"probe_{L}" for L in layers]].to_parquet(
    out / "probe_scores.parquet")

# ---- evaluation-awareness direction (CAA-style, answers prefilled) ----
lm = LM(key)
prompts, lab = [], []
for q, yes_eval, no_eval in EVAL_CAA:
    for ans, is_eval in [("Yes", yes_eval == "Yes"), ("No", yes_eval == "No")]:
        prompts.append(lm.fmt(q, ans))
        lab.append(int(is_eval))
lab = np.array(lab)
last, _ = lm.acts(prompts, [(0, 1)] * len(prompts), list(layers))
last = last.astype(np.float32)
save = {"layers": layers, "best": best}
for j, L in enumerate(layers):
    d_eval = last[lab == 1, j].mean(0) - last[lab == 0, j].mean(0)
    dirs[int(L)]["eval"] = d_eval
    for k, v in dirs[int(L)].items():
        save[f"{k}_{L}"] = v
cos = []
for L in layers:
    u = lambda v: v / np.linalg.norm(v)
    cos.append(dict(layer=int(L), cos_voice_eval=float(u(dirs[int(L)]["voice"]) @ u(dirs[int(L)]["eval"])),
                    cos_probe_eval=float(u(dirs[int(L)]["probe_w"]) @ u(dirs[int(L)]["eval"])),
                    cos_reg_eval=float(u(dirs[int(L)]["reg"]) @ u(dirs[int(L)]["eval"]))))
cos = pd.DataFrame(cos)
print(cos.round(3).to_string())
cos.to_csv(out / "direction_cosines.csv", index=False)
np.savez(out / "directions.npz", **save)
