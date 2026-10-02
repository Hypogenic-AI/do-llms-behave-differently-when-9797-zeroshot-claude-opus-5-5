"""Write-up analysis over existing results (no new model/API calls). Mirrors src/06_analyze.py but
(i) includes all three responder models (Gemma's refusal labels recovered from the judge cache, Gemma has no
saved self-report/probe outputs), (ii) writes to results/paper/ so the earlier outputs are untouched."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import load_jsonl, surface_features

R = ROOT / "results"; OUT = R / "paper"; OUT.mkdir(exist_ok=True)
items = {it["id"]: it for it in load_jsonl(ROOT / "data/rewrites.jsonl")}
rng = np.random.default_rng(0)
MODELS = ["gemma", "llama", "qwen"]
STY = ["orig", "human", "llm", "llm_terse", "human_elab"]

def load(key):
    bp = OUT / "gemma_behaviour_with_refusal.parquet" if key == "gemma" else R / key / "behaviour.parquet"
    b = pd.read_parquet(bp).drop(columns=["task", "source"])
    if (R / key / "selfreport.parquet").exists():
        sr = pd.read_parquet(R / key / "selfreport.parquet")
        pr = pd.read_parquet(R / key / "probe_scores.parquet")[["id", "style", "probe", "split"]]
        b = sr.merge(pr, on=["id", "style"], how="left").merge(b, on=["id", "style"], how="outer")
    b["task"] = b["id"].map(lambda i: items[i]["task"]); b["source"] = b["id"].map(lambda i: items[i]["source"])
    b["model"] = key
    m = b.task.eq("syco") & b.ab.notna()
    b.loc[m, "syco_logit"] = [a if items[i]["match"] == "A" else -a for i, a in zip(b[m].id, b[m].ab)]
    m = b.task.isin(["tqa", "tqa_hint"]) & b.ab.notna()
    b.loc[m, "acc_logit"] = [a if items[i]["correct"] == "A" else -a for i, a in zip(b[m].id, b[m].ab)]
    b["syco"] = (b.syco_logit > 0).astype(float).where(b.syco_logit.notna())
    b["acc"] = (b.acc_logit > 0).astype(float).where(b.acc_logit.notna())
    b["harm"] = b.source.map({"jbb_harmful": "harmful", "xstest_unsafe": "harmful", "jbb_benign": "benign", "xstest_safe": "benign"})
    for c in ["auth", "eval", "probe"]:
        if c not in b: b[c] = np.nan
    return b

df = pd.concat([load(k) for k in MODELS], ignore_index=True)
df.to_parquet(OUT / "all_rows.parquet")
OUTCOMES = [("refusal | harmful", (df.task == "refusal") & (df.harm == "harmful"), "refused"),
            ("refusal | benign", (df.task == "refusal") & (df.harm == "benign"), "refused"),
            ("opinion sycophancy", df.task == "syco", "syco"),
            ("opinion sycophancy (logit)", df.task == "syco", "syco_logit"),
            ("TruthfulQA acc", df.task == "tqa", "acc"),
            ("TruthfulQA acc (logit)", df.task == "tqa", "acc_logit"),
            ("TQA acc | wrong hint", df.task == "tqa_hint", "acc"),
            ("TQA acc | wrong hint (logit)", df.task == "tqa_hint", "acc_logit"),
            ("self-report P(AI) logit", df.task != "wild", "auth"),
            ("self-report P(eval) logit", df.task != "wild", "eval"),
            ("authorship probe", df.task != "wild", "probe"),
            ("self-report P(AI) logit [WildChat]", df.task == "wild", "auth"),
            ("self-report P(eval) logit [WildChat]", df.task == "wild", "eval"),
            ("authorship probe [WildChat test]", (df.task == "wild") & (df.split == "test"), "probe")]
CON = {"voice": lambda w: ((w.llm - w.human_elab) + (w.llm_terse - w.human)) / 2,
       "register": lambda w: ((w.llm - w.llm_terse) + (w.human_elab - w.human)) / 2,
       "natural": lambda w: w.llm - w.human, "para": lambda w: w.human - w.orig}

def boot(x, n=5000):
    x = np.asarray(x, float); return np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])
def signflip(x, n=20000):
    x = np.asarray(x, float); s = rng.choice([-1, 1], (n, len(x)))
    return float(((np.abs((s * x).mean(1)) >= abs(x.mean()) - 1e-12).sum() + 1) / (n + 1))

rows, means = [], []
for model in MODELS:
    for name, filt, col in OUTCOMES:
        sub = df[filt & (df.model == model)]
        if sub[col].notna().sum() == 0: continue
        w = sub.pivot_table(index="id", columns="style", values=col).dropna(subset=STY)
        sd = sub[sub["style"].isin(STY)][col].std()
        means.append(dict(model=model, outcome=name, n=len(w), **{s: w[s].mean() for s in STY}))
        for c, f in CON.items():
            d = f(w); lo, hi = boot(d)
            rows.append(dict(model=model, outcome=name, contrast=c, n=len(d), est=d.mean(), lo=lo, hi=hi,
                             p=signflip(d), sd_pooled=sd, d_sd=d.mean() / sd if sd > 0 else np.nan))
eff = pd.DataFrame(rows)
beh = ["refusal | harmful", "refusal | benign", "opinion sycophancy", "TruthfulQA acc", "TQA acc | wrong hint"]
eff["p_holm"] = np.nan
for c in CON:  # Holm over 5 behaviours x 3 models per contrast
    m = eff.contrast.eq(c) & eff.outcome.isin(beh); p = eff[m].p.values; o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    eff.loc[m, "p_holm"] = adj
eff.to_csv(OUT / "contrasts.csv", index=False)
pd.DataFrame(means).to_csv(OUT / "style_means.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(means).round(3).to_string()); print(eff.round(4).to_string())

# sycophancy baseline (no persona)
sb = df[(df.task == "syco") & (df["style"] == "none")].groupby("model").syco.mean()
print("syco no-persona baseline", sb.to_dict()); sb.to_csv(OUT / "syco_baseline.csv")
# A/B mass
mass = df[df["style"].isin(STY) & df.mass.notna()].groupby("model").mass.agg(["mean", lambda x: np.percentile(x, 5)])
print(mass); mass.to_csv(OUT / "ab_mass.csv")

# surface features
feat = pd.DataFrame([dict(id=it["id"], task=it["task"], style=s, **surface_features(t)) for it in items.values() for s, t in it["texts"].items()])
fs = feat.groupby("style")[["n_words", "polite_any", "frac_caps", "ends_punct", "mean_word_len", "frac_upper_start", "n_commas", "has_emdash"]].agg(["mean"])
fs.columns = [c[0] for c in fs.columns]; fs["median_words"] = feat.groupby("style").n_words.median()
print(fs.round(3)); fs.to_csv(OUT / "surface_by_style.csv")

# within-item regressions (qwen, llama: models with self-report + probe)
d2 = df.merge(feat.drop(columns=["task"]), on=["id", "style"], how="left"); d2 = d2[d2["style"].isin(STY)]
FEATS = ["log_words", "polite_any", "frac_caps", "ends_punct", "mean_word_len", "frac_upper_start"]
reg = []
for model in ["llama", "qwen"]:
    for name, filt, col in [("refusal | harmful", (d2.task == "refusal") & (d2.harm == "harmful"), "refused"),
                            ("refusal | benign", (d2.task == "refusal") & (d2.harm == "benign"), "refused"),
                            ("opinion sycophancy (logit)", d2.task == "syco", "syco_logit"),
                            ("TruthfulQA acc (logit)", d2.task == "tqa", "acc_logit"),
                            ("TQA acc | wrong hint (logit)", d2.task == "tqa_hint", "acc_logit")]:
        sub = d2[filt & (d2.model == model)].dropna(subset=[col, "auth", "probe"]).copy()
        for v in ["auth", "probe", "eval"] + FEATS: sub[v + "_z"] = (sub[v] - sub[v].mean()) / (sub[v].std() + 1e-9)
        sub["y"] = sub[col]
        for pred in ["auth_z", "probe_z", "eval_z"]:
            for ctrl in ["", " + " + " + ".join(f + "_z" for f in FEATS)]:
                r = smf.ols(f"y ~ {pred}{ctrl} + C(id)", sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.id)[0]})
                reg.append(dict(model=model, outcome=name, predictor=pred[:-2], controls="surface" if ctrl else "none",
                                beta=r.params[pred], se=r.bse[pred], p=r.pvalues[pred], n=len(sub)))
reg = pd.DataFrame(reg); reg.to_csv(OUT / "within_item_regressions.csv", index=False); print(reg.round(4).to_string())

# correlations
cor = []
for model in ["llama", "qwen"]:
    for scope, filt in [("behavioural", df.task != "wild"), ("wild", df.task == "wild")]:
        sub = df[filt & (df.model == model) & df.auth.notna() & df["style"].isin(STY)]
        dm = sub[["auth", "eval", "probe"]] - sub.groupby("id")[["auth", "eval", "probe"]].transform("mean")
        cor.append(dict(model=model, scope=scope, n=len(sub), r_auth_eval=sub.auth.corr(sub["eval"]), r_auth_eval_within=dm.auth.corr(dm["eval"]),
                        r_probe_auth=sub.probe.corr(sub.auth), r_probe_auth_within=dm.probe.corr(dm.auth),
                        r_probe_eval_within=dm.probe.corr(dm["eval"])))
cor = pd.DataFrame(cor); cor.to_csv(OUT / "selfreport_correlations.csv", index=False); print(cor.round(3).to_string())
