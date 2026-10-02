"""Figures and LaTeX tables for the paper (reads results/, writes paper_draft/figures and paper_draft/tables)."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS, ROOT

FIG = ROOT / "paper_draft" / "figures"
TAB = ROOT / "paper_draft" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
A = RESULTS / "analysis"
MODELS = ["gemma", "llama", "qwen"]
MNAME = {"gemma": "Gemma-2-9B-it", "llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen2.5-7B-Instruct"}
COL = {"gemma": "#2a78d6", "llama": "#eb6834", "qwen": "#1baf7a"}
STY = ["orig", "human", "human_elab", "llm_terse", "llm"]
SNAME = {"orig": "original", "human": "human\nshort", "human_elab": "human\nlong", "llm_terse": "LLM\nshort",
         "llm": "LLM\nlong"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200})
rng = np.random.default_rng(0)
rows = pd.read_parquet(A / "all_rows.parquet")


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    b = rng.choice(x, (2000, len(x))).mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


# ---------- Figure 1: manipulation checks ----------
ext = pd.read_csv(A / "validate_external_authorship.csv")
panels = [("auth", "Self-report: P(AI-written)\n(logit, order-averaged)"),
          ("eval", "Self-report: P(evaluation)\n(logit, order-averaged)"),
          ("probe", "Authorship probe score\n(held-out WildChat-trained)"),
          ("ext", "External rater (GPT-4.1)\nP(AI-written)")]
fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
x = np.arange(len(STY))
for ax, (col, title) in zip(axs, panels):
    if col == "ext":
        for s_i, s in enumerate(STY):
            m, lo, hi = ci(ext[ext["style"] == s].p_ai)
            ax.errorbar(s_i, m, yerr=[[m - lo], [hi - m]], fmt="o", color="#52514e", ms=5, capsize=0, lw=1.5)
    else:
        for k, mdl in enumerate(MODELS):
            sub = rows[(rows.model == mdl) & (rows["style"] != "none") & rows[col].notna()]
            sub = sub.drop_duplicates(["id", "style"])
            ms = [ci(sub[sub["style"] == s][col]) for s in STY]
            ax.errorbar(x + (k - 1) * 0.18, [m[0] for m in ms], yerr=[[m[0] - m[1] for m in ms],
                                                                       [m[2] - m[0] for m in ms]],
                        fmt="o", color=COL[mdl], ms=4.5, lw=1.5, capsize=0, label=MNAME[mdl])
    ax.set_xticks(x)
    ax.set_xticklabels([SNAME[s] for s in STY], fontsize=7.5)
    ax.set_title(title, fontsize=8.5)
axs[0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "manipulation.pdf")
fig.savefig(FIG / "manipulation.png")

# ---------- Figure 2: behavioural contrasts ----------
eff = pd.read_csv(A / "contrasts.csv")
OUTS = [("refusal | harmful", "Refusal (harmful)", 100), ("refusal | benign", "Refusal (benign/borderline)", 100),
        ("opinion sycophancy", "Opinion sycophancy", 100), ("TruthfulQA acc", "TruthfulQA accuracy", 100),
        ("TQA acc | wrong hint", "TruthfulQA acc. (wrong hint)", 100)]
CON = [("voice", "LLM voice\n(register-balanced)"), ("register", "long/explicit\n(voice-balanced)"),
       ("natural", "LLM-long vs\nhuman-short"), ("para", "paraphrase\n(human vs orig.)")]
fig, axs = plt.subplots(1, len(OUTS), figsize=(11.5, 3.0), sharey=True)
for ax, (o, title, sc) in zip(axs, OUTS):
    for k, mdl in enumerate(MODELS):
        for j, (c, _) in enumerate(CON):
            r = eff[(eff.model == mdl) & (eff.outcome == o) & (eff.contrast == c)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = j + (k - 1) * 0.22
            ax.errorbar(r.est * sc, y, xerr=[[(r.est - r.lo) * sc], [(r.hi - r.est) * sc]], fmt="o", ms=4,
                        color=COL[mdl], lw=1.4, capsize=0, label=MNAME[mdl] if j == 0 else None)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("difference (percentage points)", fontsize=7.5)
axs[0].set_yticks(range(len(CON)))
axs[0].set_yticklabels([c[1] for c in CON], fontsize=7.5)
axs[0].invert_yaxis()
axs[-1].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "contrasts.pdf")
fig.savefig(FIG / "contrasts.png")

# ---------- Figure 3: probe AUC by layer ----------
fig, axs = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
for ax, mdl in zip(axs, MODELS):
    p = pd.read_csv(RESULTS / mdl / "probe_layers.csv")
    for col, lab, c in [("auc_short", "short: human vs LLM", "#2a78d6"), ("auc_long", "long: human vs LLM", "#eb6834"),
                        ("auc_cross", "long human vs short LLM", "#1baf7a"),
                        ("auc_orig", "real WildChat vs LLM rewrite", "#4a3aa7"), ("auc_len", "length only", "#9a9890")]:
        ax.plot(p.layer, p[col], marker="o", ms=3, lw=1.6, color=c, label=lab)
    ax.set_title(MNAME[mdl], fontsize=9)
    ax.set_xlabel("layer")
axs[0].set_ylabel("held-out AUC")
axs[0].set_ylim(0.45, 1.02)
axs[-1].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "probe_auc.pdf")
fig.savefig(FIG / "probe_auc.png")

# ---------- Figure 4: steering dose-response ----------
have = [m for m in MODELS if (RESULTS / m / "steer_mc.parquet").exists()]
if have:
    import json
    items = {}
    for l in open(ROOT / "data" / "rewrites.jsonl"):
        it = json.loads(l)
        items[it["id"]] = it
    METR = [("auth", "Self-report P(AI)\n(logit, Δ vs unsteered)"), ("eval", "Self-report P(eval)\n(logit, Δ)"),
            ("syco", "Opinion sycophancy\n(logit toward user, Δ)"), ("tqa", "TruthfulQA\n(logit toward correct, Δ)"),
            ("tqa_hint", "TQA, wrong hint\n(logit toward correct, Δ)"), ("mass", "P(A)+P(B)\n(degradation check)")]
    fig, axs = plt.subplots(len(have), len(METR), figsize=(13, 2.3 * len(have)), squeeze=False)
    steer_summary = []
    for r_i, mdl in enumerate(have):
        st = pd.read_parquet(RESULTS / mdl / "steer_mc.parquet")
        ab = st[st.kind == "ab"].copy()
        sign = np.array([1 if (items[i].get("match") if t == "syco" else items[i].get("correct")) == "A" else -1
                         for i, t in zip(ab.id, ab.task)])
        ab["value"] = ab.value * sign
        frames = {"auth": st[st.kind == "auth"], "eval": st[st.kind == "eval"],
                  "syco": ab[ab.task == "syco"], "tqa": ab[ab.task == "tqa"], "tqa_hint": ab[ab.task == "tqa_hint"],
                  "mass": ab.assign(value=ab.mass)}
        for c_i, (mk, title) in enumerate(METR):
            ax = axs[r_i, c_i]
            f = frames[mk]
            base = f[f.dir == "none"].set_index("id").value
            for d, colr, lab in [("voice", "#2a78d6", "LLM-voice dir."), ("rand", "#9a9890", "random (3 seeds)"),
                                 ("reg", "#eb6834", "register dir."), ("eval", "#4a3aa7", "eval-awareness dir.")]:
                g = f[f.dir.str.startswith(d)]
                if not len(g):
                    continue
                pts = []
                for c, gg in g.groupby("c"):
                    delta = gg.groupby("id").value.mean() - base.reindex(gg.groupby("id").value.mean().index)
                    if mk == "mass":
                        delta = gg.groupby("id").value.mean()
                    m, lo, hi = ci(delta.values)
                    pts.append((c, m, lo, hi))
                    steer_summary.append(dict(model=mdl, metric=mk, dir=d, c=c, mean=m, lo=lo, hi=hi))
                pts = np.array(sorted(pts))
                ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3.5, lw=1.6, color=colr, label=lab)
                ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=colr, alpha=0.15, lw=0)
            if mk == "mass":
                ax.axhline(base.mean(), color="#0b0b0b", lw=0.8, ls="--")
            else:
                ax.axhline(0, color="#52514e", lw=0.8)
            if r_i == 0:
                ax.set_title(title, fontsize=8)
            if r_i == len(have) - 1:
                ax.set_xlabel("steering coefficient c (× ||Δμ_voice||)", fontsize=7.5)
            if c_i == 0:
                ax.set_ylabel(MNAME[mdl], fontsize=8.5)
    axs[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "steering.pdf")
    fig.savefig(FIG / "steering.png")
    pd.DataFrame(steer_summary).to_csv(A / "steering_summary.csv", index=False)

    # steering refusal table
    gl = []
    for mdl in have:
        p = RESULTS / mdl / "steer_gen.parquet"
        if not p.exists():
            continue
        g = pd.read_parquet(p)
        g["harm"] = np.where(g.source.isin(["jbb_harmful", "xstest_unsafe"]), "harmful", "benign")
        base = g[g.dir == "none"].set_index("id").refused
        for (d, c), gg in g.groupby(["dir", "c"]):
            for h in ["harmful", "benign"]:
                x = gg[gg.harm == h].set_index("id").refused
                delta = (x - base.reindex(x.index)).dropna()
                m, lo, hi = ci(delta.values)
                gl.append(dict(model=mdl, dir=d, c=c, harm=h, rate=x.mean(), delta=m, lo=lo, hi=hi, n=len(delta)))
    if gl:
        gl = pd.DataFrame(gl)
        gl.to_csv(A / "steering_refusal.csv", index=False)
print("figures written")
