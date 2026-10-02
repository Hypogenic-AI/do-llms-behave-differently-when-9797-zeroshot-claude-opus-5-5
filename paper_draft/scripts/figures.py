"""Figures + LaTeX tables for the paper from results/paper/*, results/analysis/validate_*, results/<m>/probe_layers.csv
and (Gemma probe AUCs only) the printed table in logs/pipeline3.log. No model calls."""
import re
from io import StringIO
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
P, A = ROOT / "results/paper", ROOT / "results/analysis"
FIG, TAB = ROOT / "paper_draft/figures", ROOT / "paper_draft/tables"; FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
MODELS = ["gemma", "llama", "qwen"]
MN = {"gemma": "Gemma-2-9B-it", "llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen2.5-7B-Instruct"}
COL = {"gemma": "#2a78d6", "llama": "#eb6834", "qwen": "#1baf7a"}; MK = {"gemma": "o", "llama": "s", "qwen": "D"}
STY = ["orig", "human", "human_elab", "llm_terse", "llm"]
SN = {"orig": "original", "human": "human\nshort", "human_elab": "human\nlong", "llm_terse": "LLM\nshort", "llm": "LLM\nlong"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 220, "lines.linewidth": 1.5})
rng = np.random.default_rng(0)
rows = pd.read_parquet(P / "all_rows.parquet")
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; b = rng.choice(x, (2000, len(x))).mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

# ---- Fig 1 manipulation checks ----
ext = pd.read_csv(A / "validate_external_authorship.csv")
panels = [("probe", "Authorship probe score\n(trained on WildChat split)"), ("auth", "Self-report P(AI-written)\n(logit)"),
          ("eval", "Self-report P(evaluation)\n(logit)"), ("ext", "External rater GPT-4.1\nP(AI-written)")]
fig, axs = plt.subplots(1, 4, figsize=(11, 2.7)); x = np.arange(len(STY))
for ax, (col, title) in zip(axs, panels):
    if col == "ext":
        ms = [ci(ext[ext["style"] == s].p_ai) for s in STY]
        ax.errorbar(x, [m[0] for m in ms], yerr=[[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]], fmt="o", color=MUTED, ms=5, capsize=0)
        ax.set_ylim(0, 0.5)
    else:
        for k, m in enumerate(["llama", "qwen"]):
            sub = rows[(rows.model == m) & rows["style"].isin(STY) & rows[col].notna() & (rows.task != "wild")]
            ms = [ci(sub[sub["style"] == s][col]) for s in STY]
            ax.errorbar(x + (k - .5) * .22, [v[0] for v in ms], yerr=[[v[0] - v[1] for v in ms], [v[2] - v[0] for v in ms]],
                        fmt=MK[m], color=COL[m], ms=4.5, capsize=0, label=MN[m])
    ax.set_xticks(x); ax.set_xticklabels([SN[s] for s in STY], fontsize=7.2); ax.set_title(title, fontsize=8.5)
axs[0].legend(fontsize=7, loc="center right")
fig.tight_layout(); fig.savefig(FIG / "manipulation.pdf"); fig.savefig(FIG / "manipulation.png"); plt.close(fig)

# ---- Fig 2 behavioural contrasts (forest) ----
eff = pd.read_csv(P / "contrasts.csv")
OUTS = [("refusal | harmful", "Refusal, harmful"), ("refusal | benign", "Refusal, benign/borderline"), ("opinion sycophancy", "Opinion sycophancy"),
        ("TruthfulQA acc", "TruthfulQA accuracy"), ("TQA acc | wrong hint", "TruthfulQA acc., wrong hint")]
CON = [("voice", "LLM voice\n(register-bal.)"), ("register", "long/explicit\n(voice-bal.)"), ("natural", "LLM-long vs\nhuman-short"), ("para", "paraphrase\n(human vs orig.)")]
fig, axs = plt.subplots(1, len(OUTS), figsize=(11.5, 3.1), sharey=True)
for ax, (o, t) in zip(axs, OUTS):
    for k, m in enumerate(MODELS):
        for j, (c, _) in enumerate(CON):
            r = eff[(eff.model == m) & (eff.outcome == o) & (eff.contrast == c)].iloc[0]
            y = j + (k - 1) * 0.22
            ax.errorbar(100 * r.est, y, xerr=[[100 * (r.est - r.lo)], [100 * (r.hi - r.est)]], fmt=MK[m], color=COL[m], ms=4, capsize=0,
                        mfc=COL[m] if r.p_holm < 0.05 else "white", label=MN[m] if (j == 0 and o == OUTS[0][0]) else None)
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_title(t, fontsize=8.5); ax.set_xlabel("difference (pp)")
    ax.grid(axis="x", color="#e6e6e3"); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(CON))); axs[0].set_yticklabels([c[1] for c in CON], fontsize=7.5); axs[0].invert_yaxis()
fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "contrasts.pdf"); fig.savefig(FIG / "contrasts.png"); plt.close(fig)

# ---- Fig 3 probe AUC by layer ----
log = (ROOT / "logs/pipeline3.log").read_text()
blk = log[log.index("    layer  auc_probe"):log.index("chosen layer 16")]
blk = "\n".join(l for l in blk.splitlines() if re.match(r"^\s*(layer|\d+\s)", l))
g = pd.read_csv(StringIO(blk), sep=r"\s+")
pl = {"gemma": g, "llama": pd.read_csv(ROOT / "results/llama/probe_layers.csv"), "qwen": pd.read_csv(ROOT / "results/qwen/probe_layers.csv")}
g.to_csv(P / "gemma_probe_layers_from_log.csv", index=False)
NL = {"gemma": 42, "llama": 32, "qwen": 28}
fig, axs = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
for ax, (c, t) in zip(axs, [("auc_long", "Length-matched (long): human-long vs LLM-long"), ("auc_cross", "Anti-length: human-long vs LLM-short"),
                            ("auc_orig", "Real WildChat prompt vs LLM rewrite")]):
    for m in MODELS:
        d = pl[m]; ax.plot(d.layer / NL[m], d[c], marker=MK[m], ms=3.5, color=COL[m], label=MN[m])
    ax.axhline(pl["qwen"].auc_len.iloc[0] if c != "auc_orig" else 0.5, color=MUTED, ls="--", lw=1)
    ax.set_title(t, fontsize=8.2); ax.set_xlabel("relative layer depth"); ax.set_ylim(0.45, 1.01)
axs[0].set_ylabel("held-out ROC-AUC"); axs[2].legend(fontsize=7, loc="lower right")
axs[0].text(0.02, pl["qwen"].auc_len.iloc[0] + 0.015, "length-only baseline (all pairs)", fontsize=6.5, color=MUTED)
fig.tight_layout(); fig.savefig(FIG / "probe_layers.pdf"); fig.savefig(FIG / "probe_layers.png"); plt.close(fig)

# ---- Fig 4 style means for TQA wrong hint & harmful refusal ----
sm = pd.read_csv(P / "style_means.csv")
fig, axs = plt.subplots(1, 3, figsize=(10, 2.6))
for ax, (o, t, sc) in zip(axs, [("refusal | harmful", "Refusal rate, harmful requests (%)", 100), ("refusal | benign", "Refusal rate, benign/borderline (%)", 100),
                                ("TQA acc | wrong hint", "TruthfulQA accuracy, wrong hint (%)", 100)]):
    for m in MODELS:
        r = sm[(sm.model == m) & (sm.outcome == o)].iloc[0]
        ax.plot(x, [sc * r[s] for s in STY], marker=MK[m], color=COL[m], ms=4.5, label=MN[m])
    ax.set_xticks(x); ax.set_xticklabels([SN[s] for s in STY], fontsize=7.2); ax.set_title(t, fontsize=8.5)
axs[1].legend(fontsize=7, loc="center right")
fig.tight_layout(); fig.savefig(FIG / "style_means.pdf"); fig.savefig(FIG / "style_means.png"); plt.close(fig)

# ---------------- tables ----------------
def f(v, d=2): return f"{v:.{d}f}"
# surface features
fs = pd.read_csv(P / "surface_by_style.csv", index_col=0)
L = ["\\begin{tabular}{lrrrrrr}\\toprule", "Version & mean words & median words & polite (\\%) & ends w/ punct. (\\%) & caps frac. (\\%) & ext. P(AI) \\\\ \\midrule"]
for s in STY:
    r = fs.loc[s]; L.append(f"{SN[s].replace(chr(10), ' ')} & {r.n_words:.1f} & {r.median_words:.0f} & {100*r.polite_any:.1f} & {100*r.ends_punct:.1f} & {100*r.frac_caps:.1f} & {ext[ext['style']==s].p_ai.mean():.2f} \\\\")
L += ["\\bottomrule\\end{tabular}"]; (TAB / "surface.tex").write_text("\n".join(L))
# behaviour contrasts table
L = ["\\begin{tabular}{llrrrr}\\toprule", "Outcome & Model & voice & register & natural & paraphrase \\\\ \\midrule"]
for o, t in OUTS:
    for i, m in enumerate(MODELS):
        cells = []
        for c, _ in CON:
            r = eff[(eff.model == m) & (eff.outcome == o) & (eff.contrast == c)].iloc[0]
            star = "$^{\\ast\\ast}$" if r.p_holm < 0.05 else ("$^{\\ast}$" if r.p < 0.05 else "")
            cells.append(f"{100*r.est:+.1f} [{100*r.lo:+.1f}, {100*r.hi:+.1f}]{star}")
        L.append(f"{t if i == 0 else ''} & {MN[m].split('-')[0]} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule" if o != OUTS[-1][0] else "")
L += ["\\bottomrule\\end{tabular}"]; (TAB / "contrasts.tex").write_text("\n".join(L))
# manipulation table (logit scale) for llama/qwen
L = ["\\begin{tabular}{llrrrr}\\toprule", "Readout & Model & voice & register & natural & paraphrase \\\\ \\midrule"]
for o, t in [("authorship probe", "Authorship probe"), ("self-report P(AI) logit", "Self-report P(AI)"), ("self-report P(eval) logit", "Self-report P(eval)")]:
    for i, m in enumerate(["llama", "qwen"]):
        cells = []
        for c, _ in CON:
            r = eff[(eff.model == m) & (eff.outcome == o) & (eff.contrast == c)].iloc[0]
            cells.append(f"{r.est:+.2f} ({r.d_sd:+.2f})")
        L.append(f"{t if i == 0 else ''} & {MN[m].split('-')[0]} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule\\end{tabular}"]; (TAB / "manipulation.tex").write_text("\n".join(L))
# regression table
reg = pd.read_csv(P / "within_item_regressions.csv")
RO = [("refusal | harmful", "Refusal, harmful"), ("refusal | benign", "Refusal, benign"), ("opinion sycophancy (logit)", "Sycophancy (logit)"),
      ("TruthfulQA acc (logit)", "TQA acc. (logit)"), ("TQA acc | wrong hint (logit)", "TQA acc., hint (logit)")]
L = ["\\begin{tabular}{ll" + "r" * 6 + "}\\toprule", " & & \\multicolumn{2}{c}{self-report P(AI)} & \\multicolumn{2}{c}{authorship probe} & \\multicolumn{2}{c}{self-report P(eval)} \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}", "Outcome & Model & none & +surface & none & +surface & none & +surface \\\\ \\midrule"]
def cell(r):
    st = "$^{\\ast\\ast}$" if r.p < 0.01 else ("$^{\\ast}$" if r.p < 0.05 else "")
    return f"{r.beta:+.3f}{st}"
for o, t in RO:
    for i, m in enumerate(["llama", "qwen"]):
        cs = [cell(reg[(reg.model == m) & (reg.outcome == o) & (reg.predictor == pr) & (reg.controls == ct)].iloc[0])
              for pr in ["auth", "probe", "eval"] for ct in ["none", "surface"]]
        L.append(f"{t if i == 0 else ''} & {MN[m].split('-')[0]} & " + " & ".join(cs) + " \\\\")
L += ["\\bottomrule\\end{tabular}"]; (TAB / "regressions.tex").write_text("\n".join(L))
# validation table
eq = pd.read_csv(A / "validate_equivalence.csv"); h = pd.read_csv(A / "harmfulness_ratings.csv")
L = ["\\begin{tabular}{lrrrrr}\\toprule", " & original & human short & human long & LLM short & LLM long \\\\ \\midrule"]
L.append("Equivalence (Claude Haiku 4.5, 1--5) & -- & " + " & ".join(f(eq[eq['style']==s].score.mean()) for s in ["human", "human_elab", "llm_terse", "llm"]) + " \\\\")
L.append("\\quad share rated $\\geq 4$ (\\%) & -- & " + " & ".join(f"{100*(eq[eq['style']==s].score>=4).mean():.1f}" for s in ["human", "human_elab", "llm_terse", "llm"]) + " \\\\")
for src, t in [("jbb_harmful", "JBB harmful"), ("xstest_unsafe", "XSTest unsafe"), ("jbb_benign", "JBB benign"), ("xstest_safe", "XSTest safe")]:
    L.append(f"Harmfulness rating, {t} (1--5) & " + " & ".join(f(h[(h.source==src)&(h['style']==s)].harm_rating.mean()) for s in STY) + " \\\\")
L += ["\\bottomrule\\end{tabular}"]; (TAB / "validation.tex").write_text("\n".join(L))
print("ok")
