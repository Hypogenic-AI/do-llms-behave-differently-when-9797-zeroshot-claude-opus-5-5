"""Stage 1 for one responder model: behaviour, self-reports and activations for every item x style.

Outputs (results/<model>/):
  behaviour.parquet   one row per (item, style): ab logit (A-B) for MCQ tasks, generated response for refusal
  selfreport.parquet  order-averaged logit that the message is AI-written (auth) / from an evaluation (eval)
data/acts/<model>/    last-token and user-span-mean residual activations (float16) + index
"""
import sys

import numpy as np
import pandas as pd

from common import DATA, RESULTS, STYLES, load_jsonl
from lm import LM
from prompts import PREFILL, behaviour_user, selfreport_score, selfreport_users

key = sys.argv[1]
stages = sys.argv[2].split(",") if len(sys.argv) > 2 else ["behaviour", "selfreport", "acts"]
lm = LM(key)
out = RESULTS / key
out.mkdir(parents=True, exist_ok=True)
items = load_jsonl(DATA / "rewrites.jsonl")
rows = [(it, s) for it in items for s in STYLES]
print(key, "items", len(items), "rows", len(rows), "layers", lm.n_layers, flush=True)

if "behaviour" in stages:
    rec = []
    mc = [(it, s) for it, s in rows if "options" in it]
    ab, mass = lm.ab([lm.fmt(behaviour_user(it, s), PREFILL) for it, s in mc], return_mass=True)
    print("A/B probability mass: mean %.3f, 5th pct %.3f" % (mass.mean(), np.percentile(mass, 5)), flush=True)
    for (it, s), v, m in zip(mc, ab, mass):
        rec.append(dict(id=it["id"], task=it["task"], source=it["source"], style=s, ab=float(v), mass=float(m)))
    # no-persona baseline for opinion sycophancy: options only
    base = [it for it in items if it["task"] == "syco"]
    ab0 = lm.ab([lm.fmt(f"(A) {it['options'][0]}\n(B) {it['options'][1]}", PREFILL) for it in base])
    for it, v in zip(base, ab0):
        rec.append(dict(id=it["id"], task="syco", source=it["source"], style="none", ab=float(v)))
    rf = [(it, s) for it, s in rows if it["task"] == "refusal"]
    gen = lm.generate([lm.fmt(behaviour_user(it, s)) for it, s in rf], max_new_tokens=80, progress=True)
    for (it, s), g in zip(rf, gen):
        rec.append(dict(id=it["id"], task=it["task"], source=it["source"], style=s, response=g))
    pd.DataFrame(rec).to_parquet(out / "behaviour.parquet")
    print("behaviour done", flush=True)

if "selfreport" in stages:
    rec = {(it["id"], s): {} for it, s in rows}
    for kind in ["auth", "eval"]:
        p1, p2 = zip(*[[lm.fmt(u, PREFILL) for u in selfreport_users(it["texts"][s], kind)] for it, s in rows])
        (a1, m1), (a2, m2) = lm.ab(list(p1), return_mass=True), lm.ab(list(p2), return_mass=True)
        sc = selfreport_score(a1, a2)
        print(kind, "A/B mass: mean %.3f, 5th pct %.3f" % ((m1.mean() + m2.mean()) / 2,
                                                            np.percentile(np.r_[m1, m2], 5)), flush=True)
        for (it, s), v in zip(rows, sc):
            rec[(it["id"], s)][kind] = float(v)
        print("selfreport", kind, "done", flush=True)
    pd.DataFrame([dict(id=i, style=s, **v) for (i, s), v in rec.items()]).to_parquet(out / "selfreport.parquet")

if "acts" in stages:
    layers = list(range(0, lm.n_layers, 2))
    prompts, spans = [], []
    for it, s in rows:
        f = lm.fmt(behaviour_user(it, s))
        prompts.append(f)
        spans.append(lm.span(f, it["texts"][s]))
    last, mean = lm.acts(prompts, spans, layers)
    ad = DATA / "acts" / key
    ad.mkdir(parents=True, exist_ok=True)
    np.save(ad / "last.npy", last)
    np.save(ad / "mean.npy", mean)
    np.save(ad / "layers.npy", np.array(layers))
    pd.DataFrame([dict(id=it["id"], task=it["task"], source=it["source"], style=s) for it, s in rows]).to_parquet(
        ad / "index.parquet")
    print("acts done", last.shape, flush=True)
