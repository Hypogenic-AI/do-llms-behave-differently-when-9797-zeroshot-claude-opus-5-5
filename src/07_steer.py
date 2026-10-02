"""Stage 3: activation steering with the prompt text held exactly fixed.

Base text: the human-style version of every behavioural item. At the chosen layer we add
c * ||d_voice|| * u to the residual stream at the token positions of the user's text only (not the chat
template, not the answer options, not generated tokens), where u is a unit direction:
  voice   LLM-minus-human voice direction (diff-in-means, WildChat train split; length balanced)
  rand_k  random unit directions (k = 0..2), same norm                 -> norm-matched baseline
  reg     long-minus-short register direction (voice balanced), same norm  -> 'length/formality' control
  eval    evaluation-minus-deployment direction (CAA), same norm        -> evaluation-awareness comparison
Readouts: authorship / evaluation self-reports (steering on the quoted message span), A/B logit tasks,
refusal generations (judged later by 05-style judge), A/B probability mass (degradation check).
"""
import sys

import numpy as np
import pandas as pd

from common import DATA, RESULTS, load_jsonl, pmap
from judge_refusal import judge_refusal
from lm import LM
from prompts import PREFILL, behaviour_user, selfreport_score, selfreport_users

key = sys.argv[1]
BASE = "human"
out = RESULTS / key
D = np.load(out / "directions.npz")
L = int(D["best"])
unit = lambda v: v / np.linalg.norm(v)
scale = float(np.linalg.norm(D[f"voice_{L}"]))
rs = np.random.RandomState(123)
dirs = {"voice": unit(D[f"voice_{L}"]), "reg": unit(D[f"reg_{L}"]), "eval": unit(D[f"eval_{L}"])}
for k in range(3):
    dirs[f"rand_{k}"] = unit(rs.randn(len(dirs["voice"])))

CONDS = [("none", 0.0)]
for c in [-4, -2, -1, 1, 2, 4, 8]:
    CONDS.append(("voice", c))
for c in [2, 4, 8]:
    CONDS += [(f"rand_{k}", c) for k in range(3)]
for c in [-4, 4]:
    CONDS += [("reg", c), ("eval", c)]
GEN_CONDS = [("none", 0.0), ("voice", -4), ("voice", 4), ("voice", 8), ("rand_0", 4), ("rand_1", 4), ("rand_2", 4),
             ("rand_0", 8), ("reg", 4), ("reg", -4), ("eval", 4), ("eval", -4)]

lm = LM(key)
items = [it for it in load_jsonl(DATA / "rewrites.jsonl") if it["task"] != "wild"]
print(key, "layer", L, "scale", scale, "items", len(items), flush=True)

mc_items = [it for it in items if "options" in it]
mc_prompts = [lm.fmt(behaviour_user(it, BASE), PREFILL) for it in mc_items]
mc_spans = [lm.span(p, it["texts"][BASE]) for p, it in zip(mc_prompts, mc_items)]
sr = {}
for kind in ["auth", "eval"]:
    ps = [[lm.fmt(u, PREFILL) for u in selfreport_users(it["texts"][BASE], kind)] for it in items]
    sr[kind] = [(list(p), [lm.span(x, it["texts"][BASE]) for x, it in zip(p, items)]) for p in zip(*ps)]

rec = []
for d, c in CONDS:
    spec = None if d == "none" else [(L, c * scale * dirs[d])]
    ab, mass = lm.ab(mc_prompts, steer=spec, spans=mc_spans, return_mass=True)
    srs = {}
    for kind in ["auth", "eval"]:
        (p1, s1), (p2, s2) = sr[kind]
        srs[kind] = selfreport_score(lm.ab(p1, steer=spec, spans=s1), lm.ab(p2, steer=spec, spans=s2))
    for it, v, m in zip(mc_items, ab, mass):
        rec.append(dict(dir=d, c=c, id=it["id"], task=it["task"], kind="ab", value=float(v), mass=float(m)))
    for kind in ["auth", "eval"]:
        for it, v in zip(items, srs[kind]):
            rec.append(dict(dir=d, c=c, id=it["id"], task=it["task"], kind=kind, value=float(v)))
    print(d, c, "auth", srs["auth"].mean().round(3), "eval", srs["eval"].mean().round(3), "mass",
          mass.mean().round(3), flush=True)
pd.DataFrame(rec).to_parquet(out / "steer_mc.parquet")

rf = [it for it in items if it["task"] == "refusal"]
rp = [lm.fmt(behaviour_user(it, BASE)) for it in rf]
rspan = [lm.span(p, it["texts"][BASE]) for p, it in zip(rp, rf)]
gen = []
for d, c in GEN_CONDS:
    spec = None if d == "none" else [(L, c * scale * dirs[d])]
    g = lm.generate(rp, max_new_tokens=80, steer=spec, spans=rspan)
    for it, t in zip(rf, g):
        gen.append(dict(dir=d, c=c, id=it["id"], source=it["source"], response=t))
    print("gen", d, c, flush=True)
gen = pd.DataFrame(gen)
gen["refused"] = pmap(lambda a: judge_refusal(a[0], a[1]),
                      [({it["id"]: it["orig"] for it in rf}[i], r) for i, r in zip(gen.id, gen.response)],
                      workers=24, desc="judge")
gen.to_parquet(out / "steer_gen.parquet")
print(gen.assign(h=gen.source.isin(["jbb_harmful", "xstest_unsafe"])).groupby(["h", "dir", "c"]).refused.mean())
