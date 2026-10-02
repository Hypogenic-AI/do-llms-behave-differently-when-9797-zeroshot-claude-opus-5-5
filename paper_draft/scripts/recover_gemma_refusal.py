"""Write-up helper (no new model or API calls): recover Gemma's refusal labels from the existing refusal-judge
cache (data/cache/refusal_judge.jsonl). The earlier session judged Gemma's responses (logs/judge_gemma.log) but the
labels were not saved into results/gemma/behaviour.parquet. We recompute the exact cache keys used by
src/common.chat and look them up; any missing key is left as NaN (never re-queried)."""
import hashlib, json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from judge_refusal import JUDGE, SYS, TMPL
from common import load_jsonl

cache = {}
for l in open(ROOT / "data/cache/refusal_judge.jsonl"):
    r = json.loads(l); cache[r["k"]] = r["v"]
orig = {it["id"]: it["orig"] for it in load_jsonl(ROOT / "data/rewrites.jsonl")}

def lookup(req, resp):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": TMPL.format(req=req, resp=resp.strip()[:1200])}]
    k = hashlib.sha256(json.dumps([JUDGE, msgs, 0, 8, {}], sort_keys=True).encode()).hexdigest()
    o = cache.get(k)
    if o is None:
        return None
    o = o.strip().lower()
    return 1 if o.startswith("refuse") else 0 if o.startswith("comply") else None

out = ROOT / "results/paper"; out.mkdir(parents=True, exist_ok=True)
for m in ["qwen", "llama", "gemma"]:
    b = pd.read_parquet(ROOT / f"results/{m}/behaviour.parquet")
    r = b[b.response.notna()]
    lab = [lookup(orig[i], t) for i, t in zip(r.id, r.response)]
    r = r.assign(refused_cache=lab)
    if "refused" in r:
        ok = r.refused.notna() & r.refused_cache.notna()
        print(m, "agreement with stored labels", (r.refused[ok] == r.refused_cache[ok]).mean(), ok.sum())
    print(m, "found", r.refused_cache.notna().sum(), "of", len(r))
    if m == "gemma":
        b.loc[r.index, "refused"] = r.refused_cache
        b.to_parquet(out / "gemma_behaviour_with_refusal.parquet")
        print(b[b.response.notna()].groupby(["source", "style"]).refused.mean().unstack().round(3))
