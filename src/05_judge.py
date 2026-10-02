"""Judge refusal for every generated response of the given models (behaviour.parquet -> adds `refused`)."""
import sys

import pandas as pd

from common import DATA, RESULTS, load_jsonl, pmap
from judge_refusal import judge_refusal

orig = {it["id"]: it["orig"] for it in load_jsonl(DATA / "rewrites.jsonl")}
for key in sys.argv[1:]:
    p = RESULTS / key / "behaviour.parquet"
    df = pd.read_parquet(p)
    m = df.response.notna()
    sub = df[m]
    df.loc[m, "refused"] = pmap(lambda a: judge_refusal(orig[a[0]], a[1]), list(zip(sub.id, sub.response)),
                                workers=24, desc=key)
    df.to_parquet(p)
    print(key, df[m].groupby(["source", "style"]).refused.mean().unstack().round(3))
