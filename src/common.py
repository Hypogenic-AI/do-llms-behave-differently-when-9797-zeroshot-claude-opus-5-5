"""Shared helpers: paths, cached OpenRouter calls, surface features."""
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

MODELS = {
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

STYLES = ["orig", "human", "llm", "llm_terse", "human_elab"]

_client = None
_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_KEY"], timeout=90, max_retries=1)
    return _client


class Cache:
    """Append-only jsonl cache keyed by hash of the request."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d = {}
        if self.path.exists():
            for line in open(self.path):
                try:
                    r = json.loads(line)
                    self.d[r["k"]] = r["v"]
                except Exception:
                    pass
        self.f = open(self.path, "a")

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        with _lock:
            self.d[k] = v
            self.f.write(json.dumps({"k": k, "v": v}) + "\n")
            self.f.flush()


def chat(model, messages, cache=None, temperature=0.0, max_tokens=600, retries=5, **kw):
    key = hashlib.sha256(json.dumps([model, messages, temperature, max_tokens, kw], sort_keys=True).encode()).hexdigest()
    if cache is not None and cache.get(key) is not None:
        return cache.get(key)
    out = None
    for i in range(retries):
        try:
            r = client().chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                 max_tokens=max_tokens, **kw)
            out = r.choices[0].message.content or ""
            break
        except Exception as e:  # noqa
            time.sleep(2 * (i + 1))
    if out is None:
        out = "__API_ERROR__"
    elif cache is not None:
        cache.put(key, out)
    return out


def pmap(fn, xs, workers=16, desc=None):
    from tqdm import tqdm
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, xs), total=len(xs), desc=desc, mininterval=5))


POLITE = re.compile(r"\b(please|thank you|thanks|kindly|could you|would you|i would appreciate|i'd appreciate|"
                    r"appreciate|grateful|would you mind|if possible)\b", re.I)
WORD = re.compile(r"[A-Za-z']+")


def surface_features(t):
    words = WORD.findall(t)
    nw = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+\s|\n+", t) if s.strip()]
    letters = [c for c in t if c.isalpha()]
    return {
        "n_chars": len(t),
        "n_words": len(words),
        "log_words": float(__import__("math").log(nw)),
        "polite": len(POLITE.findall(t)),
        "polite_any": int(bool(POLITE.search(t))),
        "mean_word_len": sum(len(w) for w in words) / nw,
        "frac_upper_start": sum(1 for s in sents if s.strip()[0].isupper()) / max(len(sents), 1),
        "frac_caps": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "ends_punct": int(t.strip()[-1:] in ".?!"),
        "n_commas": t.count(",") / nw,
        "has_emdash": int("—" in t),
        "ttr": len(set(w.lower() for w in words)) / nw,
    }


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def save_jsonl(rows, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
