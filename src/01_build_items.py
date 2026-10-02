"""Build the content pool: one record per item with the original request text.

Tasks
  refusal      JBB harmful/benign + XSTest safe/unsafe            (free-text request)
  syco         Anthropic model-written-evals sycophancy           (bio+question stem, fixed A/B options)
  tqa          TruthfulQA binary choice, no hint                  (question, fixed A/B options)
  tqa_hint     same items, user states belief in the wrong answer
  wild         WildChat first-turn human prompts (probe training / realism checks)
  wild_fewshot WildChat prompts reserved as style exemplars for the human-style rewriter
"""
import csv
import itertools
import json
import random
import re

from datasets import load_dataset

from common import DATA, save_jsonl

rng = random.Random(0)
items = []

# ---- refusal ----
jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
for split in ["harmful", "benign"]:
    for r in jbb[split]:
        items.append(dict(id=f"jbb_{split}_{r['Index']}", task="refusal", source=f"jbb_{split}",
                          harmful=int(split == "harmful"), category=r["Category"], orig=r["Goal"]))
for r in csv.DictReader(open(DATA / "raw" / "xstest.csv")):
    items.append(dict(id=f"xstest_{r['id']}", task="refusal", source=f"xstest_{r['label']}",
                      harmful=int(r["label"] == "unsafe"), category=r["type"], orig=r["prompt"]))

# ---- opinion sycophancy (natively LLM-written) ----
for name, n in [("nlp_survey", 100), ("philpapers2020", 100), ("political_typology_quiz", 100)]:
    rows = [json.loads(l) for l in open(DATA / "raw" / f"sycophancy_on_{name}.jsonl")]
    rng.shuffle(rows)
    k = 0
    for r in rows:
        q = r["question"]
        nm = r["answer_not_matching_behavior"]
        if isinstance(nm, list):
            if len(nm) != 1:
                continue
            nm = nm[0]
        m = re.search(r"\n\s*(?:Choices:\s*\n)?\s*\(A\) (.*)\n \(B\) (.*)\n\nAnswer:\s*$", q, flags=re.S)
        if not m or "(C)" in q:
            continue
        stem = q[: m.start()].strip()
        if not (300 < len(stem) < 1100):
            continue
        items.append(dict(id=f"syco_{name}_{k}", task="syco", source=name, orig=stem,
                          options=[m.group(1).strip(), m.group(2).strip()],
                          match=r["answer_matching_behavior"].strip()[1]))
        k += 1
        if k >= n:
            break

# ---- TruthfulQA ----
tq = list(load_dataset("truthfulqa/truthful_qa", "generation")["validation"])
rng.shuffle(tq)
k = 0
for r in tq:
    inc = r["incorrect_answers"][0] if r["incorrect_answers"] else None
    if not inc or len(r["question"]) > 200 or len(inc) > 120 or len(r["best_answer"]) > 120:
        continue
    correct_pos = rng.randint(0, 1)
    opts = [inc, inc]
    opts[correct_pos] = r["best_answer"]
    base = dict(source="truthfulqa", category=r["category"], options=opts, correct="AB"[correct_pos],
                wrong_answer=inc)
    items.append(dict(id=f"tqa_{k}", task="tqa", orig=r["question"], **base))
    items.append(dict(id=f"tqahint_{k}", task="tqa_hint", match="AB"[1 - correct_pos],
                      orig=f"{r['question']} I think the answer is \"{inc.rstrip('.')}\" but I'm really not sure.",
                      **base))
    k += 1
    if k >= 300:
        break

# ---- WildChat ----
wc = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
seen_ip, wild = set(), []
for r in itertools.islice(wc, 60000):
    if r["language"] != "English" or r["toxic"] or r["redacted"]:
        continue
    t = r["conversation"][0]["content"].strip()
    if not (25 <= len(t) <= 500) or r["hashed_ip"] in seen_ip:
        continue
    if "```" in t or t.count("\n") > 3 or re.search(r"https?://|as an ai|midjourney|\[.*\]|prompt", t, re.I):
        continue
    if sum(c.isascii() for c in t) / len(t) < 0.98:
        continue
    seen_ip.add(r["hashed_ip"])
    wild.append(t)
rng.shuffle(wild)
print("wildchat pool", len(wild))
for i, t in enumerate(wild[:700]):
    items.append(dict(id=f"wild_{i}", task="wild", source="wildchat", orig=t))
save_jsonl([dict(text=t) for t in wild[700:1000]], DATA / "wild_fewshot.jsonl")

save_jsonl(items, DATA / "items.jsonl")
from collections import Counter
print(Counter(i["task"] for i in items))
print(Counter(i["source"] for i in items))
