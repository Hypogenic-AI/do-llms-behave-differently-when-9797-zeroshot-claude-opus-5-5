# Do LLMs behave differently when the prompter is human vs another LLM?

Paper: [`paper_draft/main.pdf`](paper_draft/main.pdf) (LaTeX source: `paper_draft/main.tex`).

## Question
Suppose the same request is written in a human style in one prompt and in an LLM style in another, with the content held fixed. Do safety-relevant behaviours of the responding model change: refusal, opinion sycophancy, and TruthfulQA accuracy with and without a misleading hint? If they do, is the change driven by the model's perception that "this was written by an LLM", or only by surface features such as length and politeness?

## What was done
- **Items.** 2,250 items: JailbreakBench harmful and benign, XSTest safe and unsafe, Anthropic model-written sycophancy evals, TruthfulQA (each item also in a wrong-hint variant), and WildChat first turns.
- **2×2 content-controlled rewrites.** For each item one rewriter model wrote four versions: human voice short and long, and LLM voice short and long. The rewriter was GPT-4.1-mini, Gemini-2.5-Flash, Claude Haiku 4.5 or DeepSeek-V3.1. An LLM judge had to accept all four versions as the same request. 1,963 items were kept: 1,299 behavioural and 664 WildChat. With the original text this gives 5 versions per item. Paired contrasts separate **voice** (register-balanced), **register** (voice-balanced), the **natural** LLM-vs-human comparison, and **paraphrase** (human rewrite vs. original).
- **Responders.** Gemma-2-9B-it, Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. Refusal was labelled by a style-blind LLM judge; the MC tasks were read from A/B logits.
- **Manipulation checks.** Content equivalence was re-checked by Claude Haiku and harmfulness drift by Gemini. GPT-4.1 gave external authorship ratings. The responding models gave order-balanced self-reports of whether each prompt was AI-written and whether it came from an evaluation. A linear authorship probe was trained on the WildChat split only.
- **Analyses.** Paired bootstrap CIs, sign-flip tests with Holm correction, within-item regressions with surface controls, and cosine similarity between the authorship direction and a CAA evaluation-awareness direction.

## Main findings
1. **The manipulation registers internally, not in explicit self-report.** The authorship probe separates the voices with held-out AUC ≥ 0.99 in every layer of all three models, including length-matched and anti-length pairs; a length-only baseline reaches 0.56. Voice moves the probe by 1.9 SD. GPT-4.1 rates the LLM-voice versions as more AI-like (0.35–0.39 vs 0.15). The responding models' own self-reports still lean "human" for every version.
2. **The behavioural effects of voice are small and inconsistent.** None of the 15 register-balanced voice effects (5 behaviours × 3 models) survives Holm correction. The largest is −4.7 pp in Qwen's refusal of harmful requests (nominal p = 0.011). The sign differs across models.
3. **Effects that do survive correction come from elsewhere.**
   - Paraphrasing at all: Llama refuses harmful requests 6.0 pp less on any rewrite than on the benchmark original.
   - Long LLM-style misleading-hint TruthfulQA prompts lower accuracy in all three models (by 4.8 to 10.2 pp). This looks like a content leak: these versions repeat the wrong answer verbatim, whereas the casual rewrites compress it.
4. **The evaluation self-report, not authorship, tracks refusal.** Within items, the model's self-reported P(evaluation) predicts more refusal in both Llama and Qwen, and this survives surface controls. The probe's authorship score does not predict refusal. The authorship and CAA evaluation directions are near-orthogonal, but the CAA direction failed a validity check: it does not separate benchmark prompts from WildChat prompts.

## What is missing (honest status)
- **Activation steering was not finished.** This was the causal test: steer along the voice direction with the text held fixed, against random, register and evaluation directions (`src/07_steer.py`). No steering result file exists. Only a partial Qwen log does, with no random baselines, and it is mentioned in the paper as a limitation only.
- **Gemma has behaviour only.** Its self-reports, probe scores and directions were computed but not saved. Its probe AUCs and cosine in the paper are taken from `logs/pipeline3.log`.
- **Gemma refusal labels were recovered** from the existing judge cache. No new API calls were made; see `paper_draft/scripts/recover_gemma_refusal.py`.

## Layout
- `src/`: experiment pipeline
  - `01_build_items` → `02_rewrite` → `03_run_model` → `04_directions` → `05_judge` → `06_analyze`
  - also `07_steer` (unfinished), `08_validate`, `09_harmfulness`, `10_figures`, `11_eval_dir_check`
- `results/<model>/`: per-model behaviour, self-reports, probe scores and directions
- `results/analysis/`: validation outputs from the earlier session. The contrast files there cover Qwen only.
- `results/paper/`: full three-model analysis used in the paper (contrasts, style means, regressions, correlations, eval-direction check)
- `paper_draft/scripts/`: write-up analysis over saved results (no model or API calls)
  - `recover_gemma_refusal.py`, `analysis.py`, `eval_direction_check.py`, `figures.py`
- `data/`, `models/`: items, rewrites, caches, activations, model weights (git-ignored)
- `logs/`: run logs and the earlier session's transcript

## Rerun
```sh
uv venv && source .venv/bin/activate   # deps: torch, transformers, datasets, openai, pandas, scikit-learn, statsmodels, matplotlib
export OPENROUTER_KEY=[REDACTED] HF_TOKEN=...
cd src && python 01_build_items.py && python 02_rewrite.py && cd ..
sh run_all.sh                           # per-model stage 1, directions, steering, refusal judging (GPU)
cd src && python 08_validate.py && python 09_harmfulness.py && cd ..
python paper_draft/scripts/recover_gemma_refusal.py   # only needed when Gemma labels are missing from its parquet
python paper_draft/scripts/analysis.py
python paper_draft/scripts/eval_direction_check.py
python paper_draft/scripts/figures.py
cd paper_draft && latexmk -pdf main.tex
```
