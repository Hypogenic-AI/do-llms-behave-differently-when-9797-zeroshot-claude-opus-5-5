#!/bin/sh
# Full GPU pipeline (after src/01_build_items.py and src/02_rewrite.py). Run from repo root.
set -e
cd src
export HF_HOME=$PWD/../models/hf
for m in qwen llama gemma; do
  python 03_run_model.py $m
  python 04_directions.py $m
done
for m in qwen llama gemma; do python 07_steer.py $m; done
python 05_judge.py qwen llama gemma
