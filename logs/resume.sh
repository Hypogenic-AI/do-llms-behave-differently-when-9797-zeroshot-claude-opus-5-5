set -e
cd src
export HF_HOME=$PWD/../models/hf
python 03_run_model.py llama acts
python 04_directions.py llama
python 03_run_model.py gemma
python 04_directions.py gemma
for m in qwen llama gemma; do python 07_steer.py $m; done
python 05_judge.py qwen llama gemma
