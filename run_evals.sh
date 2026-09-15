#!/usr/bin/env bash
# Batch eval: V2 on 009/010/test_6k, plus baseline & V1 on the new test_6k slice.
set -uo pipefail
cd /home/paradox/Desktop/ai/nosh
PY=/home/paradox/Desktop/ai/ai-env/bin/python

BASE=/home/paradox/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a
V1=./checkpoints/qwen2.5-coder-1.5b-nosh-merged
V2=./checkpoints/qwen2.5-coder-1.5b-nosh-v2-merged

run () {  # name  model  file  out
  echo "=================== EVAL: $1 / $3 ==================="
  $PY eval.py "$3" --nosh-model "$2" --output "results/$1/$4"
  echo "=================== DONE: $1 / $3 (rc=$?) ==================="
}

# Headline: complex held-out set across all three models
run baseline        "$BASE" test_6k.csv      test_6k.csv
run noshfinetunedV1 "$V1"   test_6k.csv      test_6k.csv
run noshfinetunedV2 "$V2"   test_6k.csv      test_6k.csv

# Curated held-out for the new model
run noshfinetunedV2 "$V2"   cleaned/009.csv  009.csv
run noshfinetunedV2 "$V2"   cleaned/010.csv  010.csv

echo "ALL EVALS COMPLETE"
