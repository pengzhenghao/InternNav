#!/usr/bin/env bash
set -e

# Usage:
#   export GEMINI_API_KEY=...   # put this in your ~/.bashrc
#   ./scripts/eval/bash/eval_habitat_system3_scene8_torchrun.sh

cd "$(dirname "$0")/../../.."

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in your environment (e.g. ~/.bashrc)}"

export VLLM_API_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export VLLM_API_KEY="$GEMINI_API_KEY"
export MODEL_NAME="gemini-2.5-flash"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS="1"

python -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=2337 \
  scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_system3-ma_test_scene8_cfg.py


