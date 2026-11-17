#!/bin/bash
# Start a vLLM OpenAI-compatible server for Qwen3.8-27B (bf16) on both A40 GPUs.
# The venv and the weights live on the container's root disk (not persistent across restarts):
#   VENV:  python venv with vllm==0.34.0
#   MODEL: local copy of Qwen/Qwen3.8-27B (hf download Qwen/Qwen3.8-27B --local-dir $MODEL)
# --reasoning-parser splits the thinking text from the final answer in each response.
# --max-model-len 65536: few-shot prompts are ~15k tokens, plus up to 32k generated tokens.
set -euo pipefail
VENV=${VENV:?set VENV to the vllm venv folder}
MODEL=${MODEL:?set MODEL to the local Qwen3.8-27B folder}
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --port 8000 "$@"
