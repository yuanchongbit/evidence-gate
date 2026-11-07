#!/usr/bin/env bash
# v2 controls: energy-matched ablations, no-op patching, prompt formats,
# strict presence on the main absent arm, resolution, single-estimator
# per-layer transfer, and a non-Qwen backbone (SmolVLM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODELS="${MODELS:-$ROOT/models}"

# --- Qwen2.5-VL-3B -------------------------------------------------------
python -m pilot.extra44 --model "$MODELS/qwen25vl3b" \
  --run runs/v5_261.jsonl --out runs/extra44_3b.jsonl \
  --stats results/extra44_3b_stats.json
python -m pilot.patch6 --model "$MODELS/qwen25vl3b" --layers 24,32 \
  --out runs/patch6_3b.jsonl
python -m pilot.prompt_variants --model "$MODELS/qwen25vl3b" \
  --out runs/prompt_3b.jsonl
python -m pilot.presence_main --model "$MODELS/qwen25vl3b" \
  --out runs/presence_3b.jsonl
python -m pilot.resolution3b --model "$MODELS/qwen25vl3b" \
  --max-pixels 262144 --tag 512 --out runs/res3b_512.jsonl
python -m pilot.resolution3b --model "$MODELS/qwen25vl3b" \
  --max-pixels 589824 --tag 768 --out runs/res3b_768.jsonl
python -m pilot.layer_transfer --run runs/lens_261.npz \
  --out results/layer_transfer.json
python -m pilot.analyze44 --extra runs/extra44_3b.jsonl \
  --patch6 runs/patch6_3b.jsonl --prompt runs/prompt_3b.jsonl \
  --presence runs/presence_3b.jsonl --native-run runs/v5_261.jsonl \
  --out results/batch44_3b.json

# --- Qwen2.5-VL-7B -------------------------------------------------------
python -m pilot.extra44 --model "$MODELS/qwen7b" \
  --run runs/v5_261_7b.jsonl --out runs/extra44_7b.jsonl \
  --stats results/extra44_7b_stats.json
python -m pilot.patch6 --model "$MODELS/qwen7b" --layers 18,26 \
  --out runs/patch6_7b.jsonl
python -m pilot.prompt_variants --model "$MODELS/qwen7b" \
  --out runs/prompt_7b.jsonl
python -m pilot.presence_main --model "$MODELS/qwen7b" \
  --out runs/presence_7b.jsonl
python -m pilot.analyze44 --extra runs/extra44_7b.jsonl \
  --patch6 runs/patch6_7b.jsonl --prompt runs/prompt_7b.jsonl \
  --presence runs/presence_7b.jsonl --native-run runs/v5_261_7b.jsonl \
  --out results/batch44_7b.json

# --- LLaVA-OneVision-7B (non-Qwen vision stack) --------------------------
python -m pilot.extra44 --family gen --model "$MODELS/llava_ov_7b" --resize 512 \
  --lite --run runs/v5_261_llava.jsonl --out runs/extra44_llava.jsonl \
  --stats results/extra44_llava_stats.json
python -m pilot.patch6 --family gen --model "$MODELS/llava_ov_7b" --resize 512 \
  --layers 18,26 --out runs/patch6_llava.jsonl
python -m pilot.prompt_variants --family gen --model "$MODELS/llava_ov_7b" --resize 512 \
  --out runs/prompt_llava.jsonl
python -m pilot.presence_main --family gen --model "$MODELS/llava_ov_7b" --resize 512 \
  --out runs/presence_llava.jsonl
python -m pilot.analyze44 --extra runs/extra44_llava.jsonl \
  --patch6 runs/patch6_llava.jsonl --prompt runs/prompt_llava.jsonl \
  --presence runs/presence_llava.jsonl --native-run runs/v5_261_llava.jsonl \
  --out results/batch44_llava.json

# --- Scene-fit stratification of the absent arm (CPU) --------------------
python -m pilot.plausibility \
  --run runs/v5_261.jsonl runs/v5_261_7b.jsonl runs/v5_261_llava.jsonl \
  --lens runs/lens_261.npz runs/v5_261_7b.npz runs/v5_261_llava.npz \
  --out results/plausibility.json

# --- SmolVLM-Instruct (non-Qwen language backbone) -----------------------
python -m pilot.infer_gen --model "$MODELS/SmolVLM-Instruct" \
  --out runs/v5_261_smol.jsonl --resize 512
python -m pilot.extra44 --family gen --model "$MODELS/SmolVLM-Instruct" --resize 512 \
  --lite --run runs/v5_261_smol.jsonl --out runs/extra44_smol.jsonl \
  --stats results/extra44_smol_stats.json
python -m pilot.prompt_variants --family gen --model "$MODELS/SmolVLM-Instruct" \
  --resize 512 --out runs/prompt_smol.jsonl
python -m pilot.presence_main --family gen --model "$MODELS/SmolVLM-Instruct" \
  --resize 512 --out runs/presence_smol.jsonl
python -m pilot.analyze44 --extra runs/extra44_smol.jsonl \
  --prompt runs/prompt_smol.jsonl --presence runs/presence_smol.jsonl \
  --native-run runs/v5_261_smol.jsonl --out results/batch44_smol.json
