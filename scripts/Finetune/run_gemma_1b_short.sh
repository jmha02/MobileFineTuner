#!/bin/bash
set -e

# =============================================================================
# Gemma 3 1B LoRA Finetuning - Short Run (300 steps for convergence analysis)
# =============================================================================

cd /Users/yiyilu/Desktop/FT_gemma_gpt2

echo "=== Gemma 1B LoRA Short Training (300 steps convergence analysis) ==="
echo "Start Time: $(date)"

# Output directory
OUT="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gemma_1b_lora_short_300steps"
rm -rf "$OUT"
mkdir -p "$OUT"
echo "Output directory: $OUT"

# Binary with BLAS optimization
BINARY="/Users/yiyilu/Desktop/FT_gemma_gpt2/operators/build_fast/train_lora_gemma"

echo "Starting background training..."

# Run training (background)
nohup $BINARY \
  --model_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gemma-3-1b" \
  --data_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/wikitext-2-raw" \
  --output_dir "$OUT" \
  --targets attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --max_steps 300 \
  --data_fraction 0.5 \
  --lr 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  > "$OUT/train.log" 2>&1 &

PID=$!
echo "[DONE] Training started in background"
echo "  PID: $PID"
echo "  Log: $OUT/train.log"
echo ""
echo "Monitor command:"
echo "  tail -f $OUT/train.log"
echo ""
echo "Plot loss curve after training completes:"
echo "  python plot_loss_curve.py $OUT/train.log"

