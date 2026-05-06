#!/bin/bash
# Run full SVD vocabulary projection analysis for all experts

cd /media/am/AM/FlexMoRE

echo "========================================================================"
echo "RUNNING SVD VOCABULARY PROJECTION ANALYSIS"
echo "========================================================================"
echo ""

python3 src/scripts/analysis/analyze_expert_svd_vocabulary_projection.py \
  Academic \
  Code \
  Creative \
  Math \
  News \
  Reddit \
  --base-model Public \
  --model-registry src/scripts/analysis/flexolmo_models.json \
  --results-dir src/scripts/analysis/results \
  --cache-dir .cache/huggingface/hub \
  --output-dir ./svd_analysis \
  --top-k 256 \
  --sigma-threshold 0.0 \
  --kurtosis-threshold 0.0 \
  --energy-share-threshold 0.95 \
  --top-tokens 10 \
  --write-latex-tables

echo ""
echo "========================================================================"
echo "✓ Analysis complete. Results saved to ./svd_analysis/"
echo "========================================================================"
