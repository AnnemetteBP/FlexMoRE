#!/bin/bash
# Compute full SVD decomposition from delta weights
# Usage: ./compute_svds.sh academic ./output/academic_svd.pt

set -e

EXPERT=${1:-academic}
SAVE_PATH=${2:-./"svd_${EXPERT}.pt"}

echo "========================================================================"
echo "SVD COMPUTATION: $EXPERT"
echo "========================================================================"
echo ""

python3 << 'PYTHON_EOF'
import torch
from pathlib import Path
import sys

expert_name = sys.argv[1]
save_path = Path(sys.argv[2])
save_path.parent.mkdir(parents=True, exist_ok=True)

results_dir = Path("src/scripts/analysis/results")

# Load weights
print(f"[1] Loading weights for: {expert_name}")
expert_file = results_dir / "individual_mlp_weights" / f"{expert_name}_expert_mlp_tensors.pt"
public_file = results_dir / "individual_mlp_weights" / f"{expert_name}_public_mlp_tensors.pt"

if not public_file.exists():
    public_file = results_dir / "individual_mlp_weights" / "public_mlp_tensors.pt"

W_expert = torch.load(expert_file, map_location="cpu")
W_base = torch.load(public_file, map_location="cpu")

print(f"    ✓ Expert weights: {len(W_expert)} layers")
print(f"    ✓ Base weights: {len(W_base)} layers")
print()

# Compute SVD for each layer
print(f"[2] Computing SVD (layer-by-layer)...")
print()

svd_data = {}
layer_keys = sorted(W_expert.keys())

for idx, layer_key in enumerate(layer_keys, 1):
    # Compute delta
    delta = W_expert[layer_key] - W_base[layer_key]
    
    # SVD
    U, S, Vh = torch.linalg.svd(delta.to(dtype=torch.float64), full_matrices=False)
    
    # Save
    svd_data[layer_key] = {
        "U": U.cpu(),
        "S": S.cpu(),
        "Vh": Vh.cpu()
    }
    
    # Progress
    pct = int(100 * idx / len(layer_keys))
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"  [{bar}] {pct:3d}% | {layer_key}")
    
    # Checkpoint every 10 layers
    if idx % 10 == 0:
        torch.save(svd_data, save_path)

print()
print(f"[3] Saving to: {save_path}")
torch.save(svd_data, save_path)

file_size = save_path.stat().st_size / 1024 / 1024
print(f"    ✓ Saved {len(svd_data)} layers ({file_size:.1f} MB)")
print()
print("========================================================================"
print(f"✓ COMPLETE")
print("========================================================================")

PYTHON_EOF

python3 -c "
import sys
save_path = '${SAVE_PATH}'
from pathlib import Path
p = Path(save_path)
if p.exists():
    size_mb = p.stat().st_size / 1024 / 1024
    print(f'Output: {save_path} ({size_mb:.1f} MB)')
" 2>/dev/null || true
