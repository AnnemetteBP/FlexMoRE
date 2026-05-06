#!/usr/bin/env python3
"""
Compute full SVD (U, S, Vh) from delta weights with detailed terminal output.
Follow the algorithm step-by-step:
  1. Load expert weights
  2. Load base/public weights
  3. Compute delta = expert - base
  4. Run SVD decomposition
  5. Save U, S, Vh to file
"""

import torch
from pathlib import Path
import sys

def main():
    expert_name = sys.argv[1] if len(sys.argv) > 1 else "academic"
    save_path = sys.argv[2] if len(sys.argv) > 2 else f"./svd_full_{expert_name}.pt"
    
    results_dir = Path("src/scripts/analysis/results")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"FULL SVD COMPUTATION: {expert_name.upper()}")
    print(f"{'='*70}")
    
    # STEP 1: Load expert weights
    print(f"\n[STEP 1] Load expert weights")
    expert_file = results_dir / "individual_mlp_weights" / f"{expert_name}_expert_mlp_tensors.pt"
    print(f"  File: {expert_file}")
    W_expert = torch.load(expert_file, map_location="cpu")
    print(f"  ✓ Loaded {len(W_expert)} layer tensors")
    
    # STEP 2: Load base/public weights
    print(f"\n[STEP 2] Load base/public weights")
    public_file = results_dir / "individual_mlp_weights" / f"{expert_name}_public_mlp_tensors.pt"
    if not public_file.exists():
        public_file = results_dir / "individual_mlp_weights" / "public_mlp_tensors.pt"
    print(f"  File: {public_file}")
    W_base = torch.load(public_file, map_location="cpu")
    print(f"  ✓ Loaded {len(W_base)} layer tensors")
    
    # STEP 3 & 4: Compute delta and SVD for each layer
    print(f"\n[STEP 3-4] Compute delta (expert - base) and SVD decomposition")
    print(f"  Processing {len(W_expert)} layers...\n")
    
    svd_results = {}
    for idx, layer_key in enumerate(sorted(W_expert.keys()), 1):
        # Compute delta
        delta = W_expert[layer_key] - W_base[layer_key]
        
        # Run SVD
        U, S, Vh = torch.linalg.svd(delta.to(dtype=torch.float64), full_matrices=False)
        
        # Store results
        svd_results[layer_key] = {
            "U": U.cpu(),
            "S": S.cpu(),
            "Vh": Vh.cpu()
        }
        
        # Print progress
        print(f"  [{idx:2d}/{len(W_expert)}] {layer_key}")
        print(f"         delta shape: {tuple(delta.shape)}")
        print(f"         U: {tuple(U.shape)}, S: {tuple(S.shape)}, Vh: {tuple(Vh.shape)}")
    
    # STEP 5: Save to file
    print(f"\n[STEP 5] Save full SVD decomposition")
    print(f"  Path: {save_path}")
    torch.save(svd_results, save_path)
    print(f"  ✓ Saved {len(svd_results)} layers")
    
    # Summary
    file_size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"  File: {save_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Layers: {len(svd_results)}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
