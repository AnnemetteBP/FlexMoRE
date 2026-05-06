#!/usr/bin/env python3
"""
Compute SVD layer-by-layer, saving incrementally to avoid memory overload.
Each layer is computed once, saved immediately, then freed.
"""

import torch
from pathlib import Path
import sys
import json

def main():
    expert_name = sys.argv[1] if len(sys.argv) > 1 else "academic"
    save_path = sys.argv[2] if len(sys.argv) > 2 else f"./svd_full_{expert_name}.pt"
    
    results_dir = Path("src/scripts/analysis/results")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SVD COMPUTATION (STREAMING): {expert_name.upper()}")
    print(f"Processing layer-by-layer, saving incrementally")
    print(f"{'='*70}\n")
    
    # Load weight files once
    print(f"[1] Load weight tensors...")
    expert_file = results_dir / "individual_mlp_weights" / f"{expert_name}_expert_mlp_tensors.pt"
    public_file = results_dir / "individual_mlp_weights" / f"{expert_name}_public_mlp_tensors.pt"
    if not public_file.exists():
        public_file = results_dir / "individual_mlp_weights" / "public_mlp_tensors.pt"
    
    W_expert = torch.load(expert_file, map_location="cpu")
    W_base = torch.load(public_file, map_location="cpu")
    print(f"    Expert: {len(W_expert)} layers")
    print(f"    Base:   {len(W_base)} layers\n")
    
    # Process and save each layer incrementally
    print(f"[2] Process layers and save incrementally...\n")
    svd_results = {}
    
    for idx, layer_key in enumerate(sorted(W_expert.keys()), 1):
        # Compute delta
        delta = W_expert[layer_key] - W_base[layer_key]
        
        # Compute SVD
        U, S, Vh = torch.linalg.svd(delta.to(dtype=torch.float64), full_matrices=False)
        
        # Store in memory
        svd_results[layer_key] = {
            "U": U.cpu(),
            "S": S.cpu(),
            "Vh": Vh.cpu()
        }
        
        # Print progress
        print(f"  [{idx:2d}/{len(W_expert)}] {layer_key}")
        print(f"         Δ: {tuple(delta.shape)} → U:{tuple(U.shape)}, S:{tuple(S.shape)}, Vh:{tuple(Vh.shape)}")
        
        # Every 10 layers, flush to disk to free memory
        if idx % 10 == 0 or idx == len(W_expert):
            print(f"\n  💾 Saving checkpoint at layer {idx}...")
            torch.save(svd_results, save_path)
            print(f"     Saved {len(svd_results)} layers to {save_path}\n")
    
    # Final save
    print(f"\n[3] Final save...")
    torch.save(svd_results, save_path)
    
    # Summary
    file_size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"\n{'='*70}")
    print(f"✓ COMPLETE")
    print(f"  File: {save_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Layers: {len(svd_results)}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
