#!/usr/bin/env python3
"""Compute and save full SVD (U, S, Vh) from delta weights."""

import torch
from pathlib import Path
import sys

def compute_and_save_svd(expert_name: str, results_dir: Path, output_dir: Path):
    """Compute SVD from delta weights and save full decomposition."""
    
    # Load weights
    expert_file = results_dir / "individual_mlp_weights" / f"{expert_name}_expert_mlp_tensors.pt"
    public_file = results_dir / "individual_mlp_weights" / f"{expert_name}_public_mlp_tensors.pt"
    
    if not public_file.exists():
        public_file = results_dir / "individual_mlp_weights" / "public_mlp_tensors.pt"
    
    print(f"Loading {expert_file}")
    W_expert = torch.load(expert_file, map_location="cpu")
    
    print(f"Loading {public_file}")
    W_base = torch.load(public_file, map_location="cpu")
    
    # Compute SVD for all layers
    svd_results = {}
    for layer_key in W_expert.keys():
        delta = W_expert[layer_key] - W_base[layer_key]
        print(f"\nComputing SVD for {layer_key}: shape {tuple(delta.shape)}")
        
        # Compute SVD EXACTLY ONCE
        U, S, Vh = torch.linalg.svd(delta.to(dtype=torch.float64), full_matrices=False)
        
        print(f"  U: {tuple(U.shape)}")
        print(f"  S: {tuple(S.shape)}")
        print(f"  Vh: {tuple(Vh.shape)}")
        
        svd_results[layer_key] = {
            "U": U.cpu(),
            "S": S.cpu(),
            "Vh": Vh.cpu()
        }
    
    # Save full decomposition
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{expert_name}_full_svd.pt"
    torch.save(svd_results, output_file)
    
    print(f"\n✓ SAVED: {output_file}")
    print(f"  Layers: {len(svd_results)}")
    
    return output_file, svd_results


if __name__ == "__main__":
    expert_name = sys.argv[1] if len(sys.argv) > 1 else "academic"
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    results_dir = Path("/media/am/AM/FlexMoRE/src/scripts/analysis/results")
    
    if save_path:
        output_dir = Path(save_path).parent
    else:
        output_dir = Path("/media/am/AM/FlexMoRE/svd_full_decompositions")
    
    file_path, data = compute_and_save_svd(expert_name, results_dir, output_dir)
    
    if save_path:
        # Rename to the specified path
        final_path = Path(save_path)
        file_path.rename(final_path)
        file_path = final_path
    
    print(f"\n{'='*60}")
    print(f"Output file: {file_path}")
    print(f"File exists: {file_path.exists()}")
    print(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")
