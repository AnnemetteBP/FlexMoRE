#!/usr/bin/env python3
"""Quick test to verify the analyzer data loading works."""

import torch
from pathlib import Path

def test_data_loading():
    results_dir = Path("/media/am/AM/FlexMoRE/src/scripts/analysis/results")
    expert_name = "academic"

    # Test loading singular values
    svd_file = results_dir / "singular_values" / f"{expert_name}_singular_values.pt"
    if svd_file.exists():
        data = torch.load(svd_file, map_location="cpu")
        print(f"Loaded singular values for {expert_name}: {len(data)} layers")
        for key, values in data.items():
            print(f"  {key}: {len(values)} singular values")
            break  # Just show first one
        return True
    else:
        print(f"Singular values file not found: {svd_file}")
        return False

def test_delta_loading():
    results_dir = Path("/media/am/AM/FlexMoRE/src/scripts/analysis/results")
    expert_name = "academic"

    # Test loading delta weights
    expert_file = results_dir / "individual_mlp_weights" / f"{expert_name}_expert_mlp_tensors.pt"
    public_file = results_dir / "individual_mlp_weights" / f"{expert_name}_public_mlp_tensors.pt"

    if not public_file.exists():
        public_file = results_dir / "individual_mlp_weights" / "public_mlp_tensors.pt"

    if expert_file.exists() and public_file.exists():
        expert_weights = torch.load(expert_file, map_location="cpu")
        public_weights = torch.load(public_file, map_location="cpu")
        print(f"Loaded expert weights: {len(expert_weights)} layers")
        print(f"Loaded public weights: {len(public_weights)} layers")

        # Show first layer
        for key in expert_weights.keys():
            if "model.layers.0." in key:
                delta = expert_weights[key] - public_weights[key]
                print(f"Delta shape for {key}: {delta.shape}")
                break
        return True
    else:
        print(f"Weight files not found: {expert_file} or {public_file}")
        return False

if __name__ == "__main__":
    print("Testing data loading...")
    test_data_loading()
    test_delta_loading()
    print("Test complete.")