#!/usr/bin/env python3
"""Inspect the structure of the singular values file."""

import torch
from pathlib import Path

def inspect_svd_file():
    results_dir = Path("/media/am/AM/FlexMoRE/src/scripts/analysis/results")
    svd_file = results_dir / "singular_values" / "academic_singular_values.pt"

    if svd_file.exists():
        data = torch.load(svd_file, map_location="cpu")
        print(f"Type of data: {type(data)}")
        print(f"Keys: {list(data.keys())}")

        for key, value in data.items():
            print(f"Key: {key}")
            print(f"  Type: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
            elif isinstance(value, dict):
                print(f"  Dict keys: {list(value.keys())}")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {type(subvalue)} {getattr(subvalue, 'shape', 'no shape')}")
                    break
            else:
                print(f"  Value: {value}")
            print()
    else:
        print(f"File not found: {svd_file}")

if __name__ == "__main__":
    inspect_svd_file()