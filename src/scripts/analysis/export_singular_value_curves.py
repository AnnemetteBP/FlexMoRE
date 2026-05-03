import json
import sys
from pathlib import Path

import torch
import typer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.analysis.analysis_efficient_low_rank_expert import (
    checkpoint_key_is_target,
    get_checkpoint_weight_map,
    safe_open,
)


def list_target_model_keys(weight_map: dict[str, str], lora_modules: list[str]) -> list[str]:
    target_keys = []
    for key in weight_map:
        if not checkpoint_key_is_target(key, lora_modules):
            continue
        if ".experts.1." not in key:
            continue
        target_keys.append(key.removesuffix(".weight"))
    return sorted(
        target_keys,
        key=lambda model_key: (
            int(model_key.split(".")[2]),
            {"down_proj": 0, "gate_proj": 1, "up_proj": 2}[model_key.split(".")[-1]],
        ),
    )


def load_tensor_for_key(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    relative_file = weight_map[key]
    file_path = model_dir / relative_file
    if file_path.suffix == ".safetensors":
        if safe_open is None:
            raise ImportError("safetensors is required to read .safetensors checkpoints")
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    state_dict = torch.load(file_path, map_location="cpu")
    return state_dict[key]


def curve_payload(singular_values: torch.Tensor) -> dict:
    singular_values = torch.sort(singular_values.to(dtype=torch.float64).flatten(), descending=True).values
    if singular_values.numel() == 0:
        return {
            "num_singular_values": 0,
            "normalized_singular_values": [],
            "cumulative_energy": [],
        }

    sigma1 = singular_values[0].item()
    normalized = singular_values / sigma1 if sigma1 > 0 else torch.zeros_like(singular_values)
    energy = singular_values.square()
    total_energy = energy.sum()
    cumulative_energy = energy.cumsum(dim=0) / total_energy if total_energy > 0 else torch.zeros_like(energy)
    return {
        "num_singular_values": int(singular_values.numel()),
        "normalized_singular_values": normalized.tolist(),
        "cumulative_energy": cumulative_energy.tolist(),
    }


def load_existing_rows(output_root: Path) -> list[dict]:
    output_path = output_root / "news_singular_value_curves.json"
    if not output_path.exists():
        return []
    with output_path.open("r") as f:
        payload = json.load(f)
    return payload.get("rows", [])


def write_outputs(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "news_singular_value_curves.json").open("w") as f:
        json.dump({"rows": rows}, f)


def main(
    model_path: str = typer.Argument(..., help="Checkpoint path to analyze"),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news",
        help="Directory to store curve outputs",
    ),
    processes: int = typer.Option(1, help="Number of CPU threads for SVD"),
):
    torch.set_num_threads(processes)

    model_dir = Path(model_path)
    output_root = Path(output_dir)
    weight_map = get_checkpoint_weight_map(model_path)
    if weight_map is None:
        raise FileNotFoundError(f"No checkpoint index found under {model_path}")

    lora_modules = ["gate_proj", "down_proj", "up_proj"]
    target_model_keys = list_target_model_keys(weight_map, lora_modules)
    rows = load_existing_rows(output_root)
    completed = {row["model_key"] for row in rows}

    print(f"Found {len(target_model_keys)} target matrices", flush=True)
    print(f"Resuming with {len(completed)} already completed", flush=True)

    for idx, model_key in enumerate(target_model_keys, start=1):
        if model_key in completed:
            continue

        expert_key = f"{model_key}.weight"
        public_key = expert_key.replace(".experts.1.", ".experts.0.")
        print(f"[{idx}/{len(target_model_keys)}] Processing {model_key}", flush=True)

        public = load_tensor_for_key(model_dir, weight_map, public_key).to(dtype=torch.float64)
        expert = load_tensor_for_key(model_dir, weight_map, expert_key).to(dtype=torch.float64)
        delta = expert - public
        _, s, _ = torch.linalg.svd(delta, full_matrices=False)

        parts = model_key.split(".")
        row = {
            "model_key": model_key,
            "layer": int(parts[2]),
            "module": parts[-1],
            **curve_payload(s),
        }
        rows.append(row)
        write_outputs(output_root, rows)
        print(f"Saved {model_key}", flush=True)

    print(f"Wrote singular value curves for {len(rows)} matrices to {output_root}", flush=True)


if __name__ == "__main__":
    typer.run(main)
