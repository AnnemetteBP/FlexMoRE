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


def load_existing_payload(output_path: Path, model_name: str) -> dict:
    if not output_path.exists():
        return {
            "model_name": model_name,
            "num_targets_completed": 0,
            "singular_values": {},
        }
    return torch.load(output_path, map_location="cpu")


def infer_model_name(output_path: Path) -> str:
    stem = output_path.stem
    if stem.endswith("_singular_values"):
        stem = stem.removesuffix("_singular_values")
    return stem.replace("_", " ").title()


def main(
    model_path: str = typer.Argument(..., help="Checkpoint path to analyze"),
    output_path: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/math_singular_values.pt",
        help="Output .pt path for the singular-value artifact",
    ),
    processes: int = typer.Option(1, help="Number of CPU threads for SVD"),
):
    torch.set_num_threads(processes)

    model_dir = Path(model_path)
    out_path = Path(output_path)
    model_name = infer_model_name(out_path)

    weight_map = get_checkpoint_weight_map(model_path)
    if weight_map is None:
        raise FileNotFoundError(f"No checkpoint index found under {model_path}")

    lora_modules = ["gate_proj", "down_proj", "up_proj"]
    target_model_keys = list_target_model_keys(weight_map, lora_modules)

    payload = load_existing_payload(out_path, model_name)
    singular_values_map = payload.setdefault("singular_values", {})
    completed = set(singular_values_map.keys())

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

        singular_values_map[model_key] = s.detach().cpu()
        payload["num_targets_completed"] = len(singular_values_map)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        print(f"Saved {model_key}", flush=True)

    payload["num_targets"] = len(target_model_keys)
    torch.save(payload, out_path)
    print(f"Wrote singular-value artifact for {len(target_model_keys)} matrices to {out_path}", flush=True)


if __name__ == "__main__":
    typer.run(main)
