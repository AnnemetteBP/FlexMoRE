import csv
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


def energy_rank(s: torch.Tensor, tau: float) -> int:
    s = s.to(dtype=torch.float64).flatten()
    if s.numel() == 0:
        return 1

    s = torch.sort(s, descending=True).values
    energy = s.square()
    total_energy = energy.sum()
    if total_energy <= 0:
        return 1

    normalized_energy = energy / total_energy
    cumulative_energy = normalized_energy.cumsum(dim=0)
    k = int(torch.searchsorted(cumulative_energy, torch.tensor(tau, dtype=torch.float64)).item()) + 1
    return max(1, min(k, s.numel()))


def threshold_analysis(s: torch.Tensor, relative_threshold: float = 1e-3, absolute_threshold: float = 1e-8) -> dict:
    s = s.to(dtype=torch.float64).flatten()
    total = s.numel()
    if total == 0:
        return {
            "num_kept": 0,
            "total": 0,
            "fraction_kept": 0.0,
            "threshold": 0.0,
            "max_singular_value": 0.0,
        }

    max_sv = s.max().item()
    threshold = max(max_sv * relative_threshold, absolute_threshold)
    num_kept = int((s >= threshold).sum().item())
    return {
        "num_kept": num_kept,
        "total": total,
        "fraction_kept": num_kept / total,
        "threshold": threshold,
        "max_singular_value": max_sv,
    }


def summarize(rows: list[dict], field: str) -> dict:
    values = [row[field] for row in rows]
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def write_outputs(output_root: Path, rows: list[dict]) -> None:
    if not rows:
        return

    summary = {
        "overall": {
            "energy_rank_90": summarize(rows, "energy_rank_90"),
            "energy_rank_95": summarize(rows, "energy_rank_95"),
            "energy_rank_99": summarize(rows, "energy_rank_99"),
            "fraction_kept": summarize(rows, "fraction_kept"),
        },
        "per_module": {},
    }
    for module_name in ("down_proj", "gate_proj", "up_proj"):
        module_rows = [row for row in rows if row["module"] == module_name]
        if not module_rows:
            continue
        summary["per_module"][module_name] = {
            "energy_rank_90": summarize(module_rows, "energy_rank_90"),
            "energy_rank_95": summarize(module_rows, "energy_rank_95"),
            "energy_rank_99": summarize(module_rows, "energy_rank_99"),
            "fraction_kept": summarize(module_rows, "fraction_kept"),
        }

    with (output_root / "news_singular_value_diagnostics.json").open("w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)

    with (output_root / "news_singular_value_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_key",
                "layer",
                "module",
                "num_singular_values",
                "energy_rank_90",
                "energy_rank_95",
                "energy_rank_99",
                "num_kept",
                "total",
                "fraction_kept",
                "threshold",
                "max_singular_value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows(output_root: Path) -> list[dict]:
    json_path = output_root / "news_singular_value_diagnostics.json"
    if not json_path.exists():
        return []
    with json_path.open("r") as f:
        payload = json.load(f)
    return payload.get("rows", [])


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


def main(
    model_path: str = typer.Argument(..., help="Checkpoint path to analyze"),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news",
        help="Directory to store diagnostics outputs",
    ),
    processes: int = typer.Option(1, help="Number of CPU threads for SVD"),
):
    torch.set_num_threads(processes)

    model_dir = Path(model_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    weight_map = get_checkpoint_weight_map(model_path)
    if weight_map is None:
        raise FileNotFoundError(f"No checkpoint index found under {model_path}")

    lora_modules = ["gate_proj", "down_proj", "up_proj"]
    target_model_keys = list_target_model_keys(weight_map, lora_modules)

    rows = load_existing_rows(output_root)
    completed = {row["model_key"] for row in rows}
    total_targets = len(target_model_keys)
    print(f"Found {total_targets} target matrices", flush=True)
    print(f"Resuming with {len(completed)} already completed", flush=True)

    for idx, model_key in enumerate(target_model_keys, start=1):
        if model_key in completed:
            continue

        expert_key = f"{model_key}.weight"
        public_key = expert_key.replace(".experts.1.", ".experts.0.")
        print(f"[{idx}/{total_targets}] Processing {model_key}", flush=True)

        public = load_tensor_for_key(model_dir, weight_map, public_key).to(dtype=torch.float64)
        expert = load_tensor_for_key(model_dir, weight_map, expert_key).to(dtype=torch.float64)
        delta = expert - public

        _, s, _ = torch.linalg.svd(delta, full_matrices=False)
        threshold_stats = threshold_analysis(s)
        parts = model_key.split(".")
        rows.append(
            {
                "model_key": model_key,
                "layer": int(parts[2]),
                "module": parts[-1],
                "num_singular_values": int(s.numel()),
                "energy_rank_90": energy_rank(s, 0.9),
                "energy_rank_95": energy_rank(s, 0.95),
                "energy_rank_99": energy_rank(s, 0.99),
                **threshold_stats,
            }
        )
        write_outputs(output_root, rows)
        print(
            f"Saved {model_key}: er90={rows[-1]['energy_rank_90']} "
            f"er95={rows[-1]['energy_rank_95']} er99={rows[-1]['energy_rank_99']} "
            f"kept={rows[-1]['num_kept']}/{rows[-1]['total']}",
            flush=True,
        )

    print(f"Wrote diagnostics for {len(rows)} matrices to {output_root}", flush=True)


if __name__ == "__main__":
    typer.run(main)
