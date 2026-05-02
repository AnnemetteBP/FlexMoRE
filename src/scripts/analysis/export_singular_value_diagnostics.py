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
    collect_expert_triplets,
    load_target_tensors_from_checkpoint,
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


def main(
    model_path: str = typer.Argument(..., help="Checkpoint path to analyze"),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news",
        help="Directory to store diagnostics outputs",
    ),
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    expert_tensors = load_target_tensors_from_checkpoint(
        model_path,
        ["gate_proj", "down_proj", "up_proj"],
    )
    triplets = collect_expert_triplets(
        expert_tensors,
        ["gate_proj", "down_proj", "up_proj"],
    )

    rows = []
    total_triplets = len(triplets)
    for idx, (model_key, tensors) in enumerate(sorted(triplets.items()), start=1):
        print(f"[{idx}/{total_triplets}] Processing {model_key}", flush=True)
        delta = tensors["delta"].to(dtype=torch.float64)
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

    print(f"Wrote diagnostics for {len(rows)} matrices to {output_root}", flush=True)


if __name__ == "__main__":
    typer.run(main)
