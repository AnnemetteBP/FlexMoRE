import json
import math
from pathlib import Path

import torch
import typer


EXPERTS = [
    ("Math", "math"),
    ("News", "news"),
    ("Academic", "academic"),
]


def load_singular_values_map(slug: str) -> dict[str, torch.Tensor]:
    path = Path(f"/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/{slug}_singular_values.pt")
    payload = torch.load(path, map_location="cpu")
    return payload["singular_values"]


def compute_probability_rank(singular_values: torch.Tensor, probability_threshold: float) -> tuple[int, int]:
    singular_values = singular_values.to(dtype=torch.float64).flatten()
    norm = singular_values.sum()
    if norm <= 0 or singular_values.numel() == 0:
        return 1, 0

    probabilities = singular_values / norm
    kept = probabilities[probabilities > probability_threshold]
    kept_count = int(kept.numel())
    if kept_count == 0:
        return 1, 0

    kept = kept / kept.sum()
    entropy = -(kept * kept.log()).sum()
    rank = max(1, min(singular_values.numel(), math.ceil(torch.exp(entropy).item())))
    return rank, kept_count


def next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 if value == 1 else 2 ** math.ceil(math.log2(value))


def model_key_sort_key(model_key: str) -> tuple[int, int]:
    return (
        int(model_key.split(".")[2]),
        {"down_proj": 0, "gate_proj": 1, "up_proj": 2}[model_key.split(".")[-1]],
    )


def main(
    probability_threshold: float = typer.Option(
        1e-3,
        help="Probability threshold for v02 effective rank selection",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks_v02",
        help="Directory to write v02 results",
    ),
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for expert_name, slug in EXPERTS:
        sv_map = load_singular_values_map(slug)
        rows = []
        ranks = []
        kept_counts = []
        for model_key in sorted(sv_map.keys(), key=model_key_sort_key):
            rank, kept_count = compute_probability_rank(sv_map[model_key], probability_threshold)
            layer = int(model_key.split(".")[2])
            module = model_key.split(".")[-1]
            rows.append(
                {
                    "model_key": model_key,
                    "layer": layer,
                    "module": module,
                    "effective_rank_v02": rank,
                    "significant_probabilities": kept_count,
                }
            )
            ranks.append(rank)
            kept_counts.append(kept_count)

        max_rank = max(ranks)
        payload = {
            "expert": expert_name,
            "method": "v02_probability_threshold",
            "probability_threshold": probability_threshold,
            "summary": {
                "num_targets": len(rows),
                "min_effective_rank_v02": min(ranks),
                "max_effective_rank_v02": max_rank,
                "mean_effective_rank_v02": sum(ranks) / len(ranks),
                "median_effective_rank_v02": sorted(ranks)[len(ranks) // 2],
                "mean_significant_probabilities": sum(kept_counts) / len(kept_counts),
                "next_power_of_two_ge_max": next_power_of_two(max_rank),
            },
            "rows": rows,
        }

        threshold_label = f"{probability_threshold:.0e}".replace("-", "m")
        out_path = output_root / f"{slug}_probability_threshold_{threshold_label}_from_singular_values.json"
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
