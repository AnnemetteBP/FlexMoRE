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

MODULE_WEIGHTS = {
    "down_proj": 1.00,
    "gate_proj": 1.25,
    "up_proj": 1.10,
}


def load_singular_values_map(slug: str) -> dict[str, torch.Tensor]:
    path = Path(f"/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/{slug}_singular_values.pt")
    payload = torch.load(path, map_location="cpu")
    return payload["singular_values"]


def load_metric_map(slug: str) -> dict[str, dict]:
    path = Path(f"/media/am/AM/FlexMoRE/src/scripts/analysis/results/expert_mlp_metrics/{slug}_mlp_metrics.json")
    with path.open("r") as f:
        payload = json.load(f)
    return {row["model_key"]: row for row in payload["per_layer_metrics"]}


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


def normalize_feature(value: float, min_value: float, max_value: float) -> float:
    if math.isclose(min_value, max_value):
        return 0.5
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, float(normalized)))


def compute_layer_weight(
    layer_index: int,
    num_layers: int,
    early_weight: float,
    mid_weight: float,
    late_weight: float,
    last_weight: float,
) -> float:
    last_layer = num_layers - 1
    if layer_index == last_layer:
        return last_weight
    early_end = max(0, int(math.floor(0.25 * last_layer)))
    mid_end = max(early_end + 1, int(math.floor(0.75 * last_layer)))
    if layer_index <= early_end:
        return early_weight
    if layer_index <= mid_end:
        return mid_weight
    return late_weight


def main(
    probability_threshold: float = typer.Option(1e-3),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks_v06",
    ),
    num_layers: int = typer.Option(32),
    early_layer_weight: float = typer.Option(0.90),
    mid_layer_weight: float = typer.Option(1.00),
    late_layer_weight: float = typer.Option(1.15),
    last_layer_weight: float = typer.Option(1.25),
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for expert_name, slug in EXPERTS:
        sv_map = load_singular_values_map(slug)
        metric_map = load_metric_map(slug)
        rows = []
        weighted_ranks = []
        raw_ranks = []
        kept_counts = []

        cosine_distances = []
        relative_deltas = []
        provisional_rows = []
        for model_key in sorted(sv_map.keys(), key=model_key_sort_key):
            base_rank, kept_count = compute_probability_rank(sv_map[model_key], probability_threshold)
            metric = metric_map[model_key]
            layer = int(model_key.split(".")[2])
            module = model_key.split(".")[-1]
            cosine_distance = 1.0 - metric["public_expert_cosine"]
            relative_delta_norm = metric["relative_delta_norm"]
            provisional_rows.append(
                {
                    "model_key": model_key,
                    "layer": layer,
                    "module": module,
                    "effective_rank_v02": base_rank,
                    "significant_probabilities": kept_count,
                    "public_expert_cosine": metric["public_expert_cosine"],
                    "relative_delta_norm": relative_delta_norm,
                    "cosine_distance": cosine_distance,
                }
            )
            cosine_distances.append(cosine_distance)
            relative_deltas.append(relative_delta_norm)

        cosine_min, cosine_max = min(cosine_distances), max(cosine_distances)
        delta_min, delta_max = min(relative_deltas), max(relative_deltas)

        for row in provisional_rows:
            module_weight = MODULE_WEIGHTS[row["module"]]
            layer_weight = compute_layer_weight(
                row["layer"],
                num_layers,
                early_layer_weight,
                mid_layer_weight,
                late_layer_weight,
                last_layer_weight,
            )
            cosine_score = normalize_feature(row["cosine_distance"], cosine_min, cosine_max)
            delta_score = normalize_feature(row["relative_delta_norm"], delta_min, delta_max)
            similarity_weight = 0.75 + 0.25 * cosine_score + 0.25 * delta_score
            weighted_rank = max(1, math.ceil(row["effective_rank_v02"] * module_weight * layer_weight * similarity_weight))

            row["module_weight"] = module_weight
            row["layer_weight"] = layer_weight
            row["similarity_weight"] = similarity_weight
            row["effective_rank_v06"] = weighted_rank
            rows.append(row)
            raw_ranks.append(row["effective_rank_v02"])
            weighted_ranks.append(weighted_rank)
            kept_counts.append(row["significant_probabilities"])

        max_rank = max(weighted_ranks)
        payload = {
            "expert": expert_name,
            "method": "v06_probability_threshold_weighted_similarity",
            "probability_threshold": probability_threshold,
            "weights": {
                "module_weights": MODULE_WEIGHTS,
                "early_layer_weight": early_layer_weight,
                "mid_layer_weight": mid_layer_weight,
                "late_layer_weight": late_layer_weight,
                "last_layer_weight": last_layer_weight,
                "similarity_weight_formula": "0.75 + 0.25*cosine_score + 0.25*delta_score",
            },
            "summary": {
                "num_targets": len(rows),
                "min_effective_rank_v02": min(raw_ranks),
                "max_effective_rank_v02": max(raw_ranks),
                "min_effective_rank_v06": min(weighted_ranks),
                "max_effective_rank_v06": max_rank,
                "mean_effective_rank_v06": sum(weighted_ranks) / len(weighted_ranks),
                "median_effective_rank_v06": sorted(weighted_ranks)[len(weighted_ranks) // 2],
                "mean_significant_probabilities": sum(kept_counts) / len(kept_counts),
                "next_power_of_two_ge_max": next_power_of_two(max_rank),
            },
            "rows": rows,
        }

        threshold_label = f"{probability_threshold:.0e}".replace("-", "m")
        out_path = output_root / f"{slug}_probability_threshold_{threshold_label}_weighted_similarity_from_singular_values.json"
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
