import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import typer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPERTS = [
    ("Math", "math"),
    ("News", "news"),
    ("Academic", "academic"),
    ("Reddit", "reddit"),
    ("Code", "code"),
    ("Creative", "creative"),
]

DEFAULT_THRESHOLDS = [1e-4, 2e-4, 5e-4, 7e-4, 1e-3]
MODULE_WEIGHTS = {
    "down_proj": 1.00,
    "gate_proj": 1.25,
    "up_proj": 1.10,
}


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


def threshold_label(threshold: float) -> str:
    return f"{threshold:.0e}".replace("-", "m")


def latex_threshold(threshold: float) -> str:
    exponent = int(round(math.log10(threshold)))
    base = threshold / (10 ** exponent)
    if math.isclose(base, 1.0):
        return rf"$10^{{{exponent}}}$"
    if float(base).is_integer():
        base_str = str(int(base))
    else:
        base_str = str(base).rstrip("0").rstrip(".")
    return rf"${base_str}\times 10^{{{exponent}}}$"


def format_mean(value: float) -> str:
    return f"{value:.2f}"


def flatten_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.to(dtype=torch.float64).reshape(-1)
    b_flat = b.to(dtype=torch.float64).reshape(-1)
    denom = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if denom <= 0:
        return 0.0
    return float((torch.dot(a_flat, b_flat) / denom).item())


def normalize_feature(value: float, min_value: float, max_value: float) -> float:
    if math.isclose(min_value, max_value):
        return 0.5
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, float(normalized)))


def compute_layer_weight(
    layer_index: int,
    num_layers: int,
    early_weight: float = 0.90,
    mid_weight: float = 1.00,
    late_weight: float = 1.15,
    last_weight: float = 1.25,
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


def model_key_sort_key(model_key: str) -> tuple[int, int]:
    return (
        int(model_key.split(".")[2]),
        {"down_proj": 0, "gate_proj": 1, "up_proj": 2}[model_key.split(".")[-1]],
    )


def collect_similarity_rows(slug: str, sv_map: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    weights_root = Path("/media/am/AM/FlexMoRE/src/scripts/analysis/results/individual_mlp_weights")
    public_map = torch.load(weights_root / f"{slug}_public_mlp_tensors.pt", map_location="cpu")
    expert_map = torch.load(weights_root / f"{slug}_expert_mlp_tensors.pt", map_location="cpu")
    rows = []
    for model_key in sorted(sv_map.keys(), key=model_key_sort_key):
        public = public_map[model_key]
        expert = expert_map[model_key]
        delta = expert - public

        public_norm = torch.linalg.norm(public.to(dtype=torch.float64)).item()
        delta_norm = torch.linalg.norm(delta.to(dtype=torch.float64)).item()
        rows.append(
            {
                "model_key": model_key,
                "layer": int(model_key.split(".")[2]),
                "module": model_key.split(".")[-1],
                "public_expert_cosine": flatten_cosine_similarity(public, expert),
                "relative_delta_norm": delta_norm / public_norm if public_norm > 0 else 0.0,
            }
        )
    return rows


def generate_rows(thresholds: list[float]) -> list[dict]:
    rows = []
    for expert_name, slug in EXPERTS:
        print(f"Loading {expert_name} ({slug})...", flush=True)
        sv_map = load_singular_values_map(slug)
        similarity_rows = collect_similarity_rows(slug, sv_map)
        sim_by_key = {row["model_key"]: row for row in similarity_rows}
        cosine_distance_values = [1.0 - row["public_expert_cosine"] for row in similarity_rows]
        relative_delta_values = [row["relative_delta_norm"] for row in similarity_rows]
        cosine_min, cosine_max = min(cosine_distance_values), max(cosine_distance_values)
        delta_min, delta_max = min(relative_delta_values), max(relative_delta_values)

        for threshold in thresholds:
            weighted_ranks = []
            raw_ranks = []
            kept_counts = []
            for model_key in sorted(sv_map.keys(), key=model_key_sort_key):
                base_rank, kept_count = compute_probability_rank(sv_map[model_key], threshold)
                sim_row = sim_by_key[model_key]
                module_weight = MODULE_WEIGHTS[sim_row["module"]]
                layer_weight = compute_layer_weight(sim_row["layer"], num_layers=32)
                cosine_score = normalize_feature(1.0 - sim_row["public_expert_cosine"], cosine_min, cosine_max)
                delta_score = normalize_feature(sim_row["relative_delta_norm"], delta_min, delta_max)
                similarity_weight = 0.75 + 0.25 * cosine_score + 0.25 * delta_score
                weighted_rank = max(1, math.ceil(base_rank * module_weight * layer_weight * similarity_weight))
                weighted_ranks.append(weighted_rank)
                raw_ranks.append(base_rank)
                kept_counts.append(kept_count)

            rows.append(
                {
                    "threshold": threshold,
                    "threshold_label": threshold_label(threshold),
                    "expert": expert_name,
                    "slug": slug,
                    "num_targets": len(weighted_ranks),
                    "min_rank": min(weighted_ranks),
                    "mean_rank": sum(weighted_ranks) / len(weighted_ranks),
                    "max_rank": max(weighted_ranks),
                    "median_rank": sorted(weighted_ranks)[len(weighted_ranks) // 2],
                    "mean_significant_probabilities": sum(kept_counts) / len(kept_counts),
                    "selected_rank": next_power_of_two(max(weighted_ranks)),
                    "base_max_rank_v02": max(raw_ranks),
                }
            )
        print(f"Finished {expert_name} ({slug})", flush=True)
    return rows


def write_json(rows: list[dict], output_path: Path) -> None:
    payload = {
        "method": "v02v03_probability_threshold_weighted_similarity",
        "thresholds": sorted({row["threshold"] for row in rows}),
        "experts": [expert_name for expert_name, _, _ in EXPERTS],
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "threshold",
        "threshold_label",
        "expert",
        "slug",
        "num_targets",
        "min_rank",
        "mean_rank",
        "median_rank",
        "max_rank",
        "mean_significant_probabilities",
        "selected_rank",
        "base_max_rank_v02",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict], output_path: Path) -> None:
    grouped: dict[float, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["threshold"], []).append(row)

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Threshold & Expert & Min & Mean & Max & Selected \\",
        r"\midrule",
    ]

    thresholds = sorted(grouped.keys())
    for threshold_index, threshold in enumerate(thresholds):
        threshold_rows = grouped[threshold]
        for row_index, row in enumerate(threshold_rows):
            threshold_cell = latex_threshold(threshold) if row_index == 0 else ""
            lines.append(
                f"{threshold_cell} & {row['expert']} & {row['min_rank']} & {format_mean(row['mean_rank'])} & {row['max_rank']} & {row['selected_rank']} \\\\"
            )
        if threshold_index != len(thresholds) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n")


def main(
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/FlexMoRE_V02_tables",
        help="Directory for the threshold sweep artifacts",
    ),
    threshold: list[float] = typer.Option(
        DEFAULT_THRESHOLDS,
        help="Probability thresholds to sweep for the v02v03 method",
    ),
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(threshold)
    write_json(rows, output_root / "v02v03_threshold_sweep_all_experts.json")
    write_csv(rows, output_root / "v02v03_threshold_sweep_all_experts.csv")
    write_latex(rows, output_root / "v02v03_threshold_sweep_all_experts.tex")


if __name__ == "__main__":
    typer.run(main)
