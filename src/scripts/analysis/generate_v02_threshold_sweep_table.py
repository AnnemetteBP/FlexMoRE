import csv
import json
import math
from pathlib import Path

import torch
import typer


EXPERTS = [
    ("Math", "math"),
    ("News", "news"),
    ("Academic", "academic"),
    ("Reddit", "reddit"),
    ("Code", "code"),
    ("Creative", "creative"),
]

DEFAULT_THRESHOLDS = [1e-4, 2e-4, 5e-4, 7e-4, 1e-3]


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


def generate_rows(thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        for expert_name, slug in EXPERTS:
            sv_map = load_singular_values_map(slug)
            ranks = []
            kept_counts = []
            for model_key in sorted(
                sv_map.keys(),
                key=lambda mk: (
                    int(mk.split(".")[2]),
                    {"down_proj": 0, "gate_proj": 1, "up_proj": 2}[mk.split(".")[-1]],
                ),
            ):
                rank, kept_count = compute_probability_rank(sv_map[model_key], threshold)
                ranks.append(rank)
                kept_counts.append(kept_count)

            rows.append(
                {
                    "threshold": threshold,
                    "threshold_label": threshold_label(threshold),
                    "expert": expert_name,
                    "slug": slug,
                    "num_targets": len(ranks),
                    "min_rank": min(ranks),
                    "mean_rank": sum(ranks) / len(ranks),
                    "max_rank": max(ranks),
                    "median_rank": sorted(ranks)[len(ranks) // 2],
                    "mean_significant_probabilities": sum(kept_counts) / len(kept_counts),
                    "selected_rank": next_power_of_two(max(ranks)),
                }
            )
    return rows


def write_json(rows: list[dict], output_path: Path) -> None:
    payload = {
        "method": "v02_probability_threshold",
        "thresholds": sorted({row["threshold"] for row in rows}),
        "experts": [expert_name for expert_name, _ in EXPERTS],
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
    output_stem: str = typer.Option(
        "v02_threshold_sweep_all_experts",
        help="Filename stem for the generated threshold sweep artifacts",
    ),
    threshold: list[float] = typer.Option(
        DEFAULT_THRESHOLDS,
        help="Probability thresholds to sweep for the v02 method",
    ),
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(threshold)
    write_json(rows, output_root / f"{output_stem}.json")
    write_csv(rows, output_root / f"{output_stem}.csv")
    write_latex(rows, output_root / f"{output_stem}.tex")


if __name__ == "__main__":
    typer.run(main)
