import json
import sys
from pathlib import Path

import torch
import typer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.analysis.analysis_efficient_low_rank_expert import (
    compute_triplet_metrics,
    save_triplet_metrics,
    summarize_triplet_metrics,
)


EXPERTS = [
    ("Math", "math"),
    ("News", "news"),
    ("Academic", "academic"),
    ("Reddit", "reddit"),
    ("Code", "code"),
    ("Creative", "creative"),
]


def model_key_sort_key(model_key: str) -> tuple[int, int]:
    return (
        int(model_key.split(".")[2]),
        {"down_proj": 0, "gate_proj": 1, "up_proj": 2}[model_key.split(".")[-1]],
    )


def load_tensor_map(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    raise ValueError(f"Unexpected tensor-map payload in {path}")


def build_triplets(
    public_map: dict[str, torch.Tensor],
    expert_map: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    if set(public_map.keys()) != set(expert_map.keys()):
        missing_public = sorted(set(expert_map.keys()) - set(public_map.keys()))
        missing_expert = sorted(set(public_map.keys()) - set(expert_map.keys()))
        raise ValueError(
            f"Mismatched model keys. Missing in public={missing_public[:3]}, missing in expert={missing_expert[:3]}"
        )

    triplets: dict[str, dict[str, torch.Tensor]] = {}
    for model_key in sorted(public_map.keys(), key=model_key_sort_key):
        public = public_map[model_key]
        expert = expert_map[model_key]
        triplets[model_key] = {
            "public": public,
            "expert": expert,
            "delta": expert - public,
        }
    return triplets


def generate_for_expert(slug: str, weights_dir: Path, output_dir: Path) -> Path:
    public_path = weights_dir / f"{slug}_public_mlp_tensors.pt"
    expert_path = weights_dir / f"{slug}_expert_mlp_tensors.pt"
    if not public_path.exists():
        raise FileNotFoundError(public_path)
    if not expert_path.exists():
        raise FileNotFoundError(expert_path)

    public_map = load_tensor_map(public_path)
    expert_map = load_tensor_map(expert_path)
    triplets = build_triplets(public_map, expert_map)
    metric_rows = compute_triplet_metrics(triplets, effective_ranks=None)
    summary = summarize_triplet_metrics(metric_rows)
    output_path = output_dir / f"{slug}_mlp_metrics.json"
    save_triplet_metrics(metric_rows, str(output_path), summary)
    return output_path


def main(
    weights_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/individual_mlp_weights",
        help="Directory containing saved public/expert MLP tensor maps",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/expert_mlp_metrics",
        help="Directory to write per-expert MLP metric JSON files",
    ),
    expert: list[str] = typer.Option(
        [],
        help="Optional subset of expert slugs to generate (e.g. reddit code creative)",
    ),
):
    weights_root = Path(weights_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    requested = set(expert)
    for _, slug in EXPERTS:
        if requested and slug not in requested:
            continue
        output_path = generate_for_expert(slug, weights_root, output_root)
        typer.echo(json.dumps({"slug": slug, "output_path": str(output_path)}))


if __name__ == "__main__":
    typer.run(main)
