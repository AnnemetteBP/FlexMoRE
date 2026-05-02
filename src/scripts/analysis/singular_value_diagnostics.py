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


def threshold_analysis(s: torch.Tensor) -> tuple[int, int]:
    s = s.to(dtype=torch.float64).flatten()
    total = s.numel()
    if total == 0:
        return 0, 0

    max_sv = s.max().item()
    threshold = max(max_sv * 1e-3, 1e-8)
    num_kept = int((s >= threshold).sum().item())
    return num_kept, total


def diagnose_singular_values(s: torch.Tensor, label: str) -> None:
    print("Singular value diagnostics")
    print(f"Source: {label}")
    print(f"Tensor length: {s.numel()}")
    print()

    for tau in (0.9, 0.95, 0.99):
        print(f"energy_rank(tau={tau}): {energy_rank(s, tau)}")

    num_kept, total = threshold_analysis(s)
    print()
    print("threshold_analysis")
    print(f"kept singular values: {num_kept}")
    print(f"total singular values: {total}")
    print(f"fraction kept: {num_kept / total:.6f}" if total > 0 else "fraction kept: n/a")


def main(
    model_path: str | None = typer.Option(
        None,
        help="Optional checkpoint path to analyze a real expert delta matrix instead of a random tensor",
    ),
    model_key: str = typer.Option(
        "model.layers.0.mlp.experts.1.down_proj",
        help="Target expert MLP matrix key to analyze when model_path is provided",
    ),
):
    if model_path is None:
        s = torch.rand(4096)
        diagnose_singular_values(s, "random test tensor")
        return

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {model_path}")

    expert_tensors = load_target_tensors_from_checkpoint(
        model_path,
        ["gate_proj", "down_proj", "up_proj"],
    )
    triplets = collect_expert_triplets(
        expert_tensors,
        ["gate_proj", "down_proj", "up_proj"],
    )
    if model_key not in triplets:
        raise KeyError(f"Model key not found in expert triplets: {model_key}")

    delta = triplets[model_key]["delta"].to(dtype=torch.float64)
    _, s, _ = torch.linalg.svd(delta, full_matrices=False)
    diagnose_singular_values(s, model_key)


if __name__ == "__main__":
    typer.run(main)
