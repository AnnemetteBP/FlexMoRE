import csv
import json
import math
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


MODULE_BONUS = {
    "down_proj": 0.00,
    "gate_proj": 0.08,
    "up_proj": 0.04,
}


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


def compute_topk_energy_share(singular_values: torch.Tensor, k: int) -> float:
    singular_values = torch.sort(singular_values.to(dtype=torch.float64), descending=True).values
    if singular_values.numel() == 0:
        return 0.0
    energy = singular_values.square()
    total_energy = energy.sum()
    if total_energy <= 0:
        return 0.0
    topk = energy[: min(k, energy.numel())].sum()
    return float((topk / total_energy).item())


def compute_probability_entropy(singular_values: torch.Tensor) -> float:
    singular_values = singular_values.to(dtype=torch.float64).flatten()
    total = singular_values.sum()
    if total <= 0 or singular_values.numel() == 0:
        return 0.0
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    if probabilities.numel() == 0:
        return 0.0
    entropy = -(probabilities * probabilities.log()).sum().item()
    max_entropy = math.log(float(probabilities.numel()))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def compute_gini_coefficient(singular_values: torch.Tensor) -> float:
    values = torch.sort(singular_values.to(dtype=torch.float64).flatten(), descending=False).values
    if values.numel() == 0:
        return 0.0
    total = values.sum().item()
    if total <= 0:
        return 0.0
    n = values.numel()
    indices = torch.arange(1, n + 1, dtype=torch.float64)
    gini = ((2 * indices - n - 1) * values).sum().item() / (n * total)
    return max(0.0, min(1.0, float(gini)))


def flatten_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.to(dtype=torch.float64).reshape(-1)
    b_flat = b.to(dtype=torch.float64).reshape(-1)
    denom = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if denom <= 0:
        return 0.0
    return float((torch.dot(a_flat, b_flat) / denom).item())


def compute_subspace_novelty(delta_u: torch.Tensor, public_u: torch.Tensor, top_k: int) -> float:
    if delta_u.numel() == 0 or public_u.numel() == 0:
        return 0.0
    k = min(top_k, delta_u.shape[1], public_u.shape[1])
    if k <= 0:
        return 0.0
    delta_basis = delta_u[:, :k].to(dtype=torch.float64)
    public_basis = public_u[:, :k].to(dtype=torch.float64)
    overlap = delta_basis.T @ public_basis
    normalized_overlap = torch.linalg.norm(overlap, ord="fro").item() / math.sqrt(k)
    similarity = max(0.0, min(1.0, float(normalized_overlap)))
    return 1.0 - similarity


def normalize_feature(value: float, min_value: float, max_value: float) -> float:
    if math.isclose(min_value, max_value):
        return 0.5
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, float(normalized)))


def select_candidate_pool(candidate_ranks: list[int]) -> list[int]:
    compact_pool = [rank for rank in candidate_ranks if 8 <= rank <= 1024]
    if compact_pool:
        return compact_pool
    positive_candidates = [rank for rank in candidate_ranks if rank > 0]
    if positive_candidates:
        return positive_candidates
    return [1]


def assign_rank_from_score(score: float, candidate_pool: list[int]) -> int:
    if len(candidate_pool) == 1:
        return candidate_pool[0]
    index = int(round(score * (len(candidate_pool) - 1)))
    index = max(0, min(index, len(candidate_pool) - 1))
    return candidate_pool[index]


def compute_score_bounds(rows: list[dict]) -> dict[str, tuple[float, float]]:
    cosine_distance_values = [1.0 - row["public_expert_cosine"] for row in rows]
    relative_delta_values = [row["relative_delta_norm"] for row in rows]
    inverse_top50_values = [1.0 - row["top50_energy_share"] for row in rows]
    entropy_values = [row["entropy"] for row in rows]
    inverse_gini_values = [1.0 - row["gini"] for row in rows]
    novelty_values = [row["subspace_novelty"] for row in rows]
    return {
        "cosine_distance": (min(cosine_distance_values), max(cosine_distance_values)),
        "relative_delta_norm": (min(relative_delta_values), max(relative_delta_values)),
        "inverse_top50": (min(inverse_top50_values), max(inverse_top50_values)),
        "entropy": (min(entropy_values), max(entropy_values)),
        "inverse_gini": (min(inverse_gini_values), max(inverse_gini_values)),
        "novelty": (min(novelty_values), max(novelty_values)),
    }


def assign_v05_rank(row: dict, bounds: dict[str, tuple[float, float]], max_layer: int, candidate_pool: list[int]) -> tuple[float, int]:
    cosine_score = normalize_feature(1.0 - row["public_expert_cosine"], *bounds["cosine_distance"])
    delta_score = normalize_feature(row["relative_delta_norm"], *bounds["relative_delta_norm"])
    top50_score = normalize_feature(1.0 - row["top50_energy_share"], *bounds["inverse_top50"])
    entropy_score = normalize_feature(row["entropy"], *bounds["entropy"])
    gini_score = normalize_feature(1.0 - row["gini"], *bounds["inverse_gini"])
    novelty_score = normalize_feature(row["subspace_novelty"], *bounds["novelty"])
    layer_bonus = 0.05 * (row["layer"] / max_layer if max_layer > 0 else 0.0)
    module_bonus = MODULE_BONUS.get(row["module"], 0.0)

    score = (
        0.20 * cosine_score
        + 0.20 * delta_score
        + 0.15 * top50_score
        + 0.15 * entropy_score
        + 0.10 * gini_score
        + 0.20 * novelty_score
        + layer_bonus
        + module_bonus
    )
    score = max(0.0, min(1.0, score))
    return score, assign_rank_from_score(score, candidate_pool)


def write_outputs(output_root: Path, model_name: str, rows: list[dict], candidate_pool: list[int]) -> None:
    if not rows:
        return

    bounds = compute_score_bounds(rows)
    max_layer = max(row["layer"] for row in rows)
    for row in rows:
        score, rank = assign_v05_rank(row, bounds, max_layer, candidate_pool)
        row["v05_score"] = score
        row["v05_rank"] = rank

    summary = {
        "num_targets": len(rows),
        "candidate_pool": candidate_pool,
        "min_v05_rank": min(row["v05_rank"] for row in rows),
        "max_v05_rank": max(row["v05_rank"] for row in rows),
        "mean_v05_rank": sum(row["v05_rank"] for row in rows) / len(rows),
        "median_v05_rank": sorted(row["v05_rank"] for row in rows)[len(rows) // 2],
    }

    json_path = output_root / f"{model_name.lower()}_v05_ranks.json"
    csv_path = output_root / f"{model_name.lower()}_v05_ranks.csv"

    with json_path.open("w") as f:
        json.dump({"expert": model_name, "summary": summary, "rows": rows}, f, indent=2)

    fieldnames = [
        "model_key",
        "layer",
        "module",
        "public_expert_cosine",
        "relative_delta_norm",
        "top50_energy_share",
        "entropy",
        "gini",
        "subspace_novelty",
        "num_singular_values",
        "v05_score",
        "v05_rank",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows(output_root: Path, model_name: str) -> list[dict]:
    json_path = output_root / f"{model_name.lower()}_v05_ranks.json"
    if not json_path.exists():
        return []
    with json_path.open("r") as f:
        payload = json.load(f)
    return payload.get("rows", [])


def infer_model_name(output_root: Path) -> str:
    name = output_root.name.lower()
    if "math" in name:
        return "Math"
    if "news" in name:
        return "News"
    if "academic" in name or "pes2o" in name:
        return "Academic"
    return output_root.name.title()


def main(
    model_path: str = typer.Argument(..., help="Checkpoint path to analyze"),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks_v05/news",
        help="Directory to store v05 outputs",
    ),
    processes: int = typer.Option(1, help="Number of CPU threads for SVD"),
    overlap_top_k: int = typer.Option(16, help="Top-k singular vectors used for subspace novelty"),
    candidate_ranks: list[int] = typer.Option(
        [8, 16, 32, 64, 128, 256, 512, 1024],
        help="Candidate pool for v05 rank bucket assignment",
    ),
):
    torch.set_num_threads(processes)

    model_dir = Path(model_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model_name = infer_model_name(output_root)
    candidate_pool = select_candidate_pool(candidate_ranks)

    weight_map = get_checkpoint_weight_map(model_path)
    if weight_map is None:
        raise FileNotFoundError(f"No checkpoint index found under {model_path}")

    lora_modules = ["gate_proj", "down_proj", "up_proj"]
    target_model_keys = list_target_model_keys(weight_map, lora_modules)

    rows = load_existing_rows(output_root, model_name)
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

        delta_u, delta_s, _ = torch.linalg.svd(delta, full_matrices=False)
        public_u, _, _ = torch.linalg.svd(public, full_matrices=False)

        parts = model_key.split(".")
        rows.append(
            {
                "model_key": model_key,
                "layer": int(parts[2]),
                "module": parts[-1],
                "public_expert_cosine": flatten_cosine_similarity(public, expert),
                "relative_delta_norm": float(
                    torch.linalg.norm(delta).item() / torch.linalg.norm(public).item()
                    if torch.linalg.norm(public).item() > 0
                    else 0.0
                ),
                "top50_energy_share": compute_topk_energy_share(delta_s, 50),
                "entropy": compute_probability_entropy(delta_s),
                "gini": compute_gini_coefficient(delta_s),
                "subspace_novelty": compute_subspace_novelty(delta_u, public_u, overlap_top_k),
                "num_singular_values": int(delta_s.numel()),
            }
        )
        write_outputs(output_root, model_name, rows, candidate_pool)
        print(
            f"Saved {model_key}: novelty={rows[-1]['subspace_novelty']:.4f} "
            f"entropy={rows[-1]['entropy']:.4f} gini={rows[-1]['gini']:.4f}",
            flush=True,
        )

    write_outputs(output_root, model_name, rows, candidate_pool)
    print(f"Wrote v05 diagnostics for {len(rows)} matrices to {output_root}", flush=True)


if __name__ == "__main__":
    typer.run(main)
