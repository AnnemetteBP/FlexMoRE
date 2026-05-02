import csv
import json
import logging
import runpy
from pathlib import Path

import typer
from huggingface_hub import snapshot_download

from .analysis_efficient_low_rank_expert import (
    checkpoint_key_is_target,
    collect_effective_ranks,
    collect_expert_triplets,
    compute_triplet_metrics,
    get_checkpoint_weight_map,
    load_target_tensors_from_checkpoint,
    save_triplet_metrics,
    summarize_effective_ranks,
    summarize_triplet_metrics,
)

log = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

def write_csv(output_path: Path, summaries: list[dict]) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_name",
                "model_path",
                "num_targets",
                "min_effective_rank",
                "max_effective_rank",
                "mean_effective_rank",
                "median_effective_rank",
                "std_effective_rank",
            ]
        )
        for summary in summaries:
            overall_stats = summary["overall_stats"]
            writer.writerow(
                [
                    summary["model_name"],
                    summary["model_path"],
                    summary["num_targets"],
                    overall_stats["min"],
                    overall_stats["max"],
                    overall_stats["mean"],
                    overall_stats["median"],
                    overall_stats["std"],
                ]
            )


def download_target_checkpoint_files(repo_id: str, cache_root: Path, lora_modules: list[str]) -> str:
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_root),
        allow_patterns=[
            "config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model.pt",
        ],
    )

    weight_map = get_checkpoint_weight_map(local_path)
    if weight_map is None:
        return local_path

    shard_files = sorted(
        {
            shard_file
            for key, shard_file in weight_map.items()
            if checkpoint_key_is_target(key, lora_modules)
        }
    )

    if not shard_files:
        raise ValueError(f"No target expert tensors found in checkpoint index for {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_root),
        allow_patterns=[
            "config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            *shard_files,
        ],
    )
    return local_path


def main(
    model_registry_path: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/flexolmo_models.json",
        help="Path to the expert/public model registry file",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks",
        help="Directory to store individual and combined reports",
    ),
    cache_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/.cache/huggingface/hub",
        help="Directory to store downloaded Hugging Face snapshots",
    ),
    processes: int = typer.Option(
        1,
        help="Number of CPU threads to use for SVD computation",
    ),
    lora_modules: list[str] = typer.Option(
        [
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        help="List of modules to analyze",
    ),
    model_names: list[str] = typer.Option(
        [],
        help="Optional subset of expert model names to analyze",
    ),
    relative_threshold: float = typer.Option(
        1e-3,
        help="Relative singular-value threshold as a fraction of the maximum singular value",
    ),
    absolute_threshold: float = typer.Option(
        1e-8,
        help="Absolute singular-value threshold to treat tiny values as noise",
    ),
    metrics_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/expert_mlp_metrics",
        help="Directory to store per-expert MLP similarity/norm metrics JSON files",
    ),
):
    registry = runpy.run_path(model_registry_path)
    expert_models = registry["expert_models"]
    if model_names:
        requested = set(model_names)
        expert_models = {
            model_name: repo_id
            for model_name, repo_id in expert_models.items()
            if model_name in requested
        }
        missing = requested - set(expert_models.keys())
        if missing:
            raise ValueError(f"Requested model names not found in registry: {sorted(missing)}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    metrics_root = Path(metrics_dir)
    metrics_root.mkdir(parents=True, exist_ok=True)

    summaries = []

    import torch

    torch.set_num_threads(processes)

    for model_name, repo_id in expert_models.items():
        try:
            log.info(f"Downloading target checkpoint files for {model_name} from {repo_id}")
            local_path = download_target_checkpoint_files(repo_id, cache_root, lora_modules)
            log.info(f"Analyzing effective ranks for {model_name} from {local_path}")
            expert_tensors = load_target_tensors_from_checkpoint(local_path, lora_modules)
            key2usvh = {}
            effective_ranks = collect_effective_ranks(
                expert_tensors,
                lora_modules,
                key2usvh,
                rel_thresh=relative_threshold,
                abs_thresh=absolute_threshold,
            )
            triplets = collect_expert_triplets(expert_tensors, lora_modules)
            metric_rows = compute_triplet_metrics(triplets, effective_ranks)
            metric_summary = summarize_triplet_metrics(metric_rows)
            summary = summarize_effective_ranks(
                effective_ranks,
                model_name=model_name,
                model_path=repo_id,
                rel_thresh=relative_threshold,
                abs_thresh=absolute_threshold,
            )
            summaries.append(summary)

            model_slug = model_name.lower().replace(" ", "_")
            with (output_root / f"{model_slug}_effective_ranks.json").open("w") as f:
                json.dump(summary, f, indent=2)
            save_triplet_metrics(
                metric_rows,
                str(metrics_root / f"{model_slug}_mlp_metrics.json"),
                metric_summary,
            )
            log.info(f"Saved per-model report for {model_name}")
        except Exception:
            log.exception(f"Failed while analyzing {model_name}")
            raise

    combined_report = {
        "models": summaries,
    }
    with (output_root / "effective_ranks_combined.json").open("w") as f:
        json.dump(combined_report, f, indent=2)
    write_csv(output_root / "effective_ranks_summary.csv", summaries)

    log.info(f"Saved combined report to {output_root / 'effective_ranks_combined.json'}")
    log.info(f"Saved CSV summary to {output_root / 'effective_ranks_summary.csv'}")


if __name__ == "__main__":
    typer.run(main)
