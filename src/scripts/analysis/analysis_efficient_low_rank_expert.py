import json
import logging
import math
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import typer

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

log = logging.getLogger(__name__)

MODEL_KEY_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.1\.(?P<module>down_proj|gate_proj|up_proj)$"
)


def compute_effective_rank(
    singular_values: torch.Tensor,
    rel_thresh: float = 1e-3,
    abs_thresh: float = 1e-8,
) -> int:
    singular_values = singular_values.to(dtype=torch.float64)

    max_sv = singular_values.max().item()
    threshold = 0.0
    if rel_thresh > 0:
        threshold = max(threshold, max_sv * rel_thresh)
    if abs_thresh > 0:
        threshold = max(threshold, abs_thresh)

    filtered = singular_values[singular_values >= threshold]
    if filtered.numel() == 0:
        return 1

    norm = filtered.sum()
    if norm <= 0:
        return 1

    probabilities = filtered / norm
    nonzero_probabilities = probabilities[probabilities > 0]
    entropy = -(nonzero_probabilities * nonzero_probabilities.log()).sum()
    effective_rank = torch.exp(entropy).item()
    return max(1, math.ceil(effective_rank))


def checkpoint_key_is_target(key: str, lora_modules: list[str]) -> bool:
    return ".experts." in key and any(lora_module in key for lora_module in lora_modules)


def get_checkpoint_weight_map(model_path: str) -> dict[str, str] | None:
    model_dir = Path(model_path)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / index_name
        if index_path.exists():
            with open(index_path, "r") as f:
                index_data = json.load(f)
            return index_data["weight_map"]
    return None


def load_target_tensors_from_checkpoint(model_path: str, lora_modules: list[str]) -> dict[str, torch.Tensor]:
    model_dir = Path(model_path)
    weight_map = get_checkpoint_weight_map(model_path)
    if weight_map is not None:
        target_keys = [key for key in weight_map if checkpoint_key_is_target(key, lora_modules)]
        file_to_keys: dict[str, list[str]] = {}
        for key in target_keys:
            file_to_keys.setdefault(weight_map[key], []).append(key)

        tensors = {}
        for relative_file, keys in file_to_keys.items():
            file_path = model_dir / relative_file
            if file_path.suffix == ".safetensors":
                if safe_open is None:
                    raise ImportError("safetensors is required to read .safetensors checkpoints")
                with safe_open(str(file_path), framework="pt", device="cpu") as f:
                    for key in keys:
                        tensors[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(file_path, map_location="cpu")
                for key in keys:
                    tensors[key] = state_dict[key]
        return tensors

    for single_name in ("model.safetensors", "pytorch_model.bin", "model.pt"):
        file_path = model_dir / single_name
        if not file_path.exists():
            continue
        if file_path.suffix == ".safetensors":
            if safe_open is None:
                raise ImportError("safetensors is required to read .safetensors checkpoints")
            tensors = {}
            with safe_open(str(file_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if checkpoint_key_is_target(key, lora_modules):
                        tensors[key] = f.get_tensor(key)
            return tensors

        state_dict = torch.load(file_path, map_location="cpu")
        return {
            key: value
            for key, value in state_dict.items()
            if checkpoint_key_is_target(key, lora_modules)
        }

    raise FileNotFoundError(f"Could not find supported checkpoint files under {model_path}")


def split_combined_expert_weights(key: str, weights: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    if "gate_up_proj" in key:
        base_experts = list(weights[0].chunk(2, dim=0))
        experts = list(weights[1].chunk(2, dim=0))
        model_keys = [
            key.replace(".experts.gate_up_proj", ".experts.1.gate_proj"),
            key.replace(".experts.gate_up_proj", ".experts.1.up_proj"),
        ]
    elif "down_proj" in key:
        base_experts = [weights[0]]
        experts = [weights[1]]
        model_keys = [key.replace(".experts.down_proj", ".experts.1.down_proj")]
    else:
        raise AssertionError(f"Unexpected expert key {key}")

    return list(zip(base_experts, experts, model_keys))


def split_separate_expert_weights(
    key: str,
    weights: torch.Tensor,
    expert_tensors: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    if ".experts.1." not in key:
        return []

    base_key = key.replace(".experts.1.", ".experts.0.")
    if base_key not in expert_tensors:
        raise KeyError(f"Base expert key {base_key} not found for {key}")

    model_key = key.removesuffix(".weight")
    return [(expert_tensors[base_key], weights, model_key)]


def collect_effective_ranks(
    expert_tensors: dict[str, torch.Tensor],
    lora_modules: list[str],
    key2usvh: dict[str, Any],
    rel_thresh: float = 1e-3,
    abs_thresh: float = 1e-8,
) -> dict[str, int]:
    resolved_ranks = {}
    for key, weights in expert_tensors.items():
        if not checkpoint_key_is_target(key, lora_modules):
            continue
        if ".experts.gate_up_proj" in key or ".experts.down_proj" in key:
            expert_splits = split_combined_expert_weights(key, weights)
        else:
            expert_splits = split_separate_expert_weights(key, weights, expert_tensors)

        for base_expert, expert, model_key in expert_splits:
            delta_expert = expert - base_expert
            if model_key not in key2usvh:
                log.info(f"Computing SVD for key {model_key} with shape {delta_expert.shape}")
                key2usvh[model_key] = torch.linalg.svd(delta_expert, full_matrices=False)
            _, s, _ = key2usvh[model_key]
            resolved_ranks[model_key] = compute_effective_rank(
                s,
                rel_thresh=rel_thresh,
                abs_thresh=abs_thresh,
            )
            log.info(
                f"Effective rank for key {model_key}: {resolved_ranks[model_key]} "
                f"(from {s.numel()} singular values)"
            )
    return resolved_ranks


def collect_expert_triplets(
    expert_tensors: dict[str, torch.Tensor],
    lora_modules: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    triplets = {}
    for key, weights in expert_tensors.items():
        if not checkpoint_key_is_target(key, lora_modules):
            continue
        if ".experts.gate_up_proj" in key or ".experts.down_proj" in key:
            expert_splits = split_combined_expert_weights(key, weights)
        else:
            expert_splits = split_separate_expert_weights(key, weights, expert_tensors)

        for base_expert, expert, model_key in expert_splits:
            triplets[model_key] = {
                "public": base_expert.detach().cpu(),
                "expert": expert.detach().cpu(),
                "delta": (expert - base_expert).detach().cpu(),
            }
    return triplets


def cosine_similarity(flat_a: torch.Tensor, flat_b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(flat_a) * torch.linalg.vector_norm(flat_b)
    if denom <= 0:
        return 0.0
    return torch.dot(flat_a, flat_b).item() / denom.item()


def compute_triplet_metrics(
    triplets: dict[str, dict[str, torch.Tensor]],
    effective_ranks: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for model_key, tensors in sorted(triplets.items()):
        public = tensors["public"].to(dtype=torch.float64)
        expert = tensors["expert"].to(dtype=torch.float64)
        delta = tensors["delta"].to(dtype=torch.float64)

        public_flat = public.reshape(-1)
        expert_flat = expert.reshape(-1)
        delta_flat = delta.reshape(-1)

        public_norm = torch.linalg.matrix_norm(public).item()
        expert_norm = torch.linalg.matrix_norm(expert).item()
        delta_norm = torch.linalg.matrix_norm(delta).item()
        relative_delta_norm = delta_norm / public_norm if public_norm > 0 else 0.0

        rows.append(
            {
                "model_key": model_key,
                "layer": int(model_key.split(".")[2]),
                "module": model_key.split(".")[-1],
                "shape": list(delta.shape),
                "public_norm_fro": public_norm,
                "expert_norm_fro": expert_norm,
                "delta_norm_fro": delta_norm,
                "relative_delta_norm": relative_delta_norm,
                "public_expert_cosine": cosine_similarity(public_flat, expert_flat),
                "public_delta_cosine": cosine_similarity(public_flat, delta_flat),
                "expert_delta_cosine": cosine_similarity(expert_flat, delta_flat),
                "effective_rank": effective_ranks.get(model_key) if effective_ranks is not None else None,
            }
        )
    return rows


def summarize_triplet_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_by_module: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)

    metric_names = [
        "public_norm_fro",
        "expert_norm_fro",
        "delta_norm_fro",
        "relative_delta_norm",
        "public_expert_cosine",
        "public_delta_cosine",
        "expert_delta_cosine",
        "effective_rank",
    ]
    for row in metric_rows:
        module = row["module"]
        for metric_name in metric_names:
            value = row.get(metric_name)
            if value is None:
                continue
            overall[metric_name].append(value)
            grouped_by_module[module][metric_name].append(value)

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "overall": {
            metric_name: summarize(values)
            for metric_name, values in sorted(overall.items())
        },
        "per_module": {
            module: {
                metric_name: summarize(values)
                for metric_name, values in sorted(metrics.items())
            }
            for module, metrics in sorted(grouped_by_module.items())
        },
    }


def save_triplet_tensors(
    triplets: dict[str, dict[str, torch.Tensor]],
    output_dir: str,
    tensor_kinds: list[str],
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, torch.Tensor]] = {tensor_kind: {} for tensor_kind in tensor_kinds}
    for model_key, tensors in triplets.items():
        for tensor_kind in tensor_kinds:
            grouped[tensor_kind][model_key] = tensors[tensor_kind]

    for tensor_kind, tensor_map in grouped.items():
        torch.save(tensor_map, output_root / f"{tensor_kind}_mlp_tensors.pt")


def save_triplet_metrics(metric_rows: list[dict[str, Any]], output_path: str, summary: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"per_layer_metrics": metric_rows}
    if summary is not None:
        payload["summary"] = summary
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def compute_rank_stats(rank_values: list[int]) -> dict[str, float | int]:
    if not rank_values:
        raise ValueError("rank_values must not be empty")

    sorted_rank_values = sorted(rank_values)
    count = len(sorted_rank_values)
    midpoint = count // 2
    if count % 2 == 0:
        median_rank = (sorted_rank_values[midpoint - 1] + sorted_rank_values[midpoint]) / 2
    else:
        median_rank = sorted_rank_values[midpoint]

    mean_rank = sum(rank_values) / count
    variance = sum((rank - mean_rank) ** 2 for rank in rank_values) / count

    return {
        "count": count,
        "min": min(rank_values),
        "max": max(rank_values),
        "mean": mean_rank,
        "median": median_rank,
        "std": math.sqrt(variance),
    }


def summarize_effective_ranks(
    effective_ranks: dict[str, int],
    *,
    model_name: str | None = None,
    model_path: str | None = None,
    rel_thresh: float | None = None,
    abs_thresh: float | None = None,
) -> dict[str, Any]:
    per_module: dict[str, list[int]] = {}
    per_layer: dict[str, list[int]] = {}

    for key, rank in sorted(effective_ranks.items()):
        match = MODEL_KEY_PATTERN.match(key)
        if match is None:
            continue
        layer = match.group("layer")
        module = match.group("module")
        per_module.setdefault(module, []).append(rank)
        per_layer.setdefault(layer, []).append(rank)

    summary = {
        "num_targets": len(effective_ranks),
        "overall_stats": compute_rank_stats(list(effective_ranks.values())),
        "per_module_stats": {
            module: compute_rank_stats(ranks)
            for module, ranks in sorted(per_module.items())
        },
        "per_layer_stats": {
            f"layers.{layer}.mlp": compute_rank_stats(ranks)
            for layer, ranks in sorted(per_layer.items(), key=lambda item: int(item[0]))
        },
        "effective_ranks": dict(sorted(effective_ranks.items())),
    }
    if model_name is not None:
        summary["model_name"] = model_name
    if model_path is not None:
        summary["model_path"] = model_path
    if rel_thresh is not None or abs_thresh is not None:
        summary["effective_rank_config"] = {
            "relative_threshold": rel_thresh,
            "absolute_threshold": abs_thresh,
        }
    return summary

def main(
    model_path: str = typer.Argument(..., help="Path to the FlexOLMo model in HF format"),
    rank: list[int] = typer.Option(
        [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384],
        help="Rank for the low-rank adapters to be applied to each linear layer",
    ),
    processes: int = typer.Option(
        1,
        help="Number of processes for SVD computation",
    ),
    lora_modules: list[str] = typer.Option(
        [
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        help="List of modules to apply LoRA to",
    ),
    report_only: bool = typer.Option(
        False,
        help="Only compute and report effective ranks without building or saving a FlexMoRE model",
    ),
    report_path: str | None = typer.Option(
        None,
        help="Optional JSON path to save effective-rank results",
    ),
    relative_threshold: float = typer.Option(
        1e-3,
        help="Relative singular-value threshold as a fraction of the maximum singular value",
    ),
    absolute_threshold: float = typer.Option(
        1e-8,
        help="Absolute singular-value threshold to treat tiny values as noise",
    ),
    save_metrics_path: str | None = typer.Option(
        None,
        help="Optional JSON path to save per-layer similarity and norm metrics for public/expert/delta MLP tensors",
    ),
    save_tensor_dir: str | None = typer.Option(
        None,
        help="Optional directory to save raw MLP tensors for the selected tensor kinds. Can be very large.",
    ),
    save_tensor_kinds: list[str] = typer.Option(
        [],
        help="Tensor kinds to save when save_tensor_dir is set. Allowed values: public, expert, delta",
    ),
):
    log.info(f"Setting number of threads for SVD computation to {processes}")
    torch.set_num_threads(processes)

    key2usvh = {} if not os.path.exists("svd_cache.pkl") else pickle.load(open("svd_cache.pkl", "rb"))
    if report_only:
        assert 0 in rank, "report_only expects rank=0 so the effective-rank path is exercised"
        log.info(f"Loading target expert tensors directly from checkpoint at {model_path}")
        expert_tensors = load_target_tensors_from_checkpoint(model_path, lora_modules)
        resolved_ranks = collect_effective_ranks(
            expert_tensors,
            lora_modules,
            key2usvh,
            rel_thresh=relative_threshold,
            abs_thresh=absolute_threshold,
        )
        triplets = None
        if save_metrics_path or save_tensor_dir:
            triplets = collect_expert_triplets(expert_tensors, lora_modules)

        if save_metrics_path:
            assert triplets is not None
            metric_rows = compute_triplet_metrics(triplets, resolved_ranks)
            save_triplet_metrics(metric_rows, save_metrics_path, summarize_triplet_metrics(metric_rows))
            log.info(f"Saved triplet metrics to {save_metrics_path}")

        if save_tensor_dir:
            assert triplets is not None
            invalid_kinds = sorted(set(save_tensor_kinds) - {"public", "expert", "delta"})
            if invalid_kinds:
                raise ValueError(f"Unsupported save_tensor_kinds: {invalid_kinds}")
            if not save_tensor_kinds:
                raise ValueError("save_tensor_dir requires at least one save_tensor_kinds value")
            save_triplet_tensors(triplets, save_tensor_dir, save_tensor_kinds)
            log.info(f"Saved raw triplet tensors to {save_tensor_dir}")

        pickle.dump(key2usvh, open("svd_cache.pkl", "wb"))
        report = summarize_effective_ranks(
            resolved_ranks,
            model_path=model_path,
            rel_thresh=relative_threshold,
            abs_thresh=absolute_threshold,
        )
        report["mode"] = "effective_rank_report"
        output_path = report_path or f"{model_path.rstrip(os.sep)}-erank-report.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Saved effective-rank report to {output_path}")
        return

    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()

    log.info(f"Loading config from {model_path}")
    from transformers import FlexMoREConfig, FlexMoREForCausalLM, FlexOlmoForCausalLM

    model_config = json.load(open(f"{model_path}/config.json", "r"))
    model_config['model_type'] = 'flexmore'
    model_config['architectures'] = ['FlexMoREForCausalLM']
    model_config = FlexMoREConfig.from_dict(model_config)
    log.info(model_config)

    log.info(f"Loading model from {model_path}")
    expert = FlexOlmoForCausalLM.from_pretrained(model_path)
    log.info(expert.config)
    log.info(expert)
    expert_state_dict = expert.state_dict()

    for r in rank:
        resolved_ranks = {}
        if r == 0:
            log.info("Computing effective ranks for LoRA-enabled expert layers")
            resolved_ranks = collect_effective_ranks(
                expert_state_dict,
                lora_modules,
                key2usvh,
                rel_thresh=relative_threshold,
                abs_thresh=absolute_threshold,
            )
            assert resolved_ranks, "No LoRA-enabled expert layers found for effective-rank computation"
            model_rank = max(resolved_ranks.values())
            log.info(f"Using model rank {model_rank} to accommodate per-layer effective ranks")
        else:
            model_rank = r

        log.info("Initializing empty model")
        model_config.expert_ranks = [0, model_rank]
        model = FlexMoREForCausalLM(config=model_config)
        model_state_dict = model.state_dict()
        log.info(model)
        processed_keys = []
        for key in list(expert_state_dict.keys()):
            weights = expert_state_dict[key]
            if ".experts." in key and any(lora_module in key for lora_module in lora_modules):
                log.info(f"Processing key {key}")
                if "gate_up_proj" in key:
                    print(f"Shape of weights for key {key}: {weights.shape}")
                    base_experts = list(weights[0].chunk(2, dim=0))
                    print(f"Shapes of base experts for key {key}: {[be.shape for be in base_experts]}")
                    experts = list(weights[1].chunk(2, dim=0))
                    model_keys = [
                        key.replace(".experts.gate_up_proj", ".experts.1.gate_proj"),
                        key.replace(".experts.gate_up_proj", ".experts.1.up_proj"),
                    ]
                elif "down_proj" in key:
                    base_experts = [weights[0]]
                    print(f"Shape of base expert for key {key}: {base_experts[0].shape}")
                    experts = [weights[1]]
                    print(f"Shape of expert for key {key}: {experts[0].shape}")
                    model_keys = [key.replace(".experts.down_proj", ".experts.1.down_proj")]
                else:
                    assert False, f"Unexpected expert key {key}"
                for base_expert, expert, model_key in zip(base_experts, experts, model_keys):
                    base_key = model_key.replace(".experts.1.", ".experts.0.")
                    base_key = base_key.replace("_proj", "_proj.weight")
                    assert base_key in model_state_dict, f"Base key {base_key} not found in model state dict: {list(model_state_dict.keys())}"
                    assert base_expert.shape == model_state_dict[base_key].shape, f"Shape mismatch for base key {base_key}: expert shape {base_expert.shape}, model shape {model_state_dict[base_key].shape}"
                    model_state_dict[base_key] = base_expert
                    processed_keys.append(base_key)
                    delta_expert = expert - base_expert
                    # compute the low-rank adaptation
                    if model_key not in key2usvh:
                        log.info(f"Computing SVD for key {model_key} with shape {delta_expert.shape}")
                        key2usvh[model_key] = torch.linalg.svd(delta_expert, full_matrices=False)
                    u, s, vh = key2usvh[model_key]
                    layer_rank = resolved_ranks.get(model_key, r)
                    lora_u = u[:, :layer_rank]
                    lora_s = s[:layer_rank]
                    lora_vh = vh[:layer_rank, :]
                    print(f"Shapes for key {model_key}: u {u.shape}, s {s.shape}, vh {vh.shape}")
                    print(f"Shapes for LoRA key {model_key}: lora_u {lora_u.shape}, lora_s {lora_s.shape}, lora_vh {lora_vh.shape}")
                    sqrt_s = lora_s.sqrt()
                    lora_a = sqrt_s[:, None] * lora_vh
                    lora_b = lora_u * sqrt_s
                    dummy = (lora_b @ lora_a)
                    print(f"Reconstructed delta shape for key {model_key}: {dummy.shape}")
                    log.info(f"Storing LoRA adapters for key {model_key} with shapes {lora_a.shape}, {lora_b.shape}")
                    a_key = model_key.replace("_proj", f"_proj_a.weight")
                    b_key = model_key.replace("_proj", f"_proj_b.weight")
                    assert a_key in model_state_dict, f"Key {a_key} not found in model state dict: {list(model_state_dict.keys())}"
                    assert b_key in model_state_dict, f"Key {b_key} not found in model state dict: {list(model_state_dict.keys())}"
                    padded_lora_a = torch.zeros_like(model_state_dict[a_key])
                    padded_lora_b = torch.zeros_like(model_state_dict[b_key])
                    padded_lora_a[:layer_rank, :] = lora_a
                    padded_lora_b[:, :layer_rank] = lora_b
                    model_state_dict[a_key] = padded_lora_a
                    model_state_dict[b_key] = padded_lora_b
                    processed_keys.extend([a_key, b_key])
            else:
                assert key in model_state_dict, f"Key {key} not found in model state dict: {list(model_state_dict.keys())}"
                assert weights.shape == model_state_dict[key].shape, f"Shape mismatch for key {key}: expert shape {weights.shape}, model shape {model_state_dict[key].shape}"
                model_state_dict[key] = weights
                processed_keys.append(key)
        pickle.dump(key2usvh, open(f"svd_cache.pkl", "wb"))
        assert set(processed_keys) == set(model_state_dict.keys()), f"Not all keys were processed: processed {processed_keys}, model keys {list(model_state_dict.keys())}, missing {set(model_state_dict.keys()) - set(processed_keys)}"
        assert len(processed_keys) == len(model_state_dict), "Some keys were processed multiple times"
        # adapt the config
        model.config.expert_ranks = [0, model_rank]
        log.info(f"Model config after adaptation: {model.config}")
        # save the final_state_dict for the MoE in a format that the olmo_core trainer likes
        save_path = f"{model_path}-r{r}" if r != 0 else f"{model_path}-erank"
        log.info(f"Saving model to {save_path}")
        print(f"Final model state dict keys: {list(model_state_dict.keys())}")
        model.save_pretrained(save_path, state_dict=model_state_dict)
    log.info("Done")

if __name__ == "__main__":
    typer.run(main)
