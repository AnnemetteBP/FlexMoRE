from __future__ import annotations

import json
import math
import runpy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer
from huggingface_hub import snapshot_download

app = typer.Typer(add_completion=False)


def dtype_from_string(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "single"}:
        return torch.float32
    if normalized in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype string: {name}")


def expert_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        for candidate in ["model.pt", "pytorch_model.bin", "model.bin", "pytorch_model.safetensors", "model.safetensors"]:
            candidate_path = path / candidate
            if candidate_path.exists():
                path = candidate_path
                break
        else:
            return load_state_dict_from_directory(path)

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError(f"Loaded checkpoint from {path} is not a dictionary.")
    return state


def load_state_dict_from_directory(directory: Path) -> dict[str, torch.Tensor]:
    index_candidates = ["pytorch_model.bin.index.json", "model.safetensors.index.json", "pytorch_model.safetensors.index.json"]
    for index_name in index_candidates:
        index_path = directory / index_name
        if index_path.exists():
            return load_state_dict_from_index(index_path, directory)

    # Fallback: try loading any direct sharded file that contains the lm_head.
    for candidate in directory.glob("*.bin"):
        try:
            state = torch.load(candidate, map_location="cpu")
            if isinstance(state, dict) and any(key in state for key in ["lm_head.weight", "lm_head.w_out.weight", "w_out.weight", "weight"]):
                return state
        except Exception:
            continue

    for candidate in directory.glob("*.safetensors"):
        try:
            import safetensors.torch as safetensors
            state = safetensors.load_file(str(candidate), device="cpu")
            if any(key in state for key in ["lm_head.weight", "lm_head.w_out.weight", "w_out.weight", "weight"]):
                return state
        except Exception:
            continue

    raise FileNotFoundError(
        f"No checkpoint file or compatible sharded index found in directory {directory}. "
        "Expected a single checkpoint file or a HF sharded checkpoint directory with index files."
    )


def load_state_dict_from_index(index_path: Path, directory: Path) -> dict[str, torch.Tensor]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or index.get("metadata") or index.get("weights")
    if weight_map is None:
        weight_map = index

    target_keys = ["lm_head.weight", "lm_head.w_out.weight", "w_out.weight", "weight"]
    selected_keys = [key for key in target_keys if key in weight_map]
    if not selected_keys:
        raise FileNotFoundError(
            f"Index {index_path} does not reference any known lm_head keys. "
            f"Looking for {target_keys}."
        )

    state: dict[str, torch.Tensor] = {}
    loaded_files: set[Path] = set()
    for key in selected_keys:
        file_name = weight_map[key]
        if isinstance(file_name, list):
            file_name = file_name[0]
        file_path = directory / file_name
        loaded_files.add(file_path)

    for shard_path in loaded_files:
        if shard_path.suffix == ".safetensors":
            import safetensors.torch as safetensors
            shard_state = safetensors.load_file(str(shard_path), device="cpu")
        else:
            shard_state = torch.load(shard_path, map_location="cpu")
            if isinstance(shard_state, dict) and "model" in shard_state and isinstance(shard_state["model"], dict):
                shard_state = shard_state["model"]

        for key in selected_keys:
            if key in shard_state:
                state[key] = shard_state[key]

    if not state:
        raise FileNotFoundError(
            f"Could not load lm_head weights from shards listed in {index_path}."
        )
    return state


def load_model_registry(path: Path) -> dict[str, str]:
    registry = runpy.run_path(str(path))
    model_map: dict[str, str] = {}
    for key in ["public_models", "expert_models", "flex_olmo_models"]:
        if key in registry and isinstance(registry[key], dict):
            model_map.update({str(name): str(repo_id) for name, repo_id in registry[key].items()})
    return model_map


def download_checkpoint_from_hf(repo_id: str, cache_dir: Path) -> Path:
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        allow_patterns=[
            "config.json",
            "*.json",
            "*.bin",
            "*.safetensors",
            "*.index.json",
        ],
    )
    return Path(local_path)


def resolve_base_model_path(base_model: str, registry_path: Path | None, cache_dir: Path | None) -> Path:
    candidate = Path(base_model)
    if candidate.exists():
        return candidate

    if registry_path is not None:
        models = load_model_registry(registry_path)
        if base_model in models:
            repo_id = models[base_model]
            if cache_dir is None:
                raise ValueError(
                    f"Base model {base_model} is a registry key, but no cache_dir was provided."
                )
            print(f"Resolving base model {base_model} from registry using repo id {repo_id}")
            return download_checkpoint_from_hf(repo_id, cache_dir)

    raise FileNotFoundError(
        f"Base model path {base_model} not found and not available in registry {registry_path}."
    )


def write_latex_summary_tables(reports: list[dict[str, Any]], output_dir: Path) -> None:
    if not reports:
        return

    summary_rows = []
    for report in reports:
        summary_rows.append(
            {
                "expert_name": report.get("expert_name", "Unknown"),
                "layer_key": report.get("layer_key", "Unknown"),
                "svd_name": report["svd_name"],
                "basis_source": report["basis_source"],
                "top_k": report["top_k"],
                "sigma_threshold": report["sigma_threshold"],
                "kurtosis_threshold": report["kurtosis_threshold"],
                "skew_threshold": report.get("skew_threshold", 0.0),
                "energy_share_threshold": report["energy_share_threshold"],
                "total_energy": report["total_energy"],
                "estimated_rank": report["estimated_rank"],
                "rank_at_energy_share": report["rank_at_energy_share"],
                "n_meaningful_directions": report["n_meaningful_directions"],
            }
        )

        direction_df = pd.DataFrame(report["direction_rows"])
        direction_csv_path = output_dir / f"{report['svd_name']}_svd_projection_directions.csv"
        direction_tex_path = output_dir / f"{report['svd_name']}_svd_projection_directions.tex"
        direction_df.to_csv(direction_csv_path, index=False)
        direction_df.to_latex(direction_tex_path, index=False, longtable=True)
        print(f"Wrote direction table for {report['svd_name']} to {direction_tex_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "svd_vocabulary_projection_summary.csv"
    summary_tex_path = output_dir / "svd_vocabulary_projection_summary.tex"
    summary_df.to_csv(summary_csv_path, index=False)
    summary_df.to_latex(summary_tex_path, index=False)
    print(f"Wrote summary tables to {summary_tex_path}")


def load_unembedding_matrix(state_dict: dict[str, Any]) -> torch.Tensor:
    candidates = ["lm_head.weight", "lm_head.w_out.weight", "w_out.weight", "weight"]
    for key in candidates:
        if key in state_dict:
            weight = state_dict[key]
            if not isinstance(weight, torch.Tensor):
                raise ValueError(f"Checkpoint key {key} is not a tensor.")
            if weight.ndim != 2:
                raise ValueError(
                    f"Unembedding weight {key} must be 2D, got shape {tuple(weight.shape)}"
                )
            if weight.shape[0] > weight.shape[1]:
                # Typical lm_head.weight is [V, d], so transpose to [d, V].
                print(f"Loading unembedding matrix from {key}: shape={tuple(weight.shape)} -> transposing to [d, V]")
                return weight.T.clone()
            print(f"Loading unembedding matrix from {key}: shape={tuple(weight.shape)}")
            return weight.clone()
    raise KeyError(
        "Could not find a usable lm_head weight in the checkpoint. "
        "Looked for lm_head.weight, lm_head.w_out.weight, w_out.weight, weight."
    )


def load_svd_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if all(key in data for key in ("U", "S", "Vh")):
            U = data["U"]
            S = data["S"]
            Vh = data["Vh"]
        elif all(key in data for key in ("u", "s", "vh")):
            U = data["u"]
            S = data["s"]
            Vh = data["vh"]
        else:
            # Check if this is a singular values file that needs SVD recomputation
            if "singular_values" in data and isinstance(data["singular_values"], dict):
                raise ValueError(
                    f"SVD file {path} contains only singular values, not full U,S,Vh decomposition. "
                    "Please provide the path to the corresponding delta_W matrices instead, "
                    "or ensure full SVD decomposition was saved."
                )
            raise ValueError(
                f"SVD checkpoint {path} must contain keys U,S,Vh or u,s,vh. "
                f"Found keys: {list(data.keys())}"
            )
    elif isinstance(data, (tuple, list)) and len(data) == 3:
        U, S, Vh = data
    else:
        raise ValueError(
            f"SVD file {path} must contain a tuple/list of (U,S,Vh) or a dict with keys U,S,Vh."
        )

    if not all(isinstance(tensor, torch.Tensor) for tensor in (U, S, Vh)):
        raise ValueError("One or more SVD components are not torch tensors.")
    if U.ndim != 2 or S.ndim != 1 or Vh.ndim != 2:
        raise ValueError(
            f"Unexpected SVD shapes: U={tuple(U.shape)}, S={tuple(S.shape)}, Vh={tuple(Vh.shape)}"
        )
    print(f"Loaded SVD from {path}: U={tuple(U.shape)}, S={tuple(S.shape)}, Vh={tuple(Vh.shape)}")
    return U, S, Vh


def full_svd_artifact_path(expert_name: str, results_dir: Path) -> Path:
    return results_dir / "full_svd" / f"{expert_slug(expert_name)}_full_svd.pt"


def load_full_svd_artifact(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "svd_tensors" not in payload:
        raise ValueError(f"Full SVD artifact {path} is missing 'svd_tensors'.")
    svd_tensors = payload["svd_tensors"]
    if not isinstance(svd_tensors, dict):
        raise ValueError(f"Full SVD artifact {path} has invalid 'svd_tensors' payload.")

    result: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for key, value in svd_tensors.items():
        if isinstance(value, dict) and all(k in value for k in ("U", "S", "Vh")):
            result[key] = (value["U"], value["S"], value["Vh"])
        elif isinstance(value, (tuple, list)) and len(value) == 3:
            result[key] = (value[0], value[1], value[2])
        else:
            raise ValueError(f"Unexpected SVD entry format for {key} in {path}")
    return result


def save_full_svd_artifact(
    expert_name: str,
    svd_results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    results_dir: Path,
    save_dtype: torch.dtype,
    singular_values_source: str | None = None,
) -> Path:
    out_path = full_svd_artifact_path(expert_name, results_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serialized: dict[str, dict[str, torch.Tensor]] = {}
    for key, (U, S, Vh) in svd_results.items():
        serialized[key] = {
            "U": U.detach().cpu().to(dtype=save_dtype),
            "S": S.detach().cpu().to(dtype=torch.float32),
            "Vh": Vh.detach().cpu().to(dtype=save_dtype),
        }

    payload = {
        "expert_name": expert_name,
        "saved_dtype": str(save_dtype).replace("torch.", ""),
        "num_targets": len(serialized),
        "singular_values_source": singular_values_source,
        "svd_tensors": serialized,
    }
    torch.save(payload, out_path)
    print(f"Saved full SVD artifact for {expert_name} to {out_path}")
    return out_path


def load_delta_weights_for_expert(expert_name: str, results_dir: Path) -> dict[str, torch.Tensor]:
    """Load delta weights for an expert from the individual_mlp_weights directory."""
    expert_file = results_dir / "individual_mlp_weights" / f"{expert_name}_expert_mlp_tensors.pt"
    public_file = results_dir / "individual_mlp_weights" / f"{expert_name}_public_mlp_tensors.pt"

    if not public_file.exists():
        public_file = results_dir / "individual_mlp_weights" / "public_mlp_tensors.pt"

    if not expert_file.exists():
        raise FileNotFoundError(f"Expert weights file not found: {expert_file}")
    if not public_file.exists():
        raise FileNotFoundError(f"Public weights file not found: {public_file}")

    expert_weights = torch.load(expert_file, map_location="cpu")
    public_weights = torch.load(public_file, map_location="cpu")

    if set(expert_weights.keys()) != set(public_weights.keys()):
        raise ValueError(
            f"Expert and public weight files have mismatched keys for {expert_name}. "
            f"Expert keys: {sorted(expert_weights.keys())[:5]}... "
            f"Public keys: {sorted(public_weights.keys())[:5]}..."
        )

    delta_weights = {}
    for key in expert_weights.keys():
        delta_weights[key] = expert_weights[key] - public_weights[key]

    print(f"Loaded {len(delta_weights)} delta weight matrices for {expert_name}")
    return delta_weights


def compute_svd_for_delta_weights(delta_weights: dict[str, torch.Tensor], singular_values_dict: dict[str, torch.Tensor] | None = None) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute SVD for all delta weight matrices."""
    svd_results = {}
    for key, delta_w in delta_weights.items():
        print(f"Computing SVD for {key}: shape {tuple(delta_w.shape)}")
        U, S, Vh = torch.linalg.svd(delta_w.to(dtype=torch.float64), full_matrices=False)

        # If we have pre-computed singular values, use them (they might be more accurate)
        if singular_values_dict is not None and key in singular_values_dict:
            S = singular_values_dict[key].to(dtype=torch.float64)
            print(f"  Using pre-computed singular values, shape: {tuple(S.shape)}")

        svd_results[key] = (U, S, Vh)
        print(f"  SVD shapes: U={tuple(U.shape)}, S={tuple(S.shape)}, Vh={tuple(Vh.shape)}")
    return svd_results


def load_or_compute_svd_for_expert(
    expert_name: str,
    results_dir: Path,
    save_full_svd: bool = True,
    save_svd_dtype: str = "float16",
    force_recompute: bool = False,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load pre-computed SVD or compute it from delta weights."""
    full_svd_path = full_svd_artifact_path(expert_name, results_dir)
    if full_svd_path.exists() and not force_recompute:
        print(f"Loading cached full SVD artifact for {expert_name} from {full_svd_path}")
        return load_full_svd_artifact(full_svd_path)

    svd_file = results_dir / "singular_values" / f"{expert_name}_singular_values.pt"
    singular_values_source: str | None = None

    if svd_file.exists():
        try:
            # Load the file which contains metadata and singular values
            data = torch.load(svd_file, map_location="cpu")
            if isinstance(data, dict) and "singular_values" in data:
                # Only singular values available, need to recompute full SVD
                print(f"Found singular values file for {expert_name}, recomputing full SVD from delta weights")
                singular_values_dict = data["singular_values"]
                delta_weights = load_delta_weights_for_expert(expert_name, results_dir)
                svd_results = compute_svd_for_delta_weights(delta_weights, singular_values_dict)
                singular_values_source = str(svd_file)
                if save_full_svd:
                    save_full_svd_artifact(
                        expert_name,
                        svd_results,
                        results_dir,
                        dtype_from_string(save_svd_dtype),
                        singular_values_source=singular_values_source,
                    )
                return svd_results
            else:
                # Assume it's a full SVD file
                return {"full_svd": load_svd_tensors(svd_file)}
        except ValueError as e:
            if "only singular values" in str(e):
                # Recompute from delta weights
                delta_weights = load_delta_weights_for_expert(expert_name, results_dir)
                svd_results = compute_svd_for_delta_weights(delta_weights)
                if save_full_svd:
                    save_full_svd_artifact(
                        expert_name,
                        svd_results,
                        results_dir,
                        dtype_from_string(save_svd_dtype),
                    )
                return svd_results
            raise
    else:
        # No SVD file, compute from delta weights
        print(f"No SVD file found for {expert_name}, computing from delta weights")
        delta_weights = load_delta_weights_for_expert(expert_name, results_dir)
        svd_results = compute_svd_for_delta_weights(delta_weights)
        if save_full_svd:
            save_full_svd_artifact(
                expert_name,
                svd_results,
                results_dir,
                dtype_from_string(save_svd_dtype),
            )
        return svd_results


def project_to_vocab(w: torch.Tensor, U_base: torch.Tensor) -> torch.Tensor:
    print(f"project_to_vocab: w shape={tuple(w.shape)}, U_base shape={tuple(U_base.shape)}")
    if w.ndim == 1:
        w = w.unsqueeze(0)
    if w.ndim != 2:
        raise ValueError(f"Direction tensor must be 1D or 2D, got shape {tuple(w.shape)}")
    if w.shape[1] != U_base.shape[0]:
        raise ValueError(
            f"Hidden dimension mismatch: direction dim {w.shape[1]} != U_base rows {U_base.shape[0]}"
        )
    z = w @ U_base
    print(f"project_to_vocab result shape={tuple(z.shape)}")
    return z


def compute_kurtosis(z: torch.Tensor) -> torch.Tensor:
    print(f"compute_kurtosis: z shape={tuple(z.shape)}")
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if z.ndim != 2:
        raise ValueError(f"Projected tensor must be 1D or 2D, got shape {tuple(z.shape)}")
    mean = z.mean(dim=1, keepdim=True)
    std = z.std(dim=1, unbiased=False, keepdim=True)
    std = torch.where(std == 0, torch.tensor(1.0, device=std.device, dtype=std.dtype), std)
    normalized = (z - mean) / std
    kurtosis = torch.mean(normalized**4, dim=1)
    print(f"compute_kurtosis result shape={tuple(kurtosis.shape)}")
    return kurtosis


def compute_skew(z: torch.Tensor) -> torch.Tensor:
    print(f"compute_skew: z shape={tuple(z.shape)}")
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if z.ndim != 2:
        raise ValueError(f"Projected tensor must be 1D or 2D, got shape {tuple(z.shape)}")
    mean = z.mean(dim=1, keepdim=True)
    std = z.std(dim=1, unbiased=False, keepdim=True)
    std = torch.where(std == 0, torch.tensor(1.0, device=std.device, dtype=std.dtype), std)
    normalized = (z - mean) / std
    skew = torch.mean(normalized**3, dim=1)
    print(f"compute_skew result shape={tuple(skew.shape)}")
    return skew


def choose_directions(U: torch.Tensor, Vh: torch.Tensor, U_base: torch.Tensor) -> tuple[torch.Tensor, str]:
    if Vh.shape[1] == U_base.shape[0]:
        print("Using rows of Vh as the hidden-space directions")
        return Vh, "Vh"
    if U.shape[0] == U_base.shape[0]:
        print("Using columns of U as the hidden-space directions")
        return U.T, "U"
    raise ValueError(
        "Could not align SVD directions to U_base. "
        f"U shape={tuple(U.shape)}, Vh shape={tuple(Vh.shape)}, U_base shape={tuple(U_base.shape)}"
    )


def analyze_svd(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    U_base: torch.Tensor,
    top_k: int | None = None,
    sigma_threshold: float = 0.0,
    kurtosis_threshold: float = 0.0,
    skew_threshold: float = 0.0,
    energy_share_threshold: float = 0.95,
    top_token_count: int = 10,
    token_vocab: dict[int, str] | None = None,
) -> dict[str, Any]:
    print(f"analyze_svd: U={tuple(U.shape)}, S={tuple(S.shape)}, Vh={tuple(Vh.shape)}, U_base={tuple(U_base.shape)}")
    directions, basis_source = choose_directions(U, Vh, U_base)
    n_components = directions.shape[0]
    if top_k is None:
        top_k = n_components
    top_k = min(top_k, n_components)

    selected = directions[:top_k]
    sigma = S[:top_k]
    print(f"Selected top_k={top_k} directions from {basis_source}")

    z = project_to_vocab(selected, U_base)
    kurtosis = compute_kurtosis(z)
    skew = compute_skew(z)

    energy = sigma**2
    total_energy = energy.sum()
    cumulative_energy = energy.cumsum(dim=0) / total_energy
    print(f"Computed cumulative energy: shape={tuple(cumulative_energy.shape)}")

    meaningful_mask = (
        (sigma > sigma_threshold)
        & (kurtosis > kurtosis_threshold)
        & (torch.abs(skew) > skew_threshold)
    )
    meaningful_indices = torch.nonzero(meaningful_mask, as_tuple=False).flatten().tolist()
    estimated_rank = int(meaningful_mask.sum().item())

    rank_at_energy_share = int((cumulative_energy >= energy_share_threshold).nonzero(as_tuple=False).flatten().min().item() + 1) if energy_share_threshold is not None else top_k

    top_tokens_by_direction: dict[str, list[dict[str, Any]]] = {}
    if token_vocab is not None and top_token_count > 0:
        for direction_idx in range(top_k):
            token_scores = z[direction_idx]
            values, indices = torch.topk(token_scores, top_token_count, largest=True)
            tokens = [token_vocab.get(int(idx.item()), str(int(idx.item()))) for idx in indices]
            top_tokens_by_direction[str(direction_idx)] = [
                {"token": token, "score": float(value.item())}
                for token, value in zip(tokens, values)
            ]

    direction_rows = [
        {
            "index": int(i),
            "sigma": float(sigma[i].item()),
            "kurtosis": float(kurtosis[i].item()),
            "skew": float(skew[i].item()),
            "cumulative_energy": float(cumulative_energy[i].item()),
            "is_meaningful": bool(meaningful_mask[i].item()),
        }
        for i in range(top_k)
    ]

    return {
        "basis_source": basis_source,
        "n_components": int(n_components),
        "top_k": int(top_k),
        "sigma_threshold": float(sigma_threshold),
        "kurtosis_threshold": float(kurtosis_threshold),
        "skew_threshold": float(skew_threshold),
        "energy_share_threshold": float(energy_share_threshold),
        "total_energy": float(total_energy.item()),
        "estimated_rank": estimated_rank,
        "rank_at_energy_share": rank_at_energy_share,
        "n_meaningful_directions": int(estimated_rank),
        "direction_rows": direction_rows,
        "top_tokens_by_direction": top_tokens_by_direction,
    }


def load_token_vocab(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".txt", ".csv"}:
        tokens = [line.rstrip("\n") for line in text.splitlines() if line.strip() != ""]
        return {i: token for i, token in enumerate(tokens)}
    data = json.loads(text)
    if isinstance(data, list):
        return {i: str(token) for i, token in enumerate(data)}
    if isinstance(data, dict):
        # Assume token id -> token string mapping
        return {int(k): str(v) for k, v in data.items()}
    raise ValueError("Vocabulary file must be a JSON list/dict or a plain text token-per-line file.")


@app.command()
def main(
    expert_names: list[str] = typer.Argument(
        ..., help="Names of experts to analyze (e.g., Academic, Code, Creative, Math, News, Reddit)"
    ),
    base_model: str = typer.Option(
        "Public", help="Registry key for base model lm_head weights (default: Public)"
    ),
    model_registry: Path = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/flexolmo_models.json",
        "--model-registry",
        "--model_registry",
        help="Path to the expert/public model registry file.",
    ),
    results_dir: Path = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results",
        "--results-dir",
        "--results_dir",
        help="Directory containing singular_values and individual_mlp_weights subdirectories.",
    ),
    cache_dir: Path = typer.Option(
        "/media/am/AM/FlexMoRE/.cache/huggingface/hub",
        "--cache-dir",
        "--cache_dir",
        help="Hugging Face cache directory for downloading registry models.",
    ),
    output_dir: Path = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/svd_vocabulary_projection_analysis",
        "--output-dir",
        "--output_dir",
        help="Directory to write JSON and LaTeX analysis outputs.",
    ),
    top_k: int = typer.Option(256, "--top-k", "--top_k", help="Number of top SVD directions to analyze."),
    sigma_threshold: float = typer.Option(0.0, "--sigma-threshold", "--sigma_threshold", help="Minimum singular value to count a direction as meaningful."),
    kurtosis_threshold: float = typer.Option(0.0, "--kurtosis-threshold", "--kurtosis_threshold", help="Minimum kurtosis to count a direction as meaningful."),
    skew_threshold: float = typer.Option(0.0, "--skew-threshold", "--skew_threshold", help="Minimum absolute skew to count a direction as meaningful."),
    energy_share_threshold: float = typer.Option(0.95, "--energy-share-threshold", "--energy_share_threshold", help="Energy share threshold for rank estimation."),
    top_tokens: int = typer.Option(10, "--top-tokens", "--top_tokens", help="Top vocabulary tokens to return for each direction when a vocab file is provided."),
    token_vocab_path: Path | None = typer.Option(None, "--token-vocab-path", "--token_vocab_path", help="Optional token vocabulary path for top-k token decoding."),
    write_latex_tables: bool = typer.Option(True, "--write-latex-tables", "--write_latex_tables", help="Write per-SVD and summary LaTeX tables."),
    save_full_svd: bool = typer.Option(True, "--save-full-svd", "--save_full_svd", help="Persist recomputed full SVD artifacts (U, S, Vh) under results/full_svd."),
    save_svd_dtype: str = typer.Option("float16", "--save-svd-dtype", "--save_svd_dtype", help="Dtype used when saving U and Vh in full SVD artifacts."),
    force_recompute_svd: bool = typer.Option(False, "--force-recompute-svd", "--force_recompute_svd", help="Ignore cached full SVD artifacts and recompute from delta weights."),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_base_model = resolve_base_model_path(base_model, model_registry, cache_dir)
    print(f"Loading base model checkpoint from {resolved_base_model}")
    state_dict = load_state_dict_file(resolved_base_model)
    U_base = load_unembedding_matrix(state_dict)
    print(f"Loaded base unembedding matrix shape {tuple(U_base.shape)}")

    token_vocab = None
    if token_vocab_path is not None:
        token_vocab = load_token_vocab(token_vocab_path)
        print(f"Loaded token vocabulary with {len(token_vocab)} entries")

    all_reports: list[dict[str, Any]] = []
    for expert_name in expert_names:
        print(f"\n=== Analyzing expert: {expert_name} ===")

        # Load or compute SVD for this expert
        svd_results = load_or_compute_svd_for_expert(
            expert_slug(expert_name),
            results_dir,
            save_full_svd=save_full_svd,
            save_svd_dtype=save_svd_dtype,
            force_recompute=force_recompute_svd,
        )

        # Analyze each layer/module
        expert_reports = []
        for layer_key, (U, S, Vh) in svd_results.items():
            analysis = analyze_svd(
                U=U,
                S=S,
                Vh=Vh,
                U_base=U_base,
                top_k=top_k,
                sigma_threshold=sigma_threshold,
                kurtosis_threshold=kurtosis_threshold,
                skew_threshold=skew_threshold,
                energy_share_threshold=energy_share_threshold,
                top_token_count=top_tokens,
                token_vocab=token_vocab,
            )

            analysis["expert_name"] = expert_name
            analysis["layer_key"] = layer_key
            analysis["svd_name"] = f"{expert_name.lower()}_{layer_key.replace('.', '_').replace('model_layers_', '').replace('mlp_experts_1_', '')}"

            expert_reports.append(analysis)

            output_path = output_dir / f"{analysis['svd_name']}_svd_vocabulary_projection.json"
            output_path.write_text(json.dumps(analysis, indent=2))
            print(f"Wrote analysis for {layer_key} to {output_path}")

        all_reports.extend(expert_reports)

    combined_output_path = output_dir / "svd_vocabulary_projection_all_reports.json"
    combined_output_path.write_text(json.dumps(all_reports, indent=2))
    print(f"Wrote combined analysis to {combined_output_path}")

    if write_latex_tables:
        write_latex_summary_tables(all_reports, output_dir)


if __name__ == "__main__":
    app()
