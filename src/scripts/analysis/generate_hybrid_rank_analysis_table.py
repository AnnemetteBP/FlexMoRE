import json
from pathlib import Path

import pandas as pd
import torch
import typer


EXPERTS = [
    ("Math", "math"),
    ("News", "news"),
    ("Academic", "academic"),
]


def compute_energy_rank(singular_values: torch.Tensor, tau: float) -> int:
    singular_values = torch.sort(singular_values.to(dtype=torch.float64), descending=True).values
    if singular_values.numel() == 0:
        return 1
    energy = singular_values.square()
    total_energy = energy.sum()
    if total_energy <= 0:
        return 1
    cumulative = (energy / total_energy).cumsum(dim=0)
    rank = int(torch.searchsorted(cumulative, torch.tensor(tau, dtype=torch.float64)).item()) + 1
    return max(1, min(rank, singular_values.numel()))


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


def compute_spike_ratio(singular_values: torch.Tensor, i: int, j: int) -> float:
    singular_values = torch.sort(singular_values.to(dtype=torch.float64), descending=True).values
    if singular_values.numel() < j or singular_values[j - 1].item() == 0:
        return float("inf")
    return float((singular_values[i - 1] / singular_values[j - 1]).item())


def load_metric_map(slug: str) -> dict[str, dict]:
    path = Path(f"/media/am/AM/FlexMoRE/src/scripts/analysis/results/expert_mlp_metrics/{slug}_mlp_metrics.json")
    with path.open("r") as f:
        payload = json.load(f)
    return {row["model_key"]: row for row in payload["per_layer_metrics"]}


def load_singular_values_map(slug: str) -> dict[str, torch.Tensor]:
    path = Path(f"/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/{slug}_singular_values.pt")
    payload = torch.load(path, map_location="cpu")
    return payload["singular_values"]


def build_rows() -> list[dict]:
    rows = []
    for expert_name, slug in EXPERTS:
        metric_map = load_metric_map(slug)
        sv_map = load_singular_values_map(slug)
        for model_key in sorted(metric_map.keys(), key=lambda mk: (int(mk.split(".")[2]), {"down_proj": 0, "gate_proj": 1, "up_proj": 2}[mk.split(".")[-1]])):
            metric = metric_map[model_key]
            singular_values = sv_map[model_key]
            rows.append(
                {
                    "expert": expert_name,
                    "model_key": model_key,
                    "layer": metric["layer"],
                    "module": metric["module"],
                    "public_expert_cosine": metric["public_expert_cosine"],
                    "relative_delta_norm": metric["relative_delta_norm"],
                    "delta_norm_fro": metric["delta_norm_fro"],
                    "energy_rank_090": compute_energy_rank(singular_values, 0.90),
                    "energy_rank_095": compute_energy_rank(singular_values, 0.95),
                    "energy_rank_0997": compute_energy_rank(singular_values, 0.997),
                    "top10_energy_share": compute_topk_energy_share(singular_values, 10),
                    "top50_energy_share": compute_topk_energy_share(singular_values, 50),
                    "top100_energy_share": compute_topk_energy_share(singular_values, 100),
                    "spike_ratio_1_2": compute_spike_ratio(singular_values, 1, 2),
                    "spike_ratio_1_10": compute_spike_ratio(singular_values, 1, 10),
                }
            )
    return rows


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "public_expert_cosine",
        "relative_delta_norm",
        "delta_norm_fro",
        "energy_rank_090",
        "energy_rank_095",
        "energy_rank_0997",
        "top10_energy_share",
        "top50_energy_share",
        "top100_energy_share",
        "spike_ratio_1_2",
        "spike_ratio_1_10",
    ]
    rows = []
    for expert in df["expert"].unique():
        subset = df[df["expert"] == expert]
        corr = subset[metric_cols].corr(method="spearman")
        for lhs in ["public_expert_cosine", "relative_delta_norm", "delta_norm_fro"]:
            for rhs in ["energy_rank_090", "energy_rank_095", "energy_rank_0997", "top50_energy_share", "spike_ratio_1_10"]:
                rows.append(
                    {
                        "expert": expert,
                        "lhs": lhs,
                        "rhs": rhs,
                        "spearman_corr": float(corr.loc[lhs, rhs]),
                    }
                )
    return pd.DataFrame(rows)


def main(
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/hybrid_rank_analysis",
        help="Directory to write joined tables",
    ),
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "hybrid_rank_analysis_rows.csv", index=False)
    with (output_root / "hybrid_rank_analysis_rows.json").open("w") as f:
        json.dump(rows, f, indent=2)

    corr_df = correlation_table(df)
    corr_df.to_csv(output_root / "hybrid_rank_analysis_correlations.csv", index=False)

    summary = (
        df.groupby(["expert", "module"], as_index=False)
        .agg(
            public_expert_cosine_mean=("public_expert_cosine", "mean"),
            relative_delta_norm_mean=("relative_delta_norm", "mean"),
            energy_rank_095_mean=("energy_rank_095", "mean"),
            top50_energy_share_mean=("top50_energy_share", "mean"),
        )
    )
    summary.to_csv(output_root / "hybrid_rank_analysis_summary.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
