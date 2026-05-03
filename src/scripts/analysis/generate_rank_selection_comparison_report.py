import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import typer


MODULE_ORDER = ["down_proj", "gate_proj", "up_proj"]
MODULE_COLORS = {
    "down_proj": "#c44e52",
    "gate_proj": "#4c72b0",
    "up_proj": "#55a868",
}
MODELS = [
    ("Math", "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/math_singular_values.pt"),
    ("News", "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/news_singular_values.pt"),
]
PROB_THRESHOLDS = [1e-4, 3e-4, 5e-4]
ENERGY_TAUS = [0.68, 0.95, 0.997]


def configure_plot_style(base_font_size: int) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 1,
            "axes.labelsize": base_font_size,
            "legend.fontsize": base_font_size - 1,
            "xtick.labelsize": base_font_size - 1,
            "ytick.labelsize": base_font_size - 1,
        }
    )


def load_singular_values_map(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload["singular_values"]


def compute_probability_rank(singular_values: torch.Tensor, probability_threshold: float) -> tuple[int, int]:
    singular_values = singular_values.to(dtype=torch.float64)
    norm = singular_values.sum()
    if norm <= 0:
        return 1, 0
    probabilities = singular_values / norm
    nonzero_probabilities = probabilities[probabilities > probability_threshold]
    probability_norm = nonzero_probabilities.sum()
    if probability_norm <= 0:
        return 1, 0
    nonzero_probabilities = nonzero_probabilities / probability_norm
    entropy = -(nonzero_probabilities * nonzero_probabilities.log()).sum()
    rank = math.ceil(torch.exp(entropy).item())
    rank = max(1, min(rank, singular_values.numel()))
    return rank, int(nonzero_probabilities.numel())


def compute_energy_rank(singular_values: torch.Tensor, energy_tau: float) -> int:
    singular_values = torch.sort(singular_values.to(dtype=torch.float64), descending=True).values
    if singular_values.numel() == 0:
        return 1
    energy = singular_values.square()
    total_energy = energy.sum()
    if total_energy <= 0:
        return 1
    explained = energy / total_energy
    cumulative = explained.cumsum(dim=0)
    rank = int(torch.searchsorted(cumulative, torch.tensor(energy_tau, dtype=torch.float64)).item()) + 1
    return max(1, min(rank, singular_values.numel()))


def rows_for_method(model_name: str, sv_map: dict[str, torch.Tensor], method: str, value: float) -> list[dict]:
    rows = []
    for model_key in sorted(sv_map.keys(), key=lambda mk: (int(mk.split(".")[2]), {"down_proj": 0, "gate_proj": 1, "up_proj": 2}[mk.split(".")[-1]])):
        s = sv_map[model_key]
        layer = int(model_key.split(".")[2])
        module = model_key.split(".")[-1]
        if method == "probability":
            rank, aux = compute_probability_rank(s, value)
        else:
            rank = compute_energy_rank(s, value)
            aux = rank
        rows.append(
            {
                "model": model_name,
                "method": method,
                "setting": value,
                "layer": layer,
                "module": module,
                "rank": rank,
                "aux": aux,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    ranks = sorted(row["rank"] for row in rows)
    n = len(ranks)
    return {
        "mean": sum(ranks) / n,
        "median": (ranks[n // 2 - 1] + ranks[n // 2]) / 2 if n % 2 == 0 else ranks[n // 2],
        "std": pd.Series(ranks).std(ddof=1),
        "min": min(ranks),
        "max": max(ranks),
    }


def plot_grid(df: pd.DataFrame, method: str, settings: list[float], output_path: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True, sharey=True, constrained_layout=True)
    layer_ticks = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    for row_idx, model_name in enumerate(["Math", "News"]):
        for col_idx, setting in enumerate(settings):
            ax = axes[row_idx][col_idx]
            subset = df[(df["model"] == model_name) & (df["method"] == method) & (df["setting"] == setting)].copy()
            subset = subset.sort_values(["layer", "module"]).reset_index(drop=True)
            stats = summarize(subset.to_dict("records"))

            for module_name in MODULE_ORDER:
                module_df = subset[subset["module"] == module_name].sort_values("layer")
                ax.plot(
                    module_df["layer"],
                    module_df["rank"],
                    color=MODULE_COLORS[module_name],
                    linewidth=1.8,
                    label=module_name,
                )

            layer_mean = subset.groupby("layer", as_index=False)["rank"].mean()
            ax.plot(
                layer_mean["layer"],
                layer_mean["rank"],
                color="black",
                linestyle="--",
                linewidth=1.8,
                label=f"MLP mean\nmean={stats['mean']:.1f}, median={stats['median']:.1f}, std={stats['std']:.1f}",
            )
            ax.set_title(f"{model_name} | {title_prefix}={setting}")
            ax.set_xticks(layer_ticks)
            ax.grid(True, alpha=0.2)
            ax.text(
                0.98,
                0.02,
                f"min={stats['min']}\nmax={stats['max']}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=max(8, plt.rcParams["font.size"] - 1),
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
            )

    for ax in axes[-1]:
        ax.set_xlabel("MLP layer index")
    for ax in axes[:, 0]:
        ax.set_ylabel("Selected rank")
    axes[0][-1].legend(loc="upper left", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/rank_selection_comparison",
        help="Directory to write comparison outputs",
    ),
    font_size: int = typer.Option(10, help="Base plotting font size"),
):
    configure_plot_style(font_size)
    output_root = Path(output_dir)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model_name, path in MODELS:
        sv_map = load_singular_values_map(path)
        for threshold in PROB_THRESHOLDS:
            all_rows.extend(rows_for_method(model_name, sv_map, "probability", threshold))
        for tau in ENERGY_TAUS:
            all_rows.extend(rows_for_method(model_name, sv_map, "energy", tau))

    df = pd.DataFrame(all_rows)
    with (output_root / "comparison_rows.json").open("w") as f:
        json.dump(all_rows, f, indent=2)

    plot_grid(df, "probability", PROB_THRESHOLDS, figures_dir / "probability_threshold_comparison.png", "p")
    plot_grid(df, "energy", ENERGY_TAUS, figures_dir / "energy_tau_comparison.png", "tau")


if __name__ == "__main__":
    typer.run(main)
