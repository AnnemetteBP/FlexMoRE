import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer


MODEL_PATHS = {
    "Math": "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/math_singular_values.pt",
    "News": "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/news_singular_values.pt",
    "Academic": "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/academic_singular_values.pt",
    "Reddit": "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/reddit_singular_values.pt",
    "Code": "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/code_singular_values.pt",
    "Creative": "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/creative_singular_values.pt",
}

MODEL_COLORS = {
    "Math": "#1b9e77",
    "News": "#d95f02",
    "Academic": "#7570b3",
    "Reddit": "#e7298a",
    "Code": "#66a61e",
    "Creative": "#e6ab02",
}

MODULE_ORDER = ["all", "down_proj", "gate_proj", "up_proj"]
MODULE_TITLES = {
    "all": "All MLP Projections",
    "down_proj": "Down Projection",
    "gate_proj": "Gate Projection",
    "up_proj": "Up Projection",
}


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


def load_probability_frame(model_name: str, artifact_path: str) -> pd.DataFrame:
    payload = torch.load(artifact_path, map_location="cpu")
    rows = []
    for model_key, singular_values in payload["singular_values"].items():
        singular_values = singular_values.to(dtype=torch.float64)
        total = singular_values.sum()
        if total <= 0:
            continue
        probabilities = (singular_values / total).cpu().numpy()
        module = model_key.split(".")[-1]
        layer = int(model_key.split(".")[2])
        for idx, prob in enumerate(probabilities, start=1):
            rows.append(
                {
                    "model": model_name,
                    "module": module,
                    "layer": layer,
                    "index": idx,
                    "probability": float(prob),
                    "log10_probability": float(np.log10(max(prob, 1e-300))),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    values = df["probability"].to_numpy()
    q25 = float(np.quantile(values, 0.25))
    q75 = float(np.quantile(values, 0.75))
    return {
        "mean": float(values.mean()),
        "variance": float(values.var()),
        "std": float(values.std()),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
    }


def build_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module in MODULE_ORDER:
        for model in MODEL_PATHS.keys():
            subset = df[df["model"] == model]
            if module != "all":
                subset = subset[subset["module"] == module]
            stats = summarize(subset)
            rows.append(
                {
                    "Module": module,
                    "Model": model,
                    "Mean": stats["mean"],
                    "Variance": stats["variance"],
                    "Std": stats["std"],
                    "Q25": stats["q25"],
                    "Q75": stats["q75"],
                    "IQR": stats["iqr"],
                }
            )
    return pd.DataFrame(rows)


def plot_probability_histograms(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True, constrained_layout=True)

    for ax, module in zip(axes.flatten(), MODULE_ORDER):
        for model in MODEL_PATHS.keys():
            subset = df[df["model"] == model]
            if module != "all":
                subset = subset[subset["module"] == module]
            stats = summarize(subset)
            sns.kdeplot(
                data=subset,
                x="log10_probability",
                ax=ax,
                color=MODEL_COLORS[model],
                linewidth=2,
                label=(
                    f"{model}: mean={stats['mean']:.2e}, "
                    f"std={stats['std']:.2e}, "
                    f"IQR=[{stats['q25']:.2e}, {stats['q75']:.2e}]"
                ),
            )

        ax.set_title(MODULE_TITLES[module])
        ax.set_xlabel(r"$\log_{10}(p_i)$")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)

    axes[0, 0].legend(loc="upper left", frameon=True)
    fig.suptitle("Singular-Value Probability Distributions Across Experts", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_probability_boxplots(df: pd.DataFrame, output_path: Path) -> None:
    box_df = df.copy()
    box_df["module_group"] = box_df["module"]
    all_df = box_df.copy()
    all_df["module_group"] = "all"
    box_df = pd.concat([box_df, all_df], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey=True, constrained_layout=True)

    for ax, module in zip(axes.flatten(), MODULE_ORDER):
        subset = box_df[box_df["module_group"] == module]
        sns.boxplot(
            data=subset,
            x="model",
            y="log10_probability",
            order=list(MODEL_PATHS.keys()),
            palette=MODEL_COLORS,
            ax=ax,
            fliersize=1.5,
        )
        ax.set_title(MODULE_TITLES[module])
        ax.set_xlabel("")
        ax.set_ylabel(r"$\log_{10}(p_i)$")
        ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle("Singular-Value Probability Distribution Spread Across Experts", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/probability_distribution_comparison_all_experts",
        help="Directory to write comparison outputs",
    ),
    font_size: int = typer.Option(10, help="Base plotting font size"),
):
    configure_plot_style(font_size)
    output_root = Path(output_dir)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_probability_frame(model, path) for model, path in MODEL_PATHS.items()]
    df = pd.concat(frames, ignore_index=True)

    stats_table = build_stats_table(df)
    stats_table.to_csv(tables_dir / "probability_distribution_stats.csv", index=False)
    with (tables_dir / "probability_distribution_stats.json").open("w") as f:
        json.dump(stats_table.to_dict("records"), f, indent=2)

    plot_probability_histograms(df, figures_dir / "probability_distribution_kde.png")
    plot_probability_boxplots(df, figures_dir / "probability_distribution_boxplots.png")


if __name__ == "__main__":
    typer.run(main)
