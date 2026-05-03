import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer


MODULE_ORDER = ["down_proj", "gate_proj", "up_proj"]
MODULE_COLORS = {
    "down_proj": "#c44e52",
    "gate_proj": "#4c72b0",
    "up_proj": "#55a868",
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


def load_rows(input_json: Path) -> list[dict]:
    with input_json.open("r") as f:
        payload = json.load(f)
    return sorted(payload["rows"], key=lambda row: (row["layer"], MODULE_ORDER.index(row["module"])))


def stack_curves(rows: list[dict], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field], dtype=np.float64) for row in rows], axis=0)


def flatten_distribution_rows(rows: list[dict]) -> pd.DataFrame:
    flat_rows = []
    selected_indices = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096]
    for row in rows:
        normalized = row["normalized_singular_values"]
        for idx, value in enumerate(normalized, start=1):
            flat_rows.append(
                {
                    "layer": row["layer"],
                    "module": row["module"],
                    "index": idx,
                    "normalized_singular_value": value,
                    "log10_normalized_singular_value": np.log10(max(value, 1e-300)),
                    "selected_index": idx if idx in selected_indices else None,
                }
            )
    return pd.DataFrame(flat_rows)


def plot_decay(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)

    for ax, module_name in zip(axes, MODULE_ORDER):
        module_rows = [row for row in rows if row["module"] == module_name]
        curves = stack_curves(module_rows, "normalized_singular_values")
        x = np.arange(1, curves.shape[1] + 1)

        for curve in curves:
            ax.plot(x, curve, color=MODULE_COLORS[module_name], alpha=0.10, linewidth=0.9)

        median = np.median(curves, axis=0)
        q25 = np.quantile(curves, 0.25, axis=0)
        q75 = np.quantile(curves, 0.75, axis=0)

        ax.plot(x, median, color=MODULE_COLORS[module_name], linewidth=2.2, label=f"{module_name} median")
        ax.fill_between(x, q25, q75, color=MODULE_COLORS[module_name], alpha=0.20, label="IQR")
        ax.set_title(module_name)
        ax.set_xlabel("Singular value index")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", frameon=True)

    axes[0].set_ylabel("Normalized singular value ($\\sigma_i / \\sigma_1$)")
    fig.suptitle("News Expert Singular Value Decay", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_decay_quantiles(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)

    for ax, module_name in zip(axes, MODULE_ORDER):
        module_rows = [row for row in rows if row["module"] == module_name]
        curves = stack_curves(module_rows, "normalized_singular_values")
        x = np.arange(1, curves.shape[1] + 1)

        median = np.median(curves, axis=0)
        q10 = np.quantile(curves, 0.10, axis=0)
        q25 = np.quantile(curves, 0.25, axis=0)
        q75 = np.quantile(curves, 0.75, axis=0)
        q90 = np.quantile(curves, 0.90, axis=0)

        ax.plot(x, median, color=MODULE_COLORS[module_name], linewidth=2.4, label="median")
        ax.fill_between(x, q25, q75, color=MODULE_COLORS[module_name], alpha=0.25, label="25-75%")
        ax.fill_between(x, q10, q90, color=MODULE_COLORS[module_name], alpha=0.12, label="10-90%")
        ax.set_title(module_name)
        ax.set_xlabel("Singular value index")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", frameon=True)

    axes[0].set_ylabel("Normalized singular value ($\\sigma_i / \\sigma_1$)")
    fig.suptitle("News Expert Singular Value Decay Quantiles", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_energy(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)

    for ax, module_name in zip(axes, MODULE_ORDER):
        module_rows = [row for row in rows if row["module"] == module_name]
        curves = stack_curves(module_rows, "cumulative_energy")
        x = np.arange(1, curves.shape[1] + 1)

        for curve in curves:
            ax.plot(x, curve, color=MODULE_COLORS[module_name], alpha=0.10, linewidth=0.9)

        median = np.median(curves, axis=0)
        q25 = np.quantile(curves, 0.25, axis=0)
        q75 = np.quantile(curves, 0.75, axis=0)

        ax.plot(x, median, color=MODULE_COLORS[module_name], linewidth=2.2, label=f"{module_name} median")
        ax.fill_between(x, q25, q75, color=MODULE_COLORS[module_name], alpha=0.20, label="IQR")
        for tau, linestyle in ((0.9, "--"), (0.95, ":"), (0.99, "-.")):
            ax.axhline(tau, color="#444444", linestyle=linestyle, linewidth=1.0, alpha=0.8)

        ax.set_title(module_name)
        ax.set_xlabel("Singular value index")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right", frameon=True)

    axes[0].set_ylabel("Cumulative spectral energy")
    fig.suptitle("News Expert Cumulative Energy Curves", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_energy_quantiles(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)

    for ax, module_name in zip(axes, MODULE_ORDER):
        module_rows = [row for row in rows if row["module"] == module_name]
        curves = stack_curves(module_rows, "cumulative_energy")
        x = np.arange(1, curves.shape[1] + 1)

        median = np.median(curves, axis=0)
        q10 = np.quantile(curves, 0.10, axis=0)
        q25 = np.quantile(curves, 0.25, axis=0)
        q75 = np.quantile(curves, 0.75, axis=0)
        q90 = np.quantile(curves, 0.90, axis=0)

        ax.plot(x, median, color=MODULE_COLORS[module_name], linewidth=2.4, label="median")
        ax.fill_between(x, q25, q75, color=MODULE_COLORS[module_name], alpha=0.25, label="25-75%")
        ax.fill_between(x, q10, q90, color=MODULE_COLORS[module_name], alpha=0.12, label="10-90%")
        for tau, linestyle in ((0.9, "--"), (0.95, ":"), (0.99, "-.")):
            ax.axhline(tau, color="#444444", linestyle=linestyle, linewidth=1.0, alpha=0.8)
        ax.set_title(module_name)
        ax.set_xlabel("Singular value index")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right", frameon=True)

    axes[0].set_ylabel("Cumulative spectral energy")
    fig.suptitle("News Expert Cumulative Energy Quantiles", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_singular_value_distribution_histograms(rows: list[dict], output_path: Path) -> None:
    df = flatten_distribution_rows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)

    for ax, module_name in zip(axes, MODULE_ORDER):
        module_df = df[df["module"] == module_name]
        sns.histplot(
            module_df["log10_normalized_singular_value"],
            bins=80,
            stat="density",
            color=MODULE_COLORS[module_name],
            alpha=0.75,
            ax=ax,
        )
        ax.set_title(module_name)
        ax.set_xlabel(r"$\log_{10}(\sigma_i / \sigma_1)$")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Density")
    fig.suptitle("News Expert Singular-Value Distributions", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_selected_index_boxplots(rows: list[dict], output_path: Path) -> None:
    df = flatten_distribution_rows(rows)
    df = df[df["selected_index"].notna()].copy()
    df["selected_index"] = df["selected_index"].astype(int).astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True, constrained_layout=True)

    index_order = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "3072", "4096"]
    for ax, module_name in zip(axes, MODULE_ORDER):
        module_df = df[df["module"] == module_name]
        sns.boxplot(
            data=module_df,
            x="selected_index",
            y="log10_normalized_singular_value",
            color=MODULE_COLORS[module_name],
            ax=ax,
            fliersize=1.8,
        )
        ax.set_title(module_name)
        ax.set_xlabel("Singular value index")
        ax.set_xticklabels(index_order, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.2)

    axes[0].set_ylabel(r"$\log_{10}(\sigma_i / \sigma_1)$")
    fig.suptitle("News Expert Singular Values at Selected Indices", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(
    input_json: str = typer.Argument(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news/news_singular_value_curves.json",
        help="Curve JSON produced by export_singular_value_curves.py",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news/figures",
        help="Directory for generated figures",
    ),
    font_size: int = typer.Option(10, help="Base plotting font size"),
):
    configure_plot_style(font_size)
    rows = load_rows(Path(input_json))
    output_root = Path(output_dir)
    plot_decay(rows, output_root / "news_singular_value_decay.png")
    plot_decay_quantiles(rows, output_root / "news_singular_value_decay_quantiles.png")
    plot_cumulative_energy(rows, output_root / "news_cumulative_energy_curves.png")
    plot_cumulative_energy_quantiles(rows, output_root / "news_cumulative_energy_quantiles.png")
    plot_singular_value_distribution_histograms(rows, output_root / "news_singular_value_distribution_histograms.png")
    plot_selected_index_boxplots(rows, output_root / "news_singular_value_selected_index_boxplots.png")


if __name__ == "__main__":
    typer.run(main)
