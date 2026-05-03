import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer


MODULE_COLORS = {
    "down_proj": "#c44e52",
    "gate_proj": "#4c72b0",
    "up_proj": "#55a868",
}


def load_dataframe(input_json: Path) -> pd.DataFrame:
    with input_json.open("r") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["rows"])
    return df.sort_values(["layer", "module"]).reset_index(drop=True)


def layer_mean(df: pd.DataFrame, field: str) -> pd.DataFrame:
    return (
        df.groupby("layer", as_index=False)[field]
        .mean()
        .sort_values("layer")
        .reset_index(drop=True)
    )


def plot_energy_rank_panels(df: pd.DataFrame, output_path: Path) -> None:
    tau_fields = [
        ("energy_rank_90", "tau = 0.90"),
        ("energy_rank_95", "tau = 0.95"),
        ("energy_rank_99", "tau = 0.99"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)

    for ax, (field, title) in zip(axes, tau_fields):
        for module_name in ("down_proj", "gate_proj", "up_proj"):
            module_df = (
                df[df["module"] == module_name]
                .sort_values("layer")
                .reset_index(drop=True)
            )
            ax.plot(
                module_df["layer"],
                module_df[field],
                label=module_name,
                color=MODULE_COLORS[module_name],
                linewidth=2,
                alpha=0.95,
            )

        mean_df = layer_mean(df, field)
        ax.plot(
            mean_df["layer"],
            mean_df[field],
            label="MLP mean",
            color="black",
            linestyle="--",
            linewidth=1.8,
        )

        ax.set_title(title)
        ax.set_xlabel("MLP layer index")
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Energy rank")
    axes[-1].legend(loc="lower right", frameon=True)
    fig.suptitle("News Expert Energy-Rank Diagnostics", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_energy_rank_boxplots(df: pd.DataFrame, output_path: Path) -> None:
    tau_fields = [
        ("energy_rank_90", "tau = 0.90"),
        ("energy_rank_95", "tau = 0.95"),
        ("energy_rank_99", "tau = 0.99"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)

    for ax, (field, title) in zip(axes, tau_fields):
        sns.boxplot(
            data=df,
            x="module",
            y=field,
            order=["down_proj", "gate_proj", "up_proj"],
            palette=MODULE_COLORS,
            ax=ax,
            width=0.6,
            fliersize=2.5,
        )
        ax.set_title(title)
        ax.set_xlabel("MLP projection")
        ax.set_xticklabels(["down", "gate", "up"])
        ax.grid(True, axis="y", alpha=0.2)

    axes[0].set_ylabel("Energy rank")
    fig.suptitle("News Expert Energy-Rank Distributions", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_retention_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    sns.boxplot(
        data=df,
        x="module",
        y="fraction_kept",
        order=["down_proj", "gate_proj", "up_proj"],
        palette=MODULE_COLORS,
        ax=ax,
        width=0.6,
        fliersize=2.5,
    )
    ax.set_title("News Expert Threshold-Retention Distribution")
    ax.set_xlabel("MLP projection")
    ax.set_ylabel("Fraction of singular values kept")
    ax.set_xticklabels(["down", "gate", "up"])
    ax.grid(True, axis="y", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(
    input_json: str = typer.Argument(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news/news_singular_value_diagnostics.json",
        help="Diagnostics JSON produced by export_singular_value_diagnostics.py",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news/figures",
        help="Directory for generated figures",
    ),
):
    input_path = Path(input_json)
    output_root = Path(output_dir)
    df = load_dataframe(input_path)
    plot_energy_rank_panels(df, output_root / "news_energy_rank_panels.png")
    plot_energy_rank_boxplots(df, output_root / "news_energy_rank_boxplots.png")
    plot_threshold_retention_boxplot(df, output_root / "news_threshold_retention_boxplot.png")


if __name__ == "__main__":
    typer.run(main)
