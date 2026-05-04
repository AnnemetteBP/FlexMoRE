import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer


EXPERT_ORDER = ["Math", "News", "Academic", "Reddit", "Code", "Creative"]
EXPERT_COLORS = {
    "Math": "#1b9e77",
    "News": "#d95f02",
    "Academic": "#7570b3",
    "Reddit": "#e7298a",
    "Code": "#66a61e",
    "Creative": "#e6ab02",
}
MODULE_ORDER = ["down_proj", "gate_proj", "up_proj"]
MODULE_TITLES = {
    "down_proj": "Down Projection",
    "gate_proj": "Gate Projection",
    "up_proj": "Up Projection",
}
METRICS = {
    "energy_rank_095": {
        "title": "Energy Rank at Tau = 0.95",
        "ylabel": "Rank",
        "filename_prefix": "energy_rank_095_projection_focus",
        "format": ".1f",
    },
    "top50_energy_share": {
        "title": "Top-50 Energy Share",
        "ylabel": "Energy share",
        "filename_prefix": "top50_energy_share_projection_focus",
        "format": ".3f",
    },
    "spike_ratio_1_10": {
        "title": "Spectral Spike Ratio (Sigma_1 / Sigma_10)",
        "ylabel": "Ratio",
        "filename_prefix": "spike_ratio_1_10_projection_focus",
        "format": ".3f",
    },
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


def load_rows(input_path: Path) -> pd.DataFrame:
    if input_path.suffix == ".json":
        with input_path.open("r") as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(input_path)
    return df.sort_values(["expert", "layer", "module"]).reset_index(drop=True)


def overall_stats(df: pd.DataFrame, value_col: str) -> dict:
    return {
        "Mean": float(df[value_col].mean()),
        "Median": float(df[value_col].median()),
        "Std": float(df[value_col].std(ddof=1)),
        "Min": float(df[value_col].min()),
        "Max": float(df[value_col].max()),
    }


def plot_projection_focus(df: pd.DataFrame, metric_name: str, meta: dict, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, constrained_layout=True)
    layer_ticks = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    value_format = meta["format"]

    for ax, module_name in zip(axes, MODULE_ORDER):
        module_df = df[df["module"] == module_name]
        for expert_name in EXPERT_ORDER:
            subset = module_df[module_df["expert"] == expert_name].sort_values("layer")
            stats = overall_stats(subset, metric_name)
            ax.plot(
                subset["layer"],
                subset[metric_name],
                color=EXPERT_COLORS[expert_name],
                linewidth=2.0,
                label=(
                    f"{expert_name}\n"
                    f"mean={stats['Mean']:{value_format}}, "
                    f"median={stats['Median']:{value_format}}, "
                    f"std={stats['Std']:{value_format}}"
                ),
            )

        ax.set_title(MODULE_TITLES[module_name])
        ax.set_xlabel("MLP layer index")
        ax.set_xticks(layer_ticks)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel(meta["ylabel"])
    axes[-1].legend(loc="best", frameon=True)
    fig.suptitle(meta["title"], fontsize=14)
    fig.savefig(figures_dir / f"{meta['filename_prefix']}.png", bbox_inches="tight")
    plt.close(fig)


def main(
    input_path: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/hybrid_rank_analysis_all_experts/hybrid_rank_analysis_rows.csv",
        help="Joined hybrid rank analysis rows CSV or JSON",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/svd_projection_focus_report_all_experts",
        help="Directory to write figures",
    ),
    font_size: int = typer.Option(10, help="Base plotting font size"),
):
    configure_plot_style(font_size)
    df = load_rows(Path(input_path))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for metric_name, meta in METRICS.items():
        plot_projection_focus(df, metric_name, meta, output_root)


if __name__ == "__main__":
    typer.run(main)
