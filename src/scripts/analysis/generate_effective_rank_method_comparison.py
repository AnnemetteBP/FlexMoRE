import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer


MODULE_ORDER = ["down_proj", "gate_proj", "up_proj"]
MODULE_COLORS = {
    "down_proj": "#1b9e77",
    "gate_proj": "#d95f02",
    "up_proj": "#7570b3",
}


def load_report(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def per_projection_dataframe(report: dict, method_name: str) -> pd.DataFrame:
    rows = []
    for key, rank in sorted(report["effective_ranks"].items()):
        parts = key.split(".")
        rows.append(
            {
                "layer": int(parts[2]),
                "module": parts[-1],
                "rank": rank,
                "method": method_name,
            }
        )
    return pd.DataFrame(rows).sort_values(["module", "layer"]).reset_index(drop=True)


def per_layer_mean_dataframe(report: dict, method_name: str) -> pd.DataFrame:
    rows = []
    for layer_name, stats in sorted(
        report["per_layer_stats"].items(),
        key=lambda item: int(item[0].split(".")[1]),
    ):
        rows.append(
            {
                "layer": int(layer_name.split(".")[1]),
                "mean_rank": stats["mean"],
                "method": method_name,
            }
        )
    return pd.DataFrame(rows)


def comparison_table(naive: dict, thresholded: dict) -> pd.DataFrame:
    rows = []
    overall_pairs = [
        ("overall", naive["overall_stats"], thresholded["overall_stats"]),
        *[
            (module, naive["per_module_stats"][module], thresholded["per_module_stats"][module])
            for module in MODULE_ORDER
        ],
    ]
    for group_name, naive_stats, threshold_stats in overall_pairs:
        rows.append(
            {
                "group": group_name,
                "naive_mean": naive_stats["mean"],
                "threshold_mean": threshold_stats["mean"],
                "delta_mean": threshold_stats["mean"] - naive_stats["mean"],
                "naive_median": naive_stats["median"],
                "threshold_median": threshold_stats["median"],
                "delta_median": threshold_stats["median"] - naive_stats["median"],
                "naive_std": naive_stats["std"],
                "threshold_std": threshold_stats["std"],
                "delta_std": threshold_stats["std"] - naive_stats["std"],
                "naive_min": naive_stats["min"],
                "threshold_min": threshold_stats["min"],
                "naive_max": naive_stats["max"],
                "threshold_max": threshold_stats["max"],
            }
        )
    return pd.DataFrame(rows)


def plot_news_method_comparison(
    naive: dict,
    thresholded: dict,
    output_path: Path,
    title: str,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    naive_layer = per_layer_mean_dataframe(naive, "Naive")
    threshold_layer = per_layer_mean_dataframe(thresholded, "Threshold v01")
    naive_proj = per_projection_dataframe(naive, "Naive")
    threshold_proj = per_projection_dataframe(thresholded, "Threshold v01")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
    subplot_specs = [
        ("Naive", naive_proj, naive_layer, axes[0]),
        ("Threshold v01", threshold_proj, threshold_layer, axes[1]),
    ]

    for method_name, proj_df, layer_df, ax in subplot_specs:
        for module_name in MODULE_ORDER:
            module_df = proj_df[proj_df["module"] == module_name]
            color = MODULE_COLORS[module_name]
            ax.plot(
                module_df["layer"],
                module_df["rank"],
                color=color,
                linestyle="-",
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=f"MLP {module_name}",
            )

        ax.plot(
            layer_df["layer"],
            layer_df["mean_rank"],
            color="black",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            label="MLP mean",
        )
        ax.set_title(method_name)
        ax.set_xlabel("MLP layer index")
        ax.legend(loc="lower right", frameon=True, ncol=2)

    layer_ticks = list(range(0, 32, 4))
    if 31 not in layer_ticks:
        layer_ticks.append(31)
    for ax in axes:
        ax.set_xticks(layer_ticks)
    axes[0].set_ylabel("Effective rank")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(
    naive_report: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks_naive/news_effective_ranks.json",
        help="Path to the naive News effective-rank JSON report",
    ),
    threshold_report: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks_thresholds_v01/news_effective_ranks.json",
        help="Path to the thresholded News effective-rank JSON report",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks_method_comparison",
        help="Directory to store method-comparison outputs",
    ),
):
    output_root = Path(output_dir)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    naive = load_report(naive_report)
    thresholded = load_report(threshold_report)

    plot_news_method_comparison(
        naive,
        thresholded,
        figures_dir / "news_method_comparison_overlay.png",
        "News Expert: Naive vs Threshold v01 Effective Rank",
    )

    summary_df = comparison_table(naive, thresholded)
    summary_df.to_csv(tables_dir / "news_method_comparison_summary.csv", index=False)
    (tables_dir / "news_method_comparison_summary.tex").write_text(
        summary_df.to_latex(index=False, float_format=lambda value: f"{value:.2f}")
    )


if __name__ == "__main__":
    typer.run(main)
