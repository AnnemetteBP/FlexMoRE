import json
import math
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer


EXPERT_ORDER = [
    "Math",
    "News",
    "Academic",
    "Code",
    "Creative Writing",
    "Reddit",
]

EXPERT_COLORS = {
    "Math": "#1b9e77",
    "News": "#d95f02",
    "Academic": "#7570b3",
    "Code": "#e7298a",
    "Creative Writing": "#66a61e",
    "Reddit": "#e6ab02",
}

MODULE_ORDER = ["down_proj", "gate_proj", "up_proj"]
MODULE_DISPLAY = {
    "down_proj": "Down",
    "gate_proj": "Gate",
    "up_proj": "Up",
}
MODULE_LINESTYLES = {
    "down_proj": "-",
    "gate_proj": "-",
    "up_proj": "-",
}
MODULE_MARKERS = {
    "down_proj": "o",
    "gate_proj": "s",
    "up_proj": "^",
}


def load_model_registry(model_registry_path: str) -> dict[str, str]:
    registry = runpy.run_path(model_registry_path)
    return registry["expert_models"]


def slugify_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def format_float(value: float) -> str:
    return f"{value:.2f}"


def collect_reports(input_dir: Path, expert_models: dict[str, str]) -> list[dict]:
    reports = []
    for model_name in EXPERT_ORDER:
        if model_name not in expert_models:
            continue
        report_path = input_dir / f"{slugify_model_name(model_name)}_effective_ranks.json"
        if not report_path.exists():
            continue
        with report_path.open("r") as f:
            report = json.load(f)
        report["model_name"] = model_name
        report.setdefault("model_path", expert_models[model_name])
        reports.append(report)
    return reports


def overall_summary_dataframe(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for report in reports:
        stats = report["overall_stats"]
        rows.append(
            {
                "Expert": report["model_name"],
                "Targets": report["num_targets"],
                "Min": stats["min"],
                "Max": stats["max"],
                "Mean": stats["mean"],
                "Median": stats["median"],
                "Std": stats["std"],
            }
        )
    return pd.DataFrame(rows)


def per_module_dataframe(report: dict) -> pd.DataFrame:
    rows = []
    for module_name in MODULE_ORDER:
        stats = report["per_module_stats"][module_name]
        rows.append(
            {
                "Module": MODULE_DISPLAY[module_name],
                "Count": stats["count"],
                "Min": stats["min"],
                "Max": stats["max"],
                "Mean": stats["mean"],
                "Median": stats["median"],
                "Std": stats["std"],
            }
        )
    return pd.DataFrame(rows)


def per_layer_dataframe(report: dict) -> pd.DataFrame:
    rows = []
    for layer_name, stats in report["per_layer_stats"].items():
        layer_idx = int(layer_name.split(".")[1])
        rows.append(
            {
                "Layer": layer_idx,
                "Count": stats["count"],
                "Min": stats["min"],
                "Max": stats["max"],
                "Mean": stats["mean"],
                "Median": stats["median"],
                "Std": stats["std"],
            }
        )
    return pd.DataFrame(rows).sort_values("Layer").reset_index(drop=True)


def per_projection_dataframe(report: dict) -> pd.DataFrame:
    rows = []
    for key, rank in report["effective_ranks"].items():
        parts = key.split(".")
        layer_idx = int(parts[2])
        module_name = parts[-1]
        rows.append(
            {
                "Layer": layer_idx,
                "Module": module_name,
                "Rank": rank,
            }
        )
    return pd.DataFrame(rows).sort_values(["Module", "Layer"]).reset_index(drop=True)


def latex_table(df: pd.DataFrame, output_path: Path, caption: str, label: str, float_columns: list[str]) -> None:
    formatters = {column: format_float for column in float_columns}
    latex = df.to_latex(
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.2f}",
        formatters=formatters,
        caption=caption,
        label=label,
    )
    output_path.write_text(latex)


def write_tables(reports: list[dict], output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    overall_df = overall_summary_dataframe(reports)
    overall_df.to_csv(tables_dir / "overall_summary.csv", index=False)
    latex_table(
        overall_df,
        tables_dir / "overall_summary.tex",
        caption="Overall effective-rank summary across experts.",
        label="tab:effective-rank-overall",
        float_columns=["Mean", "Median", "Std"],
    )

    projection_rows = []
    for report in reports:
        for module_name in MODULE_ORDER:
            stats = report["per_module_stats"][module_name]
            projection_rows.append(
                {
                    "Expert": report["model_name"],
                    "Projection": module_name,
                    "Count": stats["count"],
                    "Min": stats["min"],
                    "Max": stats["max"],
                    "Mean": stats["mean"],
                    "Median": stats["median"],
                    "Std": stats["std"],
                }
            )
    projection_df = pd.DataFrame(projection_rows)
    projection_df.to_csv(tables_dir / "projection_summary.csv", index=False)
    latex_table(
        projection_df,
        tables_dir / "projection_summary.tex",
        caption="Projection-specific effective-rank summary across experts.",
        label="tab:effective-rank-projection-summary",
        float_columns=["Mean", "Median", "Std"],
    )

    for report in reports:
        model_slug = slugify_model_name(report["model_name"])

        module_df = per_module_dataframe(report)
        module_df.to_csv(tables_dir / f"{model_slug}_module_summary.csv", index=False)
        latex_table(
            module_df,
            tables_dir / f"{model_slug}_module_summary.tex",
            caption=f"Per-module effective-rank summary for {report['model_name']}.",
            label=f"tab:{model_slug}-module-summary",
            float_columns=["Mean", "Median", "Std"],
        )

        layer_df = per_layer_dataframe(report)
        layer_df.to_csv(tables_dir / f"{model_slug}_layer_summary.csv", index=False)
        latex_table(
            layer_df,
            tables_dir / f"{model_slug}_layer_summary.tex",
            caption=f"Per-layer MLP effective-rank summary for {report['model_name']}.",
            label=f"tab:{model_slug}-layer-summary",
            float_columns=["Mean", "Median", "Std"],
        )


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


def annotate_stats(ax, report: dict) -> None:
    stats = report["overall_stats"]
    stats_text = (
        f"min={stats['min']}\n"
        f"max={stats['max']}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=max(8, plt.rcParams["font.size"] - 1),
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )


def annotate_explicit_stats(ax, stats: dict) -> None:
    stats_text = (
        f"min={stats['min']}\n"
        f"max={stats['max']}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=max(8, plt.rcParams["font.size"] - 1),
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )


def plot_combined_layer_summary(reports: list[dict], output_dir: Path, width: float, height: float) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(width, height))
    for report in reports:
        layer_df = per_layer_dataframe(report)
        ax.plot(
            layer_df["Layer"],
            layer_df["Mean"],
            label=report["model_name"],
            color=EXPERT_COLORS[report["model_name"]],
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("MLP layer index")
    ax.set_ylabel("Effective rank")
    ax.set_title("Effective rank across MLP layers")
    ax.set_xticks(layer_df["Layer"])
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "effective_rank_overview_layers.png", bbox_inches="tight")
    plt.close(fig)


def plot_expert_subplots(
    reports: list[dict],
    output_dir: Path,
    width: float,
    height: float,
    rows: int,
    cols: int,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, sharex=True, sharey=True)
    flat_axes = axes.flatten()
    layer_ticks = list(range(0, 32, 4))
    if 31 not in layer_ticks:
        layer_ticks.append(31)

    for ax, report in zip(flat_axes, reports):
        layer_df = per_layer_dataframe(report)
        color = EXPERT_COLORS[report["model_name"]]
        overall_stats = report["overall_stats"]
        legend_label = (
            f"mean={overall_stats['mean']:.1f}, "
            f"median={overall_stats['median']:.1f}, "
            f"std={overall_stats['std']:.1f}"
        )
        ax.plot(layer_df["Layer"], layer_df["Mean"], color=color, linewidth=2.0, marker="o", markersize=3.5)
        ax.fill_between(layer_df["Layer"], layer_df["Min"], layer_df["Max"], color=color, alpha=0.15)
        ax.errorbar(
            layer_df["Layer"],
            layer_df["Mean"],
            yerr=layer_df["Std"],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            alpha=0.6,
            capsize=2,
            label=legend_label,
        )
        ax.set_title(report["model_name"])
        annotate_stats(ax, report)
        ax.set_xlabel("MLP layer index")
        ax.set_ylabel("MLP effective rank")
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels([str(tick) for tick in layer_ticks], rotation=0)
        ax.tick_params(axis="x", labelbottom=True)
        ax.legend(loc="upper left", frameon=True)

    for ax in flat_axes[len(reports):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "effective_rank_expert_subplots.png", bbox_inches="tight")
    plt.close(fig)


def plot_projection_subplots(
    reports: list[dict],
    output_dir: Path,
    width: float,
    height: float,
    rows: int,
    cols: int,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for module_name in MODULE_ORDER:
        fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, sharex=True, sharey=True)
        flat_axes = axes.flatten()
        layer_ticks = list(range(0, 32, 4))
        if 31 not in layer_ticks:
            layer_ticks.append(31)
        for ax, report in zip(flat_axes, reports):
            projection_df = per_projection_dataframe(report)
            projection_df = projection_df[projection_df["Module"] == module_name]
            color = EXPERT_COLORS[report["model_name"]]
            module_stats = report["per_module_stats"][module_name]
            legend_label = (
                f"mean={module_stats['mean']:.1f}, "
                f"median={module_stats['median']:.1f}, "
                f"std={module_stats['std']:.1f}"
            )
            ax.plot(
                projection_df["Layer"],
                projection_df["Rank"],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=legend_label,
            )
            ax.set_title(report["model_name"])
            annotate_explicit_stats(ax, module_stats)
            ax.set_xlabel(f"MLP {MODULE_DISPLAY[module_name].lower()} layer index")
            ax.set_ylabel(f"MLP {module_name} effective rank")
            ax.set_xticks(layer_ticks)
            ax.set_xticklabels([str(tick) for tick in layer_ticks], rotation=0)
            ax.tick_params(axis="x", labelbottom=True)
            ax.legend(loc="upper left", frameon=True)

        for ax in flat_axes[len(reports):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(figures_dir / f"effective_rank_{module_name}_subplots.png", bbox_inches="tight")
        plt.close(fig)


def plot_expert_module_overlay_subplots(
    reports: list[dict],
    output_dir: Path,
    width: float,
    height: float,
    rows: int,
    cols: int,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, sharex=True, sharey=True)
    flat_axes = axes.flatten()
    layer_ticks = list(range(0, 32, 4))
    if 31 not in layer_ticks:
        layer_ticks.append(31)

    for ax, report in zip(flat_axes, reports):
        projection_df = per_projection_dataframe(report)
        layer_df = per_layer_dataframe(report)
        expert_color = EXPERT_COLORS[report["model_name"]]
        for module_name in MODULE_ORDER:
            module_df = projection_df[projection_df["Module"] == module_name]
            ax.plot(
                module_df["Layer"],
                module_df["Rank"],
                label=f"MLP {module_name}",
                color=expert_color,
                linestyle=MODULE_LINESTYLES[module_name],
                marker=MODULE_MARKERS[module_name],
                markersize=3.5,
                linewidth=1.6,
                alpha=0.95,
            )

        ax.plot(
            layer_df["Layer"],
            layer_df["Mean"],
            label="MLP mean",
            color="black",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
        )
        ax.set_title(report["model_name"])
        ax.set_xlabel("MLP layer index")
        ax.set_ylabel("Effective rank")
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels([str(tick) for tick in layer_ticks], rotation=0)
        ax.tick_params(axis="x", labelbottom=True)
        ax.legend(loc="lower right", frameon=True, ncol=2)

    for ax in flat_axes[len(reports):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "effective_rank_module_overlay_subplots.png", bbox_inches="tight")
    plt.close(fig)


def write_manifest(reports: list[dict], output_dir: Path) -> None:
    manifest = {
        "experts": [report["model_name"] for report in reports],
        "expert_colors": {name: EXPERT_COLORS[name] for name in [report["model_name"] for report in reports]},
        "module_display_names": MODULE_DISPLAY,
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2))


def main(
    input_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks",
        help="Directory containing per-expert effective-rank JSON reports",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_rank_report",
        help="Directory to store generated tables and figures",
    ),
    model_registry_path: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/flexolmo_models.json",
        help="Path to the expert model registry",
    ),
    subplot_rows: int = typer.Option(2, help="Number of subplot rows for expert grid figures"),
    subplot_cols: int = typer.Option(3, help="Number of subplot columns for expert grid figures"),
    overview_width: float = typer.Option(10.0, help="Figure width for combined overview plot"),
    overview_height: float = typer.Option(5.5, help="Figure height for combined overview plot"),
    subplot_width: float = typer.Option(14.0, help="Figure width for subplot grids"),
    subplot_height: float = typer.Option(8.5, help="Figure height for subplot grids"),
    font_size: int = typer.Option(10, help="Base font size for plots"),
):
    expert_models = load_model_registry(model_registry_path)
    reports = collect_reports(Path(input_dir), expert_models)
    if not reports:
        raise ValueError(f"No effective-rank reports found under {input_dir}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    write_tables(reports, output_root)
    configure_plot_style(font_size)
    plot_combined_layer_summary(reports, output_root, overview_width, overview_height)
    plot_expert_subplots(reports, output_root, subplot_width, subplot_height, subplot_rows, subplot_cols)
    plot_projection_subplots(reports, output_root, subplot_width, subplot_height, subplot_rows, subplot_cols)
    plot_expert_module_overlay_subplots(
        reports,
        output_root,
        subplot_width,
        subplot_height,
        subplot_rows,
        subplot_cols,
    )
    write_manifest(reports, output_root)


if __name__ == "__main__":
    typer.run(main)
