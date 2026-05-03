import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer


EXPERT_ORDER = ["Math", "News", "Academic"]
EXPERT_COLORS = {
    "Math": "#1b9e77",
    "News": "#d95f02",
    "Academic": "#7570b3",
}
MODULE_ORDER = ["down_proj", "gate_proj", "up_proj"]
MODULE_COLORS = {
    "down_proj": "#c44e52",
    "gate_proj": "#4c72b0",
    "up_proj": "#55a868",
}
METRICS = {
    "public_expert_cosine": {
        "title": "Public-Expert Cosine Similarity",
        "ylabel": "Cosine similarity",
        "filename_prefix": "public_expert_cosine",
    },
    "relative_delta_norm": {
        "title": "Relative Delta Norm",
        "ylabel": "Relative delta norm",
        "filename_prefix": "relative_delta_norm",
    },
    "public_delta_cosine": {
        "title": "Public-Delta Cosine Similarity",
        "ylabel": "Cosine similarity",
        "filename_prefix": "public_delta_cosine",
    },
    "expert_delta_cosine": {
        "title": "Expert-Delta Cosine Similarity",
        "ylabel": "Cosine similarity",
        "filename_prefix": "expert_delta_cosine",
    },
}


def slugify_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def format_float(value: float) -> str:
    return f"{value:.4f}"


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


def load_report(report_path: Path, model_name: str) -> dict:
    with report_path.open("r") as f:
        payload = json.load(f)
    payload["model_name"] = model_name
    return payload


def collect_reports(input_dir: Path) -> list[dict]:
    reports = []
    for model_name in EXPERT_ORDER:
        report_path = input_dir / f"{slugify_model_name(model_name)}_mlp_metrics.json"
        if report_path.exists():
            reports.append(load_report(report_path, model_name))
    return reports


def metric_dataframe(report: dict, metric_name: str) -> pd.DataFrame:
    rows = []
    for row in report["per_layer_metrics"]:
        rows.append(
            {
                "Layer": row["layer"],
                "Module": row["module"],
                "Value": row[metric_name],
                "Expert": report["model_name"],
            }
        )
    return pd.DataFrame(rows).sort_values(["Layer", "Module"]).reset_index(drop=True)


def per_layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Layer", as_index=False)["Value"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "Mean",
                "median": "Median",
                "std": "Std",
                "min": "Min",
                "max": "Max",
                "count": "Count",
            }
        )
        .sort_values("Layer")
        .reset_index(drop=True)
    )


def overall_stats(df: pd.DataFrame) -> dict:
    return {
        "Count": int(df["Value"].count()),
        "Mean": float(df["Value"].mean()),
        "Median": float(df["Value"].median()),
        "Std": float(df["Value"].std(ddof=1)),
        "Min": float(df["Value"].min()),
        "Max": float(df["Value"].max()),
    }


def per_module_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("Module", as_index=False)["Value"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "Mean",
                "median": "Median",
                "std": "Std",
                "min": "Min",
                "max": "Max",
                "count": "Count",
            }
        )
    )
    return out.sort_values("Module").reset_index(drop=True)


def latex_table(df: pd.DataFrame, output_path: Path, caption: str, label: str, float_columns: list[str]) -> None:
    formatters = {column: format_float for column in float_columns}
    latex = df.to_latex(
        index=False,
        escape=False,
        formatters=formatters,
        caption=caption,
        label=label,
    )
    output_path.write_text(latex)


def write_tables(reports: list[dict], output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, meta in METRICS.items():
        overall_rows = []
        projection_rows = []
        for report in reports:
            df = metric_dataframe(report, metric_name)
            stats = overall_stats(df)
            overall_rows.append({"Expert": report["model_name"], **stats})

            module_df = per_module_summary(df)
            for _, row in module_df.iterrows():
                projection_rows.append(
                    {
                        "Expert": report["model_name"],
                        "Projection": row["Module"],
                        "Count": int(row["Count"]),
                        "Mean": row["Mean"],
                        "Median": row["Median"],
                        "Std": row["Std"],
                        "Min": row["Min"],
                        "Max": row["Max"],
                    }
                )

            layer_df = per_layer_summary(df)
            layer_df.to_csv(tables_dir / f"{slugify_model_name(report['model_name'])}_{meta['filename_prefix']}_layer_summary.csv", index=False)
            latex_table(
                layer_df,
                tables_dir / f"{slugify_model_name(report['model_name'])}_{meta['filename_prefix']}_layer_summary.tex",
                caption=f"Per-layer summary for {report['model_name']} using {metric_name}.",
                label=f"tab:{slugify_model_name(report['model_name'])}-{meta['filename_prefix']}-layer-summary",
                float_columns=["Mean", "Median", "Std", "Min", "Max"],
            )

        overall_df = pd.DataFrame(overall_rows)
        overall_df.to_csv(tables_dir / f"{meta['filename_prefix']}_overall_summary.csv", index=False)
        latex_table(
            overall_df,
            tables_dir / f"{meta['filename_prefix']}_overall_summary.tex",
            caption=f"Overall summary across experts for {metric_name}.",
            label=f"tab:{meta['filename_prefix']}-overall-summary",
            float_columns=["Mean", "Median", "Std", "Min", "Max"],
        )

        projection_df = pd.DataFrame(projection_rows)
        projection_df.to_csv(tables_dir / f"{meta['filename_prefix']}_projection_summary.csv", index=False)
        latex_table(
            projection_df,
            tables_dir / f"{meta['filename_prefix']}_projection_summary.tex",
            caption=f"Projection summary across experts for {metric_name}.",
            label=f"tab:{meta['filename_prefix']}-projection-summary",
            float_columns=["Mean", "Median", "Std", "Min", "Max"],
        )


def annotate_stats(ax, stats: dict) -> None:
    ax.text(
        0.98,
        0.02,
        f"min={stats['Min']:.3f}\nmax={stats['Max']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=max(8, plt.rcParams["font.size"] - 1),
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )


def plot_expert_subplots(reports: list[dict], metric_name: str, meta: dict, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(reports), figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)
    if len(reports) == 1:
        axes = [axes]

    layer_ticks = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    for ax, report in zip(axes, reports):
        df = metric_dataframe(report, metric_name)
        layer_df = per_layer_summary(df)
        stats = overall_stats(df)

        for module_name in MODULE_ORDER:
            module_df = df[df["Module"] == module_name].sort_values("Layer")
            ax.plot(
                module_df["Layer"],
                module_df["Value"],
                label=module_name,
                color=MODULE_COLORS[module_name],
                linewidth=1.8,
                alpha=0.95,
            )

        ax.plot(
            layer_df["Layer"],
            layer_df["Mean"],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=f"MLP mean\nmean={stats['Mean']:.3f}, median={stats['Median']:.3f}, std={stats['Std']:.3f}",
        )
        ax.fill_between(layer_df["Layer"], layer_df["Min"], layer_df["Max"], color="#666666", alpha=0.10)
        ax.set_title(report["model_name"], color=EXPERT_COLORS[report["model_name"]])
        ax.set_xlabel("MLP layer index")
        ax.set_xticks(layer_ticks)
        ax.grid(True, alpha=0.2)
        annotate_stats(ax, stats)

    axes[0].set_ylabel(meta["ylabel"])
    axes[-1].legend(loc="upper left", frameon=True)
    fig.suptitle(meta["title"], fontsize=14)
    fig.savefig(figures_dir / f"{meta['filename_prefix']}_expert_subplots.png", bbox_inches="tight")
    plt.close(fig)


def plot_overview(reports: list[dict], metric_name: str, meta: dict, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    for report in reports:
        df = metric_dataframe(report, metric_name)
        layer_df = per_layer_summary(df)
        ax.plot(
            layer_df["Layer"],
            layer_df["Mean"],
            label=report["model_name"],
            color=EXPERT_COLORS[report["model_name"]],
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

    ax.set_title(f"{meta['title']} Across MLP Layers")
    ax.set_xlabel("MLP layer index")
    ax.set_ylabel(meta["ylabel"])
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.2)
    fig.savefig(figures_dir / f"{meta['filename_prefix']}_overview_layers.png", bbox_inches="tight")
    plt.close(fig)


def main(
    input_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/expert_mlp_metrics",
        help="Directory containing per-expert MLP metrics JSON files",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/similarity_metrics_report_preview",
        help="Directory to write figures and tables",
    ),
    font_size: int = typer.Option(10, help="Base plotting font size"),
):
    configure_plot_style(font_size)
    reports = collect_reports(Path(input_dir))
    if not reports:
        raise FileNotFoundError(f"No expert MLP metric reports found under {input_dir}")

    output_root = Path(output_dir)
    write_tables(reports, output_root)
    for metric_name, meta in METRICS.items():
        plot_overview(reports, metric_name, meta, output_root)
        plot_expert_subplots(reports, metric_name, meta, output_root)


if __name__ == "__main__":
    typer.run(main)
