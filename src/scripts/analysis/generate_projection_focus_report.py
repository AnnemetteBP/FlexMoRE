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
MODULE_TITLES = {
    "down_proj": "Down Projection",
    "gate_proj": "Gate Projection",
    "up_proj": "Up Projection",
}
METRICS = {
    "public_expert_cosine": {
        "title": "Public-Expert Cosine Similarity",
        "ylabel": "Cosine similarity",
        "filename_prefix": "public_expert_cosine_projection_focus",
    },
    "relative_delta_norm": {
        "title": "Relative Delta Norm",
        "ylabel": "Relative delta norm",
        "filename_prefix": "relative_delta_norm_projection_focus",
    },
}


def slugify_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


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


def overall_stats(df: pd.DataFrame) -> dict:
    return {
        "Mean": float(df["Value"].mean()),
        "Median": float(df["Value"].median()),
        "Std": float(df["Value"].std(ddof=1)),
        "Min": float(df["Value"].min()),
        "Max": float(df["Value"].max()),
    }


def plot_projection_focus(reports: list[dict], metric_name: str, meta: dict, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, constrained_layout=True)
    layer_ticks = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    for ax, module_name in zip(axes, MODULE_ORDER):
        for report in reports:
            df = metric_dataframe(report, metric_name)
            subset = df[df["Module"] == module_name].sort_values("Layer")
            stats = overall_stats(subset)
            ax.plot(
                subset["Layer"],
                subset["Value"],
                color=EXPERT_COLORS[report["model_name"]],
                linewidth=2.0,
                label=(
                    f"{report['model_name']}\n"
                    f"mean={stats['Mean']:.3f}, median={stats['Median']:.3f}, std={stats['Std']:.3f}"
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
    input_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/expert_mlp_metrics",
        help="Directory containing per-expert MLP metrics JSON files",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/projection_focus_report_preview",
        help="Directory to write figures",
    ),
    font_size: int = typer.Option(10, help="Base plotting font size"),
):
    configure_plot_style(font_size)
    reports = collect_reports(Path(input_dir))
    if not reports:
        raise FileNotFoundError(f"No expert MLP metric reports found under {input_dir}")

    output_root = Path(output_dir)
    for metric_name, meta in METRICS.items():
        plot_projection_focus(reports, metric_name, meta, output_root)


if __name__ == "__main__":
    typer.run(main)
