import json
from pathlib import Path

import torch
import typer


def main(
    diagnostics_json: str = typer.Argument(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news/news_singular_value_diagnostics.json",
        help="Diagnostics JSON containing max singular values per matrix.",
    ),
    curves_json: str = typer.Argument(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_value_diagnostics/news/news_singular_value_curves.json",
        help="Curve JSON containing normalized singular values per matrix.",
    ),
    output_path: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/singular_values/news_singular_values.pt",
        help="Output path for the saved singular-value artifact.",
    ),
):
    diagnostics_path = Path(diagnostics_json)
    curves_path = Path(curves_json)
    out_path = Path(output_path)

    with diagnostics_path.open("r") as f:
        diagnostics_payload = json.load(f)
    with curves_path.open("r") as f:
        curves_payload = json.load(f)

    max_sv_by_key = {
        row["model_key"]: float(row["max_singular_value"])
        for row in diagnostics_payload["rows"]
    }

    singular_values_by_key = {}
    for row in curves_payload["rows"]:
        model_key = row["model_key"]
        max_sv = max_sv_by_key[model_key]
        normalized = torch.tensor(row["normalized_singular_values"], dtype=torch.float64)
        singular_values_by_key[model_key] = normalized * max_sv

    payload = {
        "model_name": "News",
        "num_targets": len(singular_values_by_key),
        "source_diagnostics_json": str(diagnostics_path),
        "source_curves_json": str(curves_path),
        "singular_values": singular_values_by_key,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved singular-value artifact to {out_path}")
    print(f"Targets: {len(singular_values_by_key)}")


if __name__ == "__main__":
    typer.run(main)
