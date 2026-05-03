import json
import math
from pathlib import Path

import pandas as pd
import typer


MODULE_BONUS = {
    "down_proj": 0.00,
    "gate_proj": 0.08,
    "up_proj": 0.04,
}


def normalize_feature(value: float, min_value: float, max_value: float) -> float:
    if math.isclose(min_value, max_value):
        return 0.5
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, float(normalized)))


def select_candidate_pool(candidate_ranks: list[int]) -> list[int]:
    compact_pool = [rank for rank in candidate_ranks if 8 <= rank <= 1024]
    if compact_pool:
        return compact_pool
    positive_candidates = [rank for rank in candidate_ranks if rank > 0]
    if positive_candidates:
        return positive_candidates
    return [1]


def assign_rank_from_score(score: float, candidate_pool: list[int]) -> int:
    if len(candidate_pool) == 1:
        return candidate_pool[0]
    index = int(round(score * (len(candidate_pool) - 1)))
    index = max(0, min(index, len(candidate_pool) - 1))
    return candidate_pool[index]


def compute_v04_for_expert(df: pd.DataFrame, candidate_ranks: list[int]) -> pd.DataFrame:
    candidate_pool = select_candidate_pool(sorted(set(candidate_ranks)))
    cosine_distance = 1.0 - df["public_expert_cosine"]
    inverse_top50 = 1.0 - df["top50_energy_share"]
    max_layer = int(df["layer"].max()) if not df.empty else 1

    cosine_min, cosine_max = float(cosine_distance.min()), float(cosine_distance.max())
    delta_min, delta_max = float(df["relative_delta_norm"].min()), float(df["relative_delta_norm"].max())
    energy_min, energy_max = float(df["energy_rank_095"].min()), float(df["energy_rank_095"].max())
    top50_min, top50_max = float(inverse_top50.min()), float(inverse_top50.max())

    rows = []
    for row in df.sort_values(["layer", "module"]).to_dict(orient="records"):
        cosine_score = normalize_feature(1.0 - row["public_expert_cosine"], cosine_min, cosine_max)
        delta_score = normalize_feature(row["relative_delta_norm"], delta_min, delta_max)
        energy_score = normalize_feature(row["energy_rank_095"], energy_min, energy_max)
        top50_score = normalize_feature(1.0 - row["top50_energy_share"], top50_min, top50_max)
        layer_bonus = 0.05 * (row["layer"] / max_layer if max_layer > 0 else 0.0)
        module_bonus = MODULE_BONUS.get(row["module"], 0.0)

        score = (
            0.30 * cosine_score
            + 0.30 * delta_score
            + 0.20 * energy_score
            + 0.20 * top50_score
            + layer_bonus
            + module_bonus
        )
        score = max(0.0, min(1.0, score))
        resolved_rank = assign_rank_from_score(score, candidate_pool)

        rows.append(
            {
                **row,
                "v04_score": score,
                "v04_rank": resolved_rank,
            }
        )

    return pd.DataFrame(rows)


def summary_payload(df: pd.DataFrame, candidate_pool: list[int]) -> dict:
    return {
        "num_targets": int(len(df)),
        "candidate_pool": candidate_pool,
        "min_v04_rank": int(df["v04_rank"].min()),
        "max_v04_rank": int(df["v04_rank"].max()),
        "mean_v04_rank": float(df["v04_rank"].mean()),
        "median_v04_rank": float(df["v04_rank"].median()),
        "mean_v04_score": float(df["v04_score"].mean()),
        "median_v04_score": float(df["v04_score"].median()),
    }


def main(
    input_path: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/hybrid_rank_analysis/hybrid_rank_analysis_rows.csv",
        help="Hybrid analysis rows CSV",
    ),
    output_dir: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/effective_ranks_v04",
        help="Directory to write per-expert v04 rank results",
    ),
    candidate_ranks: list[int] = typer.Option(
        [8, 16, 32, 64, 128, 256, 512, 1024],
        help="Candidate rank pool for v04 bucket assignment",
    ),
):
    input_df = pd.read_csv(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_pool = select_candidate_pool(candidate_ranks)

    for expert_name, expert_df in input_df.groupby("expert"):
        result_df = compute_v04_for_expert(expert_df.copy(), candidate_pool)
        summary = summary_payload(result_df, candidate_pool)
        rows = result_df.to_dict(orient="records")
        payload = {
            "expert": expert_name,
            "method": "v04_hybrid_matrix_aware",
            "summary": summary,
            "rows": rows,
        }

        slug = expert_name.lower()
        with (output_root / f"{slug}_v04_ranks.json").open("w") as f:
            json.dump(payload, f, indent=2)
        result_df.to_csv(output_root / f"{slug}_v04_ranks.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
