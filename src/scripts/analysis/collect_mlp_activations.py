import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import typer
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASK_GROUPS = {
    "mc9": [
        "arc_easy:mc::olmes",
        "arc_challenge:mc::olmes",
        "boolq:mc::olmes",
        "csqa:mc::olmes",
        "hellaswag:mc::olmes",
        "openbookqa:mc::olmes",
        "piqa:mc::olmes",
        "socialiqa:mc::olmes",
        "winogrande:mc::olmes",
    ],
    "gen5": [
        "coqa::olmes",
        "squad::olmes",
        "naturalqs::olmes",
        "triviaqa::olmes",
        "drop::olmes",
    ],
    "mmlu": ["mmlu:mc::olmes"],
    "mmlu_pro": ["mmlu_pro:mc::none"],
    "agi_eval": ["agi_eval_english:1shot::olmes"],
    "bbh": ["bbh:cot-v1::olmes"],
    "math2": [
        "gsm8k::olmes",
        "minerva_math_algebra::olmes",
        "minerva_math_counting_and_probability::olmes",
        "minerva_math_geometry::olmes",
        "minerva_math_intermediate_algebra::olmes",
        "minerva_math_number_theory::olmes",
        "minerva_math_prealgebra::olmes",
        "minerva_math_precalculus::olmes",
    ],
    "code4": [
        "codex_humaneval:temp0.8",
        "codex_humanevalplus:temp0.8",
        "mbpp::none",
        "mbppplus::none",
    ],
}

COMMON_TEXT_FIELDS = [
    "text",
    "prompt",
    "question",
    "instruction",
    "input",
    "passage",
    "context",
]


@dataclass
class ChannelStats:
    count_tokens: int
    sum: torch.Tensor
    abs_sum: torch.Tensor
    sq_sum: torch.Tensor
    max_abs: torch.Tensor

    @classmethod
    def create(cls, hidden_size: int) -> "ChannelStats":
        zeros = torch.zeros(hidden_size, dtype=torch.float64)
        return cls(
            count_tokens=0,
            sum=zeros.clone(),
            abs_sum=zeros.clone(),
            sq_sum=zeros.clone(),
            max_abs=zeros.clone(),
        )

    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().to(dtype=torch.float64).reshape(-1, x.shape[-1])
        self.count_tokens += int(flat.shape[0])
        self.sum += flat.sum(dim=0)
        self.abs_sum += flat.abs().sum(dim=0)
        self.sq_sum += flat.square().sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, flat.abs().max(dim=0).values)

    def to_payload(self) -> dict[str, Any]:
        denom = max(1, self.count_tokens)
        mean = self.sum / denom
        mean_abs = self.abs_sum / denom
        rms = torch.sqrt(self.sq_sum / denom)
        variance = torch.clamp((self.sq_sum / denom) - mean.square(), min=0.0)
        std = torch.sqrt(variance)
        return {
            "count_tokens": self.count_tokens,
            "hidden_size": int(mean.numel()),
            "mean": mean.cpu(),
            "mean_abs": mean_abs.cpu(),
            "rms": rms.cpu(),
            "std": std.cpu(),
            "max_abs": self.max_abs.cpu(),
        }


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def stringify_options(options: Any) -> str:
    if options is None:
        return ""
    if isinstance(options, list):
        return "\n".join(f"- {item}" for item in options)
    if isinstance(options, dict):
        return "\n".join(f"- {key}: {value}" for key, value in options.items())
    return str(options)


def record_to_text(record: dict[str, Any], text_fields: list[str] | None) -> str:
    if text_fields:
        pieces = [str(record[field]).strip() for field in text_fields if field in record and str(record[field]).strip()]
        return "\n\n".join(pieces).strip()

    if "question" in record:
        pieces = []
        if record.get("passage"):
            pieces.append(f"Passage:\n{record['passage']}")
        if record.get("context"):
            pieces.append(f"Context:\n{record['context']}")
        pieces.append(f"Question:\n{record['question']}")
        options = stringify_options(record.get("options"))
        if options:
            pieces.append(f"Options:\n{options}")
        return "\n\n".join(piece for piece in pieces if piece).strip()

    pieces = [str(record[field]).strip() for field in COMMON_TEXT_FIELDS if field in record and str(record[field]).strip()]
    if pieces:
        return "\n\n".join(pieces).strip()

    return json.dumps(record, ensure_ascii=False)


def load_text_samples(
    jsonl_path: str | None,
    text_path: str | None,
    text_fields: list[str],
    max_samples: int,
    seed: int,
    shuffle: bool,
) -> tuple[list[str], dict[str, Any]]:
    if not jsonl_path and not text_path:
        raise ValueError("Provide either --jsonl-path or --text-path")

    source_meta: dict[str, Any] = {}
    if jsonl_path:
        path = Path(jsonl_path)
        records = load_jsonl_records(path)
        samples = [record_to_text(record, text_fields or None) for record in records]
        source_meta["source_type"] = "jsonl"
        source_meta["source_path"] = str(path)
        source_meta["num_source_records"] = len(records)
    else:
        path = Path(text_path)
        with path.open("r") as f:
            samples = [line.strip() for line in f if line.strip()]
        source_meta["source_type"] = "text"
        source_meta["source_path"] = str(path)
        source_meta["num_source_records"] = len(samples)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    samples = [sample for sample in samples if sample]
    if max_samples > 0:
        samples = samples[:max_samples]

    source_meta["num_selected_samples"] = len(samples)
    return samples, source_meta


def build_hooks(model: torch.nn.Module, stats_by_module: dict[str, ChannelStats]) -> list[Any]:
    handles = []
    for module_name, module in model.named_modules():
        if not module_name.endswith(("gate_proj", "up_proj", "down_proj")):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue

        def hook_fn(_module, inputs, _output, key=module_name):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.ndim < 2:
                return
            if key not in stats_by_module:
                stats_by_module[key] = ChannelStats.create(x.shape[-1])
            stats_by_module[key].update(x)

        handles.append(module.register_forward_hook(hook_fn))
    return handles


def prepare_olmoe_compat_dir(model_path: str) -> str:
    src = Path(model_path)
    config_path = src / "config.json"
    config = json.load(config_path.open("r"))
    if config.get("model_type") != "flex_olmo":
        return model_path

    compat_root = Path(tempfile.mkdtemp(prefix="flex_olmo_compat_", dir="/tmp"))
    for name in [
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
    ]:
        src_file = src / name
        if src_file.exists():
            os.symlink(src_file, compat_root / name)

    for shard_file in src.glob("*.safetensors"):
        os.symlink(shard_file, compat_root / shard_file.name)

    compat_config = dict(config)
    compat_config["model_type"] = "olmoe"
    compat_config["architectures"] = ["OlmoeForCausalLM"]
    with (compat_root / "config.json").open("w") as f:
        json.dump(compat_config, f)

    return str(compat_root)


def main(
    model_path: str = typer.Argument(..., help="Path to the checkpoint to analyze"),
    output_path: str = typer.Option(
        "/media/am/AM/FlexMoRE/src/scripts/analysis/results/activations/mlp_activations.pt",
        help="Output .pt path for activation statistics",
    ),
    jsonl_path: str | None = typer.Option(None, help="Local JSONL file containing evaluation-style samples"),
    text_path: str | None = typer.Option(None, help="Plain text file with one sample per line"),
    text_field: list[str] = typer.Option([], help="Explicit JSON fields to concatenate into the prompt text"),
    max_samples: int = typer.Option(128, help="Maximum number of samples to process"),
    max_length: int = typer.Option(1024, help="Maximum token length per sample"),
    shuffle: bool = typer.Option(False, help="Shuffle samples before truncating to max_samples"),
    seed: int = typer.Option(0, help="Random seed used when shuffle=true"),
    processes: int = typer.Option(1, help="CPU threads for forward passes"),
    task_group: str | None = typer.Option(None, help="Optional eval-group label for metadata, e.g. mc9 or code4"),
    task_name: list[str] = typer.Option([], help="Optional specific task labels for metadata"),
):
    torch.set_num_threads(processes)

    samples, source_meta = load_text_samples(
        jsonl_path=jsonl_path,
        text_path=text_path,
        text_fields=text_field,
        max_samples=max_samples,
        seed=seed,
        shuffle=shuffle,
    )
    if not samples:
        raise ValueError("No usable samples were found")

    from transformers import OlmoeForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_model_path = prepare_olmoe_compat_dir(model_path)
    model = OlmoeForCausalLM.from_pretrained(resolved_model_path)
    model.eval()

    stats_by_module: dict[str, ChannelStats] = {}
    handles = build_hooks(model, stats_by_module)

    token_counts = []
    try:
        with torch.no_grad():
            for index, sample in enumerate(samples, start=1):
                encoded = tokenizer(
                    sample,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                token_counts.append(int(encoded["input_ids"].shape[1]))
                model(**encoded)
                if index % 10 == 0 or index == len(samples):
                    print(f"Processed {index}/{len(samples)} samples", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    payload = {
        "model_path": model_path,
        "task_group": task_group,
        "task_group_tasks": TASK_GROUPS.get(task_group) if task_group else None,
        "task_names": task_name,
        "source": source_meta,
        "config": {
            "max_samples": max_samples,
            "max_length": max_length,
            "shuffle": shuffle,
            "seed": seed,
            "processes": processes,
            "text_fields": text_field,
        },
        "num_samples_processed": len(samples),
        "token_count_summary": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
        },
        "module_stats": {
            module_name: stats.to_payload()
            for module_name, stats in sorted(stats_by_module.items())
        },
        "sample_preview": samples[: min(5, len(samples))],
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved activation statistics to {out_path}", flush=True)


if __name__ == "__main__":
    typer.run(main)
