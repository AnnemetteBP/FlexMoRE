from collections import defaultdict
import json
import logging
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    FlexMoREConfig,
    FlexMoREForCausalLM,
    FlexOlmoConfig,
    FlexOlmoForCausalLM,
)
import typer

from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

def dtype_from_string(s):
    match s.lower():
        case "float32" | "fp32":
            return torch.float32
        case "float16" | "fp16":
            return torch.float16
        case "bfloat16" | "bf16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Unsupported dtype string: {s}")


def load_config(path: str):
    config_path = f"{path}/config.json"
    config_dict = json.load(open(config_path, "r"))
    model_type = config_dict.get("model_type")
    if model_type == "flexmore":
        return FlexMoREConfig.from_dict(config_dict)
    if model_type in {"flex_olmo", "olmoe"}:
        return FlexOlmoConfig.from_dict(config_dict)
    return AutoConfig.from_pretrained(path)


def load_model(path: str, dtype):
    config_path = f"{path}/config.json"
    config_dict = json.load(open(config_path, "r"))
    model_type = config_dict.get("model_type")
    if model_type == "flexmore":
        return FlexMoREForCausalLM.from_pretrained(path, dtype=dtype)
    if model_type in {"flex_olmo", "olmoe"}:
        return FlexOlmoForCausalLM.from_pretrained(path, dtype=dtype)
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype)

def main(
    target: str = typer.Argument(..., help="Target path to save the merged model"),
    models: list[str] = typer.Argument(..., help="List of expert model paths to merge"),
    device: str = typer.Option("cpu", help="Device to load the models on"),
    dtype: str = typer.Option("bfloat16", help="Data type to load the models with"),
):
    prepare_cli_environment()
    expert_paths = models
    target_path = target
    device = torch.device(device)
    dtype = dtype_from_string(dtype)
    log.info(f"Building model config from {expert_paths[0]} with {len(expert_paths)} experts")
    model_config = load_config(expert_paths[0])
    model_config.num_experts = len(expert_paths)
    model_config.dtype = dtype
    log.info(f"Building the MoE model on {device} with dtype {dtype}")
    with torch.device(device):
        if isinstance(model_config, FlexOlmoConfig):
            model = FlexOlmoForCausalLM(config=model_config)
        elif isinstance(model_config, FlexMoREConfig):
            model = FlexMoREForCausalLM(config=model_config)
        else:
            model = AutoModelForCausalLM.from_config(model_config)
    log.info(f"Model loaded on {device} with dtype {dtype}")
    log.info(model)
    moe_state_dict = model.state_dict()
    filled_keys = defaultdict(int)
    for expert, path in enumerate(expert_paths):
        log.info(f"Loading model from {path} as expert {expert} on {device} with dtype {dtype}")
        with torch.device(device):
            expert_model = load_model(path, dtype=dtype)
        log.info(expert_model)
        assert expert_model.config.num_experts == 2, f"Expert model at {path} has num_experts={expert_model.config.num_experts}, expected 2"
        expert_state_dict = expert_model.state_dict()
        log.info(f"Expert {expert} model loaded")
        for expert_key in list(expert_state_dict.keys()):
            moe_key = expert_key
            if ".experts.0." in expert_key:
                if expert:
                    assert torch.equal(
                        moe_state_dict[moe_key], expert_state_dict[expert_key]
                    ), f"Shared key {moe_key} is different"
                    moe_key = None
            elif ".experts.1." in expert_key:
                if expert:
                    moe_key = expert_key.replace(".experts.1.", f".experts.{expert}.")
                else:
                    moe_key = None
            elif ".mlp.gate." in expert_key:
                # this is a 4096 in_features and num_experts out_features weight
                if expert:
                    assert torch.equal(
                        moe_state_dict[moe_key][:1, :],
                        expert_state_dict[expert_key][:1, :],
                    ), f"Gate weights for expert 0 are different for expert and MoE model: {moe_key}"
                    moe_state_dict[moe_key][expert:expert+1, :] = expert_state_dict[expert_key][1:2, :]
                else:
                    moe_state_dict[moe_key][:1, :] = expert_state_dict[expert_key][:1, :]
                filled_keys[moe_key] += 1
                moe_key = None
            elif expert:
                assert torch.equal(
                    moe_state_dict[moe_key], expert_state_dict[expert_key]
                ), f"Sharedf key {key} is different"
                moe_key = None
            if moe_key:
                moe_state_dict[moe_key] = expert_state_dict[expert_key]
                filled_keys[moe_key] += 1
                log.info(f"Key {expert_key} copied from expert {expert} to MoE model key {moe_key}")
            else:
                log.info(f"Key {expert_key} has not been copied from expert {expert}")
        del expert_state_dict
    assert set(moe_state_dict.keys()) == set(filled_keys.keys()), f"Not all keys have been filled: missing {set(moe_state_dict.keys()) - set(filled_keys.keys())}"
    assert all(count == 1 for key, count in filled_keys.items() if ".mlp.gate." not in key), f"Some non-gate keys have been filled multiple times: { {key: count for key, count in filled_keys.items() if '.mlp.gate.' not in key and count != 1} }"
    assert all(count == len(expert_paths) for key, count in filled_keys.items() if ".mlp.gate." in key), f"Not all gate keys have been filled correctly: { {key: count for key, count in filled_keys.items() if '.mlp.gate.' in key and count != len(expert_paths)} }"
    log.info(f"Saving the merged model to {target_path}")
    model.save_pretrained(target_path, state_dict=moe_state_dict)
    log.info(f"Model saved to {target_path}")

if __name__ == "__main__":
    typer.run(main)
