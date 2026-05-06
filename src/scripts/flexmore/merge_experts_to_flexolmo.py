import argparse
from collections import defaultdict
import logging
import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from olmo_core.utils import prepare_cli_environment
except ImportError:
    def prepare_cli_environment(*args, **kwargs):
        return None

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
    return AutoConfig.from_pretrained(path, trust_remote_code=True)


def load_model(path: str, dtype):
    # Different internal transformers forks accept either `torch_dtype` or `dtype`.
    try:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, trust_remote_code=True
        )


def raise_shared_key_mismatch(
    expert_index: int,
    model_path: str,
    moe_key: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
):
    same_shape = tuple(expected.shape) == tuple(actual.shape)
    same_dtype = expected.dtype == actual.dtype
    max_abs_diff = None
    mean_abs_diff = None
    if same_shape:
        diff = (expected.to(dtype=torch.float64) - actual.to(dtype=torch.float64)).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
        mean_abs_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0

    raise AssertionError(
        "Shared key mismatch detected during expert merge: "
        f"expert={expert_index}, "
        f"path={model_path}, "
        f"moe_key={moe_key}, "
        f"expected_shape={tuple(expected.shape)}, "
        f"actual_shape={tuple(actual.shape)}, "
        f"expected_dtype={expected.dtype}, "
        f"actual_dtype={actual.dtype}, "
        f"same_shape={same_shape}, "
        f"same_dtype={same_dtype}, "
        f"max_abs_diff={max_abs_diff}, "
        f"mean_abs_diff={mean_abs_diff}"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Merge ranked 2x7B experts into one FlexOlmo-style MoE model")
    parser.add_argument("target", help="Target path to save the merged model")
    parser.add_argument("models", nargs="+", help="List of expert model paths to merge")
    parser.add_argument("--device", default="cpu", help="Device to load the models on")
    parser.add_argument("--dtype", default="bfloat16", help="Data type to load the models with")
    return parser.parse_args()


def main():
    prepare_cli_environment()
    args = parse_args()
    expert_paths = args.models
    target_path = args.target
    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    log.info(f"Building model config from {expert_paths[0]} with {len(expert_paths)} experts")
    model_config = load_config(expert_paths[0])
    model_config.num_experts = len(expert_paths)
    if hasattr(model_config, "dtype"):
        model_config.dtype = str(dtype).replace("torch.", "")
    setattr(model_config, "torch_dtype", dtype)
    log.info(f"Building the MoE model on {device} with dtype {dtype}")
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
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
                    if not torch.equal(moe_state_dict[moe_key], expert_state_dict[expert_key]):
                        raise_shared_key_mismatch(
                            expert,
                            path,
                            moe_key,
                            moe_state_dict[moe_key],
                            expert_state_dict[expert_key],
                        )
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
                if not torch.equal(moe_state_dict[moe_key], expert_state_dict[expert_key]):
                    raise_shared_key_mismatch(
                        expert,
                        path,
                        moe_key,
                        moe_state_dict[moe_key],
                        expert_state_dict[expert_key],
                    )
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
    main()
