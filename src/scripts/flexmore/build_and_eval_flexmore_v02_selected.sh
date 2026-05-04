#!/usr/bin/env bash
set -euo pipefail

# Build and evaluate the heterogeneous FlexMoRE model chosen from the
# effective-rank v02 table:
#   Math -> r64
#   News -> r16
#   Academic/Pes2o -> r64
#   Reddit -> r32
#   Code -> r16
#   Creative -> r32
#
# The script:
# 1. Merges the ranked 2x7B expert checkpoints once into a 7-expert model.
# 2. Creates a2/a4/a7 variants by reusing the merged files and patching only
#    num_experts_per_tok in config.json.
# 3. Runs the full OLMES task-group suite on each variant.
#
# Notes:
# - Avg is not an eval group; it should be computed later from the saved group
#   results.
# - The first expert checkpoint passed to merge_experts_to_flexolmo.py is used
#   as the source of the shared/public expert weights, so we seed with the
#   uncompressed Code 2x7B checkpoint and then include the ranked Code expert
#   as the first heterogeneous expert branch.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ -z "${BASE_MODEL_ROOT:-}" ]]; then
  if [[ -d "/work/training/FlexMoRE/models" ]]; then
    BASE_MODEL_ROOT="/work/training/FlexMoRE/models"
  else
    BASE_MODEL_ROOT="${ROOT_DIR}/work/training/FlexMoRE/models"
  fi
fi

if [[ -z "${RANKED_MODEL_ROOT:-}" ]]; then
  if [[ -d "/work/training/FlexMoRE/eval_results/models" ]]; then
    RANKED_MODEL_ROOT="/work/training/FlexMoRE/eval_results/models"
  else
    RANKED_MODEL_ROOT="${ROOT_DIR}/work/training/FlexMoRE/eval_results/models"
  fi
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ROOT_DIR}/src/scripts/analysis/results/flexmore_v02_selected_models}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT_DIR}/src/scripts/analysis/results/flexmore_v02_selected_evals}"
MERGE_DEVICE="${MERGE_DEVICE:-cpu}"
MERGE_DTYPE="${MERGE_DTYPE:-bfloat16}"
GPUS="${GPUS:-1}"
LIMIT="${LIMIT:-1000}"

TASK_GROUPS=(
  mc9
  gen5
  mmlu
  mmlu_pro
  agi_eval
  bbh
  math2
  code4
)

MERGED_BASE="${ARTIFACT_ROOT}/FlexMoRE-v02-selected-merged"
A2_DIR="${ARTIFACT_ROOT}/FlexMoRE-v02-selected-a2"
A4_DIR="${ARTIFACT_ROOT}/FlexMoRE-v02-selected-a4"
A7_DIR="${ARTIFACT_ROOT}/FlexMoRE-v02-selected-a7"
MERGE_INPUT_ROOT="${ARTIFACT_ROOT}/merge_input_views"

CODE_BASE="${BASE_MODEL_ROOT}/Flex-code-2x7B-1T"
CODE_R16="${RANKED_MODEL_ROOT}/Flex-code-2x7B-1T-r16"
CREATIVE_R32="${RANKED_MODEL_ROOT}/Flex-creative-2x7B-1T-r32"
MATH_R64="${RANKED_MODEL_ROOT}/Flex-math-2x7B-1T-r64"
NEWS_R16="${RANKED_MODEL_ROOT}/Flex-news-2x7B-1T-r16"
PES2O_R64="${RANKED_MODEL_ROOT}/Flex-pes2o-2x7B-1T-r64"
REDDIT_R32="${RANKED_MODEL_ROOT}/Flex-reddit-2x7B-1T-r32"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

require_all_inputs() {
  require_path "${CODE_BASE}"
  require_path "${CODE_R16}"
  require_path "${CREATIVE_R32}"
  require_path "${MATH_R64}"
  require_path "${NEWS_R16}"
  require_path "${PES2O_R64}"
  require_path "${REDDIT_R32}"
}

patch_num_experts_per_tok() {
  local model_dir="$1"
  local active="$2"
  python3 - "${model_dir}" "${active}" <<'PY'
import json
import os
import sys

model_dir = sys.argv[1]
active = int(sys.argv[2])
config_path = os.path.join(model_dir, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
config["num_experts_per_tok"] = active
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"Patched {config_path} -> num_experts_per_tok={active}")
PY
}

prepare_model_view() {
  local src_dir="$1"
  local dst_dir="$2"
  local default_model_type="$3"
  local default_architecture="$4"

  mkdir -p "${dst_dir}"
  python3 - "${src_dir}" "${dst_dir}" "${default_model_type}" "${default_architecture}" <<'PY'
import json
import os
import shutil
import sys

src = sys.argv[1]
dst = sys.argv[2]
default_model_type = sys.argv[3]
default_architecture = sys.argv[4]

os.makedirs(dst, exist_ok=True)
for name in os.listdir(dst):
    path = os.path.join(dst, name)
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

for name in os.listdir(src):
    src_path = os.path.join(src, name)
    dst_path = os.path.join(dst, name)
    if name == "config.json":
        with open(src_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "model_type" not in config or not config["model_type"]:
            config["model_type"] = default_model_type
        if "architectures" not in config or not config["architectures"]:
            config["architectures"] = [default_architecture]
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")
    else:
        os.symlink(src_path, dst_path)
PY
}

materialize_variant() {
  local src_dir="$1"
  local dst_dir="$2"
  local active="$3"

  mkdir -p "${dst_dir}"
  python3 - "${src_dir}" "${dst_dir}" <<'PY'
import os
import shutil
import sys

src = sys.argv[1]
dst = sys.argv[2]
os.makedirs(dst, exist_ok=True)
for name in os.listdir(dst):
    path = os.path.join(dst, name)
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
for name in os.listdir(src):
    if name == "config.json":
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    else:
        os.symlink(os.path.join(src, name), os.path.join(dst, name))
PY
  patch_num_experts_per_tok "${dst_dir}" "${active}"
}

build_merged_base() {
  mkdir -p "${ARTIFACT_ROOT}"
  mkdir -p "${MERGE_INPUT_ROOT}"

  local code_base_view="${MERGE_INPUT_ROOT}/Flex-code-2x7B-1T"
  local code_r16_view="${MERGE_INPUT_ROOT}/Flex-code-2x7B-1T-r16"
  local creative_r32_view="${MERGE_INPUT_ROOT}/Flex-creative-2x7B-1T-r32"
  local math_r64_view="${MERGE_INPUT_ROOT}/Flex-math-2x7B-1T-r64"
  local news_r16_view="${MERGE_INPUT_ROOT}/Flex-news-2x7B-1T-r16"
  local pes2o_r64_view="${MERGE_INPUT_ROOT}/Flex-pes2o-2x7B-1T-r64"
  local reddit_r32_view="${MERGE_INPUT_ROOT}/Flex-reddit-2x7B-1T-r32"

  prepare_model_view "${CODE_BASE}" "${code_base_view}" "flex_olmo" "FlexOlmoForCausalLM"
  prepare_model_view "${CODE_R16}" "${code_r16_view}" "flex_olmo" "FlexOlmoForCausalLM"
  prepare_model_view "${CREATIVE_R32}" "${creative_r32_view}" "flex_olmo" "FlexOlmoForCausalLM"
  prepare_model_view "${MATH_R64}" "${math_r64_view}" "flex_olmo" "FlexOlmoForCausalLM"
  prepare_model_view "${NEWS_R16}" "${news_r16_view}" "flex_olmo" "FlexOlmoForCausalLM"
  prepare_model_view "${PES2O_R64}" "${pes2o_r64_view}" "flex_olmo" "FlexOlmoForCausalLM"
  prepare_model_view "${REDDIT_R32}" "${reddit_r32_view}" "flex_olmo" "FlexOlmoForCausalLM"

  echo "Merging selected ranked experts into ${MERGED_BASE}"
  PYTHONPATH=. python3 src/scripts/flexmore/merge_experts_to_flexolmo.py \
    "${MERGED_BASE}" \
    "${code_base_view}" \
    "${code_r16_view}" \
    "${creative_r32_view}" \
    "${math_r64_view}" \
    "${news_r16_view}" \
    "${pes2o_r64_view}" \
    "${reddit_r32_view}" \
    --device "${MERGE_DEVICE}" \
    --dtype "${MERGE_DTYPE}"
}

tasks_for_group() {
  local group="$1"
  case "${group}" in
    mc9)
      cat <<'EOF'
arc_easy:mc::olmes
arc_challenge:mc::olmes
boolq:mc::olmes
csqa:mc::olmes
hellaswag:mc::olmes
openbookqa:mc::olmes
piqa:mc::olmes
socialiqa:mc::olmes
winogrande:mc::olmes
EOF
      ;;
    gen5)
      cat <<'EOF'
coqa::olmes
squad::olmes
naturalqs::olmes
triviaqa::olmes
drop::olmes
EOF
      ;;
    mmlu)
      echo "mmlu:mc::olmes"
      ;;
    mmlu_pro)
      echo "mmlu_pro:mc::none"
      ;;
    agi_eval)
      echo "agi_eval_english:1shot::olmes"
      ;;
    bbh)
      echo "bbh:cot-v1::olmes"
      ;;
    math2)
      cat <<'EOF'
gsm8k::olmes
minerva_math_algebra::olmes
minerva_math_counting_and_probability::olmes
minerva_math_geometry::olmes
minerva_math_intermediate_algebra::olmes
minerva_math_number_theory::olmes
minerva_math_prealgebra::olmes
minerva_math_precalculus::olmes
EOF
      ;;
    code4)
      cat <<'EOF'
codex_humaneval:temp0.8
codex_humanevalplus:temp0.8
mbpp::none
mbppplus::none
EOF
      ;;
    *)
      echo "Unsupported task group: ${group}" >&2
      exit 1
      ;;
  esac
}

batch_size_for_task() {
  local task="$1"
  if [[ "${task}" == minerva_math_* || "${task}" == mbpp* || "${task}" == bigcodebench* || "${task}" == sciriff* ]]; then
    echo 1
  else
    echo 4
  fi
}

launch_group() {
  local model_path="$1"
  local variant_name="$2"
  local group="$3"
  local output_dir="${EVAL_ROOT}/${variant_name}/${group}"

  mkdir -p "${output_dir}"
  echo "Running ${group} for ${variant_name}"

  while IFS= read -r task; do
    [[ -z "${task}" ]] && continue
    local batch_size
    batch_size="$(batch_size_for_task "${task}")"
    PYTHONPATH=. python3 src/scripts/eval/launch_eval.py \
      --model "${model_path}" \
      --model-type hf \
      --task "${task}" \
      --limit "${LIMIT}" \
      --output-dir "${output_dir}" \
      --batch-size "${batch_size}" \
      --gpus "${GPUS}"
  done < <(tasks_for_group "${group}")
}

main() {
  require_all_inputs
  mkdir -p "${EVAL_ROOT}"

  build_merged_base

  materialize_variant "${MERGED_BASE}" "${A2_DIR}" 2
  materialize_variant "${MERGED_BASE}" "${A4_DIR}" 4
  materialize_variant "${MERGED_BASE}" "${A7_DIR}" 7

  for group in "${TASK_GROUPS[@]}"; do
    launch_group "${A2_DIR}" "a2" "${group}"
    launch_group "${A4_DIR}" "a4" "${group}"
    launch_group "${A7_DIR}" "a7" "${group}"
  done

  echo "Done."
  echo "Models:"
  echo "  ${A2_DIR}"
  echo "  ${A4_DIR}"
  echo "  ${A7_DIR}"
  echo "Eval outputs:"
  echo "  ${EVAL_ROOT}"
}

main "$@"
