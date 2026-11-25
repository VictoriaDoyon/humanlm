#!/bin/bash

set -Eeuo pipefail
export VLLM_USE_V2_MODEL_RUNNER=1

model="RedHatAI/gemma-4-26B-A4B-it-NVFP4" ##"google/gemma-4-31B-it"
image="vllm/vllm-openai:v0.34.0"
enable_auto_tool_choice=true
tool_call_parser="gemma4"
reasoning_parser="gemma4"
chat_template="examples/tool_chat_template_gemma4.jinja"
gpu_memory_utilizaton="0.90"
gpus="all"
fail_on_environ_validation=true
kv_cache_dtype="fp8"
port="8000"
cache="~/.cache/huggingface"
gpu_memory_utilization=""
download_dir=""
tensor_parallel_size="1"
data_parallel_size=""
reasoning_parser="gemma4"
use_rust=false

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_value() {
    (($# >= 2)) && [[ -n ${2:-} ]] || die "$1 requires a value."
}

while (($# > 0)); do 
    case "$1" in 
        --model)
            require_value"$@"
            model=$2
            shift 2
            ;;
        --image)
            require_value"$@"
            image=$2
            shift 2
            ;;
        --enable-auto-tool-choice)
            require_value"$@"
            enable_auto_tool_choice=true
            shift 2
            ;;
        --tool-call-parser)
            require_value"$@"
            tool_call_parser=$2
            shift 2
            ;;
        --reasoning-parser)
            require_value"$@"
            reasoning_parser=$2
            shift 2
            ;;
        --chat-template)
            require_value"$@"
            chat_template=$2
            shift 2
            ;;
        --gpu-memory-utilizaton)
            require_value"$@"
            gpu_memory_utilizaton=$2
            shift 2
            ;;
        --gpus)
            require_value"$@"
            gpus=$2
            shift 2
            ;;
        --fail-on-environ-validation)
            require_value"$@"
            fail_on_environ_validation=true
            shift 2
            ;;
        --kv-cache-dtype)
            require_value"$@"
            kv_cache_dtype=$2
            shift 2
            ;;
        --port)
            require_value"$@"
            port=$2
            shift 2
            ;;
        --cache)
            require_value"$@"
            cache=$2
            shift 2
            ;;
        --gpu-memory-utilization)
            require_value"$@"
            gpu_memory_utilization=$2
            shift 2
            ;;
        --download-dir)
            require_value"$@"
            download_dir=$2
            shift 2
            ;;
        --tensor-parallel-size)
            require_value"$@"
            tensor_parallel_size=$2
            shift 2
            ;;
        --data-parallel-size)
            require_value"$@"
            data_parallel_size=$2
            shift 2
            ;;
        --reasoning-parser)
            require_value"$@"
            reasoning_parser=$2
            shift 2
            ;;
    esac
done

args=(
    --gpus "$gpus"
    --ipc=host
    -p "$port":8000
    -v "$cache":/root/.cache/huggingface
    "$image" "$model"
)

optional_args=(
    tool_call_parser
    reasoning_parser
    chat_template
    gpu_memory_utilizaton
    kv_cache_dtype
    gpu_memory_utilization
    download_dir
    tensor_parallel_size
    data_parallel_size
    reasoning_parser
)

if [[ $enable_auto_tool_choice == true ]]; then
    args+=(--enable-auto-tool-choice)
fi

if [[ $fail_on_environ_validation == true ]]; then
    args+=(--fail-on-environ-validation)
fi

if [[ $use_rust == true ]]; then
    args+=(-e VLLM_USE_RUST_FRONTEND=1)
fi

for opt_arg in "${optional_args[@]}"; do
    if [[ $opt_arg != "" ]]; then
        hyphenated_name="${opt_arg//_/-}"
        args+=("--$hyphenated_name $opt_arg")
    fi
done

docker run "${args[@]}"