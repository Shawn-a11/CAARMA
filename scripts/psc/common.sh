#!/bin/bash

set -euo pipefail

require_env() {
    local name
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            echo "Required environment variable is not set: $name" >&2
            return 2
        fi
    done
}

activate_caarma_env() {
    if [[ -n "${CAARMA_VENV:-}" ]]; then
        # shellcheck disable=SC1090
        source "$CAARMA_VENV/bin/activate"
    elif [[ -n "${CAARMA_CONDA_ENV:-}" ]]; then
        local conda_base
        conda_base="$(conda info --base)"
        # shellcheck disable=SC1091
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "$CAARMA_CONDA_ENV"
    else
        echo "Set CAARMA_VENV or CAARMA_CONDA_ENV before submitting." >&2
        return 2
    fi
}

prepare_data_runtime() {
    require_env CAARMA_REPO CAARMA_RUN_ROOT
    activate_caarma_env
    mkdir -p "$CAARMA_RUN_ROOT"
    export TOKENIZERS_PARALLELISM=false
    export PYTHONFAULTHANDLER=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    cd "$CAARMA_REPO"
}

prepare_runtime() {
    prepare_data_runtime
    require_env CAARMA_HUBERT_CACHE
    mkdir -p "$CAARMA_HUBERT_CACHE"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
}
