#!/usr/bin/env bash
# Runs a small embeddings collection over py123d/CARLA scenes, using the
# env_sl venv. Two stages:
#   1. dry_run_test.py  -- pure-logic smoke test, no GPU/model/data needed
#   2. collect_embeddings_carla_py123d.py --max-scenes N -- the real thing
#
# Prerequisites: ./setup_env_sl.sh then ./install_missing_requirements.sh
#
# Override any of these before running, e.g.:
#   MAX_SCENES=20 ./run_small_carla_collection.sh
#   DTYPE=bf16 ./run_small_carla_collection.sh
# Or pass the scene count directly as the first argument, e.g.:
#   ./run_small_carla_collection.sh 200
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_NAME="${VENV_NAME:-env_sl}"
PY123D_DATA_ROOT="${PY123D_DATA_ROOT:-/home/lnery/daf/datasets_daf/py123d_datasets}"
CHECKPOINT="${CHECKPOINT:-outputs/2026_06_08_17_14_55_simlingo_seed1/checkpoints/last.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-embeddings_dataset_carla_py123d_smoketest}"
MAX_SCENES="${1:-${MAX_SCENES:-8}}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DTYPE="${DTYPE:-fp32}"

if [ ! -f "$VENV_NAME/bin/activate" ]; then
    echo "ERROR: venv '$VENV_NAME' not found. Run ./setup_env_sl.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_NAME/bin/activate"

if ! python -c "import py123d" >/dev/null 2>&1; then
    echo "ERROR: py123d not installed in $VENV_NAME. Run ./install_missing_requirements.sh first." >&2
    exit 1
fi

if [ ! -d "$PY123D_DATA_ROOT/logs/carla_train" ]; then
    echo "ERROR: no carla_train logs found under $PY123D_DATA_ROOT/logs." >&2
    echo "Set PY123D_DATA_ROOT to a py123d data root containing converted CARLA logs." >&2
    exit 1
fi
export PY123D_DATA_ROOT

if [ ! -e "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT. Set CHECKPOINT to a valid path." >&2
    echo "(can be a single .ckpt file or a DeepSpeed checkpoint directory)" >&2
    exit 1
fi

echo ">>> Running dry_run_test.py (pure-logic smoke test)"
python dry_run_test.py

echo ">>> Collecting $MAX_SCENES scenes from py123d/CARLA -> $OUTPUT_DIR (dtype=$DTYPE)"
python collect_embeddings_carla_py123d.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --max-scenes "$MAX_SCENES" \
    --batch-size "$BATCH_SIZE" \
    --dtype "$DTYPE"

echo
echo "Done. Inspect results with:"
echo "  source $VENV_NAME/bin/activate && python embedding_dataset.py $OUTPUT_DIR"
