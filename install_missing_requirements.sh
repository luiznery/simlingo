#!/usr/bin/env bash
# Installs requirements into env_sl that aren't covered by setup_env_sl.sh:
# - py123d (editable, from a local dynamo_py123d checkout), needed by
#   collect_embeddings_carla_py123d.py / py123d_carla_dataset.py.
# - a wandb upgrade: dynamo_py123d requires protobuf>=6.31, which conflicts
#   with environment.yaml's pinned wandb==0.16.3 (needs protobuf<5). timm
#   imports wandb unconditionally, so the mismatch breaks loading the vision
#   encoder ("module 'wandb.proto.wandb_internal_pb2' has no attribute
#   'Result'") even though wandb itself is never used by the collection
#   scripts. Newer wandb supports protobuf>=6, so upgrading it is enough --
#   no need to downgrade protobuf and risk breaking py123d.
# - a shapely upgrade: dynamo_py123d requires shapely>=2.0.0 (uses
#   shapely.creation, added in 2.0), which conflicts with environment.yaml's
#   pinned shapely==1.7.1. That old pin is only used by CARLA scenario-runner/
#   agent code (team_code/transfuser_utils.py, scenario_runner/*) for plain
#   Polygon construction and affinity.rotate/translate -- both unchanged in
#   Shapely 2.x -- so upgrading is safe for the collection scripts' import
#   path (simlingo_training/dataloader/dataset_base.py pulls in transfuser_utils
#   regardless of data source).
# - streamlit, for visualize_embeddings.py (not needed by the collection
#   scripts themselves, only the inspection tool).
#
# Run setup_env_sl.sh first to create env_sl itself.
# Override the checkout location with: DYNAMO_PY123D_DIR=/path ./install_missing_requirements.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_NAME="${VENV_NAME:-env_sl}"
DYNAMO_PY123D_DIR="${DYNAMO_PY123D_DIR:-/home/lnery/workspace/dynamo_py123d}"

if [ ! -f "$VENV_NAME/bin/activate" ]; then
    echo "ERROR: venv '$VENV_NAME' not found. Run ./setup_env_sl.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_NAME/bin/activate"

if python -c "import py123d" >/dev/null 2>&1; then
    echo ">>> py123d already installed, skipping"
else
    echo ">>> Installing py123d editable from $DYNAMO_PY123D_DIR"
    if [ ! -d "$DYNAMO_PY123D_DIR" ]; then
        echo "ERROR: $DYNAMO_PY123D_DIR not found. Set DYNAMO_PY123D_DIR to your dynamo_py123d checkout." >&2
        exit 1
    fi
    pip install -e "$DYNAMO_PY123D_DIR"
fi

if python -c "import wandb" >/dev/null 2>&1; then
    echo ">>> wandb import OK, skipping upgrade"
else
    echo ">>> wandb broken against py123d's protobuf version, upgrading"
    pip install --upgrade wandb
fi

if python -c "import shapely.creation" >/dev/null 2>&1; then
    echo ">>> shapely already >=2.0 (py123d-compatible), skipping upgrade"
else
    echo ">>> shapely too old for py123d (needs >=2.0), upgrading"
    pip install --upgrade "shapely>=2.0.0"
fi

if python -c "import streamlit" >/dev/null 2>&1; then
    echo ">>> streamlit already installed, skipping"
else
    echo ">>> Installing streamlit (for visualize_embeddings.py)"
    pip install streamlit
fi

echo
echo "Done. env_sl now has everything run_small_carla_collection.sh needs."
