#!/usr/bin/env bash
# Builds env_sl_qwenvl for collect_qwen_vl_embeddings.py by copying env_sl's
# already-installed site-packages into a fresh venv, then upgrading just the
# packages Qwen3-VL needs. env_sl itself is never modified.
#
# Why not upgrade env_sl in place: env_sl's transformers==4.46.3 is exactly
# pinned to match SimLingo's own InternVL2 model + a flash-attn build compiled
# against torch==2.2.0 (see setup_env_sl.sh / install_missing_requirements.sh).
# Qwen3-VL needs transformers>=4.57.0 (older versions raise
# "ValueError: model type qwen3_vl not recognized" -- see
# https://huggingface.co/docs/transformers/model_doc/qwen3_vl), so upgrading
# transformers in env_sl would risk breaking SimLingo's own model loading.
#
# Why copy instead of reinstalling from scratch: rsync-ing the ~200 already-
# resolved/compiled packages (including flash-attn, which takes a while to
# compile) is far faster than re-running pip against requirements-env_sl.txt.
# This only works because pip's installed-package metadata (RECORD files)
# uses paths relative to site-packages, not absolute paths back into env_sl/
# (confirmed by inspecting env_sl's own transformers-4.46.3.dist-info/RECORD)
# -- so the copied packages are fully self-contained in the new venv and
# `pip install --upgrade` inside it only ever touches env_sl_qwenvl/.
#
# Usage: ./setup_env_sl_qwenvl.sh
# Override with: SOURCE_VENV=env_sl NEW_VENV=env_sl_qwenvl ./setup_env_sl_qwenvl.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SOURCE_VENV="${SOURCE_VENV:-env_sl}"
NEW_VENV="${NEW_VENV:-env_sl_qwenvl}"

if [ ! -f "$SOURCE_VENV/bin/activate" ]; then
    echo "ERROR: source venv '$SOURCE_VENV' not found. Run ./setup_env_sl.sh first." >&2
    exit 1
fi

# Copying compiled binary extensions (torch, numpy, ...) directly, so the new
# venv must use the exact same Python minor version env_sl was built with --
# a different minor version has an incompatible extension-module ABI.
SOURCE_PY_VERSION=$(grep '^version' "$SOURCE_VENV/pyvenv.cfg" | cut -d= -f2 | tr -d ' ')
SOURCE_PY_MAJOR_MINOR=$(echo "$SOURCE_PY_VERSION" | cut -d. -f1,2)
PYTHON_BIN="python${SOURCE_PY_MAJOR_MINOR}"

if [ -f "$NEW_VENV/bin/activate" ]; then
    echo ">>> $NEW_VENV already exists, skipping venv creation + site-packages copy"
else
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "ERROR: $PYTHON_BIN not found on PATH (source venv is Python $SOURCE_PY_VERSION)." >&2
        echo "Install it, or pass a compatible interpreter via PYTHON_BIN and adapt this script." >&2
        exit 1
    fi

    echo ">>> Creating venv '$NEW_VENV' with $PYTHON_BIN (matching $SOURCE_VENV's Python $SOURCE_PY_VERSION)"
    "$PYTHON_BIN" -m venv "$NEW_VENV"

    SOURCE_SITE_PACKAGES=$(echo "$SOURCE_VENV"/lib/python*/site-packages)
    NEW_SITE_PACKAGES=$(echo "$NEW_VENV"/lib/python*/site-packages)

    echo ">>> Copying installed packages: $SOURCE_SITE_PACKAGES -> $NEW_SITE_PACKAGES"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$SOURCE_SITE_PACKAGES/" "$NEW_SITE_PACKAGES/"
    else
        cp -a "$SOURCE_SITE_PACKAGES/." "$NEW_SITE_PACKAGES/"
    fi
fi

# shellcheck disable=SC1091
source "$NEW_VENV/bin/activate"

echo ">>> Upgrading transformers/tokenizers/accelerate + installing qwen-vl-utils"
echo "    (deliberately unpinned -- unlike env_sl's exact pins, this venv is meant"
echo "    to track whatever transformers release actually supports Qwen3-VL)"
pip install --upgrade "transformers>=4.57.0" tokenizers accelerate qwen-vl-utils

echo ">>> Verifying Qwen3-VL is importable"
python -c "
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print('OK: Qwen3VLForConditionalGeneration + AutoProcessor importable')
" || {
    echo "ERROR: Qwen3-VL not importable with the installed transformers version." >&2
    echo "Try building from source: source $NEW_VENV/bin/activate &&" >&2
    echo "  pip install -U 'git+https://github.com/huggingface/transformers'" >&2
    exit 1
}

echo ">>> Verifying torch/CUDA still work after the site-packages copy"
python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
"

echo ">>> Verifying py123d is still importable (copied editable install)"
python -c "import py123d" >/dev/null 2>&1 && echo "OK: py123d importable" || {
    echo "WARNING: py123d not importable in $NEW_VENV." >&2
    echo "Its editable install points at an external checkout path, which should" >&2
    echo "still resolve after the copy -- if not, re-run:" >&2
    echo "  source $NEW_VENV/bin/activate && pip install -e /path/to/dynamo_py123d" >&2
}

echo
echo "Done. $SOURCE_VENV was not modified."
echo "Activate with: source $NEW_VENV/bin/activate"
echo "Then run: python collect_qwen_vl_embeddings.py --config collect_qwen_vl_config.yaml"
