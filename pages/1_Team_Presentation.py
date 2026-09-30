"""
Streamlit page: renders a set of scenes -- either a random sample or a
specific list of scene numbers you pick -- matched by (log_name, scene_uuid)
across a SimLingo/py123d dataset dir and a Qwen3-VL dataset dir, as single,
self-contained PNGs (front camera + BEV trajectory plots on top, both
models' prompt + text output below), for sharing outside the app (e.g. a
team presentation).

Reuses visualize_embeddings.py's own data-loading/rendering functions
directly (build_scene_key_index, render_scene_grid_png), so the images are
identical to what the main "Scene Viewer" page shows -- this page is just a
batch-export UI around the same rendering path, not a re-derivation of it.

Auto-discovered by Streamlit's multipage app convention: this file lives in
pages/ next to the main script, so it shows up as a second page in the
sidebar when running `streamlit run visualize_embeddings.py`.
"""

import os
import random
import re
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st

from visualize_embeddings import build_scene_key_index, render_scene_grid_png

st.set_page_config(page_title="Team Presentation Export", layout="wide")
st.title("Team presentation export")
st.caption(
    "Samples scenes present in BOTH dataset dirs below and renders each as one PNG: front camera + BEV "
    "trajectory plots on top, SimLingo's and Qwen3-VL's prompt + text output below."
)

col1, col2 = st.columns(2)
root = col1.text_input("SimLingo embeddings_dataset root", value="~/mack/datasets_mack/ScoringLingo/embeddings")
qwen_root = col2.text_input("Qwen3-VL embeddings_dataset root", value="embeddings_dataset_qwen_vl_full")

selection_mode = st.radio("Scene selection", ["Random sample", "Specific scene numbers"], horizontal=True)

if selection_mode == "Random sample":
    col3, col4 = st.columns(2)
    num_scenes = col3.number_input("Number of random scenes", min_value=1, max_value=200, value=10, step=1)
    seed = col4.number_input("Random seed (for reproducible sampling)", value=42, step=1)
    scene_numbers_text = ""
else:
    scene_numbers_text = st.text_input(
        "Scene numbers (comma/space-separated, e.g. 0, 15, 231)",
        value="",
        help="Matches manifest scene ids, zero-padded to 4 digits (e.g. 15 -> scene 0015).",
    )

root = os.path.expanduser(root)
qwen_root = os.path.expanduser(qwen_root)

if st.button("Generate samples", type="primary"):
    if not Path(root, "manifest.json").exists():
        st.error(f"No manifest.json found under {root!r}")
    elif not Path(qwen_root, "manifest.json").exists():
        st.error(f"No manifest.json found under {qwen_root!r}")
    else:
        with st.spinner("Indexing scenes for SimLingo <-> Qwen3-VL matching (reads every scene's metadata.json once)..."):
            primary_key_index = build_scene_key_index(root)
            qwen_id_by_key = {key: sid for sid, key in build_scene_key_index(qwen_root).items()}

        matched_ids = [sid for sid, key in primary_key_index.items() if key in qwen_id_by_key]
        st.write(f"{len(matched_ids)} / {len(primary_key_index)} SimLingo scenes have a matching Qwen3-VL description.")

        sampled = []
        if selection_mode == "Random sample":
            if not matched_ids:
                st.warning("No matched scenes found -- nothing to sample.")
            else:
                n = min(int(num_scenes), len(matched_ids))
                if n < num_scenes:
                    st.info(f"Only {n} matched scenes available -- sampling all of them instead of {int(num_scenes)}.")
                sampled = random.Random(int(seed)).sample(matched_ids, n)
        else:
            tokens = [t for t in re.split(r"[,\s]+", scene_numbers_text.strip()) if t]
            requested_ids, invalid_tokens = [], []
            for t in tokens:
                try:
                    requested_ids.append(f"{int(t):04d}")
                except ValueError:
                    invalid_tokens.append(t)
            requested_ids = list(dict.fromkeys(requested_ids))  # dedupe, keep order

            if invalid_tokens:
                st.warning(f"Ignoring non-numeric entries: {', '.join(invalid_tokens)}")

            missing = [sid for sid in requested_ids if sid not in primary_key_index]
            if missing:
                st.warning(f"Not found in {root!r}: {', '.join(missing)}")

            sampled = [sid for sid in requested_ids if sid in primary_key_index]
            without_qwen = [sid for sid in sampled if qwen_id_by_key.get(primary_key_index.get(sid)) is None]
            if without_qwen:
                st.info(f"No matching Qwen3-VL description (will render without that panel): {', '.join(without_qwen)}")

            if not sampled:
                st.warning("No valid scene numbers to render -- enter e.g. \"0, 15, 231\" above.")

        if sampled:
            n = len(sampled)
            data_root = os.environ.get("PY123D_DATA_ROOT")
            pngs = {}
            progress = st.progress(0.0, text="Rendering scenes...")
            for i, scene_id in enumerate(sampled):
                pngs[scene_id] = render_scene_grid_png(root, qwen_root, scene_id, primary_key_index, qwen_id_by_key, data_root)
                progress.progress((i + 1) / n, text=f"Rendered {i + 1}/{n} scenes")
            progress.empty()

            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for scene_id, png_bytes in pngs.items():
                    zf.writestr(f"{scene_id}.png", png_bytes)
            st.download_button(
                f"Download all {n} as ZIP", zip_buf.getvalue(),
                file_name="simlingo_vs_qwen_samples.zip", mime="application/zip",
            )

            for scene_id, png_bytes in pngs.items():
                st.image(png_bytes, caption=scene_id, width="stretch")
                st.download_button(f"Download {scene_id}.png", png_bytes, file_name=f"{scene_id}.png", key=f"dl_{scene_id}")
