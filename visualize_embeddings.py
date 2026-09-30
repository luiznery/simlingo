"""
Streamlit viewer for embeddings_dataset/ directories produced by
collect_embeddings.py / collect_embeddings_carla_py123d.py.

The front camera frame is never saved at collection time -- it's loaded here
on demand instead: read straight off disk via metadata's `image_path` for
SimLingo-database scenes (a real jpg that already exists there), or
re-queried live from py123d via the scene's `log_name`/`scene_uuid` for
py123d/CARLA scenes (see load_front_camera_image() /
py123d_carla_dataset.find_scene_by_ref+get_front_camera_image).

For each collected scene, shows the front camera frame side by side with a
top-down (bird's-eye, ego-frame) plot of both the model's predicted
trajectory -- `route` (20 pts, the route/path head) and `speed_wps` (10 pts,
the near-term speed-waypoints head) -- and the real expert-driven trajectory
the training/eval loss is computed against (PDM-Lite for SimLingo-database
scenes, LEAD 123D's own rule-based privileged expert for py123d/CARLA scenes;
see README.md's Setup section). All four are saved into metadata.json
(`predicted_route`/`predicted_speed_wps`/`expert_route`/`expert_waypoints`)
by EmbeddingCollector._process_batch (see collect_embeddings.py).

Ego-frame convention: x = forward, y = lateral, with y positive to the
*right* (this codebase inherits CARLA's left-handed world frame, not ISO
8855's y-left convention -- confirmed empirically: the same sign of y that
bends the projected camera overlay left also bends left here). Plotted with
x on the vertical axis (forward = up) and y increasing to the right on the
horizontal axis, so left/right in this plot match left/right in the camera
image next to it.

Optionally pass --qwen-root pointing at a Qwen3-VL embeddings_dataset dir
(see collect_qwen_vl_embeddings.py) to additionally show that model's scene
description alongside --root's own text_output, and to filter the scene
selector down to only scenes that have one. Scenes are matched across the
two dataset dirs by (log_name, scene_uuid) identity -- the same pairing
collect_qwen_vl_embeddings.py's --reference-dataset flag uses to make a
Qwen3-VL run cover the exact same scenes as an existing SimLingo/py123d run.

Run with:
    streamlit run visualize_embeddings.py -- \\
        --root embeddings_dataset_carla_py123d_smoketest \\
        --qwen-root embeddings_dataset_qwen_vl_carla_py123d_smoketest
"""

import argparse
import io
import json
import os
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from simlingo_training.utils.projection import get_camera_intrinsics, project_points

# Same "TF++ default camera" FOV assumed by every dataset source's own debug
# viz (simlingo_training/dataloader/dataset_base.py's visualise_cameras) and
# by datamodule.py's camera_intrinsics fed to the model -- there's no
# per-scene camera calibration saved, so this is the same fixed assumption
# the rest of the codebase already makes, not something specific to this tool.
CAMERA_FOV = 110

# Radius (meters) queried around the ego position for the BEV map -- see
# load_bev_map(). py123d/CARLA scenes only: no map data is saved at
# collection time, so this re-queries py123d live via the scene's
# log_name/scene_uuid (see py123d_carla_dataset.find_scene_by_ref).
MAP_QUERY_RADIUS = 60.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="embeddings_dataset_carla_py123d_smoketest")
    parser.add_argument(
        "--qwen-root", type=str, default="",
        help="Optional Qwen3-VL embeddings_dataset dir to compare against --root, matched by (log_name, scene_uuid)",
    )
    # streamlit passes its own args through; ignore anything unrecognized
    args, _ = parser.parse_known_args()
    return args


@st.cache_data
def load_manifest(root: str):
    with open(Path(root) / "manifest.json") as f:
        return json.load(f)["scenes"]


@st.cache_data
def load_scene_metadata(root: str, scene_id: str):
    scene_dir = Path(root) / "scenes" / scene_id
    with open(scene_dir / "metadata.json") as f:
        return json.load(f)


@st.cache_data
def build_scene_key_index(root: str) -> dict:
    """{scene_id: (log_name, scene_uuid)} for every scene in root -- used to
    match scenes across two different embeddings_dataset dirs (e.g. a
    SimLingo/py123d run and a Qwen3-VL run over the same scenes, see
    collect_qwen_vl_embeddings.py's --reference-dataset). manifest.json
    entries alone don't carry scene_uuid (see that script's
    load_reference_scene_keys, which reads each scene's own metadata.json
    for the same reason), so this does the same thing here, once per root,
    cached for the session."""
    index = {}
    for s in load_manifest(root):
        meta = load_scene_metadata(root, s["id"])
        log_name, scene_uuid = meta.get("log_name"), meta.get("scene_uuid")
        if log_name and scene_uuid:
            index[s["id"]] = (log_name, scene_uuid)
    return index


@st.cache_data(show_spinner=False)
def load_front_camera_image(image_path, source, log_name, scene_uuid, data_root):
    """Loads the scene's front camera frame -- not saved anywhere, so this is
    the only way to see it. SimLingo-database scenes have a real jpg on disk
    (`image_path`, from build_metadata's derive_image_path in
    collect_embeddings.py) -- read directly. py123d/CARLA scenes have no
    on-disk image at all; re-queries py123d live via the scene's
    log_name/scene_uuid instead (same re-fetch approach as load_bev_map,
    done independently here -- Scene objects aren't cached/shared between the
    two since they may not be safely picklable for st.cache_data's storage).
    Returns (PIL.Image_or_None, error_message_or_None)."""
    if image_path:
        try:
            return Image.open(image_path), None
        except FileNotFoundError:
            return None, f"image_path {image_path!r} not found on disk (dataset moved or not mounted here?)."

    if source == "py123d_carla" and log_name and scene_uuid:
        try:
            from py123d_carla_dataset import find_scene_by_ref, get_front_camera_image
        except ImportError as exc:
            return None, f"py123d not importable in this environment ({exc}). Run install_missing_requirements.sh."

        try:
            scene = find_scene_by_ref(data_root, log_name, scene_uuid)
        except AssertionError as exc:
            return None, f"{exc} (set $PY123D_DATA_ROOT before launching streamlit)"
        except Exception as exc:  # noqa: BLE001 - surface any py123d-side failure as a warning, not a crash
            return None, f"Failed to query py123d: {exc!r}"

        if scene is None:
            return None, f"No matching py123d scene found for log_name={log_name!r}, scene_uuid={scene_uuid!r}."

        return Image.fromarray(get_front_camera_image(scene)), None

    return None, "No image_path and no log_name/scene_uuid in metadata -- can't load an image for this scene."


@st.cache_data(show_spinner=False)
def load_bev_map(data_root, log_name: str, scene_uuid: str, radius: float):
    """Re-queries py123d's real HD map for one collected scene (see
    py123d_carla_dataset.find_scene_by_ref/get_bev_map_ego_frame) and caches
    the result -- re-fetching the scene by scanning its log takes several
    seconds, so this only pays that cost once per scene per session.
    Returns (bev_map_dict_or_None, error_message_or_None)."""
    try:
        from py123d_carla_dataset import find_scene_by_ref, get_bev_map_ego_frame
    except ImportError as exc:
        return None, f"py123d not importable in this environment ({exc}). Run install_missing_requirements.sh."

    try:
        scene = find_scene_by_ref(data_root, log_name, scene_uuid)
    except AssertionError as exc:
        return None, f"{exc} (set $PY123D_DATA_ROOT before launching streamlit)"
    except Exception as exc:  # noqa: BLE001 - surface any py123d-side failure as a warning, not a crash
        return None, f"Failed to query py123d: {exc!r}"

    if scene is None:
        return None, f"No matching py123d scene found for log_name={log_name!r}, scene_uuid={scene_uuid!r}."

    bev_map = get_bev_map_ego_frame(scene, radius=radius)
    if bev_map is None:
        return None, "This log has no map available via py123d's MapAPI."
    return bev_map, None


@st.cache_data(show_spinner=False)
def load_target_points(data_root, log_name: str, scene_uuid: str, target_point_source: str = "trajectory"):
    """Re-derives the near/far navigation-goal points fed to the model by
    re-querying the scene -- not saved at collection time (metadata.json
    only records which method produced them, target_point_source, not the
    actual coordinates -- see build_py123d_metadata), so there's nothing to
    read back directly; same re-fetch-per-feature tradeoff as
    load_bev_map/load_front_camera_image.

    target_point_source picks which method to re-derive with, matching
    whichever one actually produced this scene's target points at collection
    time (Py123dCarlaDataset(target_point_mode=...), see
    py123d_carla_dataset.py):
      - "route"/"drivor": the real map-based reconstructions
        (py123d_route_target_points.py's sample_route_target_points_indexed
        / _drivor respectively -- near/far ~1m apart vs. tens of meters
        apart, see that module's docstring) -- need a wider future window
        than the collection pipeline's own scenes have, so this re-fetches
        one via find_scene_with_route_lookahead rather than
        find_scene_by_ref's default window.
      - "trajectory" (default, and any older dataset collected before
        target_point_source existed): the ego-future-trajectory
        approximation (compute_target_points).

    Returns (target_points_or_None, error_message_or_None)."""
    if target_point_source in ("route", "drivor"):
        try:
            from py123d_carla_dataset import world_to_ego_frame
            from py123d_route_target_points import (
                find_scene_with_route_lookahead,
                sample_route_target_points_drivor,
                sample_route_target_points_indexed,
            )
        except ImportError as exc:
            return None, f"py123d_route_target_points not importable in this environment ({exc})."

        sampler = sample_route_target_points_drivor if target_point_source == "drivor" else sample_route_target_points_indexed

        try:
            scene, lookahead = find_scene_with_route_lookahead(data_root, log_name, scene_uuid)
        except AssertionError as exc:
            return None, f"{exc} (set $PY123D_DATA_ROOT before launching streamlit)"
        except Exception as exc:  # noqa: BLE001 - surface any py123d-side failure as a warning, not a crash
            return None, f"Failed to query py123d for route target points: {exc!r}"

        if scene is None:
            return None, f"No matching py123d scene found for log_name={log_name!r}, scene_uuid={scene_uuid!r} (route lookahead)."

        xy, valid = sampler(scene, lookahead_time_s=lookahead)
        if valid == 0.0:
            return None, f"No route could be found for this scene -- {target_point_source}-based target points unavailable here."

        ego0 = scene.get_ego_state_se3_at_iteration(0)
        origin_xy = np.array([ego0.rear_axle_se3.x, ego0.rear_axle_se3.y])
        origin_yaw = ego0.rear_axle_se3.yaw
        return np.array([world_to_ego_frame(p, origin_xy, origin_yaw) for p in xy]).tolist(), None

    try:
        from py123d_carla_dataset import compute_target_points, find_scene_by_ref
    except ImportError as exc:
        return None, f"py123d not importable in this environment ({exc}). Run install_missing_requirements.sh."

    try:
        scene = find_scene_by_ref(data_root, log_name, scene_uuid)
    except AssertionError as exc:
        return None, f"{exc} (set $PY123D_DATA_ROOT before launching streamlit)"
    except Exception as exc:  # noqa: BLE001 - surface any py123d-side failure as a warning, not a crash
        return None, f"Failed to query py123d: {exc!r}"

    if scene is None:
        return None, f"No matching py123d scene found for log_name={log_name!r}, scene_uuid={scene_uuid!r}."

    return compute_target_points(scene).tolist(), None


@st.cache_data(show_spinner=False)
def project_trajectories_via_py123d_camera(data_root, log_name: str, scene_uuid: str, points_by_label: dict):
    """Projects each trajectory's ego-frame points through the scene's real
    per-scene camera (see py123d_carla_dataset.project_ego_points) instead of
    the fixed FOV=110 pinhole model in overlay_trajectory_on_image's default
    path -- that FOV is calibrated for SimLingo's own camera, not LEAD 123D's
    (confirmed actual fov_x=65deg via camera.metadata), and the mismatch is
    large enough to hide real off-road predictions and make legitimate
    expert trajectories look like they leave the road. Re-fetching the scene
    costs several seconds (same as load_bev_map/load_front_camera_image, and
    independently paid here for the same reason -- Scene objects aren't
    cached/shared between these).
    Returns (dict[str, list[(u, v)]]_or_None, error_message_or_None)."""
    try:
        from py123d_carla_dataset import find_scene_by_ref, project_ego_points
    except ImportError as exc:
        return None, f"py123d not importable in this environment ({exc}). Run install_missing_requirements.sh."

    try:
        scene = find_scene_by_ref(data_root, log_name, scene_uuid)
    except AssertionError as exc:
        return None, f"{exc} (set $PY123D_DATA_ROOT before launching streamlit)"
    except Exception as exc:  # noqa: BLE001 - surface any py123d-side failure as a warning, not a crash
        return None, f"Failed to query py123d: {exc!r}"

    if scene is None:
        return None, f"No matching py123d scene found for log_name={log_name!r}, scene_uuid={scene_uuid!r}."

    result = {label: project_ego_points(scene, points) for label, points in points_by_label.items() if points}
    return result, None


def clean_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    # collapse the run of per-patch <IMG_CONTEXT> placeholder tokens (one per
    # vision-encoder patch -- can be hundreds long) down to a single marker,
    # the rest of the templated chat prompt is what's actually informative.
    return re.sub(r"(<IMG_CONTEXT>)+", "<IMAGE>", prompt)


# label -> (color, dot radius), shared between both projection paths below.
TRAJECTORY_STYLES = {
    "expert_waypoints": ((0, 200, 200), 4),  # teal, expert near-term waypoints
    "expert_route": ((0, 180, 0), 3),  # green, expert route
    "predicted_speed_wps": ((255, 165, 0), 4),  # orange, predicted near-term speed_wps
    "predicted_route": ((255, 0, 0), 3),  # red, predicted route
}


def _draw_pixel_points(draw: ImageDraw.ImageDraw, pts, color, radius):
    if not pts:
        return
    for x, y in pts:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    if len(pts) > 1:
        draw.line([(x, y) for x, y in pts], fill=color, width=2)


def overlay_trajectory_on_image(
    image: Image.Image,
    predicted_route=None,
    predicted_speed_wps=None,
    expert_route=None,
    expert_waypoints=None,
    pixel_points=None,
) -> Image.Image:
    """Draws the predicted + expert route/waypoints onto the front camera
    image. If `pixel_points` is given (dict of label -> [(u, v), ...],
    already projected via a scene's real camera -- see
    project_trajectories_via_py123d_camera), draws those directly. Otherwise
    projects the given ego-frame points here using the fixed FOV=110 pinhole
    model (get_camera_intrinsics/project_points), which is correct for
    SimLingo-database scenes (calibrated for that dataset's own real camera)
    but NOT for py123d/CARLA scenes -- see
    project_trajectories_via_py123d_camera's docstring."""
    overlaid = image.copy()
    draw = ImageDraw.Draw(overlaid)

    if pixel_points is not None:
        for label, (color, radius) in TRAJECTORY_STYLES.items():
            _draw_pixel_points(draw, pixel_points.get(label), color, radius)
        return overlaid

    w, h = overlaid.size
    K = np.asarray(get_camera_intrinsics(w, h, CAMERA_FOV))
    values = dict(
        expert_waypoints=expert_waypoints,
        expert_route=expert_route,
        predicted_speed_wps=predicted_speed_wps,
        predicted_route=predicted_route,
    )
    for label, (color, radius) in TRAJECTORY_STYLES.items():
        points = values[label]
        if points:
            _draw_pixel_points(draw, project_points(points, K), color, radius)

    return overlaid


# layer -> (matplotlib kind, style kwargs, legend label). Only the layers
# that matter most for driving context are drawn; lane_group/walkway/carpark/
# generic_drivable are still returned by get_bev_map_ego_frame but skipped
# here to keep the plot legible.
BEV_LAYER_STYLES = {
    "lane": ("fill", dict(color="lightgrey", alpha=0.6), "lane"),
    "intersection": ("fill", dict(color="lightblue", alpha=0.4), "intersection"),
    "crosswalk": ("fill", dict(color="gold", alpha=0.6), "crosswalk"),
    "road_edge": ("line", dict(color="black", linewidth=1.5), "road edge"),
    "road_line": ("line", dict(color="dimgrey", linewidth=1.0), "road line"),
}


def plot_trajectories(
    predicted_route,
    predicted_speed_wps,
    expert_route=None,
    expert_waypoints=None,
    target_points=None,
    target_point_source: str = "trajectory",
    bev_map=None,
    radius=MAP_QUERY_RADIUS,
    ax=None,
):
    """Pass an existing `ax` to draw into it as one panel of a larger figure
    (see render_scene_grid_png) instead of creating a standalone one -- in
    that case the caller owns the figure/layout and this returns None.
    Default (ax=None) is unchanged: creates its own figure and returns it,
    for the interactive viewer's own st.pyplot(fig) usage."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 7))

    if bev_map:
        for layer_name, (kind, style, label) in BEV_LAYER_STYLES.items():
            for i, (ego_xy, _is_surface) in enumerate(bev_map.get(layer_name, [])):
                kwargs = dict(style, zorder=0, label=label if i == 0 else None)
                if kind == "fill":
                    ax.fill(ego_xy[:, 1], ego_xy[:, 0], **kwargs)
                else:
                    ax.plot(ego_xy[:, 1], ego_xy[:, 0], **kwargs)

    def plot_one(points, color, marker, label):
        if not points:
            return
        arr = np.array(points)
        ax.plot(arr[:, 1], arr[:, 0], c=color, marker=marker, markersize=4, label=label, zorder=4)

    plot_one(expert_route, "g", "o", "expert route (20 pt)")
    plot_one(expert_waypoints, "c", "x", "expert waypoints (10 pt)")
    plot_one(predicted_route, "r", "o", "predicted route (20 pt)")
    plot_one(predicted_speed_wps, "orange", "x", "predicted speed_wps (10 pt)")

    if target_points:
        # the near/far navigation goal actually fed to the model via the
        # <TARGET_POINT> placeholder token's learned embedding (see
        # collect_embeddings.py's _process_batch) -- not derived from
        # predicted/expert route, an independent input signal. Label reflects
        # which method actually produced these coordinates (see
        # load_target_points's target_point_source param) -- "route" (real
        # map-based reconstruction) and "trajectory" (ego-future-motion
        # approximation) diverge meaningfully, see
        # py123d_route_target_points.py's module docstring.
        arr = np.array(target_points)
        label = f"target point ({target_point_source})"
        ax.scatter(arr[:, 1], arr[:, 0], c="magenta", marker="D", s=70, label=label, zorder=6)

    ax.scatter([0], [0], c="black", marker="*", s=120, label="ego (now)", zorder=5)
    ax.set_xlabel("lateral y (m), right positive →")
    ax.set_ylabel("forward x (m)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    if bev_map:
        # map objects are returned whole even if only part of them falls
        # inside the query radius (e.g. a long lane) -- clip the view so the
        # plot stays centered on the ego vehicle rather than zooming out to
        # fit some faraway lane's far end.
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius / 3, radius)
    if standalone:
        # External, below-axis legend -- fine here since this figure is just
        # this one plot (see the interactive viewer's own st.pyplot(fig) call).
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8, ncol=2)
        fig.tight_layout()
        return fig
    # In a combined figure (see render_scene_grid_png), an external
    # bbox_to_anchor legend inflates tight_layout's space reservation for
    # this one panel and starves the others sharing its row/figure -- a
    # compact in-axis legend avoids that entirely.
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)
    return None


def _truncate(text: str, max_chars: int = 500) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars].rstrip() + "..."


def _render_text_panel(ax, label: str, prompt: str, text_output: str, wrap_width: int = 100):
    """Draws a label + wrapped prompt/output block into an axis with no
    plot -- used below each image in render_scene_grid_png's grid, so a
    scene's full comparison (images + both models' language) is one
    self-contained figure/PNG instead of split across separate ones."""
    ax.axis("off")
    prompt_wrapped = textwrap.fill(_truncate(clean_prompt(prompt)) or "(no prompt)", width=wrap_width)
    output_wrapped = textwrap.fill(_truncate(text_output) or "(empty)", width=wrap_width)
    ax.text(0.0, 1.0, label, fontsize=13, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(
        0.0, 0.85, f"Prompt:\n{prompt_wrapped}\n\nOutput:\n{output_wrapped}",
        fontsize=8.5, family="monospace", transform=ax.transAxes, va="top",
    )


def render_scene_grid_png(
    root: str,
    qwen_root: str,
    scene_id: str,
    primary_key_index: dict,
    qwen_id_by_key: dict,
    data_root,
) -> bytes:
    """Renders one scene as a single, self-contained PNG for sharing outside
    the interactive app (e.g. a team presentation): front camera +
    trajectory overlay and the BEV plot on top, SimLingo's and Qwen3-VL's
    prompt + text output below each column respectively.

    Reuses the exact same data-loading/rendering functions the interactive
    viewer page uses (load_front_camera_image, project_trajectories_via_
    py123d_camera, overlay_trajectory_on_image, load_bev_map, plot_trajectories),
    unmodified -- so the images are pixel-for-pixel identical to what the
    main viewer page shows, not a re-derivation of that logic.

    primary_key_index/qwen_id_by_key: same {scene_id: (log_name, scene_uuid)}
    / {(log_name, scene_uuid): qwen_scene_id} maps main() builds from
    build_scene_key_index -- passed in rather than rebuilt per scene, since
    building them requires reading every scene's metadata.json once (slow
    for large datasets) and the caller renders many scenes per session.
    """
    metadata = load_scene_metadata(root, scene_id)
    qwen_scene_id = qwen_id_by_key.get(primary_key_index.get(scene_id))
    qwen_metadata = load_scene_metadata(qwen_root, qwen_scene_id) if qwen_scene_id else None

    log_name, scene_uuid = metadata.get("log_name"), metadata.get("scene_uuid")
    is_py123d = bool(metadata.get("source") == "py123d_carla" and log_name and scene_uuid)
    has_predicted = "predicted_route" in metadata or "predicted_speed_wps" in metadata
    has_expert = "expert_route" in metadata or "expert_waypoints" in metadata

    image, image_error = load_front_camera_image(
        metadata.get("image_path"), metadata.get("source"), log_name, scene_uuid, data_root,
    )

    display_image = image
    if image is not None and (has_predicted or has_expert):
        if is_py123d:
            points_by_label = {
                "expert_waypoints": metadata.get("expert_waypoints"),
                "expert_route": metadata.get("expert_route"),
                "predicted_speed_wps": metadata.get("predicted_speed_wps"),
                "predicted_route": metadata.get("predicted_route"),
            }
            pixel_points, proj_error = project_trajectories_via_py123d_camera(
                data_root, log_name, scene_uuid, points_by_label
            )
            if not proj_error:
                display_image = overlay_trajectory_on_image(image, pixel_points=pixel_points)
        else:
            display_image = overlay_trajectory_on_image(
                image,
                metadata.get("predicted_route"),
                metadata.get("predicted_speed_wps"),
                metadata.get("expert_route"),
                metadata.get("expert_waypoints"),
            )

    bev_map, target_points = None, None
    if is_py123d:
        bev_map, _ = load_bev_map(data_root, log_name, scene_uuid, MAP_QUERY_RADIUS)
        target_points, _ = load_target_points(
            data_root, log_name, scene_uuid, metadata.get("target_point_source", "trajectory"),
        )

    # Row 0's height is sized to match the two panels' own aspect ratios at
    # this column width (~7.5in after margins/wspace): camera is 16:9
    # (needs ~4.2in), BEV's aspect="equal" box is ~1.5:1 (needs ~5in) --
    # picking row0=5in keeps both close to their natural size instead of
    # centered with a lot of padding inside a much-taller shared row.
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 4])

    ax_cam = fig.add_subplot(gs[0, 0])
    ax_cam.axis("off")
    ax_cam.set_title("Front camera + trajectories")
    if display_image is not None:
        ax_cam.imshow(display_image)
    else:
        ax_cam.text(0.5, 0.5, image_error or "No image available", ha="center", va="center", transform=ax_cam.transAxes)

    ax_bev = fig.add_subplot(gs[0, 1])
    ax_bev.set_title("Bird's-eye view + trajectories")
    plot_trajectories(
        metadata.get("predicted_route"),
        metadata.get("predicted_speed_wps"),
        metadata.get("expert_route"),
        metadata.get("expert_waypoints"),
        target_points=target_points,
        target_point_source=metadata.get("target_point_source", "trajectory"),
        bev_map=bev_map,
        radius=MAP_QUERY_RADIUS,
        ax=ax_bev,
    )

    ax_sl = fig.add_subplot(gs[1, 0])
    _render_text_panel(ax_sl, "SimLingo", metadata.get("prompt", ""), metadata.get("text_output"))

    ax_qwen = fig.add_subplot(gs[1, 1])
    ax_qwen.axis("off")
    if qwen_metadata is not None:
        _render_text_panel(ax_qwen, "Qwen3-VL", qwen_metadata.get("prompt", ""), qwen_metadata.get("text_output"))
    else:
        ax_qwen.text(0.0, 1.0, "Qwen3-VL", fontsize=13, fontweight="bold", transform=ax_qwen.transAxes, va="top")
        ax_qwen.text(0.0, 0.85, "No matching Qwen3-VL scene.", fontsize=9, transform=ax_qwen.transAxes, va="top")

    fig.suptitle(f"scene {scene_id}   |   {log_name or metadata.get('route_id', '')}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    # bbox_inches="tight": the camera image (16:9) and BEV plot (aspect="equal",
    # ~1.5:1) each get centered within their shared GridSpec row height once
    # matplotlib shrinks their box to satisfy that aspect constraint -- rather
    # than hand-tuning figsize/height_ratios to match both aspects exactly,
    # this crops the saved PNG to the actual rendered content's bounding box.
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def main():
    args = parse_args()
    st.set_page_config(page_title="SimLingo embeddings viewer", layout="wide")
    st.title("SimLingo embeddings viewer")

    root = st.sidebar.text_input("embeddings_dataset root", value=args.root)
    if not Path(root, "manifest.json").exists():
        st.error(f"No manifest.json found under {root!r}")
        return

    scenes = load_manifest(root)
    if not scenes:
        st.warning("manifest.json has no scenes.")
        return

    embedding_types = sorted({s.get("embedding_type") for s in scenes if s.get("embedding_type")})
    selected_types = st.sidebar.multiselect("embedding_type filter", embedding_types, default=embedding_types)
    filtered = [s for s in scenes if s.get("embedding_type") in selected_types] if selected_types else scenes
    if not filtered:
        st.warning("No scenes match the selected embedding_type filter.")
        return

    name_filter = st.sidebar.text_input("Filter scenes by name (substring match, any position)", value="")
    if name_filter:
        needle = name_filter.lower()
        filtered = [
            s
            for s in filtered
            if needle in f"{s['id']} {s.get('route', '')} {s.get('frame_id', '')}".lower()
        ]
        if not filtered:
            st.warning(f"No scenes match filter {name_filter!r}.")
            return

    qwen_root = st.sidebar.text_input("Qwen3-VL embeddings_dataset root (optional)", value=args.qwen_root)
    qwen_id_by_key, primary_key_index = {}, {}
    if qwen_root:
        if Path(qwen_root, "manifest.json").exists():
            with st.spinner("Indexing scenes for SimLingo <-> Qwen3-VL matching..."):
                qwen_id_by_key = {key: sid for sid, key in build_scene_key_index(qwen_root).items()}
                primary_key_index = build_scene_key_index(root)
        else:
            st.sidebar.warning(f"No manifest.json found under {qwen_root!r} -- Qwen3-VL comparison disabled.")

    only_with_qwen = st.sidebar.checkbox(
        "Only show scenes with a Qwen3-VL description", value=False, disabled=not qwen_id_by_key,
    )
    if only_with_qwen and qwen_id_by_key:
        filtered = [s for s in filtered if primary_key_index.get(s["id"]) in qwen_id_by_key]
        if not filtered:
            st.warning("No scenes have a matching Qwen3-VL description.")
            return

    labels = [f"{s['id']}  ({s.get('route', '?')}, frame {s.get('frame_id', '?')})" for s in filtered]
    if "scene_idx" not in st.session_state:
        st.session_state.scene_idx = 0
    st.session_state.scene_idx = min(st.session_state.scene_idx, len(filtered) - 1)

    col_prev, col_sel, col_next = st.sidebar.columns([1, 3, 1])
    if col_prev.button("<"):
        st.session_state.scene_idx = max(0, st.session_state.scene_idx - 1)
    if col_next.button(">"):
        st.session_state.scene_idx = min(len(filtered) - 1, st.session_state.scene_idx + 1)
    chosen_label = col_sel.selectbox("scene", labels, index=st.session_state.scene_idx, label_visibility="collapsed")
    st.session_state.scene_idx = labels.index(chosen_label)

    scene_id = filtered[st.session_state.scene_idx]["id"]
    metadata = load_scene_metadata(root, scene_id)

    qwen_metadata = None
    if qwen_id_by_key:
        qwen_scene_id = qwen_id_by_key.get(primary_key_index.get(scene_id))
        if qwen_scene_id:
            qwen_metadata = load_scene_metadata(qwen_root, qwen_scene_id)

    st.caption(f"scene {scene_id} / {len(filtered)} (of {len(scenes)} total)")

    has_predicted = "predicted_route" in metadata or "predicted_speed_wps" in metadata
    has_expert = "expert_route" in metadata or "expert_waypoints" in metadata
    is_py123d = bool(metadata.get("source") == "py123d_carla" and metadata.get("log_name") and metadata.get("scene_uuid"))

    data_root = os.environ.get("PY123D_DATA_ROOT")
    with st.spinner("Loading front camera image..."):
        image, image_error = load_front_camera_image(
            metadata.get("image_path"),
            metadata.get("source"),
            metadata.get("log_name"),
            metadata.get("scene_uuid"),
            data_root,
        )

    col_img, col_traj = st.columns(2)
    with col_img:
        st.subheader("Front camera")
        show_overlay = st.checkbox("Overlay trajectories", value=True)
        if image is not None:
            if show_overlay and (has_predicted or has_expert):
                if is_py123d:
                    points_by_label = {
                        "expert_waypoints": metadata.get("expert_waypoints"),
                        "expert_route": metadata.get("expert_route"),
                        "predicted_speed_wps": metadata.get("predicted_speed_wps"),
                        "predicted_route": metadata.get("predicted_route"),
                    }
                    with st.spinner("Projecting trajectories via py123d's real camera..."):
                        pixel_points, proj_error = project_trajectories_via_py123d_camera(
                            data_root, metadata["log_name"], metadata["scene_uuid"], points_by_label
                        )
                    if proj_error:
                        st.warning(proj_error)
                        display_image = image
                    else:
                        display_image = overlay_trajectory_on_image(image, pixel_points=pixel_points)
                        st.caption(
                            "red = predicted route, orange = predicted speed_wps, green = expert route, "
                            "teal = expert waypoints (projected via py123d's real per-scene camera; "
                            "points outside its actual field of view are dropped)"
                        )
                else:
                    display_image = overlay_trajectory_on_image(
                        image,
                        metadata.get("predicted_route"),
                        metadata.get("predicted_speed_wps"),
                        metadata.get("expert_route"),
                        metadata.get("expert_waypoints"),
                    )
                    st.caption("red = predicted route, orange = predicted speed_wps, green = expert route, teal = expert waypoints")
            else:
                display_image = image
            st.image(display_image, width="stretch")
        else:
            st.info(image_error or "No image available for this scene.")

    with col_traj:
        st.subheader("Predicted vs. expert trajectory (ego frame)")
        show_bev_map = st.checkbox("Load BEV map (py123d/CARLA only, re-queries py123d)", value=True, disabled=not is_py123d)

        bev_map = None
        if show_bev_map and is_py123d:
            with st.spinner("Querying py123d for map data..."):
                bev_map, bev_error = load_bev_map(data_root, metadata["log_name"], metadata["scene_uuid"], MAP_QUERY_RADIUS)
            if bev_error:
                st.warning(bev_error)
        elif show_bev_map and not is_py123d:
            st.info("BEV map is only available for py123d/CARLA-sourced scenes (needs log_name/scene_uuid to re-query py123d).")

        target_points = None
        if is_py123d:
            target_point_source = metadata.get("target_point_source", "trajectory")
            with st.spinner(f"Re-querying py123d for the target point ({target_point_source})..."):
                target_points, target_points_error = load_target_points(
                    data_root, metadata["log_name"], metadata["scene_uuid"], target_point_source,
                )
            if target_points_error:
                st.warning(target_points_error)

        if has_predicted or has_expert or bev_map:
            fig = plot_trajectories(
                metadata.get("predicted_route"),
                metadata.get("predicted_speed_wps"),
                metadata.get("expert_route"),
                metadata.get("expert_waypoints"),
                target_points=target_points,
                target_point_source=metadata.get("target_point_source", "trajectory"),
                bev_map=bev_map,
                radius=MAP_QUERY_RADIUS,
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No trajectory fields in this scene's metadata (collected with an older version of collect_embeddings.py).")

    st.subheader("Language")
    col_sl_lang, col_qwen_lang = st.columns(2)
    with col_sl_lang:
        st.markdown("**SimLingo**")
        st.write(metadata.get("text_output") or "_(empty)_")
        with st.expander("Full prompt"):
            st.code(clean_prompt(metadata.get("prompt", "")), language=None)
    with col_qwen_lang:
        st.markdown("**Qwen3-VL**")
        if qwen_metadata is not None:
            st.write(qwen_metadata.get("text_output") or "_(empty)_")
            with st.expander("Full prompt"):
                st.code(clean_prompt(qwen_metadata.get("prompt", "")), language=None)
        elif qwen_root:
            st.info("No matching Qwen3-VL scene for this (log_name, scene_uuid).")
        else:
            st.info("Set a Qwen3-VL embeddings_dataset root in the sidebar to compare.")

    st.subheader("Metadata")
    col_sl_meta, col_qwen_meta = st.columns(2)
    with col_sl_meta:
        st.json(metadata, expanded=False)
    with col_qwen_meta:
        if qwen_metadata is not None:
            st.json(qwen_metadata, expanded=False)


if __name__ == "__main__":
    main()
