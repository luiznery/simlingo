"""
Adapts CARLA scenes loaded through py123d (the ``lead123d`` dataset, selected
via ``SceneFilter(datasets=["carla"])`` -- see dynamo_py123d) into the
``DatasetOutput`` shape SimLingo's own dataset classes
(simlingo_training/dataloader/dataset_driving.py) produce, so the existing,
unmodified ``DataModule.dl_collate_fn`` (image patchification + chat-template
tokenization + tensor stacking) can be reused as-is.

What this can and can't give the model, and why
------------------------------------------------
py123d's generic Scene API (see tutorials/01_scene_tutorial.ipynb and
05_scene_filter_tutorial.ipynb in dynamo_py123d) has no CARLA-specific route
or navigation-command signal -- there is no parser in dynamo_py123d for CARLA
specifically (see dynamo_py123d/ego_status_report/README.md), and no ground
truth commentary/QA/dreamer text either (those are SimLingo-specific
annotations that don't exist for arbitrary py123d scenes). So per scene we can
only build:

  - a front camera image (CameraID.PCAM_F0)
  - ego speed: ego_state.dynamic_state_se3.velocity_2d.magnitude. This is safe
    even though CARLA's dynamic_state_se3 ships with a known frame/units bug
    for the raw vx/vy/yaw_rate components (see ego_status_report/README.md) --
    a rotation preserves vector magnitude, so the speed scalar is unaffected.
  - "waypoints" / "path" / "target_point": approximated from the scene's own
    future ego trajectory (transformed into the current frame's ego-relative
    coordinates), since py123d exposes no separate planned-route modality for
    CARLA. This is a stand-in for SimLingo's route-planner-derived fields of
    the same name, not a literal reproduction of them.

Given no ground-truth commentary exists, every sample uses the
"What should the ego do next?" prompt (embedding_type="commentary"), which is
the only SimLingo prompt template that elicits free-form language without
needing ground truth (the "answer" field is never read at inference -- see
DrivingAdaptor.forward/LanguageAdaptor.forward in
simlingo_training/models/adaptors/adaptors.py, neither touches driving_label).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from simlingo_training.utils.custom_types import DatasetOutput

NUM_ROUTE_POINTS = 20  # matches BaseDataset.equal_spacing_route's hardcoded 20
PRED_LEN = 11  # matches DatasetBaseConfig.pred_len default, "0.2s apart" (see custom_types.py)
# Must evenly divide the source logs' native iteration_duration_s (0.1s / 10Hz
# for the lead123d CARLA logs) -- py123d's resampler silently returns zero
# scenes instead of erroring when it doesn't (confirmed empirically: 0.25s
# -> 0 scenes, 0.2s -> ~259k scenes against the real downloaded data).
ITERATION_DURATION_S = 0.2
FRONT_CAMERA_ID = "pcam_f0"


def _scene_cache_key(log_names: Optional[List[str]], seed: int) -> str:
    """Identifies a scene-validation cache to the exact filter it was built
    from (see Py123dCarlaDataset.__init__) -- so a cache built for one
    log_names/seed combo never gets silently reused for a different one."""
    log_names_key = ",".join(sorted(log_names)) if log_names else "ALL"
    key_src = f"{log_names_key}|seed={seed}|iter={ITERATION_DURATION_S}|pred_len={PRED_LEN}|cam={FRONT_CAMERA_ID}"
    return hashlib.sha1(key_src.encode()).hexdigest()[:16]


def equal_spacing_route(points: np.ndarray) -> np.ndarray:
    """Verbatim copy of BaseDataset.equal_spacing_route (dataset_base.py) --
    it doesn't use `self`, so this avoids depending on BaseDataset's heavy
    SimLingo-database-specific __init__ just to reach one pure function."""
    route = np.concatenate((np.zeros_like(points[:1]), points))
    shift = np.roll(route, 1, axis=0)
    shift[0] = shift[1]

    dists = np.linalg.norm(route - shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(0, len(dists)) * 1e-4

    x = np.arange(0, NUM_ROUTE_POINTS, 1)
    interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T
    return interp_points


def world_to_ego_frame(point_xy: np.ndarray, origin_xy: np.ndarray, origin_yaw: float) -> np.ndarray:
    """Rotates a world-frame point into the frame centered at
    (origin_xy, origin_yaw), i.e. origin_rotation.T @ (point - origin_translation)
    for the 2D case -- same shape of transform as BaseDataset.get_waypoints, but
    NOT numerically interchangeable with it: get_waypoints builds its rotation
    directly from CARLA's raw ego_matrix (CARLA-native, Y-right, left-handed),
    while py123d's rear_axle_se3.yaw follows py123d's own ISO 8855 convention
    (Y-left, right-handed -- see dynamo_py123d's LEAD 123D README "Coordinate
    conventions" and ego_status_report/README.md's "already correct, right-handed"
    pose note). Reconstructing a rotation matrix from that yaw with the standard
    CCW-positive formula therefore lands in the *mirrored* frame relative to
    everything else in this pipeline (project_points, the model's own trained
    waypoint/route convention, dataset_base.get_waypoints) -- confirmed
    empirically: projecting the raw (unnegated) output's lane polygons onto the
    front camera placed the ego's own lane off to the side of the visible road
    instead of under the visible traffic ahead. The final negation below
    converts py123d's Y-left into this pipeline's Y-right, matching everywhere
    else this ego frame is consumed (waypoints/path/target_points here, and
    get_bev_map_ego_frame's map query)."""
    cos_yaw, sin_yaw = np.cos(origin_yaw), np.sin(origin_yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    ego_xy = rotation.T @ (point_xy - origin_xy)
    ego_xy[1] *= -1
    return ego_xy


def ego_to_world_frame(ego_xy: np.ndarray, origin_xy: np.ndarray, origin_yaw: float, z: float = 0.0) -> np.ndarray:
    """Inverse of world_to_ego_frame: converts an ego-frame (forward, lateral)
    point in this pipeline's Y-right convention back to a world-frame 3D point
    [x, y, z]. `z` is this specific point's elevation -- callers projecting
    onto a camera should pass the real per-point elevation (see
    elevation_profile/project_ego_points) rather than a single constant,
    since assuming flat ground visibly misplaces far points on any sloped
    road (confirmed empirically: on an ~8% grade, the flat assumption placed
    a 14m-out waypoint 58 pixels higher than its real position)."""
    y_iso = -ego_xy[1]  # undo world_to_ego_frame's Y-right negation
    cos_yaw, sin_yaw = np.cos(origin_yaw), np.sin(origin_yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    world_xy = rotation @ np.array([ego_xy[0], y_iso]) + origin_xy
    return np.array([world_xy[0], world_xy[1], z])


def elevation_profile(scene, origin_xy: np.ndarray, origin_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-distance -> elevation samples built from this scene's own
    recorded ego trajectory (iterations 0..PRED_LEN, guaranteed available by
    the SceneFilter used to build these scenes -- see Py123dCarlaDataset).
    Used to interpolate a per-point z by forward distance instead of
    assuming flat ground: a reasonable proxy since road elevation is
    primarily a function of how far along the road you are, not lateral
    offset within the lane."""
    xs = [0.0]
    ego0 = scene.get_ego_state_se3_at_iteration(0)
    zs = [ego0.rear_axle_se3.z]
    for k in range(1, PRED_LEN + 1):
        ego_k = scene.get_ego_state_se3_at_iteration(k)
        if ego_k is None:
            break
        world_xy = np.array([ego_k.rear_axle_se3.x, ego_k.rear_axle_se3.y])
        ego_xy = world_to_ego_frame(world_xy, origin_xy, origin_yaw)
        xs.append(float(ego_xy[0]))
        zs.append(float(ego_k.rear_axle_se3.z))
    return np.array(xs), np.array(zs)


def project_ego_points(scene, points) -> List[Tuple[float, float]]:
    """Projects ego-frame (forward, lateral) points through this scene's
    actual front camera -- its real per-scene intrinsics and pose
    (Camera.project_points_global), not a generic assumption. This matters:
    simlingo_training/utils/projection.py's get_camera_intrinsics/project_points
    hardcode a fixed 110deg FOV calibrated for SimLingo's own real camera
    ("TF++ default camera"), but LEAD 123D's actual CARLA camera has
    fov_x=65deg (confirmed via camera.metadata.fov_x) -- a ~1.7x too-wide
    assumption there silently mis-projects every py123d/CARLA scene: points
    truly outside the real 65deg view get folded back onto the image instead
    of dropped, and in-view points land at the wrong pixel.

    Each point's elevation is interpolated from the scene's own recorded
    elevation profile (by forward distance) rather than assumed flat --
    matters on sloped roads, see ego_to_world_frame's docstring.

    Only points behind the camera (depth <= 0, where pinhole projection is
    undefined) are dropped -- points in front of it but outside the visible
    frame are kept with their real (out-of-bounds) pixel coordinates, so a
    line connecting them to an in-view point draws correctly right up to
    the image edge (PIL clips ellipses/lines drawn outside the canvas on its
    own) instead of the trajectory appearing to start abruptly mid-frame."""
    from py123d.datatypes import CameraID

    camera = scene.get_camera_at_iteration(0, camera_id=CameraID.PCAM_F0)
    ego0 = scene.get_ego_state_se3_at_iteration(0)
    origin_xy = np.array([ego0.rear_axle_se3.x, ego0.rear_axle_se3.y])
    origin_yaw = ego0.rear_axle_se3.yaw

    xs, zs = elevation_profile(scene, origin_xy, origin_yaw)
    world_pts = np.array(
        [ego_to_world_frame(np.asarray(p), origin_xy, origin_yaw, z=float(np.interp(p[0], xs, zs))) for p in points]
    )
    pixel_coords, _in_fov, depth = camera.project_points_global(world_pts)
    return [(float(u), float(v)) for (u, v), d in zip(pixel_coords, depth) if d > 0]


def encode_scene_ref(dataset: str, split: str, log_name: str, scene_uuid: str, iteration: int, speed: float, pos: np.ndarray) -> str:
    """Packs everything build_py123d_metadata() needs into the same fixed-length
    uint8 `run_id` string SimLingo's own measurement_path travels through
    (DataModule.dl_collate_fn -> encode_uint8(..., 1000)), so no extra file
    re-read is needed to recover metadata downstream."""
    return "|".join(
        [
            "carla_py123d",
            dataset,
            split,
            log_name,
            scene_uuid,
            str(iteration),
            f"{speed:.4f}",
            f"{pos[0]:.3f}",
            f"{pos[1]:.3f}",
        ]
    )


def decode_scene_ref(ref: str) -> dict:
    parts = ref.split("|")
    if len(parts) != 9 or parts[0] != "carla_py123d":
        return {"raw": ref}
    _, dataset, split, log_name, scene_uuid, iteration, speed, x, y = parts
    return {
        "dataset": dataset,
        "split": split,
        "log_name": log_name,
        "scene_uuid": scene_uuid,
        "iteration": int(iteration),
        "vehicle_speed": float(speed),
        "vehicle_pos": [float(x), float(y)],
    }


class Py123dCarlaDataset(Dataset):
    """Iterates CARLA scenes loaded via py123d and yields SimLingo-shaped
    DatasetOutput items with a "What should the ego do next?" commentary
    prompt (no ground truth answer -- see module docstring).

    Each py123d ``SceneAPI`` returned by ``get_filtered_scenes`` is already one
    fixed reference-timestamp window (iteration 0 = "now", 1..N = future) --
    matching SimLingo's own one-sample-per-frame convention, so one Scene here
    maps to one collected embedding, same as one (route, frame) does for the
    original database/simlingo/ pipeline.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        log_names: Optional[List[str]] = None,
        max_scenes: Optional[int] = None,
        cut_bottom_quarter: bool = True,
        seed: int = 0,
        scene_cache_path: Optional[str] = None,
    ):
        from py123d.api import get_filtered_scenes
        from py123d.api.scene.scene_filter import SceneFilter

        self.cut_bottom_quarter = cut_bottom_quarter

        scene_filter = SceneFilter(
            datasets=["carla"],
            log_names=log_names,
            target_iteration_duration_s=ITERATION_DURATION_S,
            future_duration_s=ITERATION_DURATION_S * PRED_LEN,
            history_duration_s=0.0,
            required_scene_modalities=[f"camera.{FRONT_CAMERA_ID}", "ego_state_se3"],
        )
        scenes = get_filtered_scenes(scene_filter, data_root=data_root)

        # get_filtered_scenes() with datasets=["carla"] alone can return
        # hundreds of thousands of candidates (259k against the reference
        # download used to test this). Validating every one of them (below)
        # decodes an image per scene and is far too slow to run unconditionally
        # -- so when max_scenes is set, shuffle first and validate lazily,
        # stopping as soon as we have enough, instead of paying O(all scenes).
        order = list(range(len(scenes)))
        random.Random(seed).shuffle(order)

        # Full (max_scenes=None) validation runs are expensive (real per-scene
        # I/O for every candidate) and every fresh process re-pays that cost
        # from scratch, including on --resume. Cache the validated
        # (log_name, scene_uuid) list -- in validated order, so resume's
        # Subset(range(resume_from, ...)) indexing stays correct -- keyed to
        # this exact filter, so a differently-scoped run (different log_names/
        # seed) doesn't reuse a mismatched cache. Only used/written for the
        # exhaustive case (max_scenes=None): a partial early-stopped list from
        # a bounded run would silently truncate a later, larger request.
        cache_file = Path(scene_cache_path) if (scene_cache_path and max_scenes is None) else None
        cache_key = _scene_cache_key(log_names, seed)

        cached_refs = None
        if cache_file is not None and cache_file.exists():
            with open(cache_file) as f:
                cache_data = json.load(f)
            if cache_data.get("key") == cache_key:
                cached_refs = cache_data["scenes"]
            else:
                print(f"scene cache at {cache_file} was built with different filter params -- ignoring and re-validating")

        if cached_refs is not None:
            by_ref = {
                (s.log_name, str(s.scene_uuid)): s
                for s in tqdm(scenes, desc="Indexing candidate scenes for cache lookup")
            }
            valid_scenes = []
            missing = 0
            for entry in tqdm(cached_refs, desc="Loading cached scenes"):
                scene = by_ref.get((entry["log_name"], entry["scene_uuid"]))
                if scene is not None:
                    valid_scenes.append(scene)
                else:
                    missing += 1
            print(f"Loaded {len(valid_scenes)} validated py123d/CARLA scenes from cache ({cache_file})" + (f", {missing} no longer found" if missing else ""))
        else:
            # Defensive validation: required_scene_modalities guarantees
            # iteration-0 camera/ego_state, but not that every future
            # iteration up to PRED_LEN is present/uncorrupted (94 CARLA logs
            # had corrupt ego_state_se3.arrow files per
            # dynamo_py123d/ego_status_report/README.md), nor that the
            # camera actually decodes -- checked once here so a bad scene
            # can't crash collate_fn's batch stacking mid-run.
            valid_scenes = []
            iterator = tqdm(order, desc="Validating py123d/CARLA scenes")
            for idx in iterator:
                if max_scenes is not None and len(valid_scenes) >= max_scenes:
                    break
                scene = scenes[idx]
                if scene.number_of_iterations < PRED_LEN + 1:
                    continue
                ego0 = scene.get_ego_state_se3_at_iteration(0)
                if ego0 is None or ego0.dynamic_state_se3 is None:
                    continue
                cam0 = scene.get_camera_at_iteration(0, camera_id=FRONT_CAMERA_ID)
                if cam0 is None:
                    continue
                future_ok = True
                for k in range(1, PRED_LEN + 1):
                    if scene.get_ego_state_se3_at_iteration(k) is None:
                        future_ok = False
                        break
                if not future_ok:
                    continue
                valid_scenes.append(scene)

            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(
                        {
                            "key": cache_key,
                            "scenes": [{"log_name": s.log_name, "scene_uuid": str(s.scene_uuid)} for s in valid_scenes],
                        },
                        f,
                    )
                print(f"Wrote {len(valid_scenes)} validated scenes to cache ({cache_file})")

        self.scenes = valid_scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> DatasetOutput:
        scene = self.scenes[index]

        image_ff_org = get_front_camera_image(scene)  # HWC uint8 RGB, full resolution, already decoded
        image = image_ff_org
        if self.cut_bottom_quarter:
            image = image[: int(image.shape[0] - (image.shape[0] * 4.8) // 16), :, :]

        # DatasetOutput.image_ff/_org_size expect [T, C, H, W] with T=1 (hist_len=1).
        image_ff = np.transpose(image[None], (0, 3, 1, 2))
        image_ff_org_size = np.transpose(image_ff_org[None], (0, 3, 1, 2))

        ego0 = scene.get_ego_state_se3_at_iteration(0)
        # .x / .y / .yaw are the accessors directly confirmed on PoseSE3
        # (pose.py); .point_2d was not, so avoided here.
        origin_xy = np.array([ego0.rear_axle_se3.x, ego0.rear_axle_se3.y])
        origin_yaw = ego0.rear_axle_se3.yaw
        speed = float(ego0.dynamic_state_se3.velocity_2d.magnitude)

        future_xy = [np.zeros(2)]  # iteration 0 is the origin -> (0, 0) in its own frame
        for k in range(1, PRED_LEN + 1):
            ego_k = scene.get_ego_state_se3_at_iteration(k)
            world_xy = np.array([ego_k.rear_axle_se3.x, ego_k.rear_axle_se3.y])
            future_xy.append(world_to_ego_frame(world_xy, origin_xy, origin_yaw))
        future_xy = np.array(future_xy)  # [PRED_LEN + 1, 2], index 0 == (0, 0)

        # Matches BaseDataset.load_waypoints's data['waypoints'] = waypoints[1:-1]
        # (drops the all-zero current frame and the final horizon point).
        waypoints = future_xy[1:-1]
        waypoints_1d_dists = np.linalg.norm(np.diff(future_xy, axis=0), axis=1)
        waypoints_1d = np.cumsum(waypoints_1d_dists)
        waypoints_1d = np.array([[d, 0.0] for d in waypoints_1d[:-1]]).reshape(-1, 2)

        # No planned-route modality exists for CARLA in py123d (see module
        # docstring) -- approximated as the same future trajectory, resampled
        # to a fixed number of equally-spaced points like SimLingo's own route.
        path = equal_spacing_route(future_xy)

        # Stand-ins for the route-planner's near/far navigation targets:
        # ~1s and full-horizon points along the future trajectory.
        near_idx = min(4, len(future_xy) - 1)
        target_point = future_xy[near_idx]
        target_point_next = future_xy[-1]
        target_points = np.array([target_point, target_point_next])

        speed_rounded = round(speed, 1)
        # Matches dataset_base.py's get_navigational_conditioning's
        # non-target_point_language branch: a literal "<TARGET_POINT>"
        # token, later swapped for a continuous embedding of target_points
        # (via placeholder_values below) by the model's own
        # replace_placeholder_tokens -- this checkpoint was trained with
        # route_as="target_point_command" (simlingo_seed1.yaml), so real
        # prompts never contain literal printed target-point digits.
        # Printing raw digits here instead (the previous behavior) fed the
        # model a text format it never saw in training AND meant
        # placeholder_values's actual target_points embedding was never
        # injected at all, since the special token was never present to
        # trigger the lookup -- silently zeroing out its navigation signal.
        prompt = (
            f"Current speed: {speed_rounded} m/s. "
            "Target waypoint: <TARGET_POINT><TARGET_POINT>. "
            "What should the ego do next?"
        )
        answer = "Waypoints:"  # never read at inference (see module docstring)

        conversation_answer = [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        conversation_all = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        scene_metadata = scene.get_scene_metadata()
        log_metadata = scene.get_log_metadata()
        run_ref = encode_scene_ref(
            dataset=scene.dataset,
            split=scene.split,
            log_name=scene.log_name,
            scene_uuid=str(scene.scene_uuid),
            iteration=0,
            speed=speed,
            pos=origin_xy,
        )
        del scene_metadata, log_metadata  # not currently surfaced further, kept for future use

        return DatasetOutput(
            conversation=conversation_all,
            answer=conversation_answer,
            image_ff=image_ff,
            image_ff_org_size=image_ff_org_size,
            waypoints=waypoints,
            waypoints_1d=waypoints_1d,
            path=path,
            target_points=target_points,
            speed=speed,
            placeholder_values={"<TARGET_POINT>": target_points},
            measurement_path=run_ref,
            dataset="carla_py123d",
        )


def get_front_camera_image(scene) -> np.ndarray:
    """Full-resolution, uncropped front camera frame for a scene's iteration
    0 -- exactly what ends up in DatasetOutput.image_ff_org_size /
    DrivingLabel.image_ff_org regardless of cut_bottom_quarter (that crop
    only applies to image_ff, the model's actual vision-encoder input, built
    separately in __getitem__). Used both there and by
    collect_embeddings_carla_py123d's collected scenes to re-fetch the same
    image on demand instead of saving it at collection time."""
    from py123d.datatypes import CameraID

    camera = scene.get_camera_at_iteration(0, camera_id=CameraID.PCAM_F0)
    return camera.image  # HWC uint8 RGB, already decoded


def find_scene_by_ref(data_root: Optional[str], log_name: str, scene_uuid: str):
    """Re-fetches the exact py123d Scene a collected embedding scene came from,
    given the log_name/scene_uuid saved in its metadata.json (see
    build_py123d_metadata / decode_scene_ref). Not saved at collection time
    (raw py123d Scene objects aren't serializable/needed downstream) --
    re-derived here on demand for e.g. BEV map visualization.

    Uses the identical SceneFilter as Py123dCarlaDataset.__init__ (same
    dataset/duration/modality params) so the log is windowed into scenes the
    same way and scene_uuid actually matches back to the same scene."""
    from py123d.api import get_filtered_scenes
    from py123d.api.scene.scene_filter import SceneFilter

    scene_filter = SceneFilter(
        datasets=["carla"],
        log_names=[log_name],
        target_iteration_duration_s=ITERATION_DURATION_S,
        future_duration_s=ITERATION_DURATION_S * PRED_LEN,
        history_duration_s=0.0,
        required_scene_modalities=[f"camera.{FRONT_CAMERA_ID}", "ego_state_se3"],
    )
    scenes = get_filtered_scenes(scene_filter, data_root=data_root)
    for scene in scenes:
        if str(scene.scene_uuid) == scene_uuid:
            return scene
    return None


BEV_MAP_LAYERS = [
    "lane",
    "lane_group",
    "intersection",
    "crosswalk",
    "walkway",
    "carpark",
    "generic_drivable",
    "road_edge",
    "road_line",
]


def get_bev_map_ego_frame(scene, radius: float = 60.0) -> Optional[dict]:
    """Queries py123d's real HD map (lanes, road edges/lines, crosswalks,
    etc. -- see dynamo_py123d/tutorials/02_map_tutorial.ipynb) within `radius`
    meters of the scene's iteration-0 ego position, and transforms every map
    object's polygon/polyline geometry from world frame into the same
    rear-axle-relative ego frame as our predicted/expert route and waypoints
    (world_to_ego_frame, matching __getitem__'s own origin_xy/origin_yaw).

    Returns None if this scene/log has no map available (map_is_per_log
    datasets don't all ship one), otherwise a dict of
    {layer_name: [ (Nx2 ego-frame ndarray, is_surface) , ... ]}.
    """
    from py123d.datatypes import BaseMapLineObject, BaseMapSurfaceObject
    from py123d.geometry import Point3D

    map_api = scene.get_map_api()
    if map_api is None:
        return None

    ego0 = scene.get_ego_state_se3_at_iteration(0)
    origin_xy = np.array([ego0.rear_axle_se3.x, ego0.rear_axle_se3.y])
    origin_yaw = ego0.rear_axle_se3.yaw
    query_point = Point3D(origin_xy[0], origin_xy[1], ego0.rear_axle_se3.z)

    map_object_dict = map_api.get_map_objects_in_radius(point=query_point, radius=radius, layers=BEV_MAP_LAYERS)

    result = {}
    for layer, objects in map_object_dict.items():
        layer_name = layer.name.lower() if hasattr(layer, "name") else str(layer)
        shapes = []
        for obj in objects:
            if isinstance(obj, BaseMapSurfaceObject):
                world_xy = np.asarray(obj.shapely_polygon.exterior.coords)[:, :2]
                is_surface = True
            elif isinstance(obj, BaseMapLineObject):
                world_xy = np.asarray(obj.polyline_2d.array)[:, :2]
                is_surface = False
            else:
                continue
            ego_xy = np.array([world_to_ego_frame(p, origin_xy, origin_yaw) for p in world_xy])
            shapes.append((ego_xy, is_surface))
        result[layer_name] = shapes
    return result


def build_py123d_metadata(batch, b_idx: int, text_output: str, num_valid_tokens: int) -> dict:
    """metadata_builder for EmbeddingCollector (see collect_embeddings.py) when
    the source dataset is Py123dCarlaDataset instead of database/simlingo/."""
    from simlingo_training.models.driving import decode_uint8

    run_ref = decode_uint8(batch.run_id)[b_idx]
    info = decode_scene_ref(run_ref)

    prompt = None
    if batch.driving_input.prompt.language_string is not None:
        prompt = batch.driving_input.prompt.language_string[b_idx]

    return {
        "source": "py123d_carla",
        "dataset": info.get("dataset"),
        "split": info.get("split"),
        "log_name": info.get("log_name"),
        "scene_uuid": info.get("scene_uuid"),
        "iteration": info.get("iteration"),
        "vehicle_speed": info.get("vehicle_speed"),
        "vehicle_pos": info.get("vehicle_pos"),
        "prompt": prompt,
        "ground_truth_text": None,  # no ground truth commentary exists for py123d/CARLA scenes
        "text_output": text_output,
        "num_valid_tokens": num_valid_tokens,
        "embedding_type": "commentary",
    }
