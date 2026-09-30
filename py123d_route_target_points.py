"""
Reconstructs CARLA-leaderboard-style sparse navigation target points
("near"/"far", the model's <TARGET_POINT> input -- see
py123d_carla_dataset.py's compute_target_points) directly from py123d's own
map, instead of approximating them from the scene's own recorded future ego
trajectory.

Why this exists: SimLingo's real target_point/target_point_next
(team_code/autopilot.py + nav_planner.py's RoutePlanner) come from CARLA's
dense GlobalRoutePlanner route (~1m hop_resolution -- see
leaderboard/leaderboard/utils/route_manipulation.py's interpolate_trajectory
default), with RoutePlanner.run_step continuously popping waypoints within
route_planner_min_distance=7.5m of the ego and returning
target_point=index[1] / next_target_point=index[2] of what remains
(team_code/config.py: route_planner_min_distance=7.5,
route_planner_max_distance=50.0). py123d's Scene API has no such route
modality for CARLA (see py123d_carla_dataset.py's module docstring), so
compute_target_points there approximates it from the ego's own future
motion, which diverges from the real signal whenever the expert deviates
from a path the actual route wouldn't have taken.

This module instead reconstructs a real map-based route (mirroring
dynamo_nav123d/src/nav123d/api/utils/route_utils.py's
get_route_lane_group_ids_from_api / _stitch_route_polyline -- ported rather
than imported, since nav123d requires numpy>=2.0, which hard-conflicts with
this repo's pinned numpy==1.23.0 used throughout simlingo_training), and
samples target points from it the same INDEXED way RoutePlanner.run_step
does -- rather than nav123d's own sample_route_target_points, which samples
by fixed arc-length offset instead of by waypoint index.

Pipeline:
  1. get_route_lane_group_ids: shortest path (networkx) on the lane-group
     connectivity graph (map_api.get_layer_graph("lane_group")) between lane
     groups near the ego now and near its recorded position
     ROUTE_LOOKAHEAD_TIME_S ahead (a real recorded future point, used only to
     pick a direction for the graph search -- not a target point itself).
  2. stitch_route_polyline: concatenates those lane groups' centerlines into
     one continuous polyline.
  3. Two alternative final sampling steps, sharing steps 1-2 above:
     - sample_route_target_points_indexed ("route" mode): resamples the
       polyline at a fixed DENSE_ROUTE_STEP_M step (matching CARLA's ~1m
       dense route), finds the index of the farthest already-reached point
       within ROUTE_POP_MIN_DISTANCE_M of the ego (mirrors
       RoutePlanner.run_step's pop loop), and returns the next 2 points by
       INDEX -- matching command_route[1]/command_route[2] in
       team_code/autopilot.py. near/far end up only ~1m apart.
     - sample_route_target_points_drivor ("drivor" mode): nav123d's own
       sample_route_target_points, ported verbatim -- Douglas-Peucker
       simplifies the polyline down to a sparse, CARLA-route-file-like
       vertex set, then picks points a FIXED arc-length offset
       (DRIVOR_POP_DISTANCE_M) past the ego's projected position. near/far
       end up tens of meters apart (DRIVOR_FALLBACK_DISTANCES_M's spacing).

Needs a scene with a future window of at least ROUTE_LOOKAHEAD_TIME_S (60s)
-- far beyond Py123dCarlaDataset's own PRED_LEN window (2s) -- so callers
must re-fetch via py123d_carla_dataset.find_scene_by_ref(...,
future_duration_s=ROUTE_LOOKAHEAD_TIME_S), not use the short-window scenes
Py123dCarlaDataset already iterates.

Standalone / opt-in only: NOT wired into Py123dCarlaDataset or either
collection script. (a) Large datasets already exist using
compute_target_points's approximation, and switching silently would make
runs inconsistent with each other; (b) many scenes won't have 60s of future
log left and must fall back anyway (see valid=0.0 below); (c) this is newly
written, not yet validated at scale.
"""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np

ROUTE_LOOKAHEAD_TIME_S = 60.0  # matches nav123d route_utils._TARGET_ROUTE_LOOKAHEAD_TIME_S
DENSE_ROUTE_STEP_M = 1.0  # matches leaderboard's interpolate_trajectory hop_resolution default
ROUTE_POP_MIN_DISTANCE_M = 7.5  # matches team_code/config.py route_planner_min_distance
ROUTE_POP_MAX_DISTANCE_M = 50.0  # matches team_code/config.py route_planner_max_distance
CANDIDATE_FALLBACK_RADII = [5.0, 20.0, 50.0]  # matches nav123d route_utils._CANDIDATE_FALLBACK_RADII
END_EXTRAPOLATION_TANGENT_M = 3.0  # matches nav123d route_utils._END_TANGENT_M


def _query_lane_group_candidates(map_api, point_2d) -> List[int]:
    """Candidate lane-group IDs at point_2d: containment first, then a radius
    fallback for ego poses slightly off any polygon (e.g. parked at a curb).
    Ported from nav123d.api.utils.route_utils._query_lane_group_candidates."""
    from py123d.datatypes import MapLayer

    contained = map_api.query_object_ids(geometry=point_2d.shapely_point, layers=[MapLayer.LANE_GROUP], predicate="intersects")
    candidate_ids = contained.get(MapLayer.LANE_GROUP, [])
    if candidate_ids:
        return [int(id_) for id_ in candidate_ids]

    for radius in CANDIDATE_FALLBACK_RADII:
        nearby = map_api.get_map_objects_in_radius(point=point_2d, radius=radius, layers=[MapLayer.LANE_GROUP])
        nearby_objects = nearby.get(MapLayer.LANE_GROUP, [])
        if nearby_objects:
            return [int(obj.object_id) for obj in nearby_objects]
    return []


def get_route_lane_group_ids(scene, lookahead_time_s: float = ROUTE_LOOKAHEAD_TIME_S) -> List[int]:
    """Infers on-route lane-group ids via shortest-path search on the
    lane-group connectivity graph, between lane groups near the ego now and
    near its recorded position `lookahead_time_s` ahead. Ported from
    nav123d.api.utils.route_utils.get_route_lane_group_ids_from_api /
    _infer_route_lane_group_ids -- CARLA-only (dropped the nuPlan
    logged-route-roadblocks branch, irrelevant here).

    `lookahead_time_s` must not exceed the future window `scene` was actually
    fetched with (see find_scene_with_route_lookahead) -- these lead123d/
    CARLA route clips run only ~10-30s total (confirmed empirically: one
    scene's log had 0 candidates left at all at future_duration_s=30s), far
    short of nav123d's own ROUTE_LOOKAHEAD_TIME_S=60s default, presumably
    tuned for longer nuPlan logs -- so the default here is a starting point
    to try, not a value every scene can satisfy.

    Returns [] if map access, the far-future ego state (scene's future
    window too short), or a lane-group path can't be found."""
    map_api = scene.get_map_api()
    if map_api is None:
        return []

    ego0 = scene.get_ego_state_se3_at_iteration(0)
    if ego0 is None:
        return []

    target_ts_us = ego0.timestamp.time_us + int(lookahead_time_s * 1e6)
    end_ego = scene.get_ego_state_se3_at_timestamp(target_ts_us, criteria="nearest")
    if end_ego is None:
        return []

    start_candidates = _query_lane_group_candidates(map_api, ego0.center_2d)
    end_candidates = _query_lane_group_candidates(map_api, end_ego.center_2d)
    if not start_candidates or not end_candidates:
        return []

    lane_group_digraph = map_api.get_layer_graph(layer="lane_group")

    best_path: List[int] = []
    for start_id in start_candidates:
        if start_id not in lane_group_digraph:
            continue
        for end_id in end_candidates:
            if end_id not in lane_group_digraph:
                continue
            try:
                path = nx.shortest_path(lane_group_digraph, source=start_id, target=end_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if not best_path or len(path) < len(best_path):
                best_path = [int(node) for node in path]
    return best_path


def stitch_route_polyline(route_lane_group_ids: List[int], map_api, anchor_point_2d):
    """Concatenates the given lane groups' centerlines into one continuous
    (N, 2) world-frame polyline. For each lane group, picks the lane whose
    centerline start is closest to the previous segment's end (or to
    anchor_point_2d for the first). Ported from
    nav123d.api.utils.route_utils._stitch_route_polyline (dropped its
    PolylineSE2/arc_boundaries return values -- not needed here, we resample
    by our own fixed step below instead of nav123d's coarser sparsify).

    Returns None if no usable centerline was found."""
    from py123d.datatypes import LaneGroup, MapLayer

    arrays: List[np.ndarray] = []
    prev_end = np.array([anchor_point_2d.x, anchor_point_2d.y], dtype=np.float64)

    for lg_id in route_lane_group_ids:
        lane_group = map_api.get_map_object_in_layer(lg_id, MapLayer.LANE_GROUP)
        if not isinstance(lane_group, LaneGroup) or not lane_group.lanes:
            continue
        best_lane = min(
            lane_group.lanes,
            key=lambda lane: float(np.linalg.norm(np.asarray(lane.centerline_2d.array[0])[:2] - prev_end)),
        )
        centerline = np.asarray(best_lane.centerline_2d.array)[:, :2].astype(np.float64)
        if centerline.shape[0] < 2:
            continue
        if arrays:
            centerline = centerline[1:]  # drop join duplicate
        arrays.append(centerline)
        prev_end = centerline[-1] if centerline.shape[0] else prev_end

    if not arrays:
        return None
    stitched = np.vstack(arrays)
    return stitched if stitched.shape[0] >= 2 else None


def _polyline_arclens(polyline_xy: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(polyline_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _densify_route(polyline_xy: np.ndarray, step_m: float = DENSE_ROUTE_STEP_M) -> np.ndarray:
    """Resamples a (possibly coarse, map-centerline-derived) polyline at a
    fixed arc-length step, mirroring the ~1m hop_resolution CARLA's own
    GlobalRoutePlanner used to build the dense route RoutePlanner.run_step
    indexes into. This dense, uniform spacing -- not nav123d's coarser,
    CARLA-route-*file*-like sparsify -- is what makes the indexed lookup in
    sample_route_target_points_indexed meaningful."""
    arclens = _polyline_arclens(polyline_xy)
    total_length = arclens[-1]
    n_steps = max(1, int(total_length // step_m))
    sample_s = np.arange(n_steps + 1) * step_m
    sample_s = sample_s[sample_s <= total_length]
    if sample_s[-1] < total_length:
        sample_s = np.append(sample_s, total_length)
    x = np.interp(sample_s, arclens, polyline_xy[:, 0])
    y = np.interp(sample_s, arclens, polyline_xy[:, 1])
    return np.stack([x, y], axis=1)


def _extend_with_extrapolation(dense_xy: np.ndarray, num_extra: int, step_m: float = DENSE_ROUTE_STEP_M) -> np.ndarray:
    """Extends dense_xy past its own end by num_extra points along the final
    heading (tangent from the last END_EXTRAPOLATION_TANGENT_M of the
    route), instead of clamping to the last point -- mirrors nav123d route_
    utils' _interp_or_extrapolate: clamping piles far goals onto the last
    vertex near the route end, which reads to a planner as 'goal reached,
    stop' instead of pulling the car through the finish."""
    if num_extra <= 0:
        return dense_xy
    arclens = _polyline_arclens(dense_xy)
    back_s = max(0.0, arclens[-1] - END_EXTRAPOLATION_TANGENT_M)
    back_xy = np.array([np.interp(back_s, arclens, dense_xy[:, 0]), np.interp(back_s, arclens, dense_xy[:, 1])])
    tangent = dense_xy[-1] - back_xy
    norm = np.linalg.norm(tangent)
    direction = tangent / norm if norm > 1e-6 else np.array([1.0, 0.0])
    extra = dense_xy[-1] + np.outer(np.arange(1, num_extra + 1), direction * step_m)
    return np.vstack([dense_xy, extra])


def _find_pop_index(dense_xy: np.ndarray, dense_arclens: np.ndarray, ego_xy: np.ndarray, min_distance: float, max_distance: float) -> int:
    """Mirrors team_code/nav_planner.py's RoutePlanner.run_step's pop logic,
    adapted for a route stitched fresh per call (see module docstring):
    CARLA's real RoutePlanner.route is a live deque continuously trimmed
    every simulation tick, so its own index 0 is always already near the
    ego -- our stitched route's index 0 can be anywhere (confirmed
    empirically: 24m BEHIND the ego for one real scene, since
    stitch_route_polyline anchors on the chosen lane's own centerline start,
    not the ego's arc-length progress along it; scanning from index 0
    unconditionally, as a first version of this function did, silently
    returned points behind the ego instead of ahead).

    So this first finds the nearest-to-ego index (the ego's actual
    arc-length position on THIS route), then applies the exact same
    'farthest still-within-min_distance, scanning forward, stop past
    max_distance' logic anchored there. Returns nearest_idx itself if no
    farther point qualifies (ego not close enough to anything further along
    -- e.g. off-route or at the route's end)."""
    distances = np.linalg.norm(dense_xy - ego_xy, axis=1)
    nearest_idx = int(np.argmin(distances))
    origin_arclen = dense_arclens[nearest_idx]

    to_pop = nearest_idx
    farthest_in_range = -np.inf
    for i in range(nearest_idx + 1, len(dense_xy)):
        if dense_arclens[i] - origin_arclen > max_distance:
            break
        distance = float(distances[i])
        if farthest_in_range < distance <= min_distance:
            farthest_in_range = distance
            to_pop = i
    return to_pop


def sample_route_target_points_indexed(
    scene,
    num_points: int = 2,
    min_distance: float = ROUTE_POP_MIN_DISTANCE_M,
    max_distance: float = ROUTE_POP_MAX_DISTANCE_M,
    step_m: float = DENSE_ROUTE_STEP_M,
    lookahead_time_s: float = ROUTE_LOOKAHEAD_TIME_S,
    fallback_distances_m=(10.9, 27.3),  # ~ this pipeline's own near/far distances, see compute_target_points
) -> Tuple[np.ndarray, float]:
    """Nav target points = the next `num_points` dense-route waypoints ahead
    of the ego, by INDEX (mirrors RoutePlanner.run_step's
    command_route[1]/command_route[2] -- see team_code/autopilot.py) rather
    than nav123d's fixed arc-length lookahead.

    Returns (xy (num_points, 2), valid) in WORLD frame -- transform with this
    pipeline's own world_to_ego_frame (py123d_carla_dataset.py) to match
    expert_route/expert_waypoints/compute_target_points's convention.
    valid=0.0 with a straight-ahead-of-the-ego (world-frame) fallback at
    fallback_distances_m when no route/map/far-future ego state was found
    (mirrors nav123d's own fallback semantics, just expressed in world frame
    here since the ego-frame transform is the caller's job).

    `lookahead_time_s` must match what `scene`'s own future window actually
    supports -- use find_scene_with_route_lookahead to fetch a scene and a
    matching lookahead together, rather than assuming ROUTE_LOOKAHEAD_TIME_S
    (60s) is available (these CARLA route clips often run far shorter).
    """
    ego0 = scene.get_ego_state_se3_at_iteration(0)
    ego_xy = np.array([ego0.rear_axle_se3.x, ego0.rear_axle_se3.y])
    origin_yaw = ego0.rear_axle_se3.yaw
    heading = np.array([np.cos(origin_yaw), np.sin(origin_yaw)])

    def _fallback():
        pts = np.array([ego_xy + d * heading for d in fallback_distances_m[:num_points]])
        while len(pts) < num_points:
            pts = np.vstack([pts, pts[-1:]])
        return pts, 0.0

    route_ids = get_route_lane_group_ids(scene, lookahead_time_s=lookahead_time_s)
    if not route_ids:
        return _fallback()

    map_api = scene.get_map_api()
    stitched = stitch_route_polyline(route_ids, map_api, ego0.center_2d)
    if stitched is None:
        return _fallback()

    dense_xy = _densify_route(stitched, step_m)
    dense_arclens = _polyline_arclens(dense_xy)

    pop_idx = _find_pop_index(dense_xy, dense_arclens, ego_xy, min_distance, max_distance)
    needed = pop_idx + num_points
    if needed >= len(dense_xy):
        dense_xy = _extend_with_extrapolation(dense_xy, needed - len(dense_xy) + 1, step_m)

    # Mirrors command_route[1]/[2]'s near-route-end collapse in
    # team_code/autopilot.py (fewer than num_points+1 points left -> repeat
    # the last available one instead of indexing out of range).
    points = np.array([dense_xy[min(pop_idx + k, len(dense_xy) - 1)] for k in range(1, num_points + 1)])
    return points, 1.0


# --------------------------------------------------------------------------
# "drivor" mode: nav123d's own sample_route_target_points, ported verbatim
# (dynamo_nav123d/src/nav123d/agents/drivoR_b2d/drivor_py123d_b2d_features.py)
# -- fixed arc-length lookahead past a Douglas-Peucker-simplified route,
# instead of sample_route_target_points_indexed's dense/by-index approach.
# Reuses this module's own get_route_lane_group_ids/stitch_route_polyline
# for the route-inference step (identical to the "indexed" mode's), so the
# only thing that differs between "route" and "drivor" modes is this final
# sampling step.
# --------------------------------------------------------------------------

DRIVOR_POP_DISTANCE_M = 5.0  # matches drivor_py123d_b2d_agent.yaml's target_point_pop_distance_m
# (= RoutePlanner min_distance in nav123d's own comment, though used here as
# a fixed arc-length offset, not a proximity-based pop threshold like
# sample_route_target_points_indexed's min_distance).
DRIVOR_FALLBACK_DISTANCES_M = (15.0, 30.0)  # first two of drivor_py123d_b2d_agent.yaml's
# target_point_lookaheads: [15.0, 30.0, 60.0] -- that config uses 3 target
# points, but this pipeline's <TARGET_POINT><TARGET_POINT> prompt (and
# DrivingInput.target_point) only ever expects 2, so only the first two are
# used as defaults here.
DRIVOR_SIMPLIFY_TOL_M = 2.0  # matches nav123d's _ROUTE_SIMPLIFY_TOL_M
DRIVOR_MAX_GAP_M = 50.0  # matches nav123d's _ROUTE_MAX_GAP_M
DRIVOR_END_TANGENT_M = 3.0  # matches nav123d's _END_TANGENT_M


def _pose_xy(pose) -> Tuple[float, float]:
    """(x, y) of a PoseSE2 returned by PolylineSE2.interpolate. Ported from
    nav123d.agents.drivoR_b2d.drivor_py123d_b2d_features._pose_xy."""
    return float(pose.x), float(pose.y)


def _interp_or_extrapolate_drivor(polyline, s: float) -> Tuple[float, float]:
    """Point at arc-length `s` on `polyline`; past the end, EXTRAPOLATE along
    the final heading rather than clamping (clamping piles the far goals
    onto the last vertex near the destination, which reads to the planner
    as 'goal reached -> stop'). Ported from nav123d.agents.drivoR_b2d.
    drivor_py123d_b2d_features._interp_or_extrapolate."""
    if s <= polyline.length:
        return _pose_xy(polyline.interpolate(s))
    end = np.asarray(_pose_xy(polyline.interpolate(polyline.length)), dtype=np.float64)
    back = np.asarray(_pose_xy(polyline.interpolate(max(0.0, polyline.length - DRIVOR_END_TANGENT_M))), dtype=np.float64)
    tangent = end - back
    n = float(np.linalg.norm(tangent))
    if n < 1e-6:
        return float(end[0]), float(end[1])
    pt = end + (s - polyline.length) * (tangent / n)
    return float(pt[0]), float(pt[1])


def _sparsify_route_drivor(polyline) -> np.ndarray:
    """Coarsen the dense stitched centerline into a sparse, CARLA-like
    (M, 2) vertex set. Mimics the leaderboard's downsample_route: Douglas-
    Peucker (LineString.simplify) keeps corners/junction vertices and drops
    collinear points, then any segment longer than DRIVOR_MAX_GAP_M is
    subdivided so long straights get ~cap-spaced vertices. Always returns
    >= 2 vertices. Ported from nav123d.agents.drivoR_b2d.
    drivor_py123d_b2d_features._sparsify_route."""
    from shapely.geometry import LineString

    xy = np.asarray(polyline.array[:, :2], dtype=np.float64)
    simplified = np.asarray(
        LineString(xy).simplify(DRIVOR_SIMPLIFY_TOL_M, preserve_topology=False).coords, dtype=np.float64
    )
    if simplified.shape[0] < 2:
        return xy
    out: List[np.ndarray] = [simplified[0]]
    for a, b in zip(simplified[:-1], simplified[1:]):
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        n_sub = int(np.ceil(seg_len / DRIVOR_MAX_GAP_M)) if seg_len > DRIVOR_MAX_GAP_M else 1
        for k in range(1, n_sub):
            out.append(a + seg * (k / n_sub))
        out.append(b)
    return np.asarray(out, dtype=np.float64)


def _route_target_arclens_drivor(vertices: np.ndarray, s0: float, num_points: int, pop_distance: float) -> List[float]:
    """Arc-lengths of the next `num_points` route VERTICES past
    `s0 + pop_distance`. `vertices` is the sparse (M, 2) route; the
    cumulative vertex arc-lengths are the candidate goals, so the distance
    to the current (first) goal VARIES as the ego advances and pops one. If
    fewer than num_points vertices remain before the route end, the tail is
    extended past the end by the last vertex gap. Ported from nav123d.
    agents.drivoR_b2d.drivor_py123d_b2d_features._route_target_arclens."""
    seg = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    vertex_arclens = np.concatenate([[0.0], np.cumsum(seg)])
    arclens = [float(a) for a in vertex_arclens if a > s0 + pop_distance][:num_points]
    if len(arclens) < num_points:
        gap = float(seg[-1]) if len(seg) else max(pop_distance, 1.0)
        last = arclens[-1] if arclens else s0 + pop_distance
        while len(arclens) < num_points:
            last += gap
            arclens.append(last)
    return arclens


def sample_route_target_points_drivor(
    scene,
    num_points: int = 2,
    pop_distance: float = DRIVOR_POP_DISTANCE_M,
    lookahead_time_s: float = ROUTE_LOOKAHEAD_TIME_S,
    fallback_distances_m=DRIVOR_FALLBACK_DISTANCES_M,
) -> Tuple[np.ndarray, float]:
    """Nav target points = the next `num_points` route VERTICES past a fixed
    arc-length offset (`pop_distance`) ahead of the ego -- mirrors nav123d's
    own sample_route_target_points (dynamo_nav123d/src/nav123d/agents/
    drivoR_b2d/drivor_py123d_b2d_features.py) verbatim in the sampling step,
    but reuses THIS module's get_route_lane_group_ids/stitch_route_polyline
    for the route-inference step instead of nav123d's own (same reasoning as
    sample_route_target_points_indexed -- nav123d isn't installed here, see
    module docstring).

    Unlike sample_route_target_points_indexed (near/far ~1m apart, matching
    however densely the real dense route happens to be sampled),
    near/far here are a FIXED distance apart (pop_distance's default spacing
    between successive vertices after Douglas-Peucker simplification --
    typically tens of meters, not ~1m).

    Returns (xy (num_points, 2), valid) in WORLD frame -- transform with
    this pipeline's own world_to_ego_frame to match
    expert_route/expert_waypoints/compute_target_points's convention.
    valid=0.0 with a straight-ahead-of-the-ego (world-frame) fallback at
    fallback_distances_m when no route/map/far-future ego state was found.

    `lookahead_time_s` must match what `scene`'s own future window actually
    supports -- use find_scene_with_route_lookahead to fetch a scene and a
    matching lookahead together (these CARLA route clips often run far
    shorter than ROUTE_LOOKAHEAD_TIME_S=60s).
    """
    from py123d.geometry import PolylineSE2

    ego0 = scene.get_ego_state_se3_at_iteration(0)
    ego_xy = np.array([ego0.rear_axle_se3.x, ego0.rear_axle_se3.y])
    origin_yaw = ego0.rear_axle_se3.yaw
    heading = np.array([np.cos(origin_yaw), np.sin(origin_yaw)])

    def _fallback():
        pts = np.array([ego_xy + d * heading for d in fallback_distances_m[:num_points]])
        while len(pts) < num_points:
            pts = np.vstack([pts, pts[-1:]])
        return pts, 0.0

    route_ids = get_route_lane_group_ids(scene, lookahead_time_s=lookahead_time_s)
    if not route_ids:
        return _fallback()

    map_api = scene.get_map_api()
    stitched = stitch_route_polyline(route_ids, map_api, ego0.center_2d)
    if stitched is None:
        return _fallback()

    route_polyline = PolylineSE2.from_array(stitched)
    if route_polyline.length < 1.0:
        return _fallback()

    vertices = _sparsify_route_drivor(route_polyline)
    sparse = PolylineSE2.from_array(vertices)
    s0 = float(sparse.project(ego0.center_2d))
    arclens = _route_target_arclens_drivor(vertices, s0, num_points, pop_distance)
    points = np.array([_interp_or_extrapolate_drivor(sparse, s) for s in arclens])
    return points, 1.0


DEFAULT_LOOKAHEAD_CANDIDATES_S = (60.0, 30.0, 20.0, 15.0, 10.0, 5.0)


def find_scene_with_route_lookahead(
    data_root, log_name: str, scene_uuid: str, candidate_lookaheads_s=DEFAULT_LOOKAHEAD_CANDIDATES_S,
):
    """Fetches the widest-future-window version of this scene that the log
    actually supports, trying `candidate_lookaheads_s` from longest to
    shortest via py123d_carla_dataset.find_scene_by_ref(...,
    future_duration_s=...).

    Needed because get_filtered_scenes requires a scene's ENTIRE requested
    future_duration_s to exist in the log to return it at all -- there's no
    way to ask for "60s, or as much as you've got": confirmed empirically
    that one of these lead123d/CARLA logs had 0 candidate scenes left at
    future_duration_s=30s even though future_duration_s=20s returned dozens,
    i.e. the whole log ran out well under 30s from that point. Real full
    leaderboard routes can run much longer, so a longer lookahead (a longer
    stitched route, closer to the real GlobalRoutePlanner's route length)
    should be preferred whenever the log actually has it.

    Returns (scene, lookahead_used_s), or (None, 0.0) if even the shortest
    candidate isn't available (log too short from this frame, or scene_uuid
    not found at all)."""
    from py123d_carla_dataset import find_scene_by_ref

    for lookahead in candidate_lookaheads_s:
        scene = find_scene_by_ref(data_root, log_name, scene_uuid, future_duration_s=lookahead)
        if scene is not None:
            return scene, lookahead
    return None, 0.0
