"""
Synthetic smoke test for the embeddings collection pipeline.

Exercises everything in collect_embeddings.py / embedding_dataset.py that does
NOT require the trained model, hydra, or the extracted CARLA dataset:
  - extract_embeddings() padding/truncation (both under- and over-length
    generations, and hidden_size != pad_to_dim)
  - parse_scene_location / derive_image_path / parse_frame_id path parsing
  - read_vehicle_pos / read_route_id against fabricated measurement json
  - save_scene / write_manifest disk layout
  - EmbeddingDataset round-trip read-back
  - py123d_carla_dataset.py's pure geometry/metadata helpers (world_to_ego_frame,
    equal_spacing_route, encode_scene_ref/decode_scene_ref)

It does NOT exercise: greedy_sample capture, the real dataloader, model
loading, or Py123dCarlaDataset.__getitem__ itself -- those need your full
training environment (torch+transformers+hydra+deepspeed+pytorch_lightning)
plus either an extracted database/simlingo/ or the py123d package installed
with $PY123D_DATA_ROOT pointing at downloaded lead123d/CARLA data. Run a real
`--limit 5` / `--max-scenes 5` collection there as the actual end-to-end check.
"""

import gzip
import json
import shutil
import tempfile
from pathlib import Path

import torch

from collect_embeddings import (
    EmbeddingCollector,
    derive_image_path,
    parse_frame_id,
    parse_scene_location,
    read_route_id,
    read_vehicle_pos,
)
from embedding_dataset import EmbeddingDataset


def make_fake_route(tmp_root: Path):
    route_root = (
        tmp_root
        / "database"
        / "simlingo"
        / "data"
        / "simlingo"
        / "training_parking_lane"
        / "Town12_short"
        / "random_weather_seed_10_balanced_150"
        / "Town12_Rep0_100_route0_01_11_22_20_59"
    )
    (route_root / "measurements").mkdir(parents=True, exist_ok=True)

    measurement = {
        "pos_global": [-454.46392822265625, 3703.322509765625],
        "ego_matrix": [[1, 0, 0, -454.46392822265625], [0, 1, 0, 3703.322509765625], [0, 0, 1, 366.41], [0, 0, 0, 1]],
        "speed": 2.91,
    }
    measurement_path = route_root / "measurements" / "0010.json.gz"
    with gzip.open(measurement_path, "wt") as f:
        json.dump(measurement, f)

    results = {"route_id": "RouteScenario_0_rep0", "scores": {"score_composed": 100.0}}
    with gzip.open(route_root / "results.json.gz", "wt") as f:
        json.dump(results, f)

    return str(measurement_path)


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="simlingo_dry_run_"))
    output_dir = tmp_root / "embeddings_dataset"

    try:
        measurement_path = make_fake_route(tmp_root)

        # --- path parsing helpers ---
        loc = parse_scene_location(measurement_path)
        check("scenario_category parsed", loc["scenario_category"] == "training_parking_lane")
        check("weather_tag parsed", loc["weather_tag"] == "random_weather_seed_10_balanced_150")
        check("route_dir parsed", loc["route_dir"] == "Town12_Rep0_100_route0_01_11_22_20_59")

        check("frame_id parsed", parse_frame_id(measurement_path) == 10)
        check("image_path derived", derive_image_path(measurement_path).endswith("/rgb/0010.jpg"))

        pos = read_vehicle_pos(measurement_path)
        check("vehicle_pos read", pos == [-454.46392822265625, 3703.322509765625, 366.41])

        route_id = read_route_id(loc["route_root"])
        check("route_id read", route_id == "RouteScenario_0_rep0")

        # --- EmbeddingCollector: shape/padding logic ---
        collector = EmbeddingCollector(
            model=None,  # not used by extract_embeddings/save_scene
            output_dir=output_dir,
            embedding_type="commentary",
            pad_token_id=0,
            max_new_tokens=100,
            pad_to_dim=2048,
        )

        # case 1: model stops early via EOS (n_gen=7 < max_new_tokens=100),
        # hidden_size=896 (the real InternVL2-1B width) < pad_to_dim=2048
        fake_call_short = {
            "token_ids": torch.arange(7).unsqueeze(0),
            "embeddings": torch.ones(1, 7, 896),
        }
        tokens_a, embeds_a, n_valid_a = collector.extract_embeddings(fake_call_short)
        check("short-seq token shape", tuple(tokens_a.shape) == (100,))
        check("short-seq embed shape", tuple(embeds_a.shape) == (100, 2048))
        check("short-seq num_valid", n_valid_a == 7)
        check("short-seq real region is 1s", torch.all(embeds_a[:7, :896] == 1.0))
        check("short-seq padded feature cols are 0", torch.all(embeds_a[:7, 896:] == 0.0))
        check("short-seq padded time rows are 0", torch.all(embeds_a[7:, :] == 0.0))
        check("short-seq padded token ids use pad_token_id", torch.all(tokens_a[7:] == 0))

        # case 2: generation runs the full max_new_tokens (n_gen == 100, no EOS)
        fake_call_full = {
            "token_ids": torch.arange(100).unsqueeze(0),
            "embeddings": torch.full((1, 100, 896), 2.0),
        }
        tokens_b, embeds_b, n_valid_b = collector.extract_embeddings(fake_call_full)
        check("full-seq num_valid", n_valid_b == 100)
        check("full-seq token ids preserved", torch.equal(tokens_b, torch.arange(100)))

        # --- metadata + save_scene + manifest ---
        for tokens, embeds, n_valid, text in [
            (tokens_a, embeds_a, n_valid_a, "The ego vehicle should change lanes."),
            (tokens_b, embeds_b, n_valid_b, "A pedestrian is crossing ahead."),
        ]:
            metadata = {
                "route_id": route_id,
                "route_dir": loc["route_dir"],
                "scenario_category": loc["scenario_category"],
                "frame_id": parse_frame_id(measurement_path),
                "vehicle_speed": 2.91,
                "vehicle_pos": pos,
                "weather": loc["weather_tag"],
                "scenario": loc["scenario_category"],
                "measurement_path": measurement_path,
                "image_path": derive_image_path(measurement_path),
                "prompt": "Current speed: 2.9 m/s. What should the ego do next?",
                "ground_truth_text": text,
                "text_output": text,
                "num_valid_tokens": n_valid,
                "embedding_type": "commentary",
            }
            collector.save_scene(tokens, embeds, metadata)
        collector.write_manifest()

        check("manifest.json written", (output_dir / "manifest.json").exists())
        check("scene 0000 dir written", (output_dir / "scenes" / "0000" / "embeddings.pt").exists())
        check("scene 0001 dir written", (output_dir / "scenes" / "0001" / "metadata.json").exists())

        # --- EmbeddingDataset round-trip ---
        dataset = EmbeddingDataset(str(output_dir))
        check("dataset length", len(dataset) == 2)

        sample0 = dataset[0]
        check("loaded token_ids shape", tuple(sample0["token_ids"].shape) == (100,))
        check("loaded token_embeddings shape", tuple(sample0["token_embeddings"].shape) == (100, 2048))
        check("loaded metadata route_id", sample0["metadata"]["route_id"] == "RouteScenario_0_rep0")
        check("loaded metadata text_output", sample0["metadata"]["text_output"] == "The ego vehicle should change lanes.")

        filtered = EmbeddingDataset(str(output_dir), embedding_type="commentary")
        check("embedding_type filter keeps both", len(filtered) == 2)
        filtered_none = EmbeddingDataset(str(output_dir), embedding_type="dreaming")
        check("embedding_type filter excludes non-matching", len(filtered_none) == 0)

        # --- py123d/CARLA adapter: pure geometry + metadata-encoding helpers ---
        # (doesn't touch Py123dCarlaDataset.__getitem__ itself -- that needs the
        # py123d package installed; this covers the math most likely to hide a bug)
        import numpy as np

        from py123d_carla_dataset import decode_scene_ref, encode_scene_ref, equal_spacing_route, world_to_ego_frame

        # facing +x (yaw=0): a point 1 unit ahead in world x is 1 unit ahead in ego x
        p_ahead = world_to_ego_frame(np.array([11.0, 10.0]), np.array([10.0, 10.0]), origin_yaw=0.0)
        check("world_to_ego_frame yaw=0 ahead", np.allclose(p_ahead, [1.0, 0.0]))

        # facing world +y (yaw=pi/2): a point offset in world +x is 90deg
        # clockwise from the heading -- which is the ego's *right* in this
        # pipeline's Y-right convention (see world_to_ego_frame's docstring) --
        # so it lands at ego (0, 1).
        p_rotated = world_to_ego_frame(np.array([11.0, 10.0]), np.array([10.0, 10.0]), origin_yaw=np.pi / 2)
        check("world_to_ego_frame yaw=90deg", np.allclose(p_rotated, [0.0, 1.0], atol=1e-9))

        # the origin itself must always map to (0, 0) regardless of heading
        p_self = world_to_ego_frame(np.array([10.0, 10.0]), np.array([10.0, 10.0]), origin_yaw=0.7)
        check("world_to_ego_frame origin maps to zero", np.allclose(p_self, [0.0, 0.0]))

        straight_line = np.stack([np.arange(12, dtype=float), np.zeros(12)], axis=1)
        route = equal_spacing_route(straight_line)
        check("equal_spacing_route output shape", route.shape == (20, 2))
        check("equal_spacing_route monotonic on a straight line", np.all(np.diff(route[:, 0]) >= -1e-9))

        ref = encode_scene_ref(
            dataset="carla", split="carla_train", log_name="Town12_Rep0_route0", scene_uuid="abc-123",
            iteration=0, speed=5.281, pos=np.array([-454.463, 3703.322]),
        )
        decoded = decode_scene_ref(ref)
        check("scene_ref round-trip log_name", decoded["log_name"] == "Town12_Rep0_route0")
        check("scene_ref round-trip speed", abs(decoded["vehicle_speed"] - 5.281) < 1e-3)
        check("scene_ref round-trip pos", np.allclose(decoded["vehicle_pos"], [-454.463, 3703.322], atol=1e-2))
        check("scene_ref round-trip malformed input falls back", "raw" in decode_scene_ref("not-a-valid-ref"))

        print("\nAll dry-run checks passed.")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
