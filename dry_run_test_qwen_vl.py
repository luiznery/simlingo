"""
Synthetic smoke test for the Qwen3-VL scene-description collection pipeline
(collect_qwen_vl_embeddings.py). Same spirit as dry_run_test.py: exercises
everything that does NOT require the actual Qwen3-VL-8B model, transformers,
qwen_vl_utils, py123d, or GPU:

  - build_prompt_text()'s formatting
  - build_qwen_vl_metadata()'s field mapping (incl. decode_scene_ref reuse)
  - _to_plain_list()'s numpy -> JSON-safe conversion
  - get_hidden_size()'s two config shapes (flat vs. get_text_config())
  - EmbeddingCollector.extract_embeddings() reused with pad_to_dim=None (the
    "use the model's actual hidden size, no padding" path -- not exercised by
    dry_run_test.py, which always passes an explicit pad_to_dim)
  - the exact settings["pad_to_dim"] resolution done in main() before calling
    write_run_config(), which guards a real TypeError (min(hidden, None))
    write_run_config hits if pad_to_dim is ever left as None
  - save_scene / write_manifest disk layout with this pipeline's metadata
    shape (log_name/iteration/dataset instead of route_id/frame_id/scenario)
  - EmbeddingDataset round-trip, filtered by embedding_type="scene_description"
  - load_reference_scene_keys() reading a fabricated reference dataset dir,
    and restrict_to_reference_scenes() matching/subsetting by
    (log_name, scene_uuid), including the case where a reference scene is
    missing from the re-validated pool

It does NOT exercise: Qwen3-VL loading/generation, Py123dCarlaDataset.__getitem__
itself, or the real image/chat-template pipeline -- those need env_sl_qwenvl
(see setup_env_sl_qwenvl.sh) plus $PY123D_DATA_ROOT pointing at downloaded
lead123d/CARLA data. Run a real `--max-scenes 5` collection there as the
actual end-to-end check.
"""

import json
import shutil
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

from collect_embeddings import EmbeddingCollector, write_run_config
from collect_qwen_vl_embeddings import (
    DEFAULT_SETTINGS,
    build_prompt_text,
    build_qwen_vl_metadata,
    get_hidden_size,
    load_reference_scene_keys,
    restrict_to_reference_scenes,
    _to_plain_list,
)
from embedding_dataset import EmbeddingDataset
from py123d_carla_dataset import encode_scene_ref


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="qwen_vl_dry_run_"))
    output_dir = tmp_root / "embeddings_dataset_qwen_vl"

    try:
        # --- build_prompt_text ---
        target_points = np.array([[15.2, -1.3], [42.7, 3.1]])
        prompt_text = build_prompt_text(5.281, target_points, DEFAULT_SETTINGS["instruction"])
        check("prompt has speed", "5.3 m/s" in prompt_text)
        check("prompt has near target point", "(15.2, -1.3)" in prompt_text)
        check("prompt has far target point", "(42.7, 3.1)" in prompt_text)
        check("prompt has instruction", DEFAULT_SETTINGS["instruction"] in prompt_text)

        # --- _to_plain_list ---
        check("_to_plain_list(None) is None", _to_plain_list(None) is None)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        plain = _to_plain_list(arr)
        check("_to_plain_list produces a plain list", isinstance(plain, list) and isinstance(plain[0], list))
        check("_to_plain_list is JSON-safe", isinstance(plain[0][0], float))

        # --- get_hidden_size: both config shapes it needs to handle ---
        flat_model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=4096))
        check("get_hidden_size flat config", get_hidden_size(flat_model) == 4096)

        composite_model = types.SimpleNamespace(
            config=types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(hidden_size=4096))
        )
        check("get_hidden_size composite config", get_hidden_size(composite_model) == 4096)

        # --- build_qwen_vl_metadata (incl. decode_scene_ref reuse) ---
        run_ref = encode_scene_ref(
            dataset="carla", split="carla_train", log_name="Town12_Rep0_route0", scene_uuid="abc-123",
            iteration=0, speed=5.281, pos=np.array([-454.463, 3703.322]),
        )
        fake_item = types.SimpleNamespace(
            measurement_path=run_ref,
            waypoints=np.array([[1.0, 0.0], [2.0, 0.1]]),
            path=np.array([[0.0, 0.0], [1.0, 0.0]]),
            eval_infos={"target_point_source": "trajectory"},
        )
        metadata = build_qwen_vl_metadata(fake_item, prompt_text, "A pedestrian is crossing ahead.", 42)
        check("metadata target_point_source", metadata["target_point_source"] == "trajectory")
        check("metadata log_name", metadata["log_name"] == "Town12_Rep0_route0")
        check("metadata embedding_type", metadata["embedding_type"] == "scene_description")
        check("metadata predicted_route is null", metadata["predicted_route"] is None)
        check("metadata predicted_speed_wps is null", metadata["predicted_speed_wps"] is None)
        check("metadata ground_truth_text is null", metadata["ground_truth_text"] is None)
        check("metadata expert_waypoints is a plain list", metadata["expert_waypoints"] == [[1.0, 0.0], [2.0, 0.1]])
        check("metadata expert_route is a plain list", metadata["expert_route"] == [[0.0, 0.0], [1.0, 0.0]])

        # eval_infos=None mirrors a scene collected before target_point_source
        # existed -- must not crash, just report None.
        fake_item_no_eval_infos = types.SimpleNamespace(
            measurement_path=run_ref, waypoints=fake_item.waypoints, path=fake_item.path, eval_infos=None,
        )
        metadata_no_eval_infos = build_qwen_vl_metadata(fake_item_no_eval_infos, prompt_text, "text", 1)
        check("metadata target_point_source is None when eval_infos is None", metadata_no_eval_infos["target_point_source"] is None)
        check("metadata vehicle_speed round-tripped", abs(metadata["vehicle_speed"] - 5.281) < 1e-3)

        # --- EmbeddingCollector.extract_embeddings with pad_to_dim=None ---
        # (the "use the model's actual hidden size" path -- dry_run_test.py
        # only ever exercises an explicit pad_to_dim)
        collector = EmbeddingCollector(
            model=None,
            output_dir=output_dir,
            embedding_type="scene_description",
            pad_token_id=0,
            max_new_tokens=100,
            pad_to_dim=None,
        )
        fake_call = {
            "token_ids": torch.arange(12).unsqueeze(0),
            "embeddings": torch.ones(1, 12, 4096),
        }
        tokens, embeds, n_valid = collector.extract_embeddings(fake_call)
        check("pad_to_dim=None token shape", tuple(tokens.shape) == (100,))
        check("pad_to_dim=None embed shape uses actual hidden size", tuple(embeds.shape) == (100, 4096))
        check("pad_to_dim=None num_valid", n_valid == 12)
        check("pad_to_dim=None real region is 1s", torch.all(embeds[:12, :] == 1.0))
        check("pad_to_dim=None padded rows are 0", torch.all(embeds[12:, :] == 0.0))

        collector.save_scene(tokens, embeds, metadata)
        collector.write_manifest()

        # --- the settings["pad_to_dim"] resolution done in main(), and the
        # write_run_config TypeError it guards against (min(hidden, None)) ---
        settings = dict(DEFAULT_SETTINGS)
        check("DEFAULT_SETTINGS pad_to_dim starts as null", settings["pad_to_dim"] is None)
        settings["pad_to_dim"] = settings["pad_to_dim"] or 4096
        check("resolved pad_to_dim is concrete", settings["pad_to_dim"] == 4096)
        write_run_config(output_dir, settings, actual_hidden_size=4096, num_scenes=1)
        check("config.yaml written", (output_dir / "config.yaml").exists())

        # --- disk layout ---
        check("manifest.json written", (output_dir / "manifest.json").exists())
        check("scene 0000 dir written", (output_dir / "scenes" / "0000" / "embeddings.pt").exists())

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        entry = manifest["scenes"][0]
        # metadata has no route_id/frame_id/scenario keys -- save_scene's
        # manifest entry must fall back to this pipeline's log_name/iteration/
        # dataset instead (same fallback already exercised by
        # collect_embeddings_carla_py123d.py's own scenes).
        check("manifest route falls back to log_name", entry["route"] == "Town12_Rep0_route0")
        check("manifest frame_id falls back to iteration", entry["frame_id"] == 0)
        check("manifest scenario falls back to dataset", entry["scenario"] == "carla")
        check("manifest embedding_type", entry["embedding_type"] == "scene_description")

        # --- load_reference_scene_keys + restrict_to_reference_scenes ---
        # (the --reference-dataset path: matching this run's scenes to an
        # existing collect_embeddings_carla_py123d.py output scene-for-scene)
        ref_dir = tmp_root / "fake_reference_dataset"
        for scene_id, log_name, scene_uuid in [("0000", "LogA", "uuid-1"), ("0001", "LogB", "uuid-2")]:
            scene_dir = ref_dir / "scenes" / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            with open(scene_dir / "metadata.json", "w") as f:
                json.dump({"log_name": log_name, "scene_uuid": scene_uuid}, f)
        with open(ref_dir / "manifest.json", "w") as f:
            json.dump({"scenes": [{"id": "0000"}, {"id": "0001"}]}, f)

        reference_keys = load_reference_scene_keys(ref_dir)
        check("reference keys count", len(reference_keys) == 2)
        check("reference keys content", reference_keys == {("LogA", "uuid-1"), ("LogB", "uuid-2")})

        fake_dataset = types.SimpleNamespace(
            scenes=[
                types.SimpleNamespace(log_name="LogA", scene_uuid="uuid-1"),
                types.SimpleNamespace(log_name="LogB", scene_uuid="uuid-2"),
                types.SimpleNamespace(log_name="LogC", scene_uuid="uuid-3"),  # not in reference set
            ]
        )
        matched = restrict_to_reference_scenes(fake_dataset, reference_keys)
        check("restrict matches only referenced scenes", set(matched.indices) == {0, 1})
        check("restrict excludes non-referenced scene", 2 not in matched.indices)

        # a reference key with no corresponding scene in the re-validated pool
        # (data changed / scene disappeared) must not crash, just be dropped
        keys_with_missing = reference_keys | {("LogD", "uuid-missing")}
        matched_with_missing = restrict_to_reference_scenes(fake_dataset, keys_with_missing)
        check("restrict tolerates a missing reference scene", set(matched_with_missing.indices) == {0, 1})

        # --- EmbeddingDataset round-trip, filtered by this pipeline's embedding_type ---
        dataset = EmbeddingDataset(str(output_dir), embedding_type="scene_description")
        check("dataset length", len(dataset) == 1)
        sample = dataset[0]
        check("loaded token_embeddings shape", tuple(sample["token_embeddings"].shape) == (100, 4096))
        check("loaded metadata text_output", sample["metadata"]["text_output"] == "A pedestrian is crossing ahead.")
        check("loaded metadata log_name", sample["metadata"]["log_name"] == "Town12_Rep0_route0")

        filtered_none = EmbeddingDataset(str(output_dir), embedding_type="commentary")
        check("embedding_type filter excludes SimLingo's 'commentary' tag", len(filtered_none) == 0)

        print("\nAll dry-run checks passed.")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
