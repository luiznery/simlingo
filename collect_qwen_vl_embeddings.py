"""
Same data source and output format as collect_embeddings_carla_py123d.py, but
running Qwen3-VL-8B-Instruct instead of SimLingo, and asking it to describe
the scene in free-form language instead of running SimLingo's own trained
commentary/waypoint heads.

Inputs are exactly what SimLingo's own model receives for these scenes --
reuses Py123dCarlaDataset's DatasetOutput unchanged, so the front camera
image (image_ff, same cut_bottom_quarter crop), ego speed, and the two
target waypoints (near ~1s / far full-horizon, ego-frame) all come from the
identical code path. Qwen3-VL has no equivalent of SimLingo's <TARGET_POINT>
placeholder-embedding injection, so speed/target points are given as plain
numbers in the prompt text instead, alongside the real image.

Output layout is unchanged from the SimLingo/py123d pipeline (manifest.json +
scenes/<id>/{embeddings.pt, token_ids.pt, metadata.json}), reusing
EmbeddingCollector's save/manifest/padding logic directly -- see
build_qwen_vl_metadata()'s docstring for the few fields that necessarily
differ (a VL captioner has no trajectory-prediction head).

The "embedding" saved per generated token is the model's own final-layer
hidden state at that step (what feeds its LM head), captured via
generate(output_hidden_states=True) -- no monkey-patching needed, unlike
SimLingo's custom greedy_sample (see collect_embeddings.py).

Needs env_sl_qwenvl (see setup_env_sl_qwenvl.sh), not env_sl -- env_sl's
transformers==4.46.3 predates Qwen3-VL support.

Run from the simlingo/ repo root, with $PY123D_DATA_ROOT set, e.g.:

    python collect_qwen_vl_embeddings.py \\
        --output-dir embeddings_dataset_qwen_vl_carla_py123d --max-scenes 200

To collect over exactly the same scenes as an existing SimLingo/py123d run
(e.g. embeddings_dataset_carla_py123d_smoketest), so the two datasets line up
scene-for-scene, pass --reference-dataset instead of --max-scenes/--log-names:

    python collect_qwen_vl_embeddings.py \\
        --output-dir embeddings_dataset_qwen_vl_carla_py123d_smoketest \\
        --reference-dataset embeddings_dataset_carla_py123d_smoketest

See collect_qwen_vl_config.yaml for a config-file based equivalent.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from collect_embeddings import DTYPE_MAP, EmbeddingCollector, write_run_config
from py123d_carla_dataset import Py123dCarlaDataset, decode_scene_ref

DEFAULT_SETTINGS = {
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "output_dir": "embeddings_dataset_qwen_vl_carla_py123d",
    "data_root": None,  # falls back to $PY123D_DATA_ROOT
    "log_names": None,  # explicit list of py123d CARLA log names, null = all
    "reference_dataset": None,  # path to an existing embeddings dataset dir
    # (e.g. one produced by collect_embeddings_carla_py123d.py) -- if set,
    # restricts collection to exactly that dataset's (log_name, scene_uuid)
    # scenes instead of drawing a fresh/independent sample, so this run lines
    # up scene-for-scene with an existing SimLingo run. Overrides log_names.
    "max_scenes": None,  # if reference_dataset is also set, caps how many of
    # its matching scenes to use (first N), rather than capping the
    # independent py123d scene pool.
    "scene_cache_path": "py123d_carla_scene_cache.json",  # same cache file/key
    # scheme as collect_embeddings_carla_py123d.py -- the underlying scene
    # set is identical (same Py123dCarlaDataset), so a cache built by one
    # script is reusable by the other as long as log_names/seed match.
    "target_point_mode": "trajectory",  # trajectory (default) | route (real
    # map-based reconstruction, see py123d_carla_dataset.py /
    # py123d_route_target_points.py -- falls back to trajectory per-scene
    # when unavailable). route costs real extra I/O per scene -- test with
    # --max-scenes small before a full run.
    "cut_bottom_quarter": True,
    "num_workers": 4,
    "resume": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "bf16",  # fp32 | fp16 | bf16 -- bf16 default here (unlike
    # collect_embeddings.py's fp32): Qwen3-VL-8B is much larger than
    # InternVL2-1B, and attn_implementation="sdpa" below means no InternViT-
    # style flash-attn dtype restriction is forcing fp32 in this pipeline.
    "max_new_tokens": 100,
    "pad_to_dim": None,  # null = use the model's actual hidden size, no padding
    "seed": 42,
    "manifest_flush_every": 25,
    "instruction": (
        "Describe this driving scene: identify the relevant objects present "
        "(other vehicles, pedestrians, cyclists, traffic signs or lights, "
        "lane markings, road edges) and any events or hazards that could "
        "affect the ego vehicle's upcoming trajectory. "
        "Be direct and concise -- keep your entire response under 100 tokens."
    ),
}


# --------------------------------------------------------------------------
# Prompt + metadata construction
# --------------------------------------------------------------------------

def build_prompt_text(speed: float, target_points, instruction: str) -> str:
    near, far = target_points[0], target_points[1]
    return (
        f"Current ego speed: {speed:.1f} m/s. "
        f"Target waypoint (ego-frame meters, x=forward, y=right): "
        f"near ({near[0]:.1f}, {near[1]:.1f}), far ({far[0]:.1f}, {far[1]:.1f}). "
        f"{instruction}"
    )


def _to_plain_list(value):
    """item.waypoints/item.path are raw numpy arrays (DatasetOutput is built
    before SimLingo's own dl_collate_fn tensor-stacking) -- json.dump can't
    serialize ndarrays directly, so this must go through numpy's own
    .tolist(), not the list() builtin (which would just give a list of
    non-serializable numpy scalars/arrays)."""
    if value is None:
        return None
    return np.asarray(value).tolist()


def build_qwen_vl_metadata(item, prompt_text: str, text_output: str, num_valid_tokens: int) -> Dict:
    """metadata_builder equivalent of py123d_carla_dataset.build_py123d_metadata
    for this pipeline -- same fields, since it's the same Py123dCarlaDataset
    scene and the same encode_scene_ref/decode_scene_ref provenance string
    (item.measurement_path). Two differences from the SimLingo/py123d run's
    metadata.json:
      - "predicted_route"/"predicted_speed_wps" are null: those are
        SimLingo's own trajectory-regression-head output, which a VL
        captioner has no equivalent of.
      - "embedding_type" is "scene_description" instead of "commentary", to
        distinguish this dataset's language task from SimLingo's trained
        commentary/qa/dreaming categories without changing the key itself
        (EmbeddingDataset's embedding_type filter works on any value).
    Everything else -- including expert_waypoints/expert_route, which come
    from the scene's own recorded future trajectory, not from either model
    -- matches exactly.
    """
    info = decode_scene_ref(item.measurement_path)
    return {
        "source": "py123d_carla",
        "dataset": info.get("dataset"),
        "split": info.get("split"),
        "log_name": info.get("log_name"),
        "scene_uuid": info.get("scene_uuid"),
        "iteration": info.get("iteration"),
        "vehicle_speed": info.get("vehicle_speed"),
        "vehicle_pos": info.get("vehicle_pos"),
        "prompt": prompt_text,
        "ground_truth_text": None,
        "text_output": text_output,
        "num_valid_tokens": num_valid_tokens,
        "embedding_type": "scene_description",
        "predicted_route": None,
        "predicted_speed_wps": None,
        "expert_waypoints": _to_plain_list(item.waypoints),
        "expert_route": _to_plain_list(item.path),
        "target_point_source": (item.eval_infos or {}).get("target_point_source"),
    }


# --------------------------------------------------------------------------
# Model loading + generation
# --------------------------------------------------------------------------

def load_qwen_vl_model(model_id: str, device: str, dtype: torch.dtype):
    """Loads Qwen3-VL-8B-Instruct. attn_implementation="sdpa" is explicit and
    deliberate: it sidesteps env_sl_qwenvl's copied flash-attn build (compiled
    against env_sl's torch==2.2.0, and never upgraded -- see
    setup_env_sl_qwenvl.sh), which this model doesn't need and shouldn't
    depend on."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa", trust_remote_code=True
    )
    model.to(device)
    model.eval()
    return model, processor


def get_hidden_size(model) -> int:
    cfg = model.config
    if hasattr(cfg, "get_text_config"):
        return cfg.get_text_config().hidden_size
    if hasattr(cfg, "text_config"):
        return cfg.text_config.hidden_size
    return cfg.hidden_size


def generate_scene_description(model, processor, image: Image.Image, prompt_text: str, device, dtype, max_new_tokens: int, pad_token_id: int):
    """Runs one image+text scene through Qwen3-VL and returns
    (generated_token_ids [1, n_gen] cpu long, last_layer_hidden [1, n_gen,
    hidden] cpu float, decoded text). n_gen may be < max_new_tokens if EOS was
    generated early -- generate() stops the loop itself for a single
    (non-batched) sequence, no manual truncation needed here."""
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = gen_out.sequences[:, prompt_len:]  # [1, n_gen]

    # One hidden-state tuple per generated step; last layer, last (only
    # newly-relevant) sequence position each time -- see module docstring.
    last_layer_hidden = [step_hs[-1][:, -1, :] for step_hs in gen_out.hidden_states]
    embeddings = torch.stack(last_layer_hidden, dim=1)  # [1, n_gen, hidden]

    text_output = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_ids.detach().cpu(), embeddings.detach().float().cpu(), text_output


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_reference_scene_keys(reference_dir) -> Set[Tuple[str, str]]:
    """Reads an existing embeddings dataset directory (e.g. produced by
    collect_embeddings_carla_py123d.py) and returns the (log_name, scene_uuid)
    pair for each of its scenes -- the same identity pair
    Py123dCarlaDataset.__init__'s own scene-cache loading matches scenes by
    (see its `by_ref = {(s.log_name, str(s.scene_uuid)): s for s in scenes}`).

    manifest.json entries alone aren't enough: their "route"/"frame_id" are
    build_py123d_metadata's log_name/iteration fallbacks, not scene_uuid (see
    EmbeddingCollector.save_scene) -- so this reads each scene's own
    metadata.json instead, which does carry scene_uuid.
    """
    reference_dir = Path(reference_dir)
    with open(reference_dir / "manifest.json") as f:
        manifest = json.load(f)

    keys = set()
    for scene in manifest["scenes"]:
        metadata_path = reference_dir / "scenes" / scene["id"] / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        log_name, scene_uuid = metadata.get("log_name"), metadata.get("scene_uuid")
        if log_name and scene_uuid:
            keys.add((log_name, scene_uuid))
    return keys


def restrict_to_reference_scenes(dataset: Py123dCarlaDataset, reference_keys: Set[Tuple[str, str]]) -> Subset:
    """Subsets an already-built Py123dCarlaDataset down to exactly the scenes
    named in reference_keys, matched by (log_name, scene_uuid) identity --
    not by re-deriving the same shuffle/seed order, which would silently
    break if py123d's underlying scene listing order ever changed."""
    matching_indices = [
        i for i, scene in enumerate(dataset.scenes)
        if (scene.log_name, str(scene.scene_uuid)) in reference_keys
    ]
    missing = len(reference_keys) - len(matching_indices)
    if missing > 0:
        print(f"WARNING: {missing} reference scene(s) not found after re-validation (data changed?)")
    return Subset(dataset, matching_indices)


def get_py123d_data_loader(
    data_root, log_names, max_scenes, cut_bottom_quarter, num_workers, seed, scene_cache_path, reference_dataset=None,
    target_point_mode="trajectory",
):
    reference_keys = None
    if reference_dataset is not None:
        reference_keys = load_reference_scene_keys(reference_dataset)
        # Narrow validation to just the logs the reference scenes actually
        # come from, instead of the whole py123d CARLA corpus -- and force
        # full (max_scenes=None) validation so every matching scene in those
        # logs is found, not just an early-stopped lazy subset of them (see
        # Py123dCarlaDataset.__init__). max_scenes, if given, is applied
        # afterward as a cap on the matched scenes instead.
        log_names = sorted({log_name for log_name, _ in reference_keys})
        print(f"--reference-dataset {reference_dataset}: matching {len(reference_keys)} scene(s) across {len(log_names)} log(s)")
        max_scenes_for_pool = None
    else:
        max_scenes_for_pool = max_scenes

    dataset = Py123dCarlaDataset(
        data_root=data_root,
        log_names=log_names,
        max_scenes=max_scenes_for_pool,
        cut_bottom_quarter=cut_bottom_quarter,
        seed=seed,
        scene_cache_path=scene_cache_path,
        target_point_mode=target_point_mode,
    )

    if reference_keys is not None:
        dataset = restrict_to_reference_scenes(dataset, reference_keys)
        if max_scenes is not None:
            dataset = Subset(dataset, range(min(max_scenes, len(dataset))))

    # batch_size=1: DatasetOutput has conversation/placeholder_values fields
    # that default collate can't stack; SimLingo's own pipeline sidesteps
    # this with dl_collate_fn, we sidestep it by just not batching.
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False,
        collate_fn=lambda batch: batch[0], pin_memory=False,
    )
    return loader, len(dataset)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML file with default settings")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None, help="py123d $PY123D_DATA_ROOT override")
    parser.add_argument("--log-names", type=str, default=None, help="Comma-separated py123d CARLA log names")
    parser.add_argument(
        "--reference-dataset", type=str, default=None,
        help="Path to an existing embeddings dataset dir (e.g. from collect_embeddings_carla_py123d.py) -- "
        "restricts collection to exactly that dataset's (log_name, scene_uuid) scenes, overriding --log-names",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--scene-cache-path", type=str, default=None)
    parser.add_argument(
        "--target-point-mode", choices=["trajectory", "route", "drivor"], default=None,
        help="trajectory (default): approximate from the scene's own future ego motion. "
        "route/drivor: real map-based reconstructions (route: near/far ~1m apart; drivor: near/far "
        "tens of meters apart), falling back to trajectory per-scene when unavailable.",
    )
    parser.add_argument("--cut-bottom-quarter", action="store_true", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--pad-to-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifest-flush-every", type=int, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    return parser


def resolve_settings(args: argparse.Namespace) -> dict:
    settings = DEFAULT_SETTINGS.copy()
    if args.config:
        with open(args.config) as f:
            settings.update(yaml.safe_load(f) or {})
    for key in list(settings.keys()):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            settings[key] = cli_val
    if isinstance(settings.get("log_names"), str):
        settings["log_names"] = [n.strip() for n in settings["log_names"].split(",") if n.strip()]
    return settings


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    settings = resolve_settings(args)

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_from = 0
    existing_manifest = None
    manifest_path = output_dir / "manifest.json"
    if settings["resume"] and manifest_path.exists():
        with open(manifest_path) as f:
            existing_manifest = json.load(f)
        resume_from = len(existing_manifest.get("scenes", []))
        print(f"Resuming after {resume_from} previously collected scenes.")

    device = settings["device"]
    dtype = DTYPE_MAP[settings["dtype"]]
    model, processor = load_qwen_vl_model(settings["model_id"], device, dtype)

    # Resolved to a concrete number (never left as None) before it's used
    # anywhere else: EmbeddingCollector.extract_embeddings treats
    # pad_to_dim=None as "use the model's actual hidden size" internally, but
    # write_run_config's mismatch-note logic below does `min(actual, pad_to_dim)`
    # unconditionally and would crash on a None pad_to_dim.
    actual_hidden_size = get_hidden_size(model)
    settings["pad_to_dim"] = settings["pad_to_dim"] or actual_hidden_size

    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0

    data_loader, num_scenes_available = get_py123d_data_loader(
        data_root=settings["data_root"],
        log_names=settings["log_names"],
        max_scenes=settings["max_scenes"],
        cut_bottom_quarter=settings["cut_bottom_quarter"],
        num_workers=settings["num_workers"],
        seed=settings["seed"],
        scene_cache_path=settings["scene_cache_path"],
        reference_dataset=settings["reference_dataset"],
        target_point_mode=settings["target_point_mode"],
    )
    print(f"py123d CARLA scenes available after filtering: {num_scenes_available}")

    if resume_from:
        # Dataset order is deterministic (shuffle=False); skip already-collected
        # scenes cheaply by re-wrapping with a Subset instead of re-running them.
        dataset = data_loader.dataset
        remaining = Subset(dataset, range(resume_from, len(dataset)))
        data_loader = DataLoader(
            remaining, batch_size=1, shuffle=False, num_workers=data_loader.num_workers,
            drop_last=False, collate_fn=data_loader.collate_fn, pin_memory=False,
        )

    collector = EmbeddingCollector(
        model=None,  # unused by extract_embeddings/save_scene, see dry_run_test.py
        output_dir=output_dir,
        embedding_type="scene_description",
        pad_token_id=pad_token_id,
        max_new_tokens=settings["max_new_tokens"],
        pad_to_dim=settings["pad_to_dim"],
        manifest_flush_every=settings["manifest_flush_every"],
        resume_scene_counter=resume_from,
    )
    if existing_manifest is not None:
        collector.manifest = existing_manifest["scenes"]

    for item in tqdm(data_loader, desc="Collecting Qwen3-VL scene descriptions"):
        try:
            image = Image.fromarray(np.transpose(item.image_ff[0], (1, 2, 0)))
            prompt_text = build_prompt_text(item.speed, item.target_points, settings["instruction"])

            token_ids, embeddings, text_output = generate_scene_description(
                model, processor, image, prompt_text, device, dtype,
                settings["max_new_tokens"], pad_token_id,
            )
            fixed_tokens, fixed_embeds, num_valid = collector.extract_embeddings(
                {"token_ids": token_ids, "embeddings": embeddings}
            )
            metadata = build_qwen_vl_metadata(item, prompt_text, text_output, num_valid)
            collector.save_scene(fixed_tokens, fixed_embeds, metadata)
        except Exception as exc:  # noqa: BLE001 - keep the run alive on bad scenes
            collector._log_error(exc)
            continue

    collector.write_manifest()

    write_run_config(
        output_dir,
        settings,
        actual_hidden_size=actual_hidden_size,
        num_scenes=collector.scene_counter,
    )

    print(f"Collected {collector.scene_counter} scenes -> {output_dir}")


if __name__ == "__main__":
    main()
