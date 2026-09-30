"""
Same embedding-collection pipeline as collect_embeddings.py, but sourcing
scenes from CARLA data loaded through py123d (dynamo_py123d's `lead123d`
dataset, `SceneFilter(datasets=["carla"])`) instead of the SimLingo-curated
database/simlingo/ dataset.

Reuses collect_embeddings.py's model loading, greedy_sample capture, and
EmbeddingCollector unchanged; only the data source and per-scene metadata
shape differ (see py123d_carla_dataset.py for why: no ground-truth commentary
or navigation route exists for arbitrary py123d/CARLA scenes, so every prompt
uses the same "What should the ego do next?" template and the target-point /
waypoint fields are approximated from the scene's own future ego trajectory).

Run from the simlingo/ repo root, with $PY123D_DATA_ROOT set (or pass
--data-root), e.g.:

    python collect_embeddings_carla_py123d.py \
        --checkpoint outputs/2026_06_08_17_14_55_simlingo_seed1/checkpoints/last.ckpt \
        --output-dir embeddings_dataset_carla_py123d --max-scenes 200

See collect_config_carla_py123d.yaml for a config-file based equivalent.
"""

import argparse
from pathlib import Path

import numpy as np
import random
import torch
import yaml
from torch.utils.data import DataLoader

from collect_embeddings import (
    DTYPE_MAP,
    EmbeddingCollector,
    load_simlingo_model,
    write_run_config,
)
from py123d_carla_dataset import Py123dCarlaDataset, build_py123d_metadata

DEFAULT_SETTINGS = {
    "checkpoint": None,
    "experiment_config": None,
    "output_dir": "embeddings_dataset_carla_py123d",
    "data_root": None,  # falls back to $PY123D_DATA_ROOT
    "log_names": None,  # explicit list of py123d CARLA log names, null = all
    "max_scenes": None,
    "scene_cache_path": "py123d_carla_scene_cache.json",  # caches the slow
    # full-dataset validation (see Py123dCarlaDataset.__init__); only used
    # when max_scenes is null. Set to null/None to disable.
    "target_point_mode": "trajectory",  # trajectory (default, unchanged from
    # every dataset collected so far) | route (real map-based reconstruction,
    # see py123d_route_target_points.py -- falls back to trajectory per-scene
    # when unavailable). route costs real extra I/O per scene (re-queries
    # each scene's full log up to 6 times trying different lookahead
    # windows) -- test with --max-scenes small before a full run.
    "cut_bottom_quarter": True,
    "batch_size": 4,
    "num_workers": 4,
    "resume": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "fp32",
    "max_new_tokens": 100,
    "pad_to_dim": None,  # null = use the model's actual hidden size, no padding
    "seed": 42,
    "manifest_flush_every": 25,
}


def get_py123d_carla_data_loader(
    cfg,
    processor,
    data_root,
    log_names,
    max_scenes,
    cut_bottom_quarter,
    batch_size,
    num_workers,
    seed,
    scene_cache_path=None,
    target_point_mode="trajectory",
):
    from simlingo_training.dataloader.datamodule import DataModule

    # Built directly (bypassing hydra.utils.instantiate) since we don't want
    # DataModule.setup()'s SimLingo-dataset-specific predict_dataset wiring --
    # only its dl_collate_fn (image patchification + chat-template
    # tokenization) is reused here, unmodified, against our own Dataset.
    data_module = DataModule(
        base_dataset=cfg.data_module.base_dataset,
        processor=processor,
        predict=True,
        driving_dataset=None,
        dreamer_dataset=None,
        qa_dataset=None,
        insteval_dataset=None,
        batch_size=batch_size,
        num_workers=num_workers,
        train_partitions=None,
        train_partitions_dreamer=None,
        use_global_img=cfg.data_module.use_global_img,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
    )

    dataset = Py123dCarlaDataset(
        data_root=data_root,
        log_names=log_names,
        max_scenes=max_scenes,
        cut_bottom_quarter=cut_bottom_quarter,
        seed=seed,
        scene_cache_path=scene_cache_path,
        target_point_mode=target_point_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=data_module.dl_collate_fn,
        pin_memory=True,
    )
    return loader, len(dataset)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML file with default settings")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None, help="py123d $PY123D_DATA_ROOT override")
    parser.add_argument("--log-names", type=str, default=None, help="Comma-separated py123d CARLA log names")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--scene-cache-path",
        type=str,
        default=None,
        help="Cache file for the full-dataset scene validation (only used when --max-scenes is unset). "
        "Pass an empty string to disable.",
    )
    parser.add_argument(
        "--target-point-mode", choices=["trajectory", "route", "drivor"], default=None,
        help="trajectory (default): approximate from the scene's own future ego motion. "
        "route/drivor: real map-based reconstructions (py123d_route_target_points.py -- route: "
        "near/far ~1m apart, mirrors CARLA's own RoutePlanner index mechanism; drivor: near/far "
        "tens of meters apart, mirrors nav123d's own fixed-arc-length-offset method), falling "
        "back to trajectory per-scene when unavailable -- costs real extra I/O per scene.",
    )
    parser.add_argument("--cut-bottom-quarter", action="store_true", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--pad-to-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manifest-flush-every", type=int, default=None)
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

    if not settings["checkpoint"]:
        parser.error("--checkpoint is required (via CLI or --config)")

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_from = 0
    existing_manifest = None
    manifest_path = output_dir / "manifest.json"
    if settings["resume"] and manifest_path.exists():
        import json

        with open(manifest_path) as f:
            existing_manifest = json.load(f)
        resume_from = len(existing_manifest.get("scenes", []))
        print(f"Resuming after {resume_from} previously collected scenes.")

    model, processor, cfg = load_simlingo_model(
        checkpoint=settings["checkpoint"],
        experiment_config=settings.get("experiment_config"),
        device=settings["device"],
        dtype=DTYPE_MAP[settings["dtype"]],
    )

    # Resolved to a concrete number (never left as None) before it's used
    # anywhere else: EmbeddingCollector.extract_embeddings treats
    # pad_to_dim=None as "use the model's actual hidden size" internally, but
    # write_run_config's mismatch-note logic below does `min(actual, pad_to_dim)`
    # unconditionally and would crash on a None pad_to_dim.
    actual_hidden_size = model.language_model.hidden_size
    settings["pad_to_dim"] = settings["pad_to_dim"] or actual_hidden_size

    data_loader, num_scenes_available = get_py123d_carla_data_loader(
        cfg,
        processor,
        data_root=settings["data_root"],
        log_names=settings["log_names"],
        max_scenes=settings["max_scenes"],
        cut_bottom_quarter=settings["cut_bottom_quarter"],
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        seed=settings["seed"],
        scene_cache_path=settings["scene_cache_path"],
        target_point_mode=settings["target_point_mode"],
    )
    print(f"py123d CARLA scenes available after filtering: {num_scenes_available}")

    if resume_from:
        # Dataset order is deterministic (shuffle=False); skip already-collected
        # scenes cheaply by re-wrapping with a Subset instead of re-running them.
        from torch.utils.data import Subset

        dataset = data_loader.dataset
        remaining = Subset(dataset, range(resume_from, len(dataset)))
        data_loader = DataLoader(
            remaining,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=data_loader.num_workers,
            drop_last=False,
            collate_fn=data_loader.collate_fn,
            pin_memory=True,
        )

    tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0

    collector = EmbeddingCollector(
        model=model,
        output_dir=output_dir,
        embedding_type="commentary",
        pad_token_id=pad_token_id,
        max_new_tokens=settings["max_new_tokens"],
        pad_to_dim=settings["pad_to_dim"],
        device=settings["device"],
        dtype=DTYPE_MAP[settings["dtype"]],
        manifest_flush_every=settings["manifest_flush_every"],
        resume_scene_counter=resume_from,
        metadata_builder=build_py123d_metadata,
    )
    if existing_manifest is not None:
        collector.manifest = existing_manifest["scenes"]

    collector.collect_from_dataset(data_loader)

    write_run_config(
        output_dir,
        settings,
        actual_hidden_size=actual_hidden_size,
        num_scenes=collector.scene_counter,
    )

    print(f"Collected {collector.scene_counter} scenes -> {output_dir}")


if __name__ == "__main__":
    main()
