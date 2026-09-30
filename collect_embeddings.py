"""
Data collection pipeline that runs SimLingo inference over scenes from
database/simlingo/ and saves the language-generation token embeddings
(the hidden states greedy_sample() feeds back into the model at every step,
normally discarded once the token id is sampled) plus per-scene metadata.

Run from the simlingo/ repo root, e.g.:

    python collect_embeddings.py \
        --checkpoint outputs/2026_06_08_17_14_55_simlingo_seed1/checkpoints/last.ckpt \
        --mode commentary --output-dir embeddings_dataset --limit 200

See collect_config.yaml for a config-file based equivalent and the
accompanying README section for full usage notes and caveats (dataset
extraction, actual embedding width, etc).
"""

import argparse
import contextlib
import copy
import gzip
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import yaml

try:
    import ujson
except ImportError:  # pragma: no cover - ujson ships with the training env; json is a safe fallback
    import json as ujson
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

DEFAULT_SETTINGS = {
    "checkpoint": None,
    "experiment_config": None,
    "output_dir": "embeddings_dataset",
    "mode": "commentary",  # commentary | qa | dreaming
    "batch_size": 4,
    "num_workers": 4,
    "limit": None,
    "resume": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "fp32",  # fp32 | fp16 | bf16
    "max_new_tokens": 100,
    "pad_to_dim": None,  # null = use the model's actual hidden size, no padding
    "seed": 42,
    "read_raw_measurements": True,
    "manifest_flush_every": 25,
}

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# --------------------------------------------------------------------------
# Embedding capture: wraps LLM.greedy_sample without touching model code.
# --------------------------------------------------------------------------

class GreedySampleCapture:
    """Monkey-patches an LLM instance's greedy_sample for the duration of the
    context so every call also records the newly generated token embeddings.

    greedy_sample() already builds these embeddings internally (it feeds them
    back in as input_embeds for the next step) but normally only returns the
    sampled token ids to the caller; this recovers the embeddings without
    modifying simlingo_training/models/language_model/llm.py.
    """

    def __init__(self, llm_module):
        self.llm_module = llm_module
        self._original = llm_module.greedy_sample
        self.calls: List[Dict[str, torch.Tensor]] = []

    def __enter__(self):
        original = self._original
        calls = self.calls

        def patched(*args, **kwargs):
            sampled_tokens, input_embeds = original(*args, **kwargs)
            n_gen = sampled_tokens.size(1)
            calls.append(
                {
                    "token_ids": sampled_tokens.detach().to("cpu"),
                    "embeddings": input_embeds[:, -n_gen:, :].detach().to("cpu"),
                }
            )
            return sampled_tokens, input_embeds

        self.llm_module.greedy_sample = patched
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.llm_module.greedy_sample = self._original

    def pop_calls(self) -> List[Dict[str, torch.Tensor]]:
        calls, self.calls = self.calls, []
        return calls


def to_device(obj, device, dtype=None):
    """Recursively moves tensors inside NamedTuples/lists/dicts (DrivingExample
    and friends) onto device, since we bypass the Lightning Trainer that would
    normally do this. If dtype is given, floating-point tensors (images,
    speed, etc.) are also cast to it -- integer/bool tensors (token ids,
    masks) are left alone. Needed because the dataset/collate pipeline always
    builds float32 tensors regardless of the model's own dtype; without this,
    casting the model to fp16/bf16 (see load_simlingo_model) crashes the
    first Conv2d/Linear layer with a weight/input dtype mismatch."""
    if torch.is_tensor(obj):
        if dtype is not None and torch.is_floating_point(obj):
            return obj.to(device=device, dtype=dtype)
        return obj.to(device)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(to_device(v, device, dtype) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device, dtype) for v in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device, dtype) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------
# Metadata helpers - derived from the dataset's on-disk conventions
# (simlingo_training/dataloader/dataset_base.py), not stored in the batch
# tensors, so best-effort reads of the raw measurement/results json are used
# to fill vehicle_pos / route_id.
# --------------------------------------------------------------------------

def parse_scene_location(measurement_path: str) -> Dict[str, Optional[str]]:
    marker = "/data/simlingo/"
    idx = measurement_path.find(marker)
    if idx == -1:
        return {"scenario_category": None, "town_variant": None, "weather_tag": None, "route_dir": None, "route_root": None}

    parts = measurement_path[idx + len(marker):].split("/")
    if len(parts) < 5:
        return {"scenario_category": None, "town_variant": None, "weather_tag": None, "route_dir": None, "route_root": None}

    scenario_category, town_variant, weather_tag, route_dir = parts[0], parts[1], parts[2], parts[3]
    route_root = measurement_path[: idx + len(marker)] + "/".join(parts[:4])
    return {
        "scenario_category": scenario_category,
        "town_variant": town_variant,
        "weather_tag": weather_tag,
        "route_dir": route_dir,
        "route_root": route_root,
    }


def read_vehicle_pos(measurement_path: str) -> Optional[List[float]]:
    try:
        with gzip.open(measurement_path, "rt") as f:
            measurement = ujson.load(f)
        pos = measurement.get("pos_global")
        if pos is None:
            return None
        ego_matrix = measurement.get("ego_matrix")
        z = float(ego_matrix[2][3]) if ego_matrix else 0.0
        return [float(pos[0]), float(pos[1]), z]
    except Exception:
        return None


def read_route_id(route_root: Optional[str]) -> Optional[str]:
    if not route_root:
        return None
    try:
        with gzip.open(os.path.join(route_root, "results.json.gz"), "rt") as f:
            results = ujson.load(f)
        return results.get("route_id")
    except Exception:
        return None


def derive_image_path(measurement_path: str) -> str:
    # Valid when hist_len == 1 (true of every experiment config shipped with
    # this repo): the current measurement frame and the current camera frame
    # share the same index, so a plain path substitution is enough.
    return measurement_path.replace("/measurements/", "/rgb/").replace(".json.gz", ".jpg")


def parse_frame_id(measurement_path: str) -> Optional[int]:
    try:
        return int(os.path.basename(measurement_path).split(".")[0])
    except Exception:
        return None


# --------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------

class EmbeddingCollector:
    def __init__(
        self,
        model,
        output_dir: Path,
        embedding_type: str,
        pad_token_id: int,
        max_new_tokens: int = 100,
        pad_to_dim: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        read_raw_measurements: bool = True,
        manifest_flush_every: int = 25,
        resume_scene_counter: int = 0,
        metadata_builder: Optional[Callable[[object, int, str, int], Dict]] = None,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        (self.output_dir / "scenes").mkdir(parents=True, exist_ok=True)
        self.embedding_type = embedding_type
        self.pad_token_id = pad_token_id
        self.max_new_tokens = max_new_tokens
        self.pad_to_dim = pad_to_dim
        self.device = device
        self.dtype = dtype
        self.read_raw_measurements = read_raw_measurements
        self.manifest_flush_every = manifest_flush_every
        # Lets alternate data sources (e.g. py123d/CARLA scenes, which have no
        # measurements.json to read metadata from) plug in their own metadata
        # shape instead of the SimLingo-database-specific build_metadata below.
        self.metadata_builder = metadata_builder

        self.scene_counter = resume_scene_counter
        self.manifest: List[Dict] = []

    def collect_from_dataset(self, data_loader):
        from simlingo_training.models.driving import decode_uint8

        self._decode_uint8 = decode_uint8

        for batch in tqdm(data_loader, desc=f"Collecting embeddings ({self.embedding_type})"):
            try:
                self._process_batch(batch)
            except Exception as exc:  # noqa: BLE001 - keep the run alive on bad scenes
                self._log_error(exc)
                continue

        self.write_manifest()

    def _process_batch(self, batch):
        batch_device = to_device(batch, self.device, dtype=self.dtype)

        with GreedySampleCapture(self.model.language_model) as capture, torch.no_grad():
            # speed_wps/route are DrivingModel.forward()'s regression-head
            # trajectory prediction (ego-frame meters, see DrivingAdaptor in
            # simlingo_training/models/adaptors/adaptors.py) -- computed on
            # every call regardless of return_language, previously discarded
            # here even though no extra forward pass is needed to get them.
            speed_wps, route, language = self.model.forward(batch_device, return_language=True)

        calls = capture.pop_calls()
        for b_idx, call in enumerate(calls):
            token_ids, embeddings, num_valid = self.extract_embeddings(call)
            text_output = language[b_idx] if b_idx < len(language) else ""
            if self.metadata_builder is not None:
                metadata = self.metadata_builder(batch, b_idx, text_output, num_valid)
            else:
                metadata = self.build_metadata(batch, b_idx, text_output, num_valid)

            metadata["predicted_route"] = route[b_idx].detach().float().cpu().tolist()
            metadata["predicted_speed_wps"] = speed_wps[b_idx].detach().float().cpu().tolist()

            # The actual expert-driven trajectory the training/eval loss is
            # computed against -- PDM-Lite for SimLingo-database scenes, LEAD
            # 123D's own rule-based privileged expert for py123d/CARLA scenes
            # (see README.md's Setup section and py123d_carla_dataset.py's
            # module docstring respectively). Already loaded for every batch
            # regardless of data source (same DrivingLabel fields feed the
            # training loss), just not previously saved here.
            driving_label = batch_device.driving_label
            if driving_label.waypoints is not None:
                metadata["expert_waypoints"] = driving_label.waypoints[b_idx].detach().float().cpu().tolist()
            if driving_label.path is not None:
                metadata["expert_route"] = driving_label.path[b_idx].detach().float().cpu().tolist()

            self.save_scene(token_ids, embeddings, metadata)

    def extract_embeddings(self, capture_call: Dict[str, torch.Tensor]):
        token_ids = capture_call["token_ids"].squeeze(0)  # [N_gen]
        embeddings = capture_call["embeddings"].squeeze(0)  # [N_gen, hidden]
        n_gen = token_ids.size(0)
        hidden = embeddings.size(-1)
        target_dim = self.pad_to_dim or hidden
        n_keep = min(n_gen, self.max_new_tokens)
        n_feat = min(hidden, target_dim)

        fixed_tokens = torch.full((self.max_new_tokens,), self.pad_token_id, dtype=torch.long)
        fixed_tokens[:n_keep] = token_ids[:n_keep]

        fixed_embeds = torch.zeros((self.max_new_tokens, target_dim), dtype=torch.float32)
        fixed_embeds[:n_keep, :n_feat] = embeddings[:n_keep, :n_feat].float()

        return fixed_tokens, fixed_embeds, n_keep

    def build_metadata(self, batch, b_idx: int, text_output: str, num_valid_tokens: int) -> Dict:
        measurement_path = self._decode_uint8(batch.run_id)[b_idx]
        loc = parse_scene_location(measurement_path)

        vehicle_speed = batch.driving_input.vehicle_speed[b_idx].item()
        ground_truth = None
        if batch.driving_label.answer.language_string is not None:
            ground_truth = batch.driving_label.answer.language_string[b_idx]
        prompt = None
        if batch.driving_input.prompt.language_string is not None:
            prompt = batch.driving_input.prompt.language_string[b_idx]

        eval_infos = None
        if batch.driving_label.eval_infos is not None:
            eval_infos = batch.driving_label.eval_infos[b_idx]
        scenario = eval_infos["mode"] if eval_infos else loc["scenario_category"]

        vehicle_pos = None
        route_id = None
        if self.read_raw_measurements:
            vehicle_pos = read_vehicle_pos(measurement_path)
            route_id = read_route_id(loc["route_root"])

        return {
            "route_id": route_id or loc["route_dir"],
            "route_dir": loc["route_dir"],
            "scenario_category": loc["scenario_category"],
            "frame_id": parse_frame_id(measurement_path),
            "vehicle_speed": vehicle_speed,
            "vehicle_pos": vehicle_pos,
            # Real per-frame CARLA weather preset isn't stored in the
            # measurement json; this is the dataset's weather/seed folder tag.
            "weather": loc["weather_tag"],
            "scenario": scenario,
            "measurement_path": measurement_path,
            "image_path": derive_image_path(measurement_path),
            "prompt": prompt,
            "ground_truth_text": ground_truth,
            "text_output": text_output,
            "num_valid_tokens": num_valid_tokens,
            "embedding_type": self.embedding_type,
        }

    def save_scene(self, token_ids: torch.Tensor, embeddings: torch.Tensor, metadata: Dict):
        scene_id = f"{self.scene_counter:04d}"
        scene_dir = self.output_dir / "scenes" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        torch.save(embeddings, scene_dir / "embeddings.pt")
        torch.save(token_ids, scene_dir / "token_ids.pt")
        with open(scene_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # metadata's exact keys depend on which metadata_builder produced it
        # (build_metadata's SimLingo-database shape vs. e.g.
        # build_py123d_metadata's route_id-less shape) -- fall back to each
        # alternate source's closest equivalent field rather than assuming
        # the SimLingo-specific keys are always present.
        self.manifest.append(
            {
                "id": scene_id,
                "route": metadata.get("route_id", metadata.get("log_name")),
                "frame_id": metadata.get("frame_id", metadata.get("iteration")),
                "scenario": metadata.get("scenario", metadata.get("dataset")),
                "embedding_type": metadata["embedding_type"],
            }
        )
        self.scene_counter += 1

        if self.scene_counter % self.manifest_flush_every == 0:
            self.write_manifest()

    def write_manifest(self):
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump({"scenes": self.manifest}, f, indent=2)

    def _log_error(self, exc: Exception):
        msg = f"[{datetime.now().isoformat()}] batch failed: {exc!r}\n"
        with open(self.output_dir / "collection_errors.log", "a") as f:
            f.write(msg)
        print(msg.strip())


# --------------------------------------------------------------------------
# Model / data loading
# --------------------------------------------------------------------------

def _gpu_needs_flash_attn_fallback(device: str) -> bool:
    """flash-attn 2.x kernels only run on Ampere+ GPUs (SM 8.0+); on older
    cards (e.g. V100/SM 7.0) they raise a bare AssertionError deep inside the
    pretrained InternVL2 code, since its config.json hardcodes
    flash_attention_2 / use_flash_attn=True regardless of hardware."""
    if not (isinstance(device, str) and device.startswith("cuda")) or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major < 8


def _dtype_needs_flash_attn_fallback(dtype: torch.dtype) -> bool:
    """Independently of GPU generation: InternViT's custom flash-attn path
    (modeling_intern_vit.py's InternAttention._flash_attn) asserts
    `qkv.dtype in [torch.float16, torch.bfloat16]` unconditionally, so fp32
    (this repo's default collection dtype, for undiminished embedding
    fidelity) always hits that bare AssertionError too, even on Ampere+/
    Hopper GPUs where the kernels themselves would otherwise run fine."""
    return dtype not in (torch.float16, torch.bfloat16)


def _disable_unsupported_flash_attention(model) -> None:
    """Post-load fallback for the custom InternViT vision tower, which reads
    the plain `use_flash_attn` instance attribute fresh every forward call --
    flipping it after the fact is enough, same math minus the unsupported
    CUDA kernel."""
    for module in model.modules():
        if hasattr(module, "use_flash_attn"):
            module.use_flash_attn = False


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Standard GQA key/value head expansion (matches transformers' own
    `repeat_kv` helpers used by their eager/sdpa attention paths) -- needed
    because torch==2.2.0's scaled_dot_product_attention has no native GQA
    support (no `enable_gqa`, added in torch 2.5)."""
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _sdpa_flash_attention_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    is_causal,
    dropout: float = 0.0,
    position_ids=None,
    softmax_scale=None,
    sliding_window=None,
    use_top_left_mask: bool = False,
    softcap=None,
    deterministic=None,
):
    """Drop-in replacement for transformers.modeling_flash_attention_utils
    ._flash_attention_forward, used when the GPU can't run flash-attn 2
    kernels (see _gpu_needs_flash_attn_fallback). Ignores flash-only extras
    (sliding_window, softcap, packed-sequence position_ids) that this
    checkpoint's Qwen2-based LLM doesn't use; mathematically equivalent
    causal self-attention via SDPA otherwise."""
    # flash-attn's layout is (batch, seq_len, num_heads, head_dim); SDPA wants
    # (batch, num_heads, seq_len, head_dim).
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    num_q_heads = query_states.shape[1]
    num_kv_heads = key_states.shape[1]
    if num_kv_heads != num_q_heads:
        n_rep = num_q_heads // num_kv_heads
        key_states = _repeat_kv(key_states, n_rep)
        value_states = _repeat_kv(value_states, n_rep)

    attn_mask = None
    if attention_mask is not None:
        attn_mask = attention_mask[:, None, None, :].to(torch.bool)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=dropout if torch.is_grad_enabled() else 0.0,
        is_causal=is_causal and attn_mask is None and query_length > 1,
        scale=softmax_scale,
    )
    return attn_output.transpose(1, 2)


@contextlib.contextmanager
def _force_sdpa_flash_attention_forward():
    """transformers.models.qwen2.modeling_qwen2 (and every other flash-attn-2
    architecture) does `from ...modeling_flash_attention_utils import
    _flash_attention_forward` at module import time, copying the name into
    its own namespace -- so this only reaches Qwen2's call site if applied
    before that module is first imported, i.e. before the `instantiate(...)`
    call below that triggers it. (Tried gating this via
    AutoModel.from_pretrained(attn_implementation=...) first: transformers'
    _autoset_attn_implementation recurses attn_implementation into
    PretrainedConfig-typed sub-configs, but InternVLChatModel's custom
    trust_remote_code config didn't actually route the setting down into its
    nested Qwen2 llm_config in practice, so the flash-attn class kept getting
    selected regardless. Patching the shared forward function directly
    sidesteps that and works no matter which attention class got picked.)"""
    import transformers.modeling_flash_attention_utils as fa_utils

    original = fa_utils._flash_attention_forward
    fa_utils._flash_attention_forward = _sdpa_flash_attention_forward
    try:
        yield
    finally:
        fa_utils._flash_attention_forward = original


def load_simlingo_model(
    checkpoint: str,
    experiment_config: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Loads a trained DrivingModel + its data_module config, mirroring
    simlingo_training/eval.py's checkpoint -> saved-hydra-config resolution."""
    from hydra import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from transformers import AutoProcessor, AutoTokenizer

    from simlingo_training.config import TrainConfig  # noqa: F401 - registers hydra ConfigStore

    # Needed so hydra.utils.get_original_cwd() (used inside the dataset
    # classes) resolves correctly; mirrors the __main__ blocks in
    # simlingo_training/dataloader/dataset_base.py and dataset_driving.py.
    initialize(config_path="simlingo_training/config", version_base="1.1")
    compose(config_name="config")  # populates HydraConfig; result unused, we load an explicit config below

    ckpt_path = Path(checkpoint)
    if experiment_config is not None:
        cfg = OmegaConf.load(experiment_config)
    else:
        auto_cfg_path = ckpt_path.parent.parent / ".hydra" / "config.yaml"
        if not auto_cfg_path.exists():
            raise FileNotFoundError(
                f"No saved experiment config found at {auto_cfg_path}. "
                "Pass --experiment-config explicitly (e.g. "
                "simlingo_training/config/experiment/simlingo_seed1.yaml composed via config.yaml)."
            )
        cfg = OmegaConf.load(auto_cfg_path)

    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)

    model_type_name = cfg.model.vision_model.variant.split("/")[1]
    cache_dir = f"pretrained/{model_type_name}"

    needs_flash_attn_fallback = _gpu_needs_flash_attn_fallback(device) or _dtype_needs_flash_attn_fallback(dtype)
    instantiate_kwargs = dict(
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False,
    )
    if needs_flash_attn_fallback:
        with _force_sdpa_flash_attention_forward():
            model = instantiate(cfg.model, **instantiate_kwargs)
        _disable_unsupported_flash_attention(model)
    else:
        model = instantiate(cfg.model, **instantiate_kwargs)

    if os.path.isdir(checkpoint):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint)
    else:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    model.to(device)
    model.to(dtype)
    model.eval()

    return model, processor, cfg


def get_data_loader(
    cfg,
    processor,
    mode: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    limit: Optional[int] = None,
    resume_from: int = 0,
):
    from hydra.utils import instantiate

    cfg = copy.deepcopy(cfg)

    cfg.data_module.base_dataset.img_augmentation = False
    cfg.data_module.base_dataset.img_shift_augmentation = False

    if mode == "commentary":
        cfg.data_module.base_dataset.use_commentary = True
        cfg.data_module.base_dataset.use_qa = False
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.insteval_dataset = None
    elif mode == "qa":
        cfg.data_module.base_dataset.use_commentary = False
        cfg.data_module.base_dataset.use_qa = True
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.insteval_dataset = None
    elif mode == "dreaming":
        cfg.data_module.base_dataset.use_safety_flag = True
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected commentary | qa | dreaming.")

    if batch_size is not None:
        cfg.data_module.batch_size = batch_size
    if num_workers is not None:
        cfg.data_module.num_workers = num_workers

    data_module = instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=True,
        _recursive_=False,
    )
    data_module.setup()

    dataset = data_module.predict_dataset
    if resume_from or limit is not None:
        end = len(dataset) if limit is None else min(len(dataset), resume_from + limit)
        dataset = Subset(dataset, range(resume_from, end))

    loader = DataLoader(
        dataset,
        batch_size=cfg.data_module.batch_size,
        shuffle=False,
        num_workers=cfg.data_module.num_workers,
        drop_last=False,
        collate_fn=data_module.dl_collate_fn,
        pin_memory=True,
    )
    return loader


# --------------------------------------------------------------------------
# Reproducibility record
# --------------------------------------------------------------------------

def get_git_commit_hash(repo_dir: str = ".") -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
    except Exception:
        return None


def write_run_config(output_dir: Path, settings: Dict, actual_hidden_size: int, num_scenes: int):
    record = dict(settings)
    record["timestamp"] = datetime.now().isoformat()
    record["git_commit"] = get_git_commit_hash()
    record["num_scenes_collected"] = num_scenes
    record["actual_language_model_hidden_size"] = actual_hidden_size
    if actual_hidden_size != settings["pad_to_dim"]:
        record["note"] = (
            f"Model hidden size ({actual_hidden_size}) != pad_to_dim ({settings['pad_to_dim']}); "
            "embeddings were zero-padded/truncated to pad_to_dim. Real signal only occupies "
            f"the first {min(actual_hidden_size, settings['pad_to_dim'])} dims."
        )
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(record, f, sort_keys=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML file with default settings (see collect_config.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mode", choices=["commentary", "qa", "dreaming"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap on number of scenes to collect")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--pad-to-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--read-raw-measurements",
        action="store_true",
        default=None,
        help="Best-effort read vehicle_pos/route_id from raw measurement/results json (extra disk IO)",
    )
    parser.add_argument("--manifest-flush-every", type=int, default=None)
    return parser


def resolve_settings(args: argparse.Namespace) -> Dict:
    settings = DEFAULT_SETTINGS.copy()
    if args.config:
        with open(args.config) as f:
            settings.update(yaml.safe_load(f) or {})
    for key in list(settings.keys()):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            settings[key] = cli_val
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

    data_loader = get_data_loader(
        cfg,
        processor,
        mode=settings["mode"],
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        limit=settings["limit"],
        resume_from=resume_from,
    )

    tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0

    collector = EmbeddingCollector(
        model=model,
        output_dir=output_dir,
        embedding_type=settings["mode"],
        pad_token_id=pad_token_id,
        max_new_tokens=settings["max_new_tokens"],
        pad_to_dim=settings["pad_to_dim"],
        device=settings["device"],
        dtype=DTYPE_MAP[settings["dtype"]],
        read_raw_measurements=settings["read_raw_measurements"],
        manifest_flush_every=settings["manifest_flush_every"],
        resume_scene_counter=resume_from,
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
