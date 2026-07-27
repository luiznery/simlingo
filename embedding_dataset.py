"""
PyTorch Dataset for loading embeddings collected by collect_embeddings.py from
an embeddings_dataset/ directory (manifest.json + scenes/<id>/{embeddings.pt,
token_ids.pt, metadata.json}).
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


def _torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        # older torch versions don't support weights_only
        return torch.load(path)


class EmbeddingDataset(Dataset):
    """Loads per-scene token embeddings + metadata produced by
    collect_embeddings.py.

    Args:
        root_dir: path to the embeddings_dataset/ directory.
        embedding_type: optional filter, e.g. "commentary", "qa", "dreaming".
    """

    def __init__(self, root_dir: str, embedding_type: Optional[str] = None):
        self.root_dir = Path(root_dir)
        with open(self.root_dir / "manifest.json") as f:
            manifest = json.load(f)

        scenes = manifest["scenes"]
        if embedding_type is not None:
            scenes = [s for s in scenes if s.get("embedding_type") == embedding_type]
        self.scenes = scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> Dict:
        scene_id = self.scenes[index]["id"]
        scene_dir = self.root_dir / "scenes" / scene_id

        token_ids = _torch_load(scene_dir / "token_ids.pt")
        token_embeddings = _torch_load(scene_dir / "embeddings.pt")

        with open(scene_dir / "metadata.json") as f:
            metadata = json.load(f)

        return {
            "token_ids": token_ids,
            "token_embeddings": token_embeddings,
            "metadata": metadata,
        }


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "embeddings_dataset"
    dataset = EmbeddingDataset(root)
    print(f"Loaded {len(dataset)} scenes from {root}")
    if len(dataset) > 0:
        sample = dataset[0]
        print("token_ids:", tuple(sample["token_ids"].shape))
        print("token_embeddings:", tuple(sample["token_embeddings"].shape))
        print("metadata:", json.dumps(sample["metadata"], indent=2))
