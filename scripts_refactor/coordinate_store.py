import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple
from urllib.parse import quote

import numpy as np


CoordinateMap = Mapping[Any, Any]


def normalize_coordinate_array(coords: Any) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape [L,3], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def coordinate_filename(protein_id: Any, fmt: str = "npy") -> str:
    suffix = ".npy" if fmt == "npy" else ".pkl"
    return quote(str(protein_id), safe="") + suffix


def write_coordinate_record(output_dir: str | os.PathLike[str], protein_id: Any, coords: Any, fmt: str = "npy") -> str:
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    arr = normalize_coordinate_array(coords)
    filename = coordinate_filename(protein_id, fmt=fmt)
    full_path = os.path.join(output_dir, filename)
    if fmt == "npy":
        np.save(full_path, arr, allow_pickle=False)
    elif fmt == "pkl":
        with open(full_path, "wb") as out:
            pickle.dump(arr, out, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError("fmt must be 'npy' or 'pkl'")
    return filename


def write_coordinate_index(index: Mapping[str, str], output_dir: str | os.PathLike[str]) -> str:
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "_index.json")
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(dict(index), fh, indent=2, ensure_ascii=False)
    return index_path


def export_coordinate_shards_from_mapping(payload: CoordinateMap, output_dir: str | os.PathLike[str], fmt: str = "npy") -> Tuple[int, str]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Coordinate payload must contain a mapping, got {type(payload).__name__}")
    index: Dict[str, str] = {}
    count = 0
    for protein_id, coords in payload.items():
        index[str(protein_id)] = write_coordinate_record(output_dir, protein_id, coords, fmt=fmt)
        count += 1
    index_path = write_coordinate_index(index, output_dir)
    return count, index_path



def _list_existing_temp_shards(temp_shard_dir: Path) -> list[str]:
    if not temp_shard_dir.exists():
        return []
    return sorted([child.name for child in temp_shard_dir.iterdir() if child.is_file() and child.name.startswith('shard_') and child.suffix == '.pkl'])


def export_coordinate_shards_from_temp_shards(
    temp_shard_dir: str | os.PathLike[str],
    shard_paths: Iterable[str],
    buffered_records: Mapping[Any, Any],
    output_dir: str | os.PathLike[str],
    fmt: str = "npy",
    reset_output_dir: bool = False,
) -> Tuple[int, str]:
    output_dir = Path(output_dir)
    if reset_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index: Dict[str, str] = {}
    count = 0
    temp_shard_dir = Path(temp_shard_dir)
    shard_paths = list(shard_paths)
    existing_shards = _list_existing_temp_shards(temp_shard_dir)
    missing = [name for name in shard_paths if name not in existing_shards]
    if missing:
        if not existing_shards:
            raise FileNotFoundError(f"Expected shard file missing during final export: {temp_shard_dir / missing[0]}")
        shard_paths = existing_shards
    elif not shard_paths:
        shard_paths = existing_shards
    for shard_name in shard_paths:
        shard_path = temp_shard_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Expected shard file missing during final export: {shard_path}")
        with open(shard_path, "rb") as fh:
            shard_data = pickle.load(fh)
        if not isinstance(shard_data, Mapping):
            raise ValueError(f"Shard {shard_path} does not contain a mapping")
        for protein_id, coords in shard_data.items():
            index[str(protein_id)] = write_coordinate_record(output_dir, protein_id, coords, fmt=fmt)
            count += 1
    for protein_id, coords in buffered_records.items():
        index[str(protein_id)] = write_coordinate_record(output_dir, protein_id, coords, fmt=fmt)
        count += 1
    index_path = write_coordinate_index(index, output_dir)
    return count, index_path
