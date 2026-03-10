"""
Extract alpha-carbon (Cα) coordinates from a folder of PDB(.gz) / mmCIF(.gz) files
and save them as a pkl file suitable for use as `coordinates_path` in the
GLProtein dataset.

Output formats:
    TSV-ID-compatible mode:
        dict[str, numpy.ndarray]
        {raw_fasta_id_token: float32 array of shape [L, 3]}

    Legacy integer-keyed mode:
        dict[int, numpy.ndarray]
        {protein_index: float32 array of shape [L, 3]}

When `--key-mode tsv_id` is used, the extractor reads the FASTA headers and
maps AlphaFold/structure accessions back to the raw FASTA ID tokens written by
`generate_tmvec_pairs_tsv.py` into anchor_id / positive_id / negative_id.

When `--triplets-tsv` is provided, the extractor further restricts work to files whose
mapped TSV ID appears in the training TSV's `anchor_id` column. This is useful when
the FASTA covers more proteins than are actually used as anchors during training.

Usage:
    python scripts_refactor/extract_ca_coords.py \
        --input-dir /path/to/structures \
        --output /path/to/coordinates.pkl \
        --key-mode tsv_id \
        --fasta /path/to/swissprot.fasta \
        --triplets-tsv /path/to/swissprot_triplets.tsv \
        --resume
"""

import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import pickle
import sys
import warnings
import time
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_local_module_path() -> None:
    current_dir = Path(__file__).resolve().parent
    candidate_dirs = [
        current_dir,
        current_dir.parent / 'src_refactor',
    ]
    for candidate in candidate_dirs:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_ensure_local_module_path()

from coordinate_store import (
    export_coordinate_shards_from_temp_shards,
    normalize_coordinate_array,
)

# Suppress noisy BioPython warnings (e.g. discontinuous chains)
warnings.filterwarnings("ignore")

try:
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)
except ImportError:
    sys.exit(
        "biopython is required.  Install it with:\n"
        "    pip install biopython\n"
        "or:\n"
        "    conda install -c conda-forge biopython"
    )


def _open_file(path: Path):
    """Return a file-like object, transparently decompressing .gz files."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def _is_cif(path: Path) -> bool:
    suffixes = {s.lower() for s in path.suffixes}
    return ".cif" in suffixes


def parse_accession(seq_id: str) -> str:
    token = seq_id.split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    if token.startswith("AF-") and "-F1" in token:
        parts = token.split("-")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return token


def read_fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def stem_to_accession(path: Path) -> str:
    stem = path.name.removesuffix(".gz")
    stem = Path(stem).stem
    return parse_accession(stem)


def read_triplet_anchor_ids(path: Path) -> set[str]:
    import csv

    anchor_ids: set[str] = set()
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None or "anchor_id" not in reader.fieldnames:
            raise ValueError(f"Triplet TSV {path} must contain an 'anchor_id' column")
        for row_idx, row in enumerate(reader, start=2):
            anchor_id = (row.get("anchor_id") or "").strip()
            if not anchor_id:
                raise ValueError(f"Triplet TSV {path} has empty anchor_id at row {row_idx}")
            anchor_ids.add(anchor_id)
    return anchor_ids


def extract_ca_coords(path: Path, model_idx: int = 0, chain_id: str | None = None) -> list[list[float]]:
    """
    Parse a PDB or mmCIF(.gz) file and return Cα coordinates as a list of
    [x, y, z] floats in residue order.

    Only standard amino acid residues are included (ATOM records / type=
    'polypeptide' residues); HETATM ligands/waters are skipped.
    """
    parser = MMCIFParser(QUIET=True) if _is_cif(path) else PDBParser(QUIET=True)

    with _open_file(path) as fh:
        content = fh.read()

    structure = parser.get_structure(path.stem, io.StringIO(content))

    try:
        model = list(structure.get_models())[model_idx]
    except IndexError as exc:
        raise ValueError(f"{path}: model index {model_idx} out of range") from exc

    coords: list[list[float]] = []
    for chain in model.get_chains():
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            if "CA" in residue:
                ca = residue["CA"]
                x, y, z = ca.get_vector().get_array().tolist()
                coords.append([x, y, z])

    return coords


def collect_structure_files(input_dir: Path) -> list[Path]:
    files = []
    for f in sorted(input_dir.iterdir()):
        if not f.is_file():
            continue
        name = f.name.lower()
        if (
            name.endswith(".pdb")
            or name.endswith(".pdb.gz")
            or name.endswith(".cif")
            or name.endswith(".cif.gz")
        ):
            files.append(f)
    return sorted(files, key=lambda p: p.name.lower())


def load_index_map(tsv_path: Path) -> dict[str, int]:
    """Read a two-column TSV: filename_stem -> integer index."""
    mapping = {}
    with open(tsv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            stem, idx = parts[0], int(parts[1])
            mapping[stem] = idx
    return mapping


def file_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name.endswith(".pdb.gz"):
        rank = 0
    elif name.endswith(".pdb"):
        rank = 1
    elif name.endswith(".cif.gz"):
        rank = 2
    elif name.endswith(".cif"):
        rank = 3
    else:
        rank = 99
    return (rank, name)


def prefilter_tsv_files(
    files: list[Path],
    accession_to_tsv_id: dict[str, str],
    allowed_tsv_ids: set[str] | None = None,
) -> tuple[list[Path], int, int, set[str]]:
    kept: list[Path] = []
    pre_skipped_not_in_fasta = 0
    pre_skipped_not_in_anchors = 0
    anchors_with_candidate_file: set[str] = set()
    for path in files:
        accession = stem_to_accession(path)
        tsv_id = accession_to_tsv_id.get(accession)
        if tsv_id is None:
            pre_skipped_not_in_fasta += 1
            continue
        if allowed_tsv_ids is not None and tsv_id not in allowed_tsv_ids:
            pre_skipped_not_in_anchors += 1
            continue
        kept.append(path)
        anchors_with_candidate_file.add(tsv_id)
    return kept, pre_skipped_not_in_fasta, pre_skipped_not_in_anchors, anchors_with_candidate_file


def dedupe_tsv_files(files: list[Path]) -> tuple[list[Path], int, list[str]]:
    best_by_accession: dict[str, Path] = {}
    dropped_examples: list[str] = []
    dropped = 0
    for path in files:
        accession = stem_to_accession(path)
        current = best_by_accession.get(accession)
        if current is None or file_priority(path) < file_priority(current):
            if current is not None:
                dropped += 1
                if len(dropped_examples) < 10:
                    dropped_examples.append(f"{current.name} -> {path.name} [{accession}]")
            best_by_accession[accession] = path
        else:
            dropped += 1
            if len(dropped_examples) < 10:
                dropped_examples.append(f"{path.name} dropped; kept {current.name} [{accession}]")
    deduped = sorted(best_by_accession.values(), key=lambda p: p.name.lower())
    return deduped, dropped, dropped_examples


def default_checkpoint_path(output_target: Path) -> Path:
    return Path(str(output_target) + ".checkpoint.pkl")


def default_shard_dir(output_target: Path) -> Path:
    return Path(str(output_target) + ".shards")


def save_pickle_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def save_checkpoint(checkpoint_path: Path, state: dict[str, Any]) -> None:
    save_pickle_atomic(checkpoint_path, state)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    with open(checkpoint_path, "rb") as fh:
        state = pickle.load(fh)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a dict")
    return state


def compute_candidate_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def config_fingerprint(
    args: argparse.Namespace,
    total_found: int,
    total_candidate_files: int,
    pre_skipped: int,
    pre_skipped_not_in_anchors: int,
    dedup_dropped: int,
    candidate_hash: str,
) -> dict[str, Any]:
    return {
        "input_dir": str(args.input_dir.resolve()),
        "output": str(args.output.resolve()),
        "output_format": args.output_format,
        "coordinate_format": getattr(args, "coordinate_format", None),
        "key_mode": args.key_mode,
        "fasta": str(args.fasta.resolve()) if args.fasta else None,
        "triplets_tsv": str(args.triplets_tsv.resolve()) if args.triplets_tsv else None,
        "index_map": str(args.index_map.resolve()) if args.index_map else None,
        "chain": args.chain,
        "model": int(args.model),
        "total_found": int(total_found),
        "total_candidate_files": int(total_candidate_files),
        "pre_skipped": int(pre_skipped),
        "pre_skipped_not_in_anchors": int(pre_skipped_not_in_anchors),
        "dedup_dropped": int(dedup_dropped),
        "candidate_hash": candidate_hash,
    }



def _existing_shard_names(shard_dir: Path) -> list[str]:
    if not shard_dir.exists():
        return []
    shard_names = [child.name for child in shard_dir.iterdir() if child.is_file() and child.name.startswith('shard_') and child.suffix == '.pkl']
    return sorted(shard_names)


def _next_shard_index(shard_paths: list[str], shard_dir: Path) -> int:
    max_index = 0
    for shard_name in list(shard_paths) + _existing_shard_names(shard_dir):
        try:
            stem = Path(shard_name).stem
            idx = int(stem.split('_')[-1])
            max_index = max(max_index, idx)
        except Exception:
            continue
    return max_index + 1


def reconcile_shard_paths(shard_dir: Path, shard_paths: list[str], *, context: str) -> list[str]:
    existing = _existing_shard_names(shard_dir)
    if not shard_paths:
        return existing
    missing = [name for name in shard_paths if name not in existing]
    if not missing:
        return shard_paths
    if not existing:
        raise FileNotFoundError(
            f"{context} references missing shard files and no shard files remain on disk. First missing shard: {shard_dir / missing[0]}"
        )
    print(
        f"[warn] {context} references {len(missing)} missing shard file(s); falling back to the {len(existing)} shard file(s) currently present in {shard_dir}. First missing shard: {shard_dir / missing[0]}"
    )
    return existing


def flush_buffer_to_shard(
    shard_dir: Path,
    buffer_result: dict[Any, np.ndarray],
    shard_paths: list[str],
) -> str | None:
    if not buffer_result:
        return None
    shard_dir.mkdir(parents=True, exist_ok=True)
    next_index = _next_shard_index(shard_paths, shard_dir)
    shard_name = f"shard_{next_index:06d}.pkl"
    shard_path = shard_dir / shard_name
    save_pickle_atomic(shard_path, buffer_result)
    shard_paths.append(shard_name)
    buffer_result.clear()
    gc.collect()
    return shard_name


def checkpoint_state(
    *,
    processed_files: set[str],
    seen_keys: set[Any],
    failed: list[str],
    skipped: int,
    saved_new: int,
    overwritten: int,
    overwrite_examples: list[str],
    shard_paths: list[str],
    buffer_result: dict[Any, np.ndarray],
    fingerprint: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    return {
        "processed_files": sorted(processed_files),
        "seen_keys": list(seen_keys),
        "failed": list(failed),
        "skipped": int(skipped),
        "saved_new": int(saved_new),
        "overwritten": int(overwritten),
        "overwrite_examples": list(overwrite_examples[:10]),
        "shard_paths": list(shard_paths),
        "buffer_result": dict(buffer_result),
        "fingerprint": fingerprint,
        "started_at": float(started_at),
    }


def finalize_output(
    *,
    output_format: str,
    shard_dir: Path,
    shard_paths: list[str],
    buffer_result: dict[Any, np.ndarray],
    output: Path,
    coordinate_format: str,
) -> tuple[int, str | None]:
    if output_format == "pkl":
        final_result: dict[Any, np.ndarray] = {}
        for shard_name in shard_paths:
            shard_path = shard_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Expected shard file missing during final merge: {shard_path}")
            with open(shard_path, "rb") as fh:
                shard_data = pickle.load(fh)
            if not isinstance(shard_data, dict):
                raise ValueError(f"Shard {shard_path} does not contain a dict")
            final_result.update(shard_data)
            del shard_data
        if buffer_result:
            final_result.update(buffer_result)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_pickle_atomic(output, final_result)
        count = len(final_result)
        del final_result
        gc.collect()
        return count, None

    count, index_path = export_coordinate_shards_from_temp_shards(
        temp_shard_dir=shard_dir,
        shard_paths=shard_paths,
        buffered_records=buffer_result,
        output_dir=output,
        fmt=coordinate_format,
        reset_output_dir=False,
    )
    gc.collect()
    return count, index_path


def remove_shard_dir(shard_dir: Path) -> None:
    if not shard_dir.exists():
        return
    for child in shard_dir.iterdir():
        if child.is_file():
            child.unlink()
    shard_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing .pdb / .pdb.gz / .cif / .cif.gz files")
    parser.add_argument("--output", required=True, type=Path, help="Output path. For --output-format pkl this is the final .pkl path; for --output-format sharded this is the final coordinate directory.")
    parser.add_argument(
        "--output-format",
        choices=["pkl", "sharded"],
        default="pkl",
        help="Final coordinate output format. Use sharded to write the low-memory coordinate store directly instead of creating a merged PKL.",
    )
    parser.add_argument(
        "--coordinate-format",
        choices=["npy", "pkl"],
        default="npy",
        help="Per-record file format for --output-format sharded (default: npy). Ignored for --output-format pkl.",
    )
    parser.add_argument(
        "--key-mode",
        choices=["index", "tsv_id"],
        default="index",
        help="Output key mode: integer index (legacy) or raw FASTA ID token matching generate_tmvec_pairs_tsv.py",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help="Optional FASTA used to map structure accessions back to raw FASTA ID tokens",
    )
    parser.add_argument(
        "--triplets-tsv",
        type=Path,
        default=None,
        help="Optional triplet TSV; when provided with --key-mode tsv_id, only anchor_id entries used by training are extracted",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=None,
        help="Optional path to write a JSON coverage report comparing requested triplet anchors against extracted coordinates.",
    )
    parser.add_argument(
        "--missing-anchor-ids-output",
        type=Path,
        default=None,
        help="Optional path to write the full list of triplet anchor IDs missing from extracted coordinates.",
    )
    parser.add_argument(
        "--require-all-triplet-anchors",
        action="store_true",
        help="If set with --triplets-tsv and --key-mode tsv_id, fail at the end if any requested triplet anchor IDs are missing from the extracted coordinate output.",
    )
    parser.add_argument(
        "--index-map",
        type=Path,
        default=None,
        help="Optional TSV mapping filename stem -> integer index (legacy index mode only)",
    )
    parser.add_argument("--chain", type=str, default=None, help="Restrict to this chain ID (default: all)")
    parser.add_argument("--model", type=int, default=0, help="MODEL index to use (default: 0)")
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log extraction progress every N newly processed candidate files for large folders (default: 500)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing extraction checkpoint if present.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to a resume checkpoint file. Defaults to <output>.checkpoint.pkl",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5000,
        help="Save resume checkpoint every N newly processed candidate files (default: 5000)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=5000,
        help="Flush buffered coordinate records to a shard every N newly saved keys (default: 2000)",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Directory for temporary shard files. Defaults to <output>.shards",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"--input-dir {args.input_dir} is not a directory")
    if args.save_every <= 0:
        sys.exit("--save-every must be > 0")
    if args.log_every <= 0:
        sys.exit("--log-every must be > 0")
    if args.flush_every <= 0:
        sys.exit("--flush-every must be > 0")
    if args.output_format == "pkl" and args.output.suffix.lower() != ".pkl":
        print(f"[warn] --output-format pkl usually writes to a .pkl path, got {args.output}")

    files = collect_structure_files(args.input_dir)
    if not files:
        sys.exit(f"No .pdb / .pdb.gz / .cif / .cif.gz files found in {args.input_dir}")

    total_found = len(files)
    print(f"[info] found {total_found} structure files in {args.input_dir}")

    accession_to_tsv_id = None
    pre_skipped = 0
    pre_skipped_not_in_anchors = 0
    dedup_dropped = 0
    dedup_examples: list[str] = []
    allowed_anchor_ids: set[str] | None = None
    anchors_with_candidate_file: set[str] = set()
    anchors_extracted_successfully: set[str] = set()
    if args.key_mode == "tsv_id":
        if args.fasta is None:
            sys.exit("--fasta is required when --key-mode tsv_id")
        fasta_ids = read_fasta_ids(args.fasta)
        accession_to_tsv_id = {}
        duplicate_accessions = set()
        for seq_id in fasta_ids:
            accession = parse_accession(seq_id)
            if accession in accession_to_tsv_id and accession_to_tsv_id[accession] != seq_id:
                duplicate_accessions.add(accession)
                continue
            accession_to_tsv_id.setdefault(accession, seq_id)
        if duplicate_accessions:
            print(
                f"[warn] duplicate accessions in FASTA; keeping first mapping for {len(duplicate_accessions)} accession(s)"
            )
        if args.triplets_tsv is not None:
            allowed_anchor_ids = read_triplet_anchor_ids(args.triplets_tsv)
            print(
                f"[info] loaded {len(allowed_anchor_ids)} unique anchor_id entries from {args.triplets_tsv}"
            )
        files, pre_skipped, pre_skipped_not_in_anchors, anchors_with_candidate_file = prefilter_tsv_files(
            files, accession_to_tsv_id, allowed_tsv_ids=allowed_anchor_ids
        )
        print(
            f"[info] prefiltered files for tsv_id: keeping {len(files)} / {total_found}, "
            f"pre-skipped {pre_skipped} not in FASTA map"
            + (f", pre-skipped {pre_skipped_not_in_anchors} not used as TSV anchors" if allowed_anchor_ids is not None else "")
        )
        files, dedup_dropped, dedup_examples = dedupe_tsv_files(files)
        print(
            f"[info] deduplicated candidate files by accession: {total_found - pre_skipped - pre_skipped_not_in_anchors} -> {len(files)} "
            f"(dropped {dedup_dropped})"
        )
        if dedup_examples:
            print("[info] dedup examples: " + "; ".join(dedup_examples[:5]))

    index_map = load_index_map(args.index_map) if args.index_map else None

    total_candidate_files = len(files)
    candidate_hash = compute_candidate_hash(files)
    checkpoint_path = args.checkpoint_path or default_checkpoint_path(args.output)
    shard_dir = args.shard_dir or default_shard_dir(args.output)
    fingerprint = config_fingerprint(
        args=args,
        total_found=total_found,
        total_candidate_files=total_candidate_files,
        pre_skipped=pre_skipped,
        pre_skipped_not_in_anchors=pre_skipped_not_in_anchors,
        dedup_dropped=dedup_dropped,
        candidate_hash=candidate_hash,
    )

    buffer_result: dict[Any, np.ndarray] = {}
    failed: list[str] = []
    skipped = 0
    saved_new = 0
    overwritten = 0
    overwrite_examples: list[str] = []
    processed_files: set[str] = set()
    seen_keys: set[Any] = set()
    shard_paths: list[str] = []
    started_at = time.time()

    if args.resume and checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path)
        checkpoint_fingerprint = state.get("fingerprint")
        if checkpoint_fingerprint != fingerprint:
            raise ValueError(
                "Checkpoint configuration does not match the current run. "
                f"Checkpoint fingerprint: {checkpoint_fingerprint}; current fingerprint: {fingerprint}"
            )
        buffer_result = dict(state.get("buffer_result", {}))
        processed_files = set(state.get("processed_files", []))
        seen_keys = set(state.get("seen_keys", []))
        failed = list(state.get("failed", []))
        skipped = int(state.get("skipped", 0))
        saved_new = int(state.get("saved_new", 0))
        overwritten = int(state.get("overwritten", 0))
        overwrite_examples = list(state.get("overwrite_examples", []))
        shard_paths = reconcile_shard_paths(shard_dir, list(state.get("shard_paths", [])), context='Checkpoint')
        if allowed_anchor_ids is not None:
            anchors_extracted_successfully = {key for key in seen_keys if key in allowed_anchor_ids}
        print(
            f"[resume] loaded checkpoint {checkpoint_path} | processed={len(processed_files)}/{total_candidate_files} "
            f"saved_new={saved_new} overwritten={overwritten} skipped={skipped} failed={len(failed)} "
            f"buffered={len(buffer_result)} shards={len(shard_paths)}"
        )
    elif args.resume:
        print(f"[resume] no checkpoint found at {checkpoint_path}; starting fresh")

    if not checkpoint_path.exists():
        save_checkpoint(
            checkpoint_path,
            checkpoint_state(
                processed_files=processed_files,
                seen_keys=seen_keys,
                failed=failed,
                skipped=skipped,
                saved_new=saved_new,
                overwritten=overwritten,
                overwrite_examples=overwrite_examples,
                shard_paths=shard_paths,
                buffer_result=buffer_result,
                fingerprint=fingerprint,
                started_at=started_at,
            ),
        )
        print(f"[checkpoint] initialized checkpoint at {checkpoint_path}")

    last_checkpoint_processed = len(processed_files)
    last_flush_saved_total = saved_new + overwritten

    for loop_idx, path in enumerate(files):
        file_token = path.name
        if file_token in processed_files:
            continue

        protein_key: Any
        if args.key_mode == "index":
            if index_map is not None:
                stem = path.name.removesuffix(".gz")
                stem = Path(stem).stem
                if stem not in index_map:
                    print(f"  [skip] {path.name}: stem '{stem}' not in index map")
                    skipped += 1
                    processed_files.add(file_token)
                    protein_key = None
                else:
                    protein_key = index_map[stem]
            else:
                protein_key = loop_idx
        else:
            accession = stem_to_accession(path)
            if accession not in accession_to_tsv_id:
                print(f"  [skip] {path.name}: accession '{accession}' not in FASTA map")
                skipped += 1
                processed_files.add(file_token)
                protein_key = None
            else:
                protein_key = accession_to_tsv_id[accession]

        if file_token not in processed_files:
            try:
                coords = extract_ca_coords(path, model_idx=args.model, chain_id=args.chain)
            except Exception as e:
                print(f"  [error] {path.name}: {e}")
                failed.append(path.name)
                processed_files.add(file_token)
            else:
                if not coords:
                    print(f"  [warn]  {path.name}: no Cα atoms found, skipping")
                    failed.append(path.name)
                    processed_files.add(file_token)
                else:
                    coords_arr = normalize_coordinate_array(coords)
                    if protein_key in seen_keys:
                        overwritten += 1
                        if len(overwrite_examples) < 10:
                            overwrite_examples.append(f"{path.name} -> {protein_key}")
                    else:
                        saved_new += 1
                        seen_keys.add(protein_key)
                    buffer_result[protein_key] = coords_arr
                    processed_files.add(file_token)
                    if allowed_anchor_ids is not None and args.key_mode == "tsv_id" and protein_key in allowed_anchor_ids:
                        anchors_extracted_successfully.add(protein_key)
                    if saved_new + overwritten <= 10:
                        print(f"  [{protein_key}] {path.name}: {coords_arr.shape[0]} residues")

        processed_count = len(processed_files)
        if processed_count == 0:
            continue

        newly_processed_since_save = processed_count - last_checkpoint_processed
        buffered_saved_total = saved_new + overwritten
        newly_saved_since_flush = buffered_saved_total - last_flush_saved_total

        if processed_count == 1 or (processed_count % args.log_every == 0):
            elapsed = time.time() - started_at
            rate = processed_count / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] processed {processed_count}/{total_candidate_files} candidate files | "
                f"saved_new={saved_new} overwritten={overwritten} skipped={skipped} failed={len(failed)} buffered={len(buffer_result)} shards={len(shard_paths)} | "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} files/s"
            )
        if newly_saved_since_flush >= args.flush_every and buffer_result:
            shard_name = flush_buffer_to_shard(shard_dir, buffer_result, shard_paths)
            print(
                f"[flush] wrote shard {shard_name} to {shard_dir} at processed={processed_count} | total shards={len(shard_paths)}"
            )
            last_flush_saved_total = saved_new + overwritten
        if newly_processed_since_save >= args.save_every:
            save_checkpoint(
                checkpoint_path,
                checkpoint_state(
                    processed_files=processed_files,
                    seen_keys=seen_keys,
                    failed=failed,
                    skipped=skipped,
                    saved_new=saved_new,
                    overwritten=overwritten,
                    overwrite_examples=overwrite_examples,
                    shard_paths=shard_paths,
                    buffer_result=buffer_result,
                    fingerprint=fingerprint,
                    started_at=started_at,
                ),
            )
            print(f"[checkpoint] saved resume state to {checkpoint_path} at processed={processed_count}")
            last_checkpoint_processed = processed_count

    processed_count = len(processed_files)
    elapsed = time.time() - started_at
    if processed_count != saved_new + overwritten + skipped + len(failed):
        raise RuntimeError(
            "Accounting mismatch: processed != saved_new + overwritten + skipped + failed "
            f"({processed_count} != {saved_new} + {overwritten} + {skipped} + {len(failed)})"
        )

    if buffer_result:
        shard_name = flush_buffer_to_shard(shard_dir, buffer_result, shard_paths)
        print(f"[flush] wrote final shard {shard_name} to {shard_dir}")
        last_flush_saved_total = saved_new + overwritten

    print(
        f"[done] total_found={total_found} | pre_skipped_not_in_fasta={pre_skipped} | pre_skipped_not_in_anchors={pre_skipped_not_in_anchors} | dedup_dropped={dedup_dropped} | "
        f"processed={processed_count}/{total_candidate_files} | saved_new={saved_new} overwritten={overwritten} "
        f"skipped={skipped} failed={len(failed)} | shards={len(shard_paths)} | elapsed={elapsed:.1f}s"
    )

    shard_paths = reconcile_shard_paths(shard_dir, shard_paths, context='Final export')
    unique_key_count, index_path = finalize_output(
        output_format=args.output_format,
        shard_dir=shard_dir,
        shard_paths=shard_paths,
        buffer_result=buffer_result,
        output=args.output,
        coordinate_format=args.coordinate_format,
    )
    if args.output_format == "pkl":
        print(f"\nSaved {unique_key_count} unique protein keys -> {args.output}")
    else:
        print(f"\nSaved {unique_key_count} unique protein keys -> shard directory {args.output}")
        if index_path is not None:
            print(f"Index written to {index_path}")
    if overwrite_examples:
        print("Overwrite examples: " + "; ".join(overwrite_examples[:10]))
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:10])}" + (" ..." if len(failed) > 10 else ""))

    if allowed_anchor_ids is not None and args.key_mode == "tsv_id":
        requested_anchor_ids = set(allowed_anchor_ids)
        missing_anchor_ids = sorted(requested_anchor_ids - anchors_extracted_successfully)
        anchors_missing_no_file = sorted(requested_anchor_ids - anchors_with_candidate_file)
        anchors_failed_after_match = sorted(anchors_with_candidate_file - anchors_extracted_successfully)
        print(f"[coverage] requested anchors: {len(requested_anchor_ids)}")
        print(f"[coverage] anchors with candidate structure file: {len(anchors_with_candidate_file)}")
        print(f"[coverage] anchors extracted successfully: {len(anchors_extracted_successfully)}")
        print(f"[coverage] missing with no matching structure file: {len(anchors_missing_no_file)}")
        print(f"[coverage] matched file but extraction failed / no CA: {len(anchors_failed_after_match)}")
        if anchors_missing_no_file:
            print("[coverage] first missing-no-file anchors: " + ", ".join(anchors_missing_no_file[:10]))
        if anchors_failed_after_match:
            print("[coverage] first failed-after-match anchors: " + ", ".join(anchors_failed_after_match[:10]))
        coverage_report = {
            "requested_anchor_count": len(requested_anchor_ids),
            "anchors_with_candidate_file_count": len(anchors_with_candidate_file),
            "anchors_extracted_successfully_count": len(anchors_extracted_successfully),
            "missing_anchor_count": len(missing_anchor_ids),
            "anchors_missing_no_file_count": len(anchors_missing_no_file),
            "anchors_failed_after_match_count": len(anchors_failed_after_match),
            "first_missing_anchor_ids": missing_anchor_ids[:20],
            "first_missing_no_file_anchor_ids": anchors_missing_no_file[:20],
            "first_failed_after_match_anchor_ids": anchors_failed_after_match[:20],
        }
        default_coverage_path = (args.output / "_coverage.json") if args.output_format == "sharded" else Path(str(args.output) + ".coverage.json")
        coverage_path = args.coverage_report or default_coverage_path
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(coverage_path, "w", encoding="utf-8") as fh:
            json.dump(coverage_report, fh, indent=2, ensure_ascii=False)
        print(f"[coverage] wrote coverage report to {coverage_path}")
        if args.missing_anchor_ids_output is not None:
            args.missing_anchor_ids_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.missing_anchor_ids_output, "w", encoding="utf-8") as fh:
                for anchor_id in missing_anchor_ids:
                    fh.write(f"{anchor_id}\n")
            print(f"[coverage] wrote missing anchor IDs to {args.missing_anchor_ids_output}")
        if args.require_all_triplet_anchors and missing_anchor_ids:
            raise ValueError(
                "Not all requested triplet anchors were covered by extracted coordinates. "
                f"Requested: {len(requested_anchor_ids)}; extracted: {len(anchors_extracted_successfully)}; missing: {len(missing_anchor_ids)}"
            )

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[cleanup] removed checkpoint file {checkpoint_path}")
    remove_shard_dir(shard_dir)
    if shard_dir.exists():
        print(f"[warn] shard directory not fully removed: {shard_dir}")
    else:
        print(f"[cleanup] removed shard directory {shard_dir}")


if __name__ == "__main__":
    main()
