"""
Extract alpha-carbon (Cα) coordinates from a folder of PDB(.gz) / mmCIF(.gz) files
and save them as a pkl file suitable for use as `coordinates_path` in the
GLProtein dataset.

Output formats:
    TSV-ID-compatible mode:
        dict[str, list[list[float]]]
        {raw_fasta_id_token: [[x, y, z], ...]}

    Legacy integer-keyed mode:
        dict[int, list[list[float]]]
        {protein_index: [[x, y, z], ...]}


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
import gzip
import io
import os
import pickle
import sys
import warnings
import time
from pathlib import Path
from typing import Any

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
        reader = csv.DictReader(fh, delimiter="	")
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

    Parameters
    ----------
    path       : Path to the structure file.
    model_idx  : Which MODEL to use (0-indexed).  AlphaFold files have one model.
    chain_id   : If given, restrict to this chain; otherwise use all chains.
    """
    if _is_cif(path):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    with _open_file(path) as fh:
        content = fh.read()

    structure = parser.get_structure(path.stem, io.StringIO(content))

    try:
        model = list(structure.get_models())[model_idx]
    except IndexError:
        raise ValueError(f"{path}: model index {model_idx} out of range")

    coords = []
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
) -> tuple[list[Path], int, int]:
    kept: list[Path] = []
    pre_skipped_not_in_fasta = 0
    pre_skipped_not_in_anchors = 0
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
    return kept, pre_skipped_not_in_fasta, pre_skipped_not_in_anchors


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


def default_checkpoint_path(output_path: Path) -> Path:
    return Path(str(output_path) + ".checkpoint.pkl")


def save_checkpoint(checkpoint_path: Path, state: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        pickle.dump(state, fh)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    with open(checkpoint_path, "rb") as fh:
        state = pickle.load(fh)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a dict")
    return state


def checkpoint_state(
    result: dict[Any, list[list[float]]],
    processed_files: set[str],
    failed: list[str],
    skipped: int,
    saved_new: int,
    overwritten: int,
    overwrite_examples: list[str],
    total_candidate_files: int,
    pre_skipped: int,
    pre_skipped_not_in_anchors: int,
    dedup_dropped: int,
    started_at: float,
) -> dict[str, Any]:
    return {
        "result": result,
        "processed_files": sorted(processed_files),
        "failed": list(failed),
        "skipped": int(skipped),
        "saved_new": int(saved_new),
        "overwritten": int(overwritten),
        "overwrite_examples": list(overwrite_examples),
        "total_candidate_files": int(total_candidate_files),
        "pre_skipped": int(pre_skipped),
        "pre_skipped_not_in_anchors": int(pre_skipped_not_in_anchors),
        "dedup_dropped": int(dedup_dropped),
        "started_at": float(started_at),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing .pdb / .pdb.gz / .cif / .cif.gz files")
    parser.add_argument("--output", required=True, type=Path, help="Output .pkl path")
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
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"--input-dir {args.input_dir} is not a directory")
    if args.save_every <= 0:
        sys.exit("--save-every must be > 0")
    if args.log_every <= 0:
        sys.exit("--log-every must be > 0")

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
        allowed_anchor_ids = None
        if args.triplets_tsv is not None:
            allowed_anchor_ids = read_triplet_anchor_ids(args.triplets_tsv)
            print(
                f"[info] loaded {len(allowed_anchor_ids)} unique anchor_id entries from {args.triplets_tsv}"
            )
        files, pre_skipped, pre_skipped_not_in_anchors = prefilter_tsv_files(
            files, accession_to_tsv_id, allowed_tsv_ids=allowed_anchor_ids
        )
        print(
            f"[info] prefiltered files for tsv_id: keeping {len(files)} / {total_found}, "
            f"pre-skipped {pre_skipped} not in FASTA map"
            + (f", pre-skipped {pre_skipped_not_in_anchors} not used as TSV anchors" if allowed_anchor_ids is not None else "")
        )
        files, dedup_dropped, dedup_examples = dedupe_tsv_files(files)
        print(
            f"[info] deduplicated candidate files by accession: {total_found - pre_skipped} -> {len(files)} "
            f"(dropped {dedup_dropped})"
        )
        if dedup_examples:
            print("[info] dedup examples: " + "; ".join(dedup_examples[:5]))

    index_map = load_index_map(args.index_map) if args.index_map else None

    total_candidate_files = len(files)
    checkpoint_path = args.checkpoint_path or default_checkpoint_path(args.output)

    result: dict[Any, list[list[float]]] = {}
    failed: list[str] = []
    skipped = 0
    saved_new = 0
    overwritten = 0
    overwrite_examples: list[str] = []
    processed_files: set[str] = set()
    started_at = time.time()

    if args.resume:
        if checkpoint_path.exists():
            state = load_checkpoint(checkpoint_path)
            result = state.get("result", {})
            processed_files = set(state.get("processed_files", []))
            failed = list(state.get("failed", []))
            skipped = int(state.get("skipped", 0))
            saved_new = int(state.get("saved_new", len(result)))
            overwritten = int(state.get("overwritten", 0))
            overwrite_examples = list(state.get("overwrite_examples", []))
            ck_total_candidate = state.get("total_candidate_files")
            ck_pre_skipped = state.get("pre_skipped")
            ck_pre_skipped_not_in_anchors = state.get("pre_skipped_not_in_anchors")
            ck_dedup_dropped = state.get("dedup_dropped")
            if ck_total_candidate is not None and int(ck_total_candidate) != total_candidate_files:
                print(
                    f"[warn] checkpoint candidate file count {ck_total_candidate} differs from current {total_candidate_files}; continuing with current file list"
                )
            if ck_pre_skipped is not None and int(ck_pre_skipped) != pre_skipped:
                print(
                    f"[warn] checkpoint pre-skipped count {ck_pre_skipped} differs from current {pre_skipped}; continuing with current file list"
                )
            if ck_pre_skipped_not_in_anchors is not None and int(ck_pre_skipped_not_in_anchors) != pre_skipped_not_in_anchors:
                print(
                    f"[warn] checkpoint anchor-prefilter count {ck_pre_skipped_not_in_anchors} differs from current {pre_skipped_not_in_anchors}; continuing with current file list"
                )
            if ck_dedup_dropped is not None and int(ck_dedup_dropped) != dedup_dropped:
                print(
                    f"[warn] checkpoint dedup-dropped count {ck_dedup_dropped} differs from current {dedup_dropped}; continuing with current file list"
                )
            print(
                f"[resume] loaded checkpoint {checkpoint_path} | processed={len(processed_files)}/{total_candidate_files} "
                f"saved_new={saved_new} overwritten={overwritten} skipped={skipped} failed={len(failed)}"
            )
        else:
            print(f"[resume] no checkpoint found at {checkpoint_path}; starting fresh")

    last_checkpoint_processed = len(processed_files)

    for loop_idx, path in enumerate(files):
        file_token = path.name
        if file_token in processed_files:
            continue

        if args.key_mode == "index":
            if index_map is not None:
                stem = path.name.removesuffix(".gz")
                stem = Path(stem).stem
                if stem not in index_map:
                    print(f"  [skip] {path.name}: stem '{stem}' not in index map")
                    skipped += 1
                    processed_files.add(file_token)
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
            else:
                protein_key = accession_to_tsv_id[accession]

        if file_token in processed_files:
            pass
        else:
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
                    if protein_key in result:
                        overwritten += 1
                        if len(overwrite_examples) < 10:
                            overwrite_examples.append(f"{path.name} -> {protein_key}")
                    else:
                        saved_new += 1
                    result[protein_key] = coords
                    processed_files.add(file_token)
                    if saved_new + overwritten <= 10:
                        print(f"  [{protein_key}] {path.name}: {len(coords)} residues")

        processed_count = len(processed_files)
        if processed_count == 0:
            continue

        newly_processed_since_save = processed_count - last_checkpoint_processed
        if processed_count == 1 or (processed_count % args.log_every == 0):
            elapsed = time.time() - started_at
            rate = processed_count / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] processed {processed_count}/{total_candidate_files} candidate files | "
                f"saved_new={saved_new} overwritten={overwritten} skipped={skipped} failed={len(failed)} | "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} files/s"
            )
        if newly_processed_since_save >= args.save_every:
            save_checkpoint(
                checkpoint_path,
                checkpoint_state(
                    result=result,
                    processed_files=processed_files,
                    failed=failed,
                    skipped=skipped,
                    saved_new=saved_new,
                    overwritten=overwritten,
                    overwrite_examples=overwrite_examples,
                    total_candidate_files=total_candidate_files,
                    pre_skipped=pre_skipped,
                    pre_skipped_not_in_anchors=pre_skipped_not_in_anchors,
                    dedup_dropped=dedup_dropped,
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

    print(
        f"[done] total_found={total_found} | pre_skipped_not_in_fasta={pre_skipped} | pre_skipped_not_in_anchors={pre_skipped_not_in_anchors} | dedup_dropped={dedup_dropped} | "
        f"processed={processed_count}/{total_candidate_files} | saved_new={saved_new} overwritten={overwritten} "
        f"skipped={skipped} failed={len(failed)} | unique_keys={len(result)} | elapsed={elapsed:.1f}s"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(result, fh)

    print(f"\nSaved {len(result)} unique protein keys -> {args.output}")
    if overwrite_examples:
        print("Overwrite examples: " + "; ".join(overwrite_examples[:10]))
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:10])}" + (" ..." if len(failed) > 10 else ""))

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[cleanup] removed checkpoint file {checkpoint_path}")


if __name__ == "__main__":
    main()
