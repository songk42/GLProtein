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

Usage:
    TSV-ID-compatible mode:
    python scripts_refactor/extract_ca_coords.py \
        --input-dir /path/to/structures \
        --output /path/to/coordinates.pkl \
        --key-mode tsv_id \
        --fasta /path/to/swissprot.fasta

    Legacy integer-keyed mode:
    python scripts_refactor/extract_ca_coords.py \
        --input-dir /path/to/structures \
        --output /path/to/coordinates.pkl \
        --key-mode index \
        [--index-map /path/to/id_map.tsv]
"""

import argparse
import gzip
import io
import pickle
import sys
import warnings
import time
from pathlib import Path

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
        help="Log extraction progress every N files for large folders (default: 500)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"--input-dir {args.input_dir} is not a directory")

    files = collect_structure_files(args.input_dir)
    if not files:
        sys.exit(f"No .pdb / .pdb.gz / .cif / .cif.gz files found in {args.input_dir}")

    print(f"[info] found {len(files)} structure files in {args.input_dir}")

    accession_to_tsv_id = None
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

    index_map = load_index_map(args.index_map) if args.index_map else None

    result = {}
    failed = []
    skipped = 0
    started_at = time.time()

    for file_idx, path in enumerate(files):
        if file_idx == 0 or ((file_idx + 1) % max(1, args.log_every) == 0):
            elapsed = time.time() - started_at
            rate = (file_idx + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] processed {file_idx + 1}/{len(files)} files | "
                f"saved={len(result)} skipped_or_failed={len(failed) + skipped} | "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} files/s"
            )
        if args.key_mode == "index":
            if index_map is not None:
                stem = path.name.removesuffix(".gz")
                stem = Path(stem).stem
                if stem not in index_map:
                    print(f"  [skip] {path.name}: stem '{stem}' not in index map")
                    skipped += 1
                    continue
                protein_key = index_map[stem]
            else:
                protein_key = file_idx
        else:
            accession = stem_to_accession(path)
            if accession not in accession_to_tsv_id:
                print(f"  [skip] {path.name}: accession '{accession}' not in FASTA map")
                skipped += 1
                continue
            protein_key = accession_to_tsv_id[accession]

        try:
            coords = extract_ca_coords(path, model_idx=args.model, chain_id=args.chain)
        except Exception as e:
            print(f"  [error] {path.name}: {e}")
            failed.append(path.name)
            continue

        if not coords:
            print(f"  [warn]  {path.name}: no Cα atoms found, skipping")
            failed.append(path.name)
            continue

        result[protein_key] = coords
        if file_idx < 10:
            print(f"  [{protein_key}] {path.name}: {len(coords)} residues")

    elapsed = time.time() - started_at
    print(
        f"[done] processed {len(files)} files | saved={len(result)} skipped={skipped} failed={len(failed)} | "
        f"elapsed={elapsed:.1f}s"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(result, fh)

    print(f"\nSaved {len(result)} proteins -> {args.output}")
    if failed:
        print(f"Skipped / failed ({len(failed)}): {', '.join(failed[:10])}" + (" ..." if len(failed) > 10 else ""))


if __name__ == "__main__":
    main()