"""
Extract alpha-carbon (Cα) coordinates from a folder of PDB / mmCIF(.gz) files
and save them as a pkl file suitable for use as `coordinates_path` in the
GLProtein dataset.

Output format:
    dict[int, list[list[float]]]
    {protein_index: [[x, y, z], ...]}  -- one [x,y,z] per residue, in chain order

Protein indices are assigned by sorting the input files alphabetically (0-based).
If your files need to map to specific SwissProt indices, pass --index-map, a
two-column TSV: <filename_stem>  <integer_index>

Usage:
    python scripts/extract_ca_coords.py \\
        --input-dir /path/to/structures \\
        --output coordinates.pkl \\
        [--index-map id_map.tsv] \\
        [--chain A]          # restrict to a specific chain (default: all chains)
        [--model 0]          # which MODEL to use (default: 0, i.e. first)

"""

import argparse
import gzip
import io
import os
import pickle
import sys
import warnings
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
            # Skip HETATM records (waters, ligands, etc.)
            if residue.id[0] != " ":
                continue
            if "CA" in residue:
                ca = residue["CA"]
                x, y, z = ca.get_vector().get_array().tolist()
                coords.append([x, y, z])

    return coords


def collect_structure_files(input_dir: Path) -> list[Path]:
    extensions = {".pdb", ".cif", ".cif.gz"}
    files = []
    for f in sorted(input_dir.iterdir()):
        name = f.name.lower()
        if name.endswith(".pdb") or name.endswith(".cif") or name.endswith(".cif.gz"):
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
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing .pdb / .cif / .cif.gz files")
    parser.add_argument("--output", required=True, type=Path, help="Output .pkl path")
    parser.add_argument("--index-map", type=Path, default=None,
                        help="Optional TSV mapping filename stem -> integer index")
    parser.add_argument("--chain", type=str, default=None, help="Restrict to this chain ID (default: all)")
    parser.add_argument("--model", type=int, default=0, help="MODEL index to use (default: 0)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"--input-dir {args.input_dir} is not a directory")

    files = collect_structure_files(args.input_dir)
    if not files:
        sys.exit(f"No .pdb / .cif / .cif.gz files found in {args.input_dir}")

    index_map = load_index_map(args.index_map) if args.index_map else None

    result: dict[int, list[list[float]]] = {}
    failed = []

    for file_idx, path in enumerate(files):
        # Determine the integer key for this protein
        if index_map is not None:
            # Strip all suffixes to get the stem (handles .cif.gz)
            stem = path.name.removesuffix(".gz")
            stem = Path(stem).stem
            if stem not in index_map:
                print(f"  [skip] {path.name}: stem '{stem}' not in index map")
                continue
            protein_idx = index_map[stem]
        else:
            protein_idx = file_idx

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

        result[protein_idx] = coords
        print(f"  [{protein_idx:>6}] {path.name}: {len(coords)} residues")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(result, fh)

    print(f"\nSaved {len(result)} proteins -> {args.output}")
    if failed:
        print(f"Skipped / failed ({len(failed)}): {', '.join(failed[:10])}" +
              (" ..." if len(failed) > 10 else ""))


if __name__ == "__main__":
    main()
