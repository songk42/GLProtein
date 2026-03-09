import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import os
import random
import subprocess
import tarfile
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    cur_id, cur = None, []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append(''.join(cur))
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
    if cur_id is not None:
        ids.append(cur_id)
        seqs.append(''.join(cur))
    return ids, seqs


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def maybe_truncate(seq: str, max_len: Optional[int]) -> str:
    if max_len is None or max_len <= 0:
        return seq
    return seq[:max_len]


def parse_accession(seq_id: str) -> str:
    token = seq_id.split()[0]
    if '|' in token:
        parts = token.split('|')
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    if token.startswith('AF-') and '-F1' in token:
        return token.split('-')[1]
    return token


def predicted_tm_from_cosine(cosine: np.ndarray) -> np.ndarray:
    """TM-Vec-style calibration: treat cosine similarity of normalized embeddings as predicted TM-score."""
    cosine = np.asarray(cosine, dtype=np.float32)
    return np.clip(cosine, 0.0, 1.0)


def build_faiss_index(embs: np.ndarray):
    import faiss  # type: ignore

    index = faiss.IndexFlatIP(int(embs.shape[1]))
    index.add(np.asarray(embs, dtype=np.float32))
    return index


def batched_topk_inner_product(embs: np.ndarray, query_idx: int, top_k: int, batch_size: int = 16384) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback top-k search without materializing a full NxN similarity matrix."""
    query = embs[query_idx:query_idx + 1]
    best_scores = np.full((top_k,), -np.inf, dtype=np.float32)
    best_indices = np.full((top_k,), -1, dtype=np.int64)
    n = embs.shape[0]
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        scores = (embs[start:stop] @ query.T).reshape(-1)
        candidate_indices = np.arange(start, stop, dtype=np.int64)
        merged_scores = np.concatenate([best_scores, scores], axis=0)
        merged_indices = np.concatenate([best_indices, candidate_indices], axis=0)
        order = np.argpartition(-merged_scores, kth=min(top_k - 1, len(merged_scores) - 1))[:top_k]
        best_scores = merged_scores[order]
        best_indices = merged_indices[order]
        final_order = np.argsort(-best_scores)
        best_scores = best_scores[final_order]
        best_indices = best_indices[final_order]
    return best_scores, best_indices


_STANDARD_AA = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O',
}


def _build_structure_candidates(seq_id: str, accession: str, filename_template: Optional[str], extension_order: Sequence[str]) -> List[str]:
    candidates: List[str] = []
    if filename_template:
        candidates.append(filename_template.format(id=seq_id, accession=accession))
    stem_candidates = OrderedDict.fromkeys([
        seq_id,
        accession,
        f'AF-{accession}-F1-model_v6',
        f'AF-{accession}-F1-model_v4',
        f'AF-{accession}-F1-model_v3',
    ])
    for stem in stem_candidates:
        for ext in extension_order:
            candidates.append(f'{stem}{ext}')
            candidates.append(f'{stem}{ext}.gz')
    return candidates


def resolve_structure_path(structure_dir: str, seq_id: str, accession: str, filename_template: Optional[str], extension_order: Sequence[str]) -> Optional[str]:
    structure_dir_path = Path(structure_dir)
    for candidate in _build_structure_candidates(seq_id, accession, filename_template, extension_order):
        candidate_path = structure_dir_path / candidate
        if candidate_path.exists():
            return str(candidate_path)
    return None


class StructureTMScorer:
    def __init__(self, structure_dir: str, filename_template: Optional[str], tm_score_impl: str, tm_score_norm: str):
        self.structure_dir = structure_dir
        self.filename_template = filename_template
        self.tm_score_impl = tm_score_impl
        self.tm_score_norm = tm_score_norm
        self._cache: Dict[str, Optional[Tuple[np.ndarray, str, str]]] = {}
        self._tmtools_ready = False
        self._external_binary = None
        self._archive = None
        self._archive_members: Dict[str, str] = {}
        self._temp_dir_obj = None
        self._temp_dir = None
        structure_path = Path(structure_dir)
        if structure_path.is_file() and structure_path.suffix == '.tar':
            self._archive = tarfile.open(structure_dir, mode='r')
            self._archive_members = self._index_archive_members(self._archive)
        if tm_score_impl == 'tmtools':
            from Bio.PDB import MMCIFParser, PDBParser  # noqa: F401
            from tmtools import tm_align  # noqa: F401
            self._tmtools_ready = True
        elif tm_score_impl == 'usalign':
            if shutil.which('USalign') is not None:
                self._external_binary = 'USalign'
            elif shutil.which('TMalign') is not None:
                self._external_binary = 'TMalign'
            else:
                raise RuntimeError('tm_score_impl=usalign requested, but neither USalign nor TMalign was found in PATH')
        else:
            raise ValueError(f'Unsupported tm_score_impl: {tm_score_impl}')

    @staticmethod
    def _index_archive_members(archive: tarfile.TarFile) -> Dict[str, str]:
        member_map: Dict[str, str] = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            basename = Path(member.name).name
            member_map.setdefault(basename, member.name)
        return member_map

    def _resolve_archive_member(self, seq_id: str, accession: str) -> Optional[str]:
        extension_order = ('.pdb', '.cif', '.mmcif')
        for candidate in _build_structure_candidates(seq_id, accession, self.filename_template, extension_order):
            if candidate in self._archive_members:
                return self._archive_members[candidate]
        return None

    def _extract_archive_member_to_bytes(self, member_name: str) -> Tuple[bytes, str]:
        extracted = self._archive.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f'Could not extract archive member: {member_name}')
        raw = extracted.read()
        suffix = ''.join(Path(member_name).suffixes)
        if suffix.endswith('.gz'):
            raw = gzip.decompress(raw)
            suffix = suffix[:-3]
        return raw, suffix

    def _materialize_archive_member(self, seq_id: str, accession: str) -> Optional[str]:
        member_name = self._resolve_archive_member(seq_id, accession)
        if member_name is None:
            return None
        raw, suffix = self._extract_archive_member_to_bytes(member_name)
        if self._temp_dir is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory(prefix='tm_score_structures_')
            self._temp_dir = self._temp_dir_obj.name
        safe_name = Path(member_name).name
        if safe_name.endswith('.gz'):
            safe_name = safe_name[:-3]
        out_path = Path(self._temp_dir) / safe_name
        if not out_path.exists():
            out_path.write_bytes(raw)
        return str(out_path)

    def _parse_structure_from_path_or_bytes(self, structure_id: str, path: Optional[str] = None, raw: Optional[bytes] = None, raw_suffix: Optional[str] = None):
        from Bio.PDB import MMCIFParser, PDBParser

        if path is not None:
            parse_suffix = ''.join(Path(path).suffixes).lower()
        else:
            parse_suffix = (raw_suffix or '').lower()
        if parse_suffix.endswith(('.cif', '.mmcif')):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        if path is not None:
            return parser.get_structure(structure_id, path)
        handle = io.StringIO(raw.decode('utf-8'))
        return parser.get_structure(structure_id, handle)

    @staticmethod
    def _score_from_tmtools_result(result, norm: str) -> float:
        values = {
            'chain1': float(result.tm_norm_chain1),
            'chain2': float(result.tm_norm_chain2),
            'avg': float((result.tm_norm_chain1 + result.tm_norm_chain2) / 2.0),
            'max': float(max(result.tm_norm_chain1, result.tm_norm_chain2)),
            'min': float(min(result.tm_norm_chain1, result.tm_norm_chain2)),
        }
        if norm not in values:
            raise ValueError(f'Unsupported tm_score_norm: {norm}')
        return values[norm]

    @staticmethod
    def _parse_usalign_score(stdout: str, norm: str) -> float:
        import re

        matches = re.findall(r'TM-score=\s*([0-9]*\.?[0-9]+)', stdout)
        if not matches:
            raise RuntimeError('Could not parse TM-score from USalign/TMalign output')
        scores = [float(m) for m in matches[:2]]
        if len(scores) == 1:
            scores.append(scores[0])
        if norm == 'chain1':
            return scores[0]
        if norm == 'chain2':
            return scores[1]
        if norm == 'avg':
            return sum(scores[:2]) / 2.0
        if norm == 'max':
            return max(scores[:2])
        if norm == 'min':
            return min(scores[:2])
        raise ValueError(f'Unsupported tm_score_norm: {norm}')

    def _load_structure(self, seq_id: str, accession: str) -> Optional[Tuple[np.ndarray, str, str]]:
        cache_key = accession or seq_id
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached is None:
                return None
            coords, seq, path = cached
            return coords, seq, path

        extension_order = ('.pdb', '.cif', '.mmcif')
        path: Optional[str] = None
        raw: Optional[bytes] = None
        raw_suffix: Optional[str] = None
        if self._archive is not None:
            member_name = self._resolve_archive_member(seq_id, accession)
            if member_name is None:
                self._cache[cache_key] = None
                return None
            if self.tm_score_impl == 'usalign':
                path = self._materialize_archive_member(seq_id, accession)
                if path is None:
                    self._cache[cache_key] = None
                    return None
            else:
                raw, raw_suffix = self._extract_archive_member_to_bytes(member_name)
                path = f'{self.structure_dir}::{member_name}'
        else:
            resolved_path = resolve_structure_path(self.structure_dir, seq_id, accession, self.filename_template, extension_order)
            if resolved_path is None:
                self._cache[cache_key] = None
                return None
            if resolved_path.endswith('.gz'):
                raw = gzip.decompress(Path(resolved_path).read_bytes())
                raw_suffix = ''.join(Path(resolved_path).suffixes[:-1])
                path = resolved_path
            else:
                path = resolved_path

        structure = self._parse_structure_from_path_or_bytes(accession or seq_id, None if raw is not None and self.tm_score_impl == 'tmtools' else path, raw=raw, raw_suffix=raw_suffix)

        coords = []
        seq = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().upper()
                    if resname not in _STANDARD_AA or 'CA' not in residue:
                        continue
                    coords.append(residue['CA'].coord)
                    seq.append(_STANDARD_AA[resname])
                if coords:
                    break
            if coords:
                break

        if not coords:
            self._cache[cache_key] = None
            return None

        out = (np.asarray(coords, dtype=np.float64), ''.join(seq), path or '')
        self._cache[cache_key] = out
        return out

    def score_pair(self, anchor_id: str, anchor_accession: str, other_id: str, other_accession: str) -> Optional[float]:
        anchor = self._load_structure(anchor_id, anchor_accession)
        other = self._load_structure(other_id, other_accession)
        if anchor is None or other is None:
            return None
        anchor_coords, anchor_seq, anchor_path = anchor
        other_coords, other_seq, other_path = other

        if self.tm_score_impl == 'tmtools':
            from tmtools import tm_align

            result = tm_align(anchor_coords, other_coords, anchor_seq, other_seq)
            return self._score_from_tmtools_result(result, self.tm_score_norm)

        proc = subprocess.run(
            [self._external_binary, anchor_path, other_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return self._parse_usalign_score(proc.stdout, self.tm_score_norm)


# shutil imported lazily to keep module import cheap when exact TM is not used.
import shutil

logger = logging.getLogger(__name__)



def load_checkpoint(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_checkpoint(path: str, state: Dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def build_checkpoint_state(args, *, next_anchor_index: int, rows: int, skipped: int, anchors_processed: int, anchors_with_triplets: int, missing_exact_positive: int, missing_exact_negative: int, positive_scores_logged: List[float], negative_scores_logged: List[float], positive_score_mode: str, negative_score_mode: str, faiss_enabled: bool, num_sequences: int, embedding_dim: int, start_time: float) -> Dict:
    elapsed = max(time.time() - start_time, 0.0)
    return {
        'version': 3,
        'out_tsv': os.path.abspath(args.out_tsv),
        'swiss_fasta': os.path.abspath(args.swiss_fasta),
        'swiss_tmvec_emb_npy': os.path.abspath(args.swiss_tmvec_emb_npy),
        'max_proteins': args.max_proteins,
        'max_protein_seq_length': args.max_protein_seq_length,
        'top_k_pos': args.top_k_pos,
        'triplets_per_anchor': args.triplets_per_anchor,
        'positive_search_k': args.positive_search_k,
        'positive_tmscore_min': args.positive_tmscore_min,
        'neg_random_pool': args.neg_random_pool,
        'negative_tmscore_max': args.negative_tmscore_max,
        'negative_pick_strategy': args.negative_pick_strategy,
        'exact_tm_score_structures_dir': os.path.abspath(args.exact_tm_score_structures_dir) if args.exact_tm_score_structures_dir else None,
        'tm_score_impl': args.tm_score_impl if args.exact_tm_score_structures_dir else None,
        'tm_score_norm': args.tm_score_norm if args.exact_tm_score_structures_dir else None,
        'next_anchor_index': int(next_anchor_index),
        'rows_written': int(rows),
        'skipped_anchors': int(skipped),
        'anchors_processed': int(anchors_processed),
        'anchors_with_triplets': int(anchors_with_triplets),
        'missing_exact_positive_scores': int(missing_exact_positive),
        'missing_exact_negative_scores': int(missing_exact_negative),
        'positive_scores_logged_tail': positive_scores_logged[-1000:],
        'negative_scores_logged_tail': negative_scores_logged[-1000:],
        'positive_score_mode': positive_score_mode,
        'negative_score_mode': negative_score_mode,
        'use_faiss': bool(faiss_enabled),
        'num_sequences': int(num_sequences),
        'embedding_dim': int(embedding_dim),
        'elapsed_seconds_before_resume': float(elapsed),
    }


def maybe_save_checkpoint(path: str, every_anchors: int, processed_anchor_count: int, state: Dict, tsv_handle=None) -> None:
    if every_anchors and every_anchors > 0 and processed_anchor_count > 0 and processed_anchor_count % every_anchors == 0:
        if tsv_handle is not None:
            flush_tsv_file(tsv_handle)
        save_checkpoint(path, state)


def count_tsv_data_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8', newline='') as f:
        total = sum(1 for _ in f)
    return max(total - 1, 0)


def flush_tsv_file(handle) -> None:
    if handle is None or handle.closed:
        return
    handle.flush()
    os.fsync(handle.fileno())


def truncate_tsv_data_rows(path: str, keep_data_rows: int) -> None:
    tmp_path = f"{path}.truncate.tmp"
    with open(path, 'r', encoding='utf-8', newline='') as src, open(tmp_path, 'w', encoding='utf-8', newline='') as dst:
        for lineno, line in enumerate(src):
            if lineno == 0:
                dst.write(line)
                continue
            if lineno <= keep_data_rows:
                dst.write(line)
            else:
                break
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp_path, path)


def recover_resume_state_from_tsv(path: str, ids: List[str]) -> Dict[str, object]:
    if not os.path.exists(path):
        return {
            'resume_anchor_index': 0,
            'rows_written': 0,
            'anchors_processed': 0,
            'anchors_with_triplets': 0,
            'skipped_anchors': 0,
            'positive_scores_logged_tail': [],
            'negative_scores_logged_tail': [],
            'recovered_by_truncating_last_anchor': False,
        }

    id_to_index: Dict[str, int] = {}
    for idx, seq_id in enumerate(ids):
        if seq_id not in id_to_index:
            id_to_index[seq_id] = idx

    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='	')
        header = next(reader, None)
        if header is None:
            return {
                'resume_anchor_index': 0,
                'rows_written': 0,
                'anchors_processed': 0,
                'anchors_with_triplets': 0,
                'skipped_anchors': 0,
                'positive_scores_logged_tail': [],
                'negative_scores_logged_tail': [],
                'recovered_by_truncating_last_anchor': False,
            }
        try:
            anchor_idx = header.index('anchor_id')
            pos_idx = header.index('positive_score')
            neg_idx = header.index('negative_score')
        except ValueError as exc:
            raise ValueError(f'Existing TSV is missing required columns: {exc}')

        data_rows = 0
        group_count = 0
        last_group_anchor_id: Optional[str] = None
        last_group_start_row = 1
        prev_anchor_id: Optional[str] = None

        for data_rows, row in enumerate(reader, start=1):
            aid = row[anchor_idx]
            if prev_anchor_id is None or aid != prev_anchor_id:
                group_count += 1
                last_group_anchor_id = aid
                last_group_start_row = data_rows
                prev_anchor_id = aid

    if data_rows == 0 or last_group_anchor_id is None:
        return {
            'resume_anchor_index': 0,
            'rows_written': 0,
            'anchors_processed': 0,
            'anchors_with_triplets': 0,
            'skipped_anchors': 0,
            'positive_scores_logged_tail': [],
            'negative_scores_logged_tail': [],
            'recovered_by_truncating_last_anchor': False,
        }

    if last_group_anchor_id not in id_to_index:
        raise ValueError(f'Last TSV anchor_id {last_group_anchor_id} is not present in the current FASTA')

    keep_rows = last_group_start_row - 1
    resume_anchor_index = id_to_index[last_group_anchor_id]

    pos_tail: List[float] = []
    neg_tail: List[float] = []
    if keep_rows > 0:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter='	')
            next(reader, None)
            for rowno, row in enumerate(reader, start=1):
                if rowno > keep_rows:
                    break
                try:
                    pos_tail.append(float(row[pos_idx]))
                    neg_tail.append(float(row[neg_idx]))
                except (TypeError, ValueError, IndexError):
                    pass
        pos_tail = pos_tail[-1000:]
        neg_tail = neg_tail[-1000:]

    anchors_with_triplets = max(group_count - 1, 0)
    anchors_processed = resume_anchor_index
    skipped_anchors = max(anchors_processed - anchors_with_triplets, 0)

    return {
        'resume_anchor_index': int(resume_anchor_index),
        'rows_written': int(keep_rows),
        'anchors_processed': int(anchors_processed),
        'anchors_with_triplets': int(anchors_with_triplets),
        'skipped_anchors': int(skipped_anchors),
        'positive_scores_logged_tail': pos_tail,
        'negative_scores_logged_tail': neg_tail,
        'recovered_by_truncating_last_anchor': True,
    }


def validate_resume_args(checkpoint_state: Dict, args) -> None:
    expected = {
        'max_proteins': args.max_proteins,
        'max_protein_seq_length': args.max_protein_seq_length,
        'top_k_pos': args.top_k_pos,
        'triplets_per_anchor': args.triplets_per_anchor,
        'positive_search_k': args.positive_search_k,
        'positive_tmscore_min': args.positive_tmscore_min,
        'neg_random_pool': args.neg_random_pool,
        'negative_tmscore_max': args.negative_tmscore_max,
        'negative_pick_strategy': args.negative_pick_strategy,
        'use_faiss': bool(args.use_faiss),
        'exact_tm_score_structures_dir': os.path.abspath(args.exact_tm_score_structures_dir) if args.exact_tm_score_structures_dir else None,
        'tm_score_impl': args.tm_score_impl if args.exact_tm_score_structures_dir else None,
        'tm_score_norm': args.tm_score_norm if args.exact_tm_score_structures_dir else None,
    }
    for key, value in expected.items():
        if checkpoint_state.get(key) != value:
            raise ValueError(f'Checkpoint {key} mismatch: {checkpoint_state.get(key)} vs {value}')


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float32)
    return {
        'min': float(arr.min()),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'p75': float(np.percentile(arr, 75)),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description='Mine explicit anchor/positive/negative triplets from precomputed TM-Vec embeddings.')
    ap.add_argument('--swiss_fasta', required=True)
    ap.add_argument('--swiss_tmvec_emb_npy', required=True)
    ap.add_argument('--swiss_tmvec_metadata_npy', default=None)
    ap.add_argument('--out_tsv', required=True)
    ap.add_argument('--out_metadata_json', default=None)
    ap.add_argument('--top_k_pos', type=int, default=5)
    ap.add_argument('--triplets_per_anchor', type=int, default=1)
    ap.add_argument('--positive_search_k', type=int, default=64, help='Nearest-neighbor candidate pool to retrieve before positive reranking/filtering.')
    ap.add_argument('--positive_tmscore_min', type=float, default=0.0, help='Optional minimum predicted/actual TM-score for positives.')
    ap.add_argument('--neg_random_pool', type=int, default=256)
    ap.add_argument('--negative_tmscore_max', type=float, default=0.2, help='Pairs at or below this TM-score are treated as structurally dissimilar.')
    ap.add_argument('--neg_similarity_max', type=float, default=None, help='Deprecated alias for --negative_tmscore_max.')
    ap.add_argument('--negative_pick_strategy', choices=['random', 'hardest', 'easiest'], default='hardest')
    ap.add_argument('--max_proteins', type=int, default=None)
    ap.add_argument('--max_protein_seq_length', type=int, default=None)
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--use_faiss', action='store_true', help='Use a FAISS inner-product index for positive candidate mining when available.')
    ap.add_argument('--exact_tm_score_structures_dir', default=None, help='Optional directory of PDB/mmCIF structures, .pdb.gz/.cif.gz files, or an AlphaFold archive')
    ap.add_argument('--structure_filename_template', default=None, help='Optional filename template, e.g. AF-{accession}-F1-model_v4.pdb or {id}.pdb.')
    ap.add_argument('--tm_score_impl', choices=['tmtools', 'usalign'], default='tmtools', help='Exact TM-score backend used when --exact_tm_score_structures_dir is provided.')
    ap.add_argument('--tm_score_norm', choices=['chain1', 'chain2', 'avg', 'max', 'min'], default='chain1', help='How to choose the normalized TM-score when the backend reports both directions.')
    ap.add_argument('--log_every_anchors', type=int, default=1000, help='Print progress every N processed anchors. 0 disables periodic progress logging.')
    ap.add_argument('--checkpoint_path', default=None, help='Path to resumable mining checkpoint JSON. Defaults to <out_tsv>.checkpoint.json.')
    ap.add_argument('--checkpoint_every_anchors', type=int, default=1000, help='Save a resumable checkpoint every N processed anchors. 0 disables checkpoint saves.')
    ap.add_argument('--resume', action='store_true', help='Resume from checkpoint_path if it exists and append to an existing TSV.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    if args.neg_similarity_max is not None:
        args.negative_tmscore_max = args.neg_similarity_max

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    ids, seqs = read_fasta(args.swiss_fasta)
    embs = np.load(args.swiss_tmvec_emb_npy)
    if args.max_proteins is not None:
        ids = ids[:args.max_proteins]
        seqs = seqs[:args.max_proteins]
        embs = embs[:args.max_proteins]
    if len(seqs) != int(embs.shape[0]):
        raise ValueError(f'FASTA/embedding mismatch: {len(seqs)} vs. {embs.shape[0]}')

    normalized_embs = normalize_rows(embs)
    accessions = [parse_accession(seq_id) for seq_id in ids]
    n = len(seqs)
    all_indices = np.arange(n)

    faiss_index = None
    faiss_enabled = False
    if args.use_faiss:
        try:
            faiss_index = build_faiss_index(normalized_embs)
            faiss_enabled = True
        except Exception as exc:  # pragma: no cover - best-effort optional dependency
            print(f'Could not initialize FAISS ({exc}); falling back to streamed NumPy search.')

    structure_scorer = None
    if args.exact_tm_score_structures_dir:
        structure_scorer = StructureTMScorer(
            structure_dir=args.exact_tm_score_structures_dir,
            filename_template=args.structure_filename_template,
            tm_score_impl=args.tm_score_impl,
            tm_score_norm=args.tm_score_norm,
        )

    out_path = args.out_tsv
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    checkpoint_path = args.checkpoint_path or (out_path + '.checkpoint.json')

    logger.info(
        'Triplet mining config | num_sequences=%d | embedding_dim=%d | out_tsv=%s | top_k_pos=%d | triplets_per_anchor=%d | positive_search_k=%d | positive_tmscore_min=%.3f | neg_random_pool=%d | negative_tmscore_max=%.3f | negative_pick_strategy=%s | use_faiss=%s | exact_tm_score=%s | tm_score_impl=%s | tm_score_norm=%s | max_protein_seq_length=%s | max_proteins=%s',
        len(seqs),
        int(normalized_embs.shape[1]),
        out_path,
        args.top_k_pos,
        args.triplets_per_anchor,
        args.positive_search_k,
        args.positive_tmscore_min,
        args.neg_random_pool,
        args.negative_tmscore_max,
        args.negative_pick_strategy,
        faiss_enabled,
        bool(args.exact_tm_score_structures_dir),
        args.tm_score_impl if args.exact_tm_score_structures_dir else None,
        args.tm_score_norm if args.exact_tm_score_structures_dir else None,
        args.max_protein_seq_length,
        args.max_proteins,
    )
    logger.info('Checkpoint config | checkpoint_path=%s | checkpoint_every_anchors=%s | resume=%s', checkpoint_path, args.checkpoint_every_anchors, args.resume)

    rows = 0
    skipped = 0
    missing_exact_positive = 0
    missing_exact_negative = 0
    positive_scores_logged: List[float] = []
    negative_scores_logged: List[float] = []
    positive_score_mode = 'predicted_tmvec_tmscore'
    negative_score_mode = 'predicted_tmvec_tmscore'
    anchors_processed = 0
    anchors_with_triplets = 0
    resume_elapsed_seconds = 0.0
    resume_anchor_index = 0

    if args.resume:
        checkpoint_state = load_checkpoint(checkpoint_path)
        if checkpoint_state is not None:
            if os.path.abspath(out_path) != checkpoint_state.get('out_tsv'):
                raise ValueError(f'Checkpoint out_tsv mismatch: {checkpoint_state.get("out_tsv")} vs {os.path.abspath(out_path)}')
            if os.path.abspath(args.swiss_fasta) != checkpoint_state.get('swiss_fasta'):
                raise ValueError('Checkpoint swiss_fasta mismatch')
            if os.path.abspath(args.swiss_tmvec_emb_npy) != checkpoint_state.get('swiss_tmvec_emb_npy'):
                raise ValueError('Checkpoint swiss_tmvec_emb_npy mismatch')
            validate_resume_args(checkpoint_state, args)
            if not os.path.exists(out_path):
                raise FileNotFoundError(f'Checkpoint exists but TSV does not: {out_path}')
            resume_anchor_index = int(checkpoint_state.get('next_anchor_index', 0))
            rows = int(checkpoint_state.get('rows_written', 0))
            skipped = int(checkpoint_state.get('skipped_anchors', 0))
            anchors_processed = int(checkpoint_state.get('anchors_processed', 0))
            anchors_with_triplets = int(checkpoint_state.get('anchors_with_triplets', 0))
            missing_exact_positive = int(checkpoint_state.get('missing_exact_positive_scores', 0))
            missing_exact_negative = int(checkpoint_state.get('missing_exact_negative_scores', 0))
            positive_scores_logged = list(checkpoint_state.get('positive_scores_logged_tail', []))
            negative_scores_logged = list(checkpoint_state.get('negative_scores_logged_tail', []))
            positive_score_mode = checkpoint_state.get('positive_score_mode', positive_score_mode)
            negative_score_mode = checkpoint_state.get('negative_score_mode', negative_score_mode)
            resume_elapsed_seconds = float(checkpoint_state.get('elapsed_seconds_before_resume', 0.0))
            if resume_anchor_index < 0:
                raise ValueError(f'Invalid resume_anchor_index in checkpoint: {resume_anchor_index}')
            if resume_anchor_index > n:
                raise ValueError(f'Checkpoint next_anchor_index {resume_anchor_index} exceeds current sequence count {n}')
            existing_rows = count_tsv_data_rows(out_path)
            if existing_rows > rows:
                logger.warning('Resume TSV/checkpoint data row mismatch: %d vs %d. Truncating TSV and resuming from checkpoint anchor %d.', existing_rows, rows, resume_anchor_index)
                truncate_tsv_data_rows(out_path, rows)
                existing_rows = rows
            elif existing_rows < rows:
                logger.warning('Resume TSV/checkpoint row mismatch: %d vs %d. Recovering from the TSV itself and truncating the last anchor block.', existing_rows, rows)
                recovered = recover_resume_state_from_tsv(out_path, ids)
                recovered_rows = int(recovered['rows_written'])
                truncate_tsv_data_rows(out_path, recovered_rows)
                resume_anchor_index = int(recovered['resume_anchor_index'])
                rows = recovered_rows
                anchors_processed = int(recovered['anchors_processed'])
                anchors_with_triplets = int(recovered['anchors_with_triplets'])
                skipped = int(recovered['skipped_anchors'])
                positive_scores_logged = list(recovered['positive_scores_logged_tail'])
                negative_scores_logged = list(recovered['negative_scores_logged_tail'])
                missing_exact_positive = 0
                missing_exact_negative = 0
                positive_score_mode = 'predicted_tmvec_tmscore'
                negative_score_mode = 'predicted_tmvec_tmscore'
                existing_rows = recovered_rows
                logger.warning('Recovered resume state from TSV | resume_anchor_index=%d | rows_written=%d | anchors_processed=%d | anchors_with_triplets=%d | skipped=%d', resume_anchor_index, rows, anchors_processed, anchors_with_triplets, skipped)
            if resume_anchor_index == n:
                logger.info('Checkpoint indicates completion (next_anchor_index=%d, num_sequences=%d).', resume_anchor_index, n)
                return
            logger.info('Resuming triplet mining from checkpoint | checkpoint_path=%s | next_anchor_index=%d | rows_written=%d | anchors_processed=%d', checkpoint_path, resume_anchor_index, rows, anchors_processed)
        else:
            logger.info('Resume requested but no checkpoint found at %s; starting fresh.', checkpoint_path)

    start_time = time.time() - resume_elapsed_seconds

    def _log_progress(force: bool = False):
        if not force and (args.log_every_anchors is None or args.log_every_anchors <= 0):
            return
        if not force and anchors_processed == 0:
            return
        elapsed = max(time.time() - start_time, 1e-9)
        anchors_per_sec = anchors_processed / elapsed
        rows_per_sec = rows / elapsed if rows > 0 else 0.0
        total_processed = min(max(anchors_processed, 0), n)
        pct = (total_processed / n * 100.0) if n > 0 else 100.0
        remaining = max(n - total_processed, 0)
        eta_seconds = remaining / anchors_per_sec if anchors_per_sec > 0 else float('inf')
        pos_summary = summarize(positive_scores_logged[-1000:])
        neg_summary = summarize(negative_scores_logged[-1000:])
        logger.info(
            'Triplet mining progress | anchors=%d/%d (%.2f%%) | rows_written=%d | anchors_with_triplets=%d | skipped=%d | missing_exact_pos=%d | missing_exact_neg=%d | elapsed=%.1fs | anchors_per_sec=%.2f | rows_per_sec=%.2f | eta_seconds=%s | pos_mean=%s | neg_mean=%s',
            min(max(anchors_processed, 0), n), n, pct, rows, anchors_with_triplets, skipped, missing_exact_positive, missing_exact_negative, elapsed, anchors_per_sec, rows_per_sec,
            'inf' if eta_seconds == float('inf') else f'{eta_seconds:.1f}',
            None if not pos_summary else f"{pos_summary['mean']:.4f}",
            None if not neg_summary else f"{neg_summary['mean']:.4f}",
        )

    file_mode = 'a' if args.resume and resume_anchor_index > 0 and os.path.exists(out_path) else 'w'
    logger.info('Triplet mining start mode | file_mode=%s | resume_anchor_index=%d | num_sequences=%d | max_proteins=%s', file_mode, resume_anchor_index, n, args.max_proteins)
    with open(out_path, file_mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='	')
        if file_mode == 'w':
            writer.writerow([
                'anchor_id', 'positive_id', 'negative_id',
                'anchor_seq', 'positive_seq', 'negative_seq',
                'positive_score', 'negative_score'
            ])

        for i in range(resume_anchor_index, n):
            anchors_processed += 1
            if faiss_enabled:
                top_scores, top_indices = faiss_index.search(normalized_embs[i:i + 1], k=min(args.positive_search_k + 1, n))
                nn_scores = top_scores[0]
                nn_indices = top_indices[0]
            else:
                nn_scores, nn_indices = batched_topk_inner_product(normalized_embs, i, top_k=min(args.positive_search_k + 1, n))

            positive_candidates = []
            for cand_score, cand_idx in zip(nn_scores.tolist(), nn_indices.tolist()):
                if cand_idx < 0 or cand_idx == i:
                    continue
                predicted_score = float(predicted_tm_from_cosine(np.array([cand_score]))[0])
                if predicted_score < args.positive_tmscore_min:
                    continue
                score = predicted_score
                score_mode = 'predicted_tmvec_tmscore'
                if structure_scorer is not None:
                    exact_score = structure_scorer.score_pair(ids[i], accessions[i], ids[cand_idx], accessions[cand_idx])
                    if exact_score is not None:
                        score = float(exact_score)
                        score_mode = f'exact_{args.tm_score_impl}_tm_score'
                    else:
                        missing_exact_positive += 1
                positive_candidates.append((cand_idx, score, score_mode))

            positive_candidates.sort(key=lambda x: x[1], reverse=True)
            positive_candidates = [x for x in positive_candidates if x[1] >= args.positive_tmscore_min]
            chosen_positives = positive_candidates[:max(args.top_k_pos, args.triplets_per_anchor)]
            if not chosen_positives:
                skipped += 1
                if args.log_every_anchors and args.log_every_anchors > 0 and anchors_processed % args.log_every_anchors == 0:
                    _log_progress()
                maybe_save_checkpoint(checkpoint_path, args.checkpoint_every_anchors, anchors_processed, build_checkpoint_state(args, next_anchor_index=i + 1, rows=rows, skipped=skipped, anchors_processed=anchors_processed, anchors_with_triplets=anchors_with_triplets, missing_exact_positive=missing_exact_positive, missing_exact_negative=missing_exact_negative, positive_scores_logged=positive_scores_logged, negative_scores_logged=negative_scores_logged, positive_score_mode=positive_score_mode, negative_score_mode=negative_score_mode, faiss_enabled=faiss_enabled, num_sequences=len(seqs), embedding_dim=int(normalized_embs.shape[1]), start_time=start_time), tsv_handle=f)
                continue

            forbidden = {i, *[idx for idx, _, _ in chosen_positives]}
            neg_sample_size = min(max(args.neg_random_pool, args.triplets_per_anchor), max(0, n - len(forbidden)))
            if neg_sample_size <= 0:
                skipped += 1
                if args.log_every_anchors and args.log_every_anchors > 0 and anchors_processed % args.log_every_anchors == 0:
                    _log_progress()
                maybe_save_checkpoint(checkpoint_path, args.checkpoint_every_anchors, anchors_processed, build_checkpoint_state(args, next_anchor_index=i + 1, rows=rows, skipped=skipped, anchors_processed=anchors_processed, anchors_with_triplets=anchors_with_triplets, missing_exact_positive=missing_exact_positive, missing_exact_negative=missing_exact_negative, positive_scores_logged=positive_scores_logged, negative_scores_logged=negative_scores_logged, positive_score_mode=positive_score_mode, negative_score_mode=negative_score_mode, faiss_enabled=faiss_enabled, num_sequences=len(seqs), embedding_dim=int(normalized_embs.shape[1]), start_time=start_time), tsv_handle=f)
                continue
            negative_pool = rng.sample([idx for idx in all_indices.tolist() if idx not in forbidden], k=neg_sample_size)

            negative_candidates = []
            neg_scores = predicted_tm_from_cosine(normalized_embs[negative_pool] @ normalized_embs[i])
            for cand_idx, predicted_score in zip(negative_pool, neg_scores.tolist()):
                score = float(predicted_score)
                score_mode = 'predicted_tmvec_tmscore'
                if structure_scorer is not None:
                    exact_score = structure_scorer.score_pair(ids[i], accessions[i], ids[cand_idx], accessions[cand_idx])
                    if exact_score is not None:
                        score = float(exact_score)
                        score_mode = f'exact_{args.tm_score_impl}_tm_score'
                    else:
                        missing_exact_negative += 1
                if score <= args.negative_tmscore_max:
                    negative_candidates.append((cand_idx, score, score_mode))

            if not negative_candidates:
                fallback_scores = []
                for cand_idx, predicted_score in zip(negative_pool, neg_scores.tolist()):
                    fallback_scores.append((cand_idx, float(predicted_score), 'predicted_tmvec_tmscore'))
                fallback_scores.sort(key=lambda x: x[1])
                negative_candidates = fallback_scores[:1]

            if not negative_candidates:
                skipped += 1
                if args.log_every_anchors and args.log_every_anchors > 0 and anchors_processed % args.log_every_anchors == 0:
                    _log_progress()
                maybe_save_checkpoint(checkpoint_path, args.checkpoint_every_anchors, anchors_processed, build_checkpoint_state(args, next_anchor_index=i + 1, rows=rows, skipped=skipped, anchors_processed=anchors_processed, anchors_with_triplets=anchors_with_triplets, missing_exact_positive=missing_exact_positive, missing_exact_negative=missing_exact_negative, positive_scores_logged=positive_scores_logged, negative_scores_logged=negative_scores_logged, positive_score_mode=positive_score_mode, negative_score_mode=negative_score_mode, faiss_enabled=faiss_enabled, num_sequences=len(seqs), embedding_dim=int(normalized_embs.shape[1]), start_time=start_time), tsv_handle=f)
                continue

            if args.negative_pick_strategy == 'hardest':
                negative_candidates.sort(key=lambda x: x[1], reverse=True)
            elif args.negative_pick_strategy == 'easiest':
                negative_candidates.sort(key=lambda x: x[1])
            else:
                rng.shuffle(negative_candidates)

            wrote_any_triplet = False
            for rank, (p_idx, p_score, p_mode) in enumerate(chosen_positives[:args.triplets_per_anchor]):
                n_idx, n_score, n_mode = negative_candidates[min(rank, len(negative_candidates) - 1)]
                writer.writerow([
                    ids[i], ids[p_idx], ids[n_idx],
                    maybe_truncate(seqs[i], args.max_protein_seq_length),
                    maybe_truncate(seqs[p_idx], args.max_protein_seq_length),
                    maybe_truncate(seqs[n_idx], args.max_protein_seq_length),
                    f'{float(p_score):.6f}',
                    f'{float(n_score):.6f}',
                ])
                rows += 1
                wrote_any_triplet = True
                positive_scores_logged.append(float(p_score))
                negative_scores_logged.append(float(n_score))
                positive_score_mode = p_mode
                negative_score_mode = n_mode
            if wrote_any_triplet:
                anchors_with_triplets += 1
            if args.log_every_anchors and args.log_every_anchors > 0 and anchors_processed % args.log_every_anchors == 0:
                _log_progress()
            maybe_save_checkpoint(checkpoint_path, args.checkpoint_every_anchors, anchors_processed, build_checkpoint_state(args, next_anchor_index=i + 1, rows=rows, skipped=skipped, anchors_processed=anchors_processed, anchors_with_triplets=anchors_with_triplets, missing_exact_positive=missing_exact_positive, missing_exact_negative=missing_exact_negative, positive_scores_logged=positive_scores_logged, negative_scores_logged=negative_scores_logged, positive_score_mode=positive_score_mode, negative_score_mode=negative_score_mode, faiss_enabled=faiss_enabled, num_sequences=len(seqs), embedding_dim=int(normalized_embs.shape[1]), start_time=start_time), tsv_handle=f)

        flush_tsv_file(f)

    meta_path = args.out_metadata_json or (out_path[:-4] + '.metadata.json' if out_path.endswith('.tsv') else out_path + '.metadata.json')
    save_checkpoint(checkpoint_path, build_checkpoint_state(
        args,
        next_anchor_index=n,
        rows=rows,
        skipped=skipped,
        anchors_processed=anchors_processed,
        anchors_with_triplets=anchors_with_triplets,
        missing_exact_positive=missing_exact_positive,
        missing_exact_negative=missing_exact_negative,
        positive_scores_logged=positive_scores_logged,
        negative_scores_logged=negative_scores_logged,
        positive_score_mode=positive_score_mode,
        negative_score_mode=negative_score_mode,
        faiss_enabled=faiss_enabled,
        num_sequences=len(seqs),
        embedding_dim=int(normalized_embs.shape[1]),
        start_time=start_time,
    ))

    meta = {
        'swiss_fasta': os.path.abspath(args.swiss_fasta),
        'swiss_tmvec_emb_npy': os.path.abspath(args.swiss_tmvec_emb_npy),
        'swiss_tmvec_metadata_npy': os.path.abspath(args.swiss_tmvec_metadata_npy) if args.swiss_tmvec_metadata_npy else None,
        'num_sequences': len(seqs),
        'embedding_dim': int(normalized_embs.shape[1]),
        'num_triplets': rows,
        'num_skipped_anchors': skipped,
        'top_k_pos': args.top_k_pos,
        'triplets_per_anchor': args.triplets_per_anchor,
        'positive_search_k': args.positive_search_k,
        'positive_tmscore_min': args.positive_tmscore_min,
        'neg_random_pool': args.neg_random_pool,
        'negative_tmscore_max': args.negative_tmscore_max,
        'negative_pick_strategy': args.negative_pick_strategy,
        'max_protein_seq_length': args.max_protein_seq_length,
        'seed': args.seed,
        'use_faiss': faiss_enabled,
        'exact_tm_score_structures_dir': os.path.abspath(args.exact_tm_score_structures_dir) if args.exact_tm_score_structures_dir else None,
        'tm_score_impl': args.tm_score_impl if args.exact_tm_score_structures_dir else None,
        'tm_score_norm': args.tm_score_norm if args.exact_tm_score_structures_dir else None,
        'positive_score_mode': positive_score_mode,
        'negative_score_mode': negative_score_mode,
        'missing_exact_positive_scores': missing_exact_positive,
        'missing_exact_negative_scores': missing_exact_negative,
        'positive_score_summary': summarize(positive_scores_logged),
        'negative_score_summary': summarize(negative_scores_logged),
        'triplets_tsv_path': os.path.abspath(out_path),
        'triplets_tsv_sha256': sha256_file(out_path),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    _log_progress(force=True)
    elapsed = max(time.time() - start_time, 1e-9)
    logger.info(
        'Triplet mining finished | anchors_processed=%d | total_anchors=%d | rows_written=%d | anchors_with_triplets=%d | skipped=%d | skipped_fraction=%.4f | missing_exact_pos=%d | missing_exact_neg=%d | elapsed=%.1fs | anchors_per_sec=%.2f | rows_per_sec=%.2f | positive_score_summary=%s | negative_score_summary=%s | metadata_json=%s',
        anchors_processed, n, rows, anchors_with_triplets, skipped, (skipped / n if n else 0.0), missing_exact_positive, missing_exact_negative, elapsed, anchors_processed / elapsed, rows / elapsed if rows > 0 else 0.0, summarize(positive_scores_logged), summarize(negative_scores_logged), meta_path
    )
    print(f'Wrote {rows} triplets to {out_path}')
    print(f'Wrote metadata to {meta_path}')
    if positive_scores_logged:
        print('Positive score summary:', summarize(positive_scores_logged))
    if negative_scores_logged:
        print('Negative score summary:', summarize(negative_scores_logged))


if __name__ == '__main__':
    main()
