import argparse
import json
import os
import pickle
from typing import Dict, Any

import numpy as np

from coordinate_store import export_coordinate_shards_from_mapping


def export_coordinate_shards(input_pkl: str, output_dir: str, fmt: str = 'npy') -> None:
    with open(input_pkl, 'rb') as fh:
        payload = pickle.load(fh)
    count, index_path = export_coordinate_shards_from_mapping(payload, output_dir, fmt=fmt)
    print(f'Wrote {count} coordinate shards to {output_dir}')
    print(f'Wrote index to {index_path}')


def export_aa_vocab(mol2vec_model_path: str, output_path: str) -> None:
    from dataset import _build_aa_vocab_from_mol2vec

    aa_vocab = _build_aa_vocab_from_mol2vec(mol2vec_model_path)
    serializable: Dict[str, Any] = {key: np.asarray(value, dtype=np.float32) for key, value in aa_vocab.items()}
    ext = os.path.splitext(output_path)[1].lower()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if ext in {'.pkl', '.pickle'}:
        with open(output_path, 'wb') as fh:
            pickle.dump(serializable, fh, protocol=pickle.HIGHEST_PROTOCOL)
    elif ext == '.json':
        with open(output_path, 'w', encoding='utf-8') as fh:
            json.dump({k: v.tolist() for k, v in serializable.items()}, fh, indent=2, ensure_ascii=False)
    elif ext == '.npy':
        np.save(output_path, serializable, allow_pickle=True)
    else:
        raise ValueError('aa vocab output path must end with .pkl, .pickle, .json, or .npy')
    print(f'Wrote amino-acid vocab with {len(serializable)} residue codes to {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare low-memory auxiliary assets for GLProtein pretraining. For coordinates, this is mainly a legacy/backfill converter from a full coordinate PKL into the final sharded store used by --coordinates_dir.')
    parser.add_argument('--coordinates_pkl', type=str, default=None, help='Input coordinate PKL to shard into per-anchor files.')
    parser.add_argument('--coordinates_out_dir', type=str, default=None, help='Output directory for sharded coordinates.')
    parser.add_argument('--coordinate_format', type=str, choices=['npy', 'pkl'], default='npy', help='Shard file format.')
    parser.add_argument('--aa_vec_model_path', type=str, default=None, help='Mol2vec Word2Vec model path.')
    parser.add_argument('--aa_vec_vocab_out', type=str, default=None, help='Output path for tiny amino-acid vocab file.')
    args = parser.parse_args()

    did_work = False
    if args.coordinates_pkl or args.coordinates_out_dir:
        if not (args.coordinates_pkl and args.coordinates_out_dir):
            raise ValueError('--coordinates_pkl and --coordinates_out_dir must be provided together')
        export_coordinate_shards(args.coordinates_pkl, args.coordinates_out_dir, fmt=args.coordinate_format)
        did_work = True
    if args.aa_vec_model_path or args.aa_vec_vocab_out:
        if not (args.aa_vec_model_path and args.aa_vec_vocab_out):
            raise ValueError('--aa_vec_model_path and --aa_vec_vocab_out must be provided together')
        export_aa_vocab(args.aa_vec_model_path, args.aa_vec_vocab_out)
        did_work = True
    if not did_work:
        raise ValueError('Nothing to do. Provide coordinate or aa-vocab export arguments.')


if __name__ == '__main__':
    main()
