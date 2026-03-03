import argparse
import gzip
import os
import random
from typing import Iterator, List, Tuple
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

try:
    from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
    from tm_vec.tm_vec_utils import encode
except Exception as e:
    raise ImportError("Failed to import tm_vec")

try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


def iter_uniprot_sequences(dat_path: str) -> Iterator[str]:
    """
    UniProt .dat/.dat.gz parser:
    - finds 'SQ   SEQUENCE' section
    - reads sequence lines until '//'
    - returns raw amino acid string
    """
    opener = gzip.open if dat_path.endswith(".gz") else open
    with opener(dat_path, "rt", encoding="utf-8", errors="ignore") as f:
        in_seq = False
        seq_parts: List[str] = []
        for line in f:
            if line.startswith("SQ   SEQUENCE"):
                in_seq = True
                seq_parts = []
                continue
            if in_seq:
                if line.startswith("//"):
                    seq = "".join(seq_parts).replace(" ", "").replace("\n", "")
                    if seq:
                        yield seq
                    in_seq = False
                    seq_parts = []
                else:
                    letters = "".join([c for c in line if c.isalpha()])
                    if letters:
                        seq_parts.append(letters)


def load_tmvec_models(ckpt: str, config_json: str, prot_t5_name: str, device: str):
    cfg = trans_basic_block_Config.from_json(config_json)
    tmvec_model = trans_basic_block.load_from_checkpoint(ckpt, config=cfg).to(device)
    t5 = T5EncoderModel.from_pretrained(prot_t5_name).to(device)
    tok = T5Tokenizer.from_pretrained(prot_t5_name, do_lower_case=False)
    # tm-vec expects tokenizer.batch_encode_plus in some versions
    if not hasattr(tok, "batch_encode_plus"):
        tok.batch_encode_plus = tok.__call__

    tmvec_model.eval()
    t5.eval()
    return tmvec_model, t5, tok


@torch.no_grad()
def encode_in_batches(
    seqs: List[str],
    tmvec_model,
    t5,
    tok,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """
    Uses tm-vec encode() in batches, returns numpy array (N, D).
    """
    embs: List[np.ndarray] = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        out = encode(batch, tmvec_model, t5, tok, device)
        out = np.asarray(out, dtype=np.float32)
        embs.append(out)
    return np.concatenate(embs, axis=0)


def build_index(embs: np.ndarray, use_faiss: bool):
    """
    Build a nearest neighbor search index over normalized embeddings.
    Similarity score is cosine after normalization.
    """
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs_n = embs / norms

    if use_faiss:
        if not _HAVE_FAISS:
            raise RuntimeError("use_faise=true but _HAVE_FAISS=false")
        d = embs_n.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embs_n.astype(np.float32))
        return ("faiss", index, embs_n)

    return ("bruteforce", None, embs_n)


def knn_search(kind, index, embs_n: np.ndarray, query_n: np.ndarray, top_k: int):
    """
    Returns KNN (scores, indices) for each query.
    """
    if kind == "faiss":
        scores, idxs = index.search(query_n.astype(np.float32), top_k)
        return scores, idxs

    scores = query_n @ embs_n.T
    idxs = np.argsort(-scores, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(scores, idxs, axis=1)
    return top_scores, idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniprot_dat", required=True, help="Path to uniprot_sprot.dat or .dat.gz")
    ap.add_argument("--out_tsv", required=True, help="Output TSV path")
    ap.add_argument("--tmvec_ckpt", required=True, help="TM-Vec checkpoint (.ckpt)")
    ap.add_argument("--tmvec_config", required=True, help="TM-Vec params JSON")
    ap.add_argument("--prot_t5_name", default="Rostlab/prot_t5_xl_uniref50")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_proteins", type=int, default=2000,
                    help="Limit number of proteins read from UniProt")
    ap.add_argument("--min_len", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=2000)
    ap.add_argument("--encode_batch_size", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--pairs_per_anchor", type=int, default=1,
                    help="How many positive pairs to emit per anchor (<= top_k)")
    ap.add_argument("--use_faiss", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load sequences
    seqs: List[str] = []
    for s in iter_uniprot_sequences(args.uniprot_dat):
        if len(s) < args.min_len:
            continue
        if len(s) > args.max_len:
            s = s[: args.max_len]
        seqs.append(s)
        if args.max_proteins and len(seqs) >= args.max_proteins:
            break

    if len(seqs) < 2:
        raise ValueError("Need at least 2 sequences")
    print(f"Loaded {len(seqs)} sequences.")

    # Load TM-Vec models
    tmvec_model, t5, tok = load_tmvec_models(args.tmvec_ckpt, args.tmvec_config, args.prot_t5_name, args.device)

    # Encode all sequences
    embs = encode_in_batches(
        seqs, tmvec_model, t5, tok, device=args.device, batch_size=args.encode_batch_size
    )
    print(f"Embeddings shape: {embs.shape}")

    # Build KNN index
    kind, index, embs_n = build_index(embs, use_faiss=args.use_faiss)

    # For each anchor, retrieve top_k+1 sequences and write pairs
    query_n = embs_n
    k_search = min(args.top_k + 1, len(seqs))
    scores, idxs = knn_search(kind, index, embs_n, query_n, k_search)

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    out_lines = 0
    with open(args.out_tsv, "w", encoding="utf-8") as out:
        for i in range(len(seqs)):
            neighbors = idxs[i].tolist()
            # Remove self
            neighbors = [j for j in neighbors if j != i]
            if not neighbors:
                continue

            for j in neighbors[: max(1, min(args.pairs_per_anchor, len(neighbors)))]:
                out.write(f"{seqs[i]}\t{seqs[j]}\n")
                out_lines += 1
    print(f"Wrote {out_lines} pairs to: {args.out_tsv}")


if __name__ == "__main__":
    main()
