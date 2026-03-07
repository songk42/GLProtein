import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

try:
    from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
    from tm_vec.tm_vec_utils import encode
except Exception:
    raise ImportError("Failed to import tm_vec")

try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def cuda_mem_str() -> str:
    if not torch.cuda.is_available():
        return "cuda=n/a"
    try:
        device = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        return f"cuda_alloc={alloc:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB"
    except Exception:
        return "cuda=unavailable"


def resolve_torch_dtype(model_dtype: str, device: str, tmvec_half: bool):
    if model_dtype == "auto":
        if device == "cuda" and tmvec_half:
            return torch.float16
        return torch.float32
    if model_dtype == "float16":
        return torch.float16
    return torch.float32


def iter_uniprot_sequences(dat_path: str) -> Iterator[str]:
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


def load_tmvec_models(
    ckpt: str,
    config_json: str,
    prot_t5_name: str,
    device: str,
    model_dtype: str,
    tmvec_half: bool,
):
    torch_dtype = resolve_torch_dtype(model_dtype, device, tmvec_half)

    log("Loading TM-Vec config")
    cfg = trans_basic_block_Config.from_json(config_json)

    log("Loading TM-Vec checkpoint")
    tmvec_model = trans_basic_block.load_from_checkpoint(ckpt, config=cfg)
    if tmvec_half and device == "cuda":
        log("Keeping TM-Vec in float32 for compatibility")
    tmvec_model = tmvec_model.to(device)
    log(f"TM-Vec loaded | dtype={next(tmvec_model.parameters()).dtype} | {cuda_mem_str()}")

    log("Loading ProtT5 encoder")
    t5_kwargs = {}
    if device == "cuda":
        t5_kwargs["torch_dtype"] = torch_dtype
        t5_kwargs["low_cpu_mem_usage"] = True
    t5 = T5EncoderModel.from_pretrained(prot_t5_name, **t5_kwargs).to(device)
    if device == "cuda" and torch_dtype == torch.float16:
        t5 = t5.half()
    log(f"ProtT5 loaded | dtype={next(t5.parameters()).dtype} | {cuda_mem_str()}")

    tok = T5Tokenizer.from_pretrained(prot_t5_name, do_lower_case=False)
    if not hasattr(tok, "batch_encode_plus"):
        tok.batch_encode_plus = tok.__call__

    tmvec_model.eval()
    t5.eval()
    return tmvec_model, t5, tok


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _emb_ckpt_paths(embed_checkpoint_dir: str) -> Tuple[str, str]:
    emb_path = os.path.join(embed_checkpoint_dir, "embeddings.float32.npy")
    meta_path = os.path.join(embed_checkpoint_dir, "checkpoint_meta.json")
    return emb_path, meta_path


@torch.no_grad()
def encode_in_batches_checkpointed(
    seqs: List[str],
    tmvec_model,
    t5,
    tok,
    device: str,
    batch_size: int,
    log_every: int,
    checkpoint_dir: Optional[str],
    resume_embeddings: bool,
    emb_dim: int = 512,
) -> np.ndarray:
    num_seqs = len(seqs)
    num_batches = math.ceil(num_seqs / batch_size)
    t0 = time.time()

    if not checkpoint_dir:
        embs: List[np.ndarray] = []
        for batch_idx, i in enumerate(range(0, num_seqs, batch_size), start=1):
            batch = seqs[i:i + batch_size]
            b0 = time.time()
            out = encode(batch, tmvec_model, t5, tok, device)
            out = np.asarray(out, dtype=np.float32)
            embs.append(out)
            if batch_idx == 1 or batch_idx == num_batches or (log_every and batch_idx % log_every == 0):
                elapsed = time.time() - t0
                avg_per_batch = elapsed / batch_idx
                eta = avg_per_batch * (num_batches - batch_idx)
                log(
                    f"Batch {batch_idx}/{num_batches} | "
                    f"seqs {min(i + len(batch), num_seqs)}/{num_seqs} | "
                    f"last_batch={time.time() - b0:.2f}s | avg_batch={avg_per_batch:.2f}s | "
                    f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)} | {cuda_mem_str()}"
                )
        return np.concatenate(embs, axis=0)

    os.makedirs(checkpoint_dir, exist_ok=True)
    emb_path, meta_path = _emb_ckpt_paths(checkpoint_dir)

    start_batch = 0
    if resume_embeddings and os.path.exists(emb_path) and os.path.exists(meta_path):
        meta = _read_json(meta_path)
        expected = {
            "num_seqs": num_seqs,
            "batch_size": batch_size,
            "emb_dim": emb_dim,
        }
        for k, v in expected.items():
            if meta.get(k) != v:
                raise ValueError(
                    f"Embedding checkpoint mismatch for {k}: current run expects {v!r} but got {meta.get(k)!r}"
                )
        start_batch = int(meta.get("completed_batches", 0))
        log(f"Resuming embeddings from batch {start_batch + 1}/{num_batches}")
        embs_mm = np.memmap(emb_path, dtype=np.float32, mode="r+", shape=(num_seqs, emb_dim))
    else:
        log("Creating new embedding checkpoint")
        embs_mm = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(num_seqs, emb_dim))
        _write_json(
            meta_path,
            {
                "num_seqs": num_seqs,
                "batch_size": batch_size,
                "emb_dim": emb_dim,
                "completed_batches": 0,
                "completed_seqs": 0,
                "dtype": "float32",
                "created_at": _now(),
            },
        )

    for batch_idx, i in enumerate(range(0, num_seqs, batch_size), start=1):
        batch_end = min(i + batch_size, num_seqs)
        if batch_idx <= start_batch:
            if batch_idx == start_batch:
                log(f"Skipped already-checkpointed batches through {batch_end}/{num_seqs} sequences")
            continue

        batch = seqs[i:batch_end]
        b0 = time.time()
        out = encode(batch, tmvec_model, t5, tok, device)
        out = np.asarray(out, dtype=np.float32)
        if out.ndim != 2 or out.shape[0] != len(batch) or out.shape[1] != emb_dim:
            raise ValueError(
                f"Unexpected embedding shape from encode(): got {out.shape}, expected ({len(batch)}, {emb_dim})"
            )
        embs_mm[i:batch_end] = out
        embs_mm.flush()
        _write_json(
            meta_path,
            {
                "num_seqs": num_seqs,
                "batch_size": batch_size,
                "emb_dim": emb_dim,
                "completed_batches": batch_idx,
                "completed_seqs": batch_end,
                "dtype": "float32",
                "updated_at": _now(),
            },
        )

        if batch_idx == 1 or batch_idx == num_batches or (log_every and batch_idx % log_every == 0):
            effective_done = batch_idx
            elapsed = time.time() - t0
            avg_per_batch = elapsed / max(1, effective_done - start_batch)
            remaining = num_batches - batch_idx
            eta = avg_per_batch * remaining
            log(
                f"Batch {batch_idx}/{num_batches} | "
                f"seqs {batch_end}/{num_seqs} | "
                f"last_batch={time.time() - b0:.2f}s | avg_batch={avg_per_batch:.2f}s | "
                f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)} | checkpointed | {cuda_mem_str()}"
            )

    final = np.array(embs_mm, dtype=np.float32, copy=True)
    del embs_mm
    return final


def build_index(
    embs: np.ndarray,
    use_faiss: bool,
    approx_index: str,
    ivf_nlist: int,
    hnsw_m: int,
    hnsw_ef_search: int,
    hnsw_ef_construction: int,
):
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs_n = embs / norms

    if use_faiss:
        if not _HAVE_FAISS:
            raise RuntimeError("use_faiss=true but faiss is not installed")
        d = embs_n.shape[1]
        xb = embs_n.astype(np.float32)

        if approx_index == "flat":
            log("Building exact FAISS IndexFlatIP")
            index = faiss.IndexFlatIP(d)
            index.add(xb)
            return ("faiss_flat", index, embs_n)

        if approx_index == "ivfflat":
            nlist = max(1, min(ivf_nlist, xb.shape[0]))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            log("Training FAISS IndexIVFFlat")
            index.train(xb)
            index.add(xb)
            index.nprobe = min(max(1, int(round(math.sqrt(nlist)))), nlist)
            log("Built approximate FAISS IndexIVFFlat")
            return ("faiss_ivfflat", index, embs_n)

        if approx_index == "hnsw":
            log("Building approximate FAISS IndexHNSWFlat")
            index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = hnsw_ef_construction
            index.add(xb)
            index.hnsw.efSearch = hnsw_ef_search
            log("Built approximate FAISS IndexHNSWFlat")
            return ("faiss_hnsw", index, embs_n)

        raise ValueError(f"Unsupported approx_index: {approx_index}")

    log("Using brute-force numpy search")
    return ("bruteforce", None, embs_n)


def knn_search(kind, index, embs_n: np.ndarray, query_n: np.ndarray, top_k: int, query_chunk_size: int, log_every_chunks: int):
    num_queries = query_n.shape[0]
    if kind.startswith("faiss"):
        score_chunks = []
        idx_chunks = []
        num_chunks = math.ceil(num_queries / query_chunk_size)
        t0 = time.time()
        for chunk_idx, start in enumerate(range(0, num_queries, query_chunk_size), start=1):
            end = min(start + query_chunk_size, num_queries)
            q = query_n[start:end].astype(np.float32)
            c0 = time.time()
            scores, idxs = index.search(q, top_k)
            score_chunks.append(scores)
            idx_chunks.append(idxs)
            if chunk_idx == 1 or chunk_idx == num_chunks or (log_every_chunks and chunk_idx % log_every_chunks == 0):
                elapsed = time.time() - t0
                avg = elapsed / chunk_idx
                eta = avg * (num_chunks - chunk_idx)
                log(
                    f"Chunk {chunk_idx}/{num_chunks} | "
                    f"queries {end}/{num_queries} | last_chunk={time.time() - c0:.2f}s | "
                    f"avg_chunk={avg:.2f}s | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
                )
        return np.concatenate(score_chunks, axis=0), np.concatenate(idx_chunks, axis=0)

    log("Starting brute-force matrix multiplication")
    scores = query_n @ embs_n.T
    idxs = np.argsort(-scores, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(scores, idxs, axis=1)
    return top_scores, idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniprot_dat", required=True, help="Path to uniprot_sprot.dat or .dat.gz")
    ap.add_argument("--out_tsv", required=True, help="Output TSV path")
    ap.add_argument("--out_emb_npy", default=None, help="Output .npy path for precomputed TM-Vec embeddings")
    ap.add_argument("--emb_dtype", default="float16", choices=["float16", "float32"], help="Dtype for saved embeddings")
    ap.add_argument("--tmvec_ckpt", required=True, help="TM-Vec checkpoint (.ckpt)")
    ap.add_argument("--tmvec_config", required=True, help="TM-Vec params JSON")
    ap.add_argument("--prot_t5_name", default="Rostlab/prot_t5_xl_uniref50")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_proteins", type=int, default=None, help="Optional limit on accepted proteins")
    ap.add_argument("--min_len", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=2000)
    ap.add_argument("--encode_batch_size", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--pairs_per_anchor", type=int, default=1, help="How many positive pairs to emit per anchor (<= top_k)")
    ap.add_argument("--use_faiss", action="store_true")
    ap.add_argument("--approx_index", default="ivfflat", choices=["flat", "ivfflat", "hnsw"], help="FAISS index type")
    ap.add_argument("--ivf_nlist", type=int, default=4096, help="Number of IVF cells for ivfflat")
    ap.add_argument("--hnsw_m", type=int, default=32, help="Graph degree for HNSW")
    ap.add_argument("--hnsw_ef_search", type=int, default=128, help="efSearch for HNSW")
    ap.add_argument("--hnsw_ef_construction", type=int, default=200, help="efConstruction for HNSW")
    ap.add_argument("--model_dtype", default="float32", choices=["auto", "float16", "float32"], help="Precision for loading ProtT5 (auto: float16 on CUDA, float32 on CPU)")
    ap.add_argument("--tmvec_half", action="store_true", help="Whether to cast TM-Vec model to float16 on CUDA (currently disabled)")
    ap.add_argument("--embed_checkpoint_dir", default=None, help="Directory for embedding checkpoint")
    ap.add_argument("--resume_embeddings", action="store_true", help="Resume from existing embedding checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sequence_log_every", type=int, default=50000, help="Log every N accepted sequences while reading UniProt")
    ap.add_argument("--encode_log_every", type=int, default=25, help="Log every N encode batches")
    ap.add_argument("--knn_query_chunk_size", type=int, default=4096, help="Number of queries per FAISS search call")
    ap.add_argument("--knn_log_every", type=int, default=10, help="Log every N FAISS query chunks")
    ap.add_argument("--write_log_every", type=int, default=50000, help="Log every N anchors while writing TSV")
    args = ap.parse_args()

    if args.resume_embeddings and not args.embed_checkpoint_dir:
        raise ValueError("--resume_embeddings requires --embed_checkpoint_dir")

    random.seed(args.seed)
    np.random.seed(args.seed)

    t_start = time.time()
    log(f"Starting run with args: {' '.join(sys.argv[1:])}")
    log(f"Torch version={torch.__version__} | CUDA available={torch.cuda.is_available()} | {cuda_mem_str()}")

    seqs: List[str] = []
    read_t0 = time.time()
    for s in iter_uniprot_sequences(args.uniprot_dat):
        if len(s) < args.min_len:
            continue
        if len(s) > args.max_len:
            s = s[: args.max_len]
        seqs.append(s)
        if len(seqs) == 1 or (args.sequence_log_every and len(seqs) % args.sequence_log_every == 0):
            log(f"Loaded {len(seqs)} sequences")
        if args.max_proteins is not None and args.max_proteins > 0 and len(seqs) >= args.max_proteins:
            log("Reached max_proteins")
            break

    if len(seqs) < 2:
        raise ValueError("Need at least 2 sequences")
    log(f"Finished loading {len(seqs)} sequences in {format_seconds(time.time() - read_t0)}")

    tmvec_model, t5, tok = load_tmvec_models(
        args.tmvec_ckpt,
        args.tmvec_config,
        args.prot_t5_name,
        args.device,
        args.model_dtype,
        args.tmvec_half,
    )

    enc_t0 = time.time()
    embs = encode_in_batches_checkpointed(
        seqs,
        tmvec_model,
        t5,
        tok,
        device=args.device,
        batch_size=args.encode_batch_size,
        log_every=args.encode_log_every,
        checkpoint_dir=args.embed_checkpoint_dir,
        resume_embeddings=args.resume_embeddings,
        emb_dim=512,
    )
    log(f"Embeddings shape: {embs.shape} | encode_time={format_seconds(time.time() - enc_t0)}")

    if args.out_emb_npy is not None:
        dtype = np.float16 if args.emb_dtype == "float16" else np.float32
        est_pair_gb = (len(seqs) * max(1, args.pairs_per_anchor) * 2 * embs.shape[1] * np.dtype(dtype).itemsize) / (1024 ** 3)
        log(f"Estimated pair embedding file size: ~{est_pair_gb:.2f} GB")

    index_t0 = time.time()
    kind, index, embs_n = build_index(
        embs,
        use_faiss=args.use_faiss,
        approx_index=args.approx_index,
        ivf_nlist=args.ivf_nlist,
        hnsw_m=args.hnsw_m,
        hnsw_ef_search=args.hnsw_ef_search,
        hnsw_ef_construction=args.hnsw_ef_construction,
    )
    log(f"Index ready in {format_seconds(time.time() - index_t0)}")

    query_n = embs_n
    k_search = min(args.top_k + 1, len(seqs))
    knn_t0 = time.time()
    scores, idxs = knn_search(kind, index, embs_n, query_n, k_search, args.knn_query_chunk_size, args.knn_log_every)
    log(f"KNN search complete in {format_seconds(time.time() - knn_t0)} | scores_shape={scores.shape} idxs_shape={idxs.shape}")

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    out_lines = 0
    pair_indices: List[Tuple[int, int]] = []
    write_t0 = time.time()
    with open(args.out_tsv, "w", encoding="utf-8") as out:
        for i in range(len(seqs)):
            neighbors = idxs[i].tolist()
            neighbors = [j for j in neighbors if j != i and j >= 0]
            if not neighbors:
                continue

            for j in neighbors[: max(1, min(args.pairs_per_anchor, len(neighbors)))]:
                out.write(f"{seqs[i]}\t{seqs[j]}\n")
                pair_indices.append((i, j))
                out_lines += 1

            if i == 0 or i + 1 == len(seqs) or (args.write_log_every and (i + 1) % args.write_log_every == 0):
                elapsed = time.time() - write_t0
                avg = elapsed / (i + 1)
                eta = avg * (len(seqs) - (i + 1))
                log(
                    f"Anchors {i + 1}/{len(seqs)} | pairs={out_lines} | "
                    f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
                )
    log(f"Wrote {out_lines} pairs")

    if args.out_emb_npy is not None:
        save_t0 = time.time()
        dtype = np.float16 if args.emb_dtype == "float16" else np.float32
        pair_embs = np.empty((len(pair_indices) * 2, embs.shape[1]), dtype=dtype)
        for k, (a_idx, p_idx) in enumerate(pair_indices):
            pair_embs[2 * k] = embs[a_idx].astype(dtype, copy=False)
            pair_embs[2 * k + 1] = embs[p_idx].astype(dtype, copy=False)
            if k == 0 or k + 1 == len(pair_indices) or ((k + 1) % max(1, args.write_log_every) == 0):
                elapsed = time.time() - save_t0
                avg = elapsed / (k + 1)
                eta = avg * (len(pair_indices) - (k + 1))
                log(
                    f"Pairs {k + 1}/{len(pair_indices)} | "
                    f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
                )
        os.makedirs(os.path.dirname(args.out_emb_npy) or ".", exist_ok=True)
        np.save(args.out_emb_npy, pair_embs)
        log("Saved pair-order embeddings")

    log(f"Total runtime: {format_seconds(time.time() - t_start)}")


if __name__ == "__main__":
    main()
