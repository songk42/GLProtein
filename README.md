# GLProtein: Global-and-Local Structure Aware Protein Representation Learning

Implementation of the EMNLP 2025 Findings paper "[GLProtein: Global-and-Local Structure Aware Protein Representation Learning](https://arxiv.org/abs/2506.06294)".

GLProtein pre-trains a protein language model with three complementary structural signals:
- **Global structure** — triplet contrastive loss using TM-Vec structural similarity
- **Local 3D structure** — Gaussian-basis-kernel encoding of pairwise Cα distances as attention bias
- **Substructure** — mol2vec embeddings of amino acid molecules

![Architecture](figures/archi.png)

---

## Dependencies

**Hardware:** 32 GB RAM, NVIDIA GPU (L4 or equivalent)

**Python packages:**
```
python >= 3.9
torch >= 1.9
transformers >= 4.5.1
biopython
lmdb
gensim
mol2vec
rdkit
accelerate
seqeval
tmtools
tqdm
faiss-cpu        # or faiss-gpu for GPU acceleration
```

Create and activate a virtual environment, then install:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

For downstream evaluation, also install:
```bash
pip install tape-proteins scikit-multilearn PyYAML
# PPI task only — install per https://pytorch-geometric.readthedocs.io
pip install torch-geometric
```

The `tape_proteins` library is missing P@L/5 and P@L/2 metrics for contact prediction. Apply the patch:
```bash
cp replace_code/tape/modeling_utils.py $(python -c "import tape; print(tape.__file__.replace('__init__.py','models/modeling_utils.py'))")
```

---

## Data

### Pre-training data

1. **SwissProt TM-Vec embeddings and sequences** — download from [Zenodo](https://zenodo.org/records/11199459) and place in `data/pretrain_data/`:
   ```bash
   wget -c -P data/pretrain_data https://zenodo.org/records/11199459/files/swiss_large.npy
   wget -c -P data/pretrain_data https://zenodo.org/records/11199459/files/swissprot_seq.fasta
   ```

2. **AlphaFold SwissProt structures** — download from [AlphaFold DB](https://alphafold.ebi.ac.uk/download#swissprot-section) and extract to `data/pretrain_data/alphafold/`:
   ```bash
   wget -c -P data/pretrain_data https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar
   mkdir -p data/pretrain_data/alphafold
   tar -xf data/pretrain_data/swissprot_pdb_v6.tar -C data/pretrain_data/alphafold
   ```

3. **Mol2Vec model** — download and place in `data/pretrain_data/`:
   ```bash
   wget -c -P data/pretrain_data https://raw.githubusercontent.com/samoturk/mol2vec/master/examples/models/model_300dim.pkl
   ```

### Downstream task data

| Task set | Download |
|----------|----------|
| TAPE tasks (SS3/SS8, contact, remote homology, fluorescence, stability) | [Google Drive](https://drive.google.com/file/d/1snEAixeRokQW0wrJxLWtNA7m8VrzXN5A/view?usp=sharing) |
| PROBE tasks (semantic similarity, binding affinity) | [Google Drive](https://drive.google.com/file/d/1Sy0ldh_0fhAPatffTYJ7CENp3pbZHfyu/view?usp=sharing) |
| PPI task | Included in the TAPE download above |

---

## Pre-trained Models

| Model | Description | Location |
|-------|-------------|----------|
| [ProtBERT](https://huggingface.co/Rostlab/prot_bert) | Protein sequence encoder (initialisation) | HuggingFace |
| Mol2Vec | 300-dim amino acid substructure embeddings | `data/pretrain_data/model_300dim.pkl` |

A fine-tuned GLProtein checkpoint is available at: [Google Drive](https://drive.google.com/file/d) *(update link when available)*.

---

## Preprocessing

All preprocessing scripts are in `script/`. Each supports `--resume` to continue interrupted runs.

### 1. Generate structural triplets

Uses the precomputed SwissProt TM-Vec embeddings from Zenodo to build anchor/positive/negative triplets. For a small test run (10k proteins):

```bash
python script/generate_tmvec_pairs_tsv.py \
    --swiss_fasta data/pretrain_data/swissprot_seq.fasta \
    --swiss_tmvec_emb_npy data/pretrain_data/swiss_large.npy \
    --out_tsv data/pretrain_data/tmvec_triplets.tsv \
    --out_metadata_json data/pretrain_data/tmvec_triplets.metadata.json \
    --top_k_pos 5 \
    --triplets_per_anchor 1 \
    --positive_search_k 64 \
    --negative_tmscore_max 0.2 \
    --negative_pick_strategy hardest \
    --use_faiss \
    --seed 2021 \
    --max_proteins 10000 \
    --resume
```

For the full SwissProt dataset, set `--max_proteins 300000` (or remove the flag to use all).

Key arguments:
- `--swiss_fasta` — SwissProt FASTA file
- `--swiss_tmvec_emb_npy` — precomputed TM-Vec embeddings (from Zenodo)
- `--out_tsv` — output triplet TSV
- `--top_k_pos` — top-K neighbours considered as positives
- `--negative_tmscore_max` — TM-score ceiling for hard negatives
- `--negative_pick_strategy` — `hardest` (closest negative) or `random`
- `--use_faiss` — use FAISS for fast approximate k-NN (recommended)

### 2. Extract Cα coordinates

Reads the AlphaFold structure files and writes sharded per-protein coordinate arrays. Pass `--triplets-tsv` to restrict extraction to only the proteins used as anchors in training:

```bash
python script/extract_ca_coords.py \
    --input-dir data/pretrain_data/alphafold \
    --output data/pretrain_data/coordinates \
    --output-format sharded \
    --coordinate-format npy \
    --key-mode tsv_id \
    --fasta data/pretrain_data/swissprot_seq.fasta \
    --triplets-tsv data/pretrain_data/tmvec_triplets.tsv \
    --resume
```

Key arguments:
- `--input-dir` — directory of `.pdb` / `.cif` / `.cif.gz` files
- `--output` — output directory for sharded coordinate files
- `--key-mode tsv_id` — map AlphaFold accessions back to FASTA ID tokens used in the triplet TSV
- `--triplets-tsv` — restrict extraction to anchor proteins in this TSV
- `--chain` — restrict to a specific chain ID (default: all chains)

### 3. Prepare auxiliary assets

Export the mol2vec amino-acid vocabulary as a fast-loading pickle (avoids loading the full mol2vec model at training time):

```bash
python script/prepare_memory_assets.py \
    --aa_vec_model_path data/pretrain_data/model_300dim.pkl \
    --aa_vec_vocab_out data/pretrain_data/aa_vocab.pkl
```

---

## Training

### Pre-training

```bash
python run_pretrain_refactor.py \
    --model_protein_seq_data True \
    --use_tmvec_loss True \
    --pretrain_data_dir data/pretrain_data \
    --output_dir outputs/glprotein \
    --tmvec_triplets_tsv tmvec_triplets.tsv \
    --coordinates_dir coordinates \
    --aa_vec_vocab_path aa_vocab.pkl \
    --filter_triplets_to_coordinate_coverage True \
    --coordinate_cache_size 32 \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --lm_learning_rate 1e-5 \
    --lm_warmup_ratio 0.167 \
    --fp16 \
    --seed 2021 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --save_steps 1000 \
    --max_steps 300000 \
    --per_device_train_batch_size 1 \
    --triplet_microbatch_size 1 \
    --max_tokens_per_batch 2048 \
    --max_protein_seq_length 1024 \
    --save_total_limit 10 \
    --auto_resume_from_latest True
```

---

## Evaluation

### TAPE tasks

Run the hyperparameter sweep across all tasks:
```bash
bash script/run_sweep.sh \
    --model outputs/glprotein/checkpoint-final/encoder \
    --tasks contact,ss3,ss8 \
    --lrs 1e-5,1e-4 \
    --batch_sizes 2,4 \
    --epochs 15
```

Or fine-tune a single task directly:
```bash
python run_downstream.py \
    --model_name_or_path outputs/glprotein/checkpoint-final/encoder \
    --task_name contact \
    --data_dir data/downstream_datasets/contact \
    --output_dir outputs/contact \
    --do_train True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.08 \
    --fp16 True
```

Supported task names: `ss3`, `ss8`, `contact`, `remote_homology`, `fluorescence`, `stability`.

Individual task scripts (using defaults from the sweep) are in `script/`:
```bash
bash script/run_contact.sh
bash script/run_ss3.sh
bash script/run_ss8.sh
bash script/run_remote_homology.sh
bash script/run_fluorescence.sh
bash script/run_stability.sh
```

### PROBE tasks

Semantic similarity inference and binding affinity estimation ([PROBE](https://github.com/kansil/PROBE)):

1. Set the model and data paths in `src/benchmark/PROBE/extract_embeddings.py`
2. Extract embeddings:
   ```bash
   python src/benchmark/PROBE/extract_embeddings.py
   ```
3. Set paths in `src/benchmark/PROBE/bin/probe_config.yaml`
4. Run evaluation:
   ```bash
   python src/benchmark/PROBE/bin/PROBE.py
   ```

### PPI task

Protein-protein interaction prediction ([GNN-PPI](https://github.com/lvguofeng/GNN_PPI)):

1. Set model and data paths in `src/benchmark/GNN_PPI/extract_protein_embeddings.py`
2. Extract embeddings:
   ```bash
   python src/benchmark/GNN_PPI/extract_protein_embeddings.py
   ```
3. Set paths in `src/benchmark/GNN_PPI/run.py`
4. Train and evaluate:
   ```bash
   python src/benchmark/GNN_PPI/run.py
   ```

> PyTorch Geometric is required for the PPI task. Follow the [installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) matching your CUDA version.

---

## Acknowledgements

This implementation builds on [OntoProtein](https://github.com/zjunlp/OntoProtein) and [KeAP](https://github.com/RL4M/KeAP). The global structure component uses [TM-Vec](https://github.com/tymor22/tm-vec). Substructure encoding uses [mol2vec](https://github.com/samoturk/mol2vec). Downstream evaluation uses [TAPE](https://github.com/songlab-cal/tape), [PROBE](https://github.com/kansil/PROBE), and [GNN-PPI](https://github.com/lvguofeng/GNN_PPI).

---

## Citation

```bibtex
@article{liu2025glprotein,
  title={GLProtein: Global-and-Local Structure Aware Protein Representation Learning},
  author={Liu, Yunqing and Fan, Wenqi and Wei, Xiaoyong and Li, Qing},
  journal={arXiv preprint arXiv:2506.06294},
  year={2025}
}
```
