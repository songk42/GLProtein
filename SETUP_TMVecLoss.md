# GLProtein + Global Structure Triplet Loss

## (0) Prerequisites

- Python 3.9+
- NVIDIA L4 GPU

---

## (1) Create and activate a virtual environment

From the repo root (the folder that contains `run_pretrain_refactor.py`), run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

---

## (2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## (3) Download data files

Construction of global structure triplets needs the following files:

- `swiss_large.npy`: TM-Vec embeddings of SwissProt
- `swissprot_seq.fasta`: Annotated sequences of SwissProt

Download them from https://zenodo.org/records/11199459 and place them in `/data/pretrain_data`.

Extraction of local structure coordinates needs the following file:
- `swissprot_pdb_v6.tar`: PDB files of SwissProt

Download it from https://alphafold.ebi.ac.uk/download#swissprot-section and decompress its contents to `/data/pretrain_data/alphafold`.

Pre-training with local structure coordinates needs the following file:
- `model_300dim.pkl`: pre-trained mol2vec model

Download it from https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl and place it in `/data/pretrain_data`.

You can also do all of the above with the following commands:

```bash
wget -c -P data/pretrain_data https://zenodo.org/records/11199459/files/swiss_large.npy
wget -c -P data/pretrain_data https://zenodo.org/records/11199459/files/swissprot_seq.fasta
wget -c -P data/pretrain_data https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar
wget -c -P data/pretrain_data https://raw.githubusercontent.com/samoturk/mol2vec/master/examples/models/model_300dim.pkl
mkdir data/pretrain_data/alphafold
tar -xvf data/pretrain_data/swissprot_pdb_v6.tar -C data/pretrain_data/alphafold
```

---

## (4) Triplet construction test

Run the following to construct `tmvec_triplets_small.tsv` in `/data/pretrain_data`:

```bash
python scripts_refactor/generate_tmvec_pairs_tsv.py \
  --swiss_fasta data/pretrain_data/swissprot_seq.fasta \
  --swiss_tmvec_emb_npy data/pretrain_data/swiss_large.npy \
  --out_tsv data/pretrain_data/tmvec_triplets_small.tsv \
  --out_metadata_json data/pretrain_data/tmvec_triplets_small.metadata.json \
  --top_k_pos 5 \
  --triplets_per_anchor 1 \
  --positive_search_k 64 \
  --negative_tmscore_max 0.2 \
  --negative_pick_strategy hardest \
  --use_faiss \
  --seed 2021 \
  --max_proteins 10000 \
  --log_every_anchors 1000 \
  --resume
```

---

## (5) Coordinate extraction test

Run the following to construct `coordinates_small.pkl` in `/data/pretrain_data`:

```bash
python scripts_refactor/extract_ca_coords.py \
  --input-dir data/pretrain_data/alphafold \
  --output data/pretrain_data/coordinates_small.pkl \
  --key-mode tsv_id \
  --fasta data/pretrain_data/swissprot_seq.fasta \
  --triplets-tsv data/pretrain_data/tmvec_triplets_small.tsv \
  --resume
```

---

## (6) Pre-training test

```bash
python run_pretrain_refactor.py \
  --model_protein_seq_data True \
  --use_tmvec_loss True \
  --output_dir outputs/glprotein_small \
  --pretrain_data_dir data/pretrain_data \
  --tmvec_triplets_tsv tmvec_triplets_small.tsv \
  --coordinates_path coordinates_small.pkl \
  --aa_vec_model_path model_300dim.pkl \
  --weight_decay 0.01 \
  --lr_scheduler_type linear \
  --lm_learning_rate 1e-5 \
  --lm_warmup_ratio 0.167 \
  --fp16 \
  --seed 2021 \
  --per_device_train_batch_size 2 \
  --logging_steps 1 \
  --save_steps 4 \
  --max_steps 8 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing True \
  --triplet_microbatch_size 1 \
  --max_tokens_per_batch 2048 \
  --max_protein_seq_length 1024 \
  --save_total_limit 2 \
  --filter_triplets_to_coordinate_coverage True \
  --auto_resume_from_latest True
```

---

## (7) Full triplet construction

```bash
python scripts_refactor/generate_tmvec_pairs_tsv.py \
  --swiss_fasta data/pretrain_data/swissprot_seq.fasta \
  --swiss_tmvec_emb_npy data/pretrain_data/swiss_large.npy \
  --out_tsv data/pretrain_data/tmvec_triplets_full.tsv \
  --out_metadata_json data/pretrain_data/tmvec_triplets_full.metadata.json \
  --top_k_pos 5 \
  --triplets_per_anchor 1 \
  --positive_search_k 64 \
  --negative_tmscore_max 0.2 \
  --negative_pick_strategy hardest \
  --use_faiss \
  --seed 2021 \
  --max_proteins 300000 \ # Edit this based on your dataset size
  --log_every_anchors 100 \
  --resume
```

Checkpoints are saved during training. If interrupted, make sure `--resume` is included in the command and re-run.

---

## (8) Full coordinate extraction

```bash
python scripts_refactor/extract_ca_coords.py \
  --input-dir data/pretrain_data/alphafold \
  --output data/pretrain_data/coordinates_full.pkl \
  --key-mode tsv_id \
  --fasta data/pretrain_data/swissprot_seq.fasta \
  --triplets-tsv data/pretrain_data/tmvec_triplets_full.tsv \
  --resume
```

Checkpoints are saved during training. If interrupted, make sure `--resume` is included in the command and re-run.

---

## (9) Full pre-training

```bash
python run_pretrain_refactor.py \
  --model_protein_seq_data True \
  --use_tmvec_loss True \
  --output_dir outputs/glprotein_full \
  --pretrain_data_dir data/pretrain_data \
  --tmvec_triplets_tsv tmvec_triplets_full.tsv \
  --weight_decay 0.01 \
  --lr_scheduler_type linear \
  --lm_learning_rate 1e-5 \
  --lm_warmup_ratio 0.167 \
  --fp16 \
  --seed 2021 \
  --per_device_train_batch_size 2 \
  --logging_steps 50 \
  --save_steps 1000 \
  --max_steps 300000 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing True \
  --triplet_microbatch_size 1 \
  --max_tokens_per_batch 2048 \
  --max_protein_seq_length 1024 \
  --save_total_limit 10 \
  --filter_triplets_to_coordinate_coverage True \
  --auto_resume_from_latest True
```
