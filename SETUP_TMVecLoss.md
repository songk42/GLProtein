# GLProtein + Global Structure Triplet Loss

## (0) Prerequisites

- Python 3.9+
- CUDA GPU
- To use Google Colab, upload `GLProtein_TMVec_Colab.ipynb` and follow its instructions.

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

---

## (3) Obtain Data

Construction of global structure triplets needs the following files:

- `swiss_large.npy`: TM-Vec embeddings of SwissProt
- `swissprot_seq.fasta`: Annotated sequences of SwissProt

Download them from https://zenodo.org/records/11199459 and place them in `/data/pretrain_data`.

Pre-training needs the following file:
- `swissprot_pdb_v6.tar`: PDB files of SwissProt

Download it from https://alphafold.ebi.ac.uk/download#swissprot-section and place it in `/data/pretrain_data`.

---

## (4) Triplet Construction Test

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
  --max_proteins 15000 \
  --log_every_anchors 1000 \
  --resume
```

---

## (5) Pre-Training Test

```bash
python run_pretrain_refactor.py \
  --output_dir outputs/glprotein_triplet_small \
  --pretrain_data_dir data/pretrain_data \
  --model_protein_seq_data True \
  --use_tmvec_loss True \
  --tmvec_triplets_tsv tmvec_triplets_small.tsv \
  --weight_decay 0.01 \
  --lr_scheduler_type linear \
  --lm_learning_rate 1e-5 \
  --lm_warmup_ratio 0.167 \
  --fp16 \
  --dataloader_pin_memory \
  --seed 2021 \
  --per_device_train_batch_size 2 \
  --logging_steps 1 \
  --save_steps 10 \
  --max_steps 30 \
  --gradient_accumulation_steps 5 \
  --gradient_checkpointing True \
  --triplet_microbatch_size 1 \
  --max_tokens_per_batch 4096 \
  --max_protein_seq_length 1024
```

---

## (6) Full triplet construction

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
  --max_proteins 300000 \
  --log_every_anchors 100 \
  --resume
```

Checkpoints are saved during training. If interrupted, make sure `--resume` is included in the command and re-run.

---

## (7) Full pre-training

```bash
python run_pretrain_refactor.py \
  --output_dir outputs/glprotein_triplet_full \
  --pretrain_data_dir data/pretrain_data \
  --model_protein_seq_data True \
  --use_tmvec_loss True \
  --tmvec_triplets_tsv tmvec_triplets_full.tsv \
  --weight_decay 0.01 \
  --lr_scheduler_type linear \
  --lm_learning_rate 1e-5 \
  --lm_warmup_ratio 0.167 \
  --fp16 \
  --dataloader_pin_memory \
  --seed 2021 \
  --per_device_train_batch_size 4 \
  --logging_steps 10 \
  --save_steps 500 \
  --max_steps 300000 \
  --gradient_accumulation_steps 256 \
  --gradient_checkpointing True \
  --triplet_microbatch_size 1 \
  --max_tokens_per_batch 4096 \
  --max_protein_seq_length 1024
```
