# Setup instructions for GLProtein (refactor) + TMVecLoss wiring

These steps prepare the environment and required data to run:

- `run_pretrain_refactor.py` (MLM-only) or
- `run_pretrain_refactor.py` with `--use_tmvec_loss True` (adds global structure TM-Vec contrastive loss)

---

## (0) Prerequisites

- Python 3.9+
- A CUDA GPU is required for TM-Vec encoding with `Rostlab/prot_t5_xl_uniref50`.
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

## (2) Install base dependencies

```bash
pip install -r requirements.txt
```

---

## (3) Install TM-Vec from GitHub

```bash
pip install git+https://github.com/tymor22/tm-vec.git
python -c "import tm_vec; print('tm_vec imported OK')"
```

---

## (4) Obtain TM-Vec checkpoint + config JSON

`TMVecLoss` needs the following files:

- `tm_vec_model_ckpt`: TM-Vec model checkpoint
- `tm_vec_model_config_json`: TM-Vec model params JSON

Download them from https://figshare.com/s/e414d6a52fd471d86d69 and place them in `/assets/tmvec`.

Pass these paths to `run_pretrain_refactor.py`:
- `--tmvec_model_ckpt assets/tmvec/tm_vec_cath_model.ckpt`
- `--tmvec_model_config_json assets/tmvec/tm_vec_cath_model_params.json`

---

## (5) Prepare data

Download UniProt data from https://drive.google.com/file/d/1fsfE8kG6oBJor7tr2RJfOyFGAMj6woVM, decompress it and save it in `/data/pretrain_data`.

Run the following to create the TSV and NPY files for TM-Vec in `/data/pretrain_data`:
```bash
python generate_tmvec_pairs_tsv.py \
  --uniprot_dat data/pretrain_data/uniprot_sprot.dat \
  --out_tsv data/pretrain_data/tmvec_pairs.tsv \
  --out_emb_npy data/pretrain_data/tmvec_pairs_emb.npy \
  --tmvec_ckpt assets/tmvec/tm_vec_cath_model.ckpt \
  --tmvec_config assets/tmvec/tm_vec_cath_model_params.json \
  --device cuda \
  --top_k 5 \
  --use_faiss \
  --seed 2021 \
  --embed_checkpoint_dir embed_ckpt \
  --resume_embeddings \
  --max_proteins 1000
```

---

## (6) Verify pretraining with TMVecLoss
```bash
python run_pretrain_refactor.py \
  --output_dir outputs/mlm_plus_tmvec_precomputed \
  --use_tmvec_loss True \
  --tmvec_pairs_tsv tmvec_pairs.tsv \
  --tmvec_pairs_emb_npy tmvec_pairs_emb.npy \
  --per_device_train_batch_size 4 \
  --weight_decay 0.01 \
  --optimize_memory True \
  --lr_scheduler_type linear \
  --lm_learning_rate 1e-5 \
  --lm_warmup_ratio 0.167 \
  --seed 2021 \
  --fp16 \
  --dataloader_pin_memory \
  --max_steps 10 \
  --gradient_accumulation_steps 2 \
  --protein_seq_sample_limit 8
```

---

## (7) Full pretraining

> Note: The following arguments have not been tested.

To prepare the full dataset for TMVecLoss, run `generate_tmvec_pairs_tsv.py` without `--max_proteins`.

To run full pretraining, run `run_pretrain_refactor.py` with `--max_steps 300000`, `--gradient_accumulation_steps 256` and no `--protein_seq_sample_limit`.

---

## (8) Troubleshooting
- If you see an error about `sequence` missing, you are not using `ProteinSeqPairDataset` (or your collator did not pass through `sequence`).
- If you see an error about batch size needing to be even, set `--per_device_train_batch_size` to an even number.
