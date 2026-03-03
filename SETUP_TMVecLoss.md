# Setup instructions for GLProtein (refactor) + TMVecLoss wiring

These steps prepare the environment and required data to run:

- `run_pretrain_refactor.py` (MLM-only) or
- `run_pretrain_refactor.py` with `--use_tmvec_loss True` (adds global structure TM-Vec contrastive loss)

> Note: TMVecLoss expects paired batches: `(0,1), (2,3), ...` are positive pairs.
> This is enforced by `ProteinSeqPairDataset`, which reads a TSV file of paired sequences.

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

Create a TSV file where each line is `<anchor_sequence>\t<positive_sequence>`. Sequences should be raw amino-acid strings from UniProt. If using Google Colab, follow instructions in `GLProtein_TMVec_Colab.ipynb` to generate sequence pairs and save the file in `/data/pretrain_data`.

---

## (6) Run pretraining (MLM-only)

### (6.1) Quick verification

```bash
python run_pretrain_refactor.py \
  --output_dir outputs/mlm_only_quick \
  --per_device_train_batch_size 2 \
  --protein_seq_sample_limit 5 \
  --max_steps 10
```

### (6.2) Full pretraining

```bash
python run_pretrain_refactor.py \
  --output_dir outputs/mlm_only_full \
  --max_steps <SET_AS_PAPER>
```

---

## (7) Run pretraining with TMVecLoss (global structure)

### (7.1) Quick verification
```bash
python run_pretrain_refactor.py \
  --output_dir outputs/mlm_plus_tmvec_quick \
  --per_device_train_batch_size 2 \
  --protein_seq_sample_limit 5 \
  --max_steps 10 \
  --tmvec_pairs_tsv data/pretrain_data/tmvec_pairs.tsv \
  --use_tmvec_loss True \
  --tmvec_model_ckpt assets/tmvec/tm_vec_cath_model.ckpt \
  --tmvec_model_config_json assets/tmvec/tm_vec_cath_model_params.json \
  --tmvec_device cuda
```

### (7.2) Full pretraining

```bash
python run_pretrain_refactor.py \
  --output_dir outputs/mlm_plus_tmvec_full \
  --max_steps <SET_AS_PAPER> \
  --tmvec_pairs_tsv data/pretrain_data/tmvec_pairs.tsv \
  --use_tmvec_loss True \
  --tmvec_model_ckpt assets/tmvec/tm_vec_cath_model.ckpt \
  --tmvec_model_config_json assets/tmvec/tm_vec_cath_model_params.json \
  --tmvec_device cuda
```

---

## (8) Troubleshooting
- If you see an error about `sequence` missing, you are not using `ProteinSeqPairDataset` (or your collator did not pass through `sequence`).
- If you see an error about batch size needing to be even, set
  `--per_device_train_batch_size` to an even number.
