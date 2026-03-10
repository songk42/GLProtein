Here is a concrete memory-optimization patch plan based on your current `src_refactor_crossattn_maskfix.zip`.

## Main memory hotspots in this codebase

From the current files, the biggest avoidable RAM users are:

* `dataset.py`

  * `ProteinSeqTripletDataset` loads the entire triplet TSV into `self.triplets`
  * it also loads the entire coordinate pickle into `self.protein_cor`
  * `_build_coordinates()` and `_build_aa_vec()` return Python lists, which creates extra copies later
  * `_build_aa_vocab_from_mol2vec()` loads the full Word2Vec model at training startup

* `dataloader.py`

  * `_collate_batch_for_protein_cor()` and `_collate_batch_for_aa_vec()` convert each sample through Python list → NumPy → Torch, creating transient copies

* `trainer.py`

  * `self.loss_recorder` grows for the whole run
  * `loss_trace.json` and CSV rewriting also depend on keeping the whole trace in memory

Below is the patch plan I recommend, ordered by impact.

---

# Phase 1: biggest RAM savings with minimal pipeline disruption

## 1. Replace in-memory triplet rows with line-offset indexing

### File

`src_refactor/dataset.py`

### Current issue

`ProteinSeqTripletDataset.__init__()` reads the full TSV into:

```python
self.triplets = [...]
```

For a large triplet TSV, this stores every sequence string and metadata row in RAM.

### Patch

Refactor `ProteinSeqTripletDataset` so it stores:

* `self.triplets_tsv`
* `self.headers`
* `self.row_offsets`
* minimal cached metadata only:

  * `self.example_lengths`
  * `self.anchor_ids` if needed for validation

### Exact design

During `__init__()`:

1. open the TSV once
2. detect whether it has a header
3. record the byte offset of every usable row in `self.row_offsets`
4. compute `self.example_lengths` while scanning
5. collect only the minimal anchor-id set needed for validation

Then in `__getitem__()`:

* seek to the corresponding byte offset
* parse only that row
* tokenize and build local structure for just that sample

### Why this helps

This removes the largest “all triplets in RAM” structure while preserving the dataset API.

### What stays the same

* same TSV format
* same triplet route
* same anchor-only local structure behavior

---

## 2. Stop returning Python lists for coordinates and aa_vec

### File

`src_refactor/dataset.py`

### Current issue

These methods currently return nested Python lists:

* `_build_coordinates()`
* `_build_aa_vec()`

That creates extra Python-object overhead and extra conversions in the collator.

### Patch

Make both return `np.ndarray(dtype=np.float32)` instead.

### Exact changes

#### `_build_coordinates()`

Change return type from:

```python
List[List[float]]
```

to:

```python
np.ndarray
```

Use:

```python
return cor.astype(np.float32, copy=False)
```

#### `_build_aa_vec()`

Change return type from:

```python
List[List[float]]
```

to:

```python
np.ndarray
```

Return:

```python
np.asarray(aa_vec[:residue_count], dtype=np.float32)
```

### Dataclass update

In `ProteinSeqTripletInputFeatures`, loosen the type hints for:

* `anchor_coordinates`
* `anchor_aa_vec`

to allow arrays, or just leave them as optional generic payloads.

### Why this helps

It reduces:

* Python object count
* per-sample conversion overhead
* transient RAM spikes during collation

---

## 3. Make the collator consume arrays directly

### File

`src_refactor/dataloader.py`

### Current issue

The collator helpers do conversions like:

```python
torch.tensor(np.array(e.anchor_coordinates), dtype=torch.float)
```

That adds another copy per sample.

### Patch

Update:

* `_collate_batch_for_protein_cor()`
* `_collate_batch_for_aa_vec()`

to assume that triplet samples already carry `np.ndarray`.

### Exact changes

#### For coordinates

Replace per-example conversion with:

```python
arr = np.asarray(e.anchor_coordinates, dtype=np.float32)
tensor = torch.from_numpy(arr)
```

#### For aa_vec

Use:

```python
arr = np.asarray(e.anchor_aa_vec, dtype=np.float32)
tensor = torch.from_numpy(arr)
```

### Further improvement

When padding, prefer:

* allocate the final padded tensor once
* copy sample tensors into it

instead of building intermediate lists of tensors and then padding them.

This can be a second pass if needed, but even switching to `torch.from_numpy()` is already useful.

---

## 4. Stop keeping the entire loss history in RAM

### File

`src_refactor/trainer.py`

### Current issue

The resumable pipeline now does:

```python
self.loss_recorder.append(all_loss)
```

for every step.

On long runs this becomes large, and it is unnecessary because you already write JSONL.

### Patch

Change the logging design to be streaming-first.

### Exact changes

#### Keep

* `loss_trace.jsonl` as the append-only source of truth

#### Change

Replace full-history storage with a bounded buffer, e.g.:

```python
collections.deque(maxlen=1000)
```

or keep only:

* last N points
* running aggregates

### Checkpoint behavior

At checkpoint time:

* do not rewrite the full historical `loss_trace.json`
* instead either:

  * copy the JSONL file path reference into the checkpoint, or
  * write only the recent buffer and/or summary stats

### Recommended minimal patch

* keep `loss_trace.jsonl`
* keep `loss_trace.csv` only in `output_dir`, not every checkpoint
* remove full-run `self.loss_recorder` growth

### Why this helps

It is not the main 2-step crash cause, but it prevents long-run RAM growth.

---

# Phase 2: medium complexity, large additional savings

## 5. Add a lightweight coordinate store mode

### File

`src_refactor/dataset.py`

### Current issue

The dataset still does:

```python
self.protein_cor = pickle.load(f)
```

This loads the whole coordinate PKL into RAM.

### Patch

Add an optional coordinate storage mode that supports lazy loading.

### Recommended first implementation

Support a new on-disk format:

* `coordinates_shard_dir/`

  * one small pickle or `.npy` per anchor
* plus an index file if needed

Then add an argument like:

```python
coordinates_dir: Optional[str] = None
```

If `coordinates_dir` is set:

* do not load `self.protein_cor`
* instead load coordinates on demand in `_build_coordinates()`

### Why this is the best next step

It avoids a large in-memory Python dict and makes the pipeline much friendlier on 16 GB RAM.

### Why not force this immediately

It requires a companion preprocessing/export script, so I would treat it as the next structural upgrade after Phase 1.

---

## 6. Add a tiny precomputed aa-vocab file path

### File

`src_refactor/dataset.py`

### Current issue

`_build_aa_vocab_from_mol2vec()` loads the whole gensim model just to derive a tiny residue vocabulary.

### Patch

Add support for:

```python
aa_vec_vocab_path: Optional[str] = None
```

If provided:

* load the already-precomputed residue vectors directly from a tiny pickle / npy file

Fallback:

* keep current mol2vec model path behavior

### Exact design

Expected file content:

```python
Dict[str, np.ndarray]
```

for:

* A/R/N/D/...
* B/Z/X

### Why this helps

It removes unnecessary model-loading overhead at training time.

---

# Phase 3: optional but useful refinements

## 7. Add a “metadata-only validation scan” path for large triplet files

### File

`src_refactor/dataset.py`

### Current issue

Validation currently depends on in-memory triplet structures.

### Patch

When switching to line-offset indexing, perform validation during the initial TSV scan using only:

* row offsets
* anchor IDs
* example lengths

This preserves your strict mismatch checks without bringing the full TSV into RAM.

This fits naturally with Phase 1.

---

## 8. Reduce repeated token-to-residue work in `_build_aa_vec()`

### File

`src_refactor/dataset.py`

### Current issue

`_build_aa_vec()` repeatedly calls:

* `tokenizer.convert_ids_to_tokens(...)`
* Python string handling
* dict lookups

### Patch

Add a one-time token-id → residue-code lookup table at dataset init.

For example:

```python
self.token_id_to_residue = {token_id: 'A', ...}
```

Then `_build_aa_vec()` becomes:

* simple integer lookup
* vector gather

### Why this helps

It reduces Python overhead and per-sample temporary allocations.

---

## 9. Keep DataLoader workers at 0 for the triplet route on low-RAM VMs

### File

`src_refactor/trainer.py`

### Current issue

Your current protein-seq DataLoader construction appears not to pass `num_workers` in the triplet path, which is actually good for memory.

### Patch recommendation

Keep it that way for low-RAM environments, or explicitly force:

```python
num_workers=0
```

for the triplet local-structure route unless the user opts in.

This is more of a stability guard than a memory optimization patch.

---

# Recommended implementation order

## First patch set I would implement

1. `dataset.py`

   * switch triplet dataset from full in-memory rows to line offsets
   * return NumPy arrays from `_build_coordinates()` and `_build_aa_vec()`

2. `dataloader.py`

   * consume those arrays with `torch.from_numpy()`
   * reduce intermediate copies

3. `trainer.py`

   * replace `self.loss_recorder` with bounded-memory logging
   * keep JSONL as the main persistent trace

This should give the best memory reduction without changing the training objective.

## Second patch set

4. `dataset.py`

   * support lazy coordinate loading from a shard directory
5. `dataset.py`

   * support tiny precomputed aa-vocab files

---

# Expected impact by patch

## Highest impact

* line-offset triplet dataset
* lazy / sharded coordinate loading

## High impact

* NumPy arrays instead of Python lists
* `torch.from_numpy()` in collator

## Moderate impact

* tiny aa-vocab preload
* token-id lookup table

## Low but still worth doing

* bounded in-memory loss logging

---

# Files that should change

## Definitely

* `src_refactor/dataset.py`
* `src_refactor/dataloader.py`
* `src_refactor/trainer.py`

## Likely later

* `src_refactor/training_args.py`

  * if you add `coordinates_dir` / `aa_vec_vocab_path`

* possibly a new preprocessing/export script

  * for sharded coordinates
  * for tiny aa-vocab export

---

# Practical recommendation

For the next implementation step, I strongly recommend we do **only Phase 1 first**:

* line-offset triplet dataset
* array-based local-structure payloads
* streaming loss logging

That gives you the best immediate chance of making the current code usable on a 16 GB VM without forcing a new data format right away.

Then, if RAM is still too high, the next step should be the **lazy coordinate store**.
