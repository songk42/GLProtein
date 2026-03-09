import json
import os
import math
import csv
import random
import shutil
import collections
import time
from tqdm import trange
from packaging import version
from typing import Optional, Tuple, Union, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, Sampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, PreTrainedModel, logging, BertPreTrainedModel, BertModel,T5ForConditionalGeneration
# from transformers.deepspeed import deepspeed_init
# from transformers.training_args import ShardedDDPOption, ParallelMode
from transformers.trainer_pt_utils import get_parameter_names, IterableDatasetShard
# from transformers.optimization import Adafactor, AdamW
from transformers.optimization import Adafactor
from torch.optim import AdamW
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.file_utils import is_apex_available, is_sagemaker_mp_enabled

from src_refactor.dataset import GoGoDataset, ProteinGoDataset, ProteinSeqDataset, ProteinSeqTripletDataset
from src_refactor.dataloader import DataCollatorForLanguageModeling, DataCollatorForGoGo, DataCollatorForProteinGo
from src_refactor.models import GLProtein, KnowledgeDecoder, GLProteinLoss, TMVecLoss, GlobalStructureTripletLoss
from src_refactor.optimization import get_scheduler


logger = logging.get_logger(__name__)




def _sorted_checkpoints(output_dir: str) -> List[str]:
    checkpoints: List[Tuple[int, str]] = []
    if not output_dir or not os.path.isdir(output_dir):
        return []
    prefix = f"{PREFIX_CHECKPOINT_DIR}-"
    for name in os.listdir(output_dir):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if not suffix.isdigit():
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            checkpoints.append((int(suffix), path))
    checkpoints.sort(key=lambda x: x[0])
    return [path for _, path in checkpoints]


def _checkpoint_step(checkpoint_path: str) -> int:
    base = os.path.basename(os.path.normpath(checkpoint_path))
    if base.startswith(f"{PREFIX_CHECKPOINT_DIR}-"):
        suffix = base[len(f"{PREFIX_CHECKPOINT_DIR}-"):]
        if suffix.isdigit():
            return int(suffix)
    return 0

# if is_apex_available():
#     from apex import amp


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

# Data parallelism: sharded_ddp
# Model parallelism: deepspeed



class PairBatchSampler(Sampler[List[int]]):
    """Batch sampler that shuffles pair rows while preserving anchor/positive adjacency."""

    def __init__(self, dataset, pairs_per_batch: int, generator: Optional[torch.Generator] = None, drop_last: Optional[bool] = None):
        if pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be > 0")
        if len(dataset) % 2 != 0:
            raise ValueError("Dataset flattened length must be even")
        self.dataset = dataset
        self.num_pairs = len(dataset) // 2
        self.pairs_per_batch = int(pairs_per_batch)
        self.generator = generator
        self.drop_last = bool(getattr(dataset, '_drop_last_for_pairs', False) if drop_last is None else drop_last)

    def __iter__(self):
        if self.generator is None:
            perm = torch.randperm(self.num_pairs).tolist()
        else:
            perm = torch.randperm(self.num_pairs, generator=self.generator).tolist()
        batch = []
        for pair_idx in perm:
            base_idx = 2 * pair_idx
            batch.extend([base_idx, base_idx + 1])
            if len(batch) == self.pairs_per_batch * 2:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_pairs // self.pairs_per_batch
        return math.ceil(self.num_pairs / self.pairs_per_batch)


class TokenBudgetBatchSampler(BatchSampler):
    """Batch sampler that caps padded tokens per batch using example lengths."""

    def __init__(self, dataset, max_tokens: int, drop_last: bool, generator: Optional[torch.Generator] = None, bucket_size_multiplier: int = 20, max_batch_size: Optional[int] = None):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not hasattr(dataset, 'get_example_length'):
            raise ValueError("TokenBudgetBatchSampler requires dataset.get_example_length(index)")
        self.dataset = dataset
        self.max_tokens = int(max_tokens)
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.max_batch_size = int(max_batch_size) if max_batch_size is not None and int(max_batch_size) > 0 else None
        anchor_batch = self.max_batch_size if self.max_batch_size is not None else 1
        self.bucket_size = max(anchor_batch, int(bucket_size_multiplier) * anchor_batch)

    def _fits(self, current_batch: List[int], current_max_len: int, next_len: int) -> bool:
        proposed_count = len(current_batch) + 1
        if self.max_batch_size is not None and proposed_count > self.max_batch_size:
            return False
        proposed_max_len = max(current_max_len, next_len)
        return proposed_max_len * proposed_count <= self.max_tokens

    def __iter__(self):
        n = len(self.dataset)
        if self.generator is None:
            perm = torch.randperm(n).tolist()
        else:
            perm = torch.randperm(n, generator=self.generator).tolist()

        pooled_batches = []
        for start in range(0, n, self.bucket_size):
            pool = perm[start:start + self.bucket_size]
            pool.sort(key=lambda idx: self.dataset.get_example_length(idx))

            current_batch: List[int] = []
            current_max_len = 0
            for idx in pool:
                next_len = int(self.dataset.get_example_length(idx))
                if current_batch and not self._fits(current_batch, current_max_len, next_len):
                    pooled_batches.append(current_batch)
                    current_batch = []
                    current_max_len = 0

                # Always allow at least one over-budget example to form a singleton batch.
                current_batch.append(idx)
                current_max_len = max(current_max_len, next_len)

            if current_batch:
                pooled_batches.append(current_batch)

        if self.drop_last:
            pooled_batches = [batch for batch in pooled_batches if len(batch) > 1 or (batch and int(self.dataset.get_example_length(batch[0])) * len(batch) <= self.max_tokens)]

        if pooled_batches:
            if self.generator is None:
                order = torch.randperm(len(pooled_batches)).tolist()
            else:
                order = torch.randperm(len(pooled_batches), generator=self.generator).tolist()
            for i in order:
                batch = pooled_batches[i]
                if batch and (not self.drop_last or len(batch) > 0):
                    yield batch

    def __len__(self):
        lengths = sorted(int(self.dataset.get_example_length(idx)) for idx in range(len(self.dataset)))
        total = 0
        current_count = 0
        current_max_len = 0
        for ex_len in lengths:
            proposed_count = current_count + 1
            proposed_max_len = max(current_max_len, ex_len)
            exceeds_token_budget = proposed_max_len * proposed_count > self.max_tokens
            exceeds_batch_size = self.max_batch_size is not None and proposed_count > self.max_batch_size
            if current_count and (exceeds_token_budget or exceeds_batch_size):
                total += 1
                current_count = 0
                current_max_len = 0
                proposed_count = 1
                proposed_max_len = ex_len
            current_count = proposed_count
            current_max_len = proposed_max_len
        if current_count and not self.drop_last:
            total += 1
        elif current_count and total == 0:
            total = 1
        return total


class LengthBucketBatchSampler(BatchSampler):
    """Batch sampler that groups examples with similar lengths to reduce padding and VRAM spikes."""

    def __init__(self, dataset, batch_size: int, drop_last: bool, generator: Optional[torch.Generator] = None, bucket_size_multiplier: int = 20):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not hasattr(dataset, 'get_example_length'):
            raise ValueError("LengthBucketBatchSampler requires dataset.get_example_length(index)")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.bucket_size = max(self.batch_size, int(bucket_size_multiplier) * self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        if self.generator is None:
            perm = torch.randperm(n).tolist()
        else:
            perm = torch.randperm(n, generator=self.generator).tolist()
        pooled_batches = []
        for start in range(0, n, self.bucket_size):
            pool = perm[start:start + self.bucket_size]
            pool.sort(key=lambda idx: self.dataset.get_example_length(idx))
            for bstart in range(0, len(pool), self.batch_size):
                batch = pool[bstart:bstart + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    pooled_batches.append(batch)
        if self.generator is None:
            order = torch.randperm(len(pooled_batches)).tolist()
        else:
            order = torch.randperm(len(pooled_batches), generator=self.generator).tolist()
        for i in order:
            batch = pooled_batches[i]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

class GLProteinTrainer(Trainer):
    """
    GLProtein implement the pretraining of protein language model with knowledge injection, which class 
    inherits from `transformer.Trainer`.
    Note we dont use go_go data

    Args:
        model: The model to train, evaluate or use for prediction
        args: The hyper-parameters for training, which will default to a basic instance :class:`~transformers.TrainingArguments`
        protein_seq_dataset: The instance of :class:`ProteinSeqDataset` 
        protein_go_dataset: The instance of :class:`ProteinGoDataset`
        go_go_dataset: The instance of :class:`GoGoDataset`
        protein_seq_data_collator: Data collator for :obj:`protein_seq_dataset`
        protein_go_data_collator: Data collator for :obj:`protein_go_dataset`
        go_go_data_collator: Data collator for :obj:`go_go_dataset`
    """

    def __init__(
        self,
        model: Union[nn.Module, PreTrainedModel],
        args,
        protein_seq_dataset: ProteinSeqDataset = None,
        protein_go_dataset: ProteinGoDataset = None,
        go_go_dataset: GoGoDataset = None,
        protein_seq_data_collator: DataCollatorForLanguageModeling = None,
        protein_go_data_collator: DataCollatorForProteinGo = None,
        go_go_data_collator: DataCollatorForGoGo = None,
    ):
        super().__init__(
            model=model,
            args=args
        )
        # note model is of class ontoproteinModel
        self.protein_seq_dataset = protein_seq_dataset
        self.protein_go_dataset = protein_go_dataset
        self.go_go_dataset = go_go_dataset
        self.protein_seq_data_collator = protein_seq_data_collator
        self.protein_go_data_collator = protein_go_data_collator
        self.go_go_data_collator = go_go_data_collator

        self.model_loss = GLProteinLoss(pfi_weight = self.args.pfi_lambda, mlm_lambda=self.args.mlm_lambda,
            num_protein_go_neg_sample=self.args.num_protein_go_neg_sample)

        # Optional global structure component (TM-Vec loss)
        self.tmvec_loss = None
        self.triplet_structure_loss = None
        if getattr(self.args, "use_tmvec_loss", False):
            if isinstance(self.protein_seq_dataset, ProteinSeqTripletDataset):
                self.triplet_structure_loss = GlobalStructureTripletLoss(
                    margin=getattr(self.args, "triplet_margin", 0.2),
                    distance_type=getattr(self.args, "triplet_distance_type", "l2"),
                )
            else:
                self.tmvec_loss = TMVecLoss(
                    temperature=self.args.tmvec_temperature,
                    distill_weight=getattr(self.args, "tmvec_distill_weight", 0.0),
                )

        self.use_amp = False
        self.loss_recorder = []
        self.loss_trace_file = os.path.join(self.args.output_dir, "loss_trace.jsonl") if getattr(self.args, "output_dir", None) else None


    def _append_loss_trace(self, record: Dict[str, Any]) -> None:
        if not self.loss_trace_file:
            return
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(self.loss_trace_file, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_loss_trace_csv(self, output_dir: str) -> None:
        if not self.loss_recorder:
            return
        fieldnames: List[str] = []
        for row in self.loss_recorder:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        csv_path = os.path.join(output_dir, 'loss_trace.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.loss_recorder:
                writer.writerow(row)

    def _save_rng_state(self, output_dir: str) -> None:
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state['cuda'] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, os.path.join(output_dir, 'rng_state.pth'))

    def _load_rng_state(self, checkpoint_dir: str) -> None:
        rng_path = os.path.join(checkpoint_dir, 'rng_state.pth')
        if not os.path.exists(rng_path):
            return
        rng_state = torch.load(rng_path, map_location='cpu')
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and 'cuda' in rng_state:
            torch.cuda.set_rng_state_all(rng_state['cuda'])

    def _load_loss_trace(self, checkpoint_dir: str) -> None:
        loss_path = os.path.join(checkpoint_dir, 'loss_trace.json')
        if not os.path.exists(loss_path):
            self.loss_recorder = []
            return
        with open(loss_path, 'r', encoding='utf-8') as handle:
            self.loss_recorder = json.load(handle)
        if self.loss_trace_file:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(self.loss_trace_file, 'w', encoding='utf-8') as handle:
                for row in self.loss_recorder:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _rotate_checkpoints(self) -> None:
        save_total_limit = getattr(self.args, 'save_total_limit', None)
        if save_total_limit is None or save_total_limit <= 0:
            return
        checkpoints = _sorted_checkpoints(self.args.output_dir)
        excess = len(checkpoints) - int(save_total_limit)
        for checkpoint in checkpoints[:max(0, excess)]:
            shutil.rmtree(checkpoint, ignore_errors=True)

    def _load_non_deepspeed_checkpoint(self, checkpoint_dir: str) -> None:
        logger.info("Loading model state from checkpoint %s", checkpoint_dir)
        self._load_from_checkpoint(checkpoint_dir)
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        if os.path.exists(trainer_state_path):
            self.state = TrainerState.load_from_json(trainer_state_path)
        self._load_optimizer_and_scheduler(checkpoint_dir)
        self._load_rng_state(checkpoint_dir)
        self._load_loss_trace(checkpoint_dir)


    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Rewrite '~transformers.Trainer.train'
        """

        args = self.args

        # print('args:', args)
        # import ipdb; ipdb.set_trace()

        self.is_in_train = True
        
        # Keeping track whether we can len() on the train dataset.
        train_dataset_is_sized = isinstance(self.protein_seq_dataset, collections.abc.Sized) or isinstance(self.protein_go_dataset, collections.abc.Sized)

        # Dataloader
        protein_seq_dataloader, protein_go_dataloader = self.get_train_dataloader()

        # protein_seq_dataloader = None
        
        total_train_protein_seq_batch_size = args.train_protein_seq_batch_size * args.gradient_accumulation_steps * args.world_size
        total_train_protein_go_batch_size = args.train_protein_go_batch_size * args.gradient_accumulation_steps * args.world_size

        if train_dataset_is_sized:
            num_protein_seq_update_steps_per_epoch = max(len(protein_seq_dataloader) // args.gradient_accumulation_steps, 1) if protein_seq_dataloader else -1
            num_protein_go_update_steps_per_epoch = max(len(protein_go_dataloader) // args.gradient_accumulation_steps, 1) if protein_go_dataloader else -1

            if args.max_steps > 0:
                max_protein_seq_steps = args.max_steps
                num_protein_seq_epochs = args.max_steps // num_protein_seq_update_steps_per_epoch + int(
                    args.max_steps % num_protein_seq_update_steps_per_epoch > 0
                ) if num_protein_seq_update_steps_per_epoch else 0
                num_protein_seq_train_samples = args.max_steps * total_train_protein_seq_batch_size

                # max_protein_go_steps = args.max_steps
                # num_protein_go_epochs = args.max_steps // num_protein_go_update_steps_per_epoch + int(
                #     args.max_steps % num_protein_go_update_steps_per_epoch > 0
                # ) if num_protein_go_update_steps_per_epoch else 0
                # num_protein_go_train_samples = args.max_steps * total_train_protein_go_batch_size

            else:
                max_protein_seq_steps = math.ceil(args.num_protein_seq_epochs * num_protein_seq_update_steps_per_epoch)
                num_protein_seq_epochs = math.ceil(args.num_protein_seq_epochs)
                num_protein_seq_train_samples = len(self.protein_seq_dataset) * args.num_protein_seq_epochs
            
                # max_protein_go_steps = math.ceil(args.num_protein_go_epochs * num_protein_go_update_steps_per_epoch)
                # num_protein_go_epochs = math.ceil(args.num_protein_go_epochs)
                # num_protein_go_train_samples = len(self.protein_go_dataset) * args.num_protein_go_epochs
        else:
            raise NotImplementedError("Not support dataset which don't implement `__len__`.")
        
        # delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        delay_optimizer_creation = False

        # TODO: Only support same max steps of training on the three dataset at present.
        # assert max_protein_seq_steps == max_protein_go_steps, "Only support same max_steps on the two dataset"
        max_steps = max_protein_seq_steps

        if args.deepspeed:
            # print('Using deepspeed')
            # import ipdb; ipdb.set_trace()

            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )

            # print('Optimizer:', optimizer)
            # import ipdb; ipdb.set_trace()

            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        if resume_from_checkpoint and not self.deepspeed:
            self._load_non_deepspeed_checkpoint(resume_from_checkpoint)
        else:
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train
        num_protein_seq_examples = (
            self.num_examples(protein_seq_dataloader) if train_dataset_is_sized else total_train_protein_seq_batch_size * max_steps
        )

        #debug
        # print("num_protein_seq_examples",num_protein_seq_examples)
        # import pdb
        # pdb.set_trace()
        
        # num_protein_go_examples = (
        #     self.num_examples(protein_go_dataloader) if train_dataset_is_sized else total_train_protein_go_batch_size * max_steps
        # )

        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {num_protein_seq_examples} | {num_protein_go_examples}")
        # logger.info(f"  Num Epochs = {num_protein_seq_epochs} | {num_protein_go_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_protein_seq_batch_size} | {total_train_protein_go_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        start_time = time.time()
        raw_steps_trained = int(self.state.global_step)
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        if isinstance(protein_seq_dataloader, DataLoader) and isinstance(protein_seq_dataloader.sampler, DistributedSampler):
            protein_seq_dataloader.sampler.set_epoch(0)

        protein_seq_iter = iter(protein_seq_dataloader) if protein_seq_dataloader else None
        # protein_go_iter = iter(protein_go_dataloader) if protein_go_dataloader else None

        num_protein_seq_steps_per_epoch = max(len(protein_seq_dataloader), 1) if protein_seq_dataloader else -1
        # num_protein_go_steps_per_epoch = max(len(protein_go_dataloader), 1) if protein_go_dataloader else -1

        # debug
        # print("num_protein_seq_steps_per_epoch",num_protein_seq_steps_per_epoch)
        # import pdb
        # pdb.set_trace()

        # record epoch for update of seed on dataloaders.
        cur_protein_seq_epoch = 0
        # cur_protein_go_epoch = 0

        if raw_steps_trained > 0 and num_protein_seq_steps_per_epoch > 0:
            epochs_trained = raw_steps_trained
            cur_protein_seq_epoch = raw_steps_trained // num_protein_seq_steps_per_epoch
            steps_trained_in_current_epoch = raw_steps_trained % num_protein_seq_steps_per_epoch
            self.state.epoch = raw_steps_trained / max(num_protein_seq_steps_per_epoch, 1)
            if isinstance(protein_seq_dataloader.sampler, DistributedSampler):
                protein_seq_dataloader.sampler.set_epoch(cur_protein_seq_epoch)
            elif isinstance(protein_seq_dataloader.dataset, IterableDatasetShard):
                protein_seq_dataloader.dataset.set_epoch(cur_protein_seq_epoch)
            protein_seq_iter = iter(protein_seq_dataloader) if protein_seq_dataloader else None
            for _ in range(steps_trained_in_current_epoch):
                if protein_seq_iter is not None:
                    next(protein_seq_iter)
            logger.info(
                "Resuming training from step %d (epoch index %d, step offset %d within epoch)",
                raw_steps_trained,
                cur_protein_seq_epoch,
                steps_trained_in_current_epoch,
            )
        else:
            self.state.epoch = 0

        train_iterator = range(
            epochs_trained, max_steps
        )

        for step in train_iterator:
            # tempt = time.time()

            # update the seed of dataloader
            if num_protein_seq_steps_per_epoch != -1 and (step + 1) % num_protein_seq_steps_per_epoch == 0:
                cur_protein_seq_epoch += 1
                if isinstance(protein_seq_dataloader.sampler, DistributedSampler):
                    protein_seq_dataloader.sampler.set_epoch(cur_protein_seq_epoch)
                elif isinstance(protein_seq_dataloader.dataset, IterableDatasetShard):
                    protein_seq_dataloader.dataset.set_epoch(cur_protein_seq_epoch)
                protein_seq_iter = iter(protein_seq_dataloader)

            # if num_protein_go_steps_per_epoch != -1 and (step + 1) % num_protein_go_steps_per_epoch == 0:
            #     cur_protein_go_epoch += 1
            #     if isinstance(protein_go_dataloader.sampler, DistributedSampler):
            #         protein_go_dataloader.sampler.set_epoch(cur_protein_go_epoch)
            #     elif isinstance(protein_go_dataloader.dataset, IterableDatasetShard):
            #         protein_go_dataloader.dataset.set_epoch(cur_protein_go_epoch)
            #     protein_go_iter = iter(protein_go_dataloader)

            protein_seq_inputs = None
            protein_go_inputs = None
            go_go_inputs = None

            if protein_seq_iter:
                protein_seq_inputs = next(protein_seq_iter)
            
            # if protein_go_iter:
            #     # protein_go_inputs = protein_go_iter.next()
            #     protein_go_inputs = next(protein_go_iter)
            # import ipdb;ipdb.set_trace()


            # #debug
            # print("protein_go_inputs",protein_go_inputs)
            # print("protein_seq_inputs",protein_seq_inputs)
            # import pdb
            # pdb.set_trace() 

            if (
                ((step + 1) % args.gradient_accumulation_steps != 0)
                and args.local_rank != -1
                and args._no_sync_in_gradient_accumulation
            ):
                # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                with model.no_sync():
                    loss, all_loss = self.training_step(model, protein_seq_inputs, protein_go_inputs, go_go_inputs)
                    tr_loss += loss
            else:
                loss, all_loss = self.training_step(model, protein_seq_inputs, protein_go_inputs, go_go_inputs)
                tr_loss += loss

            # record loss.
            if args.local_rank == -1 or args.local_rank == 0:
                all_loss['global_step'] = int(self.state.global_step)
                all_loss['learning_rate'] = self.get_learning_rate()
                all_loss = dict(all_loss)
                logger.info("loss and lr dict: %s",str(all_loss))
                print(all_loss)
                self.loss_recorder.append(all_loss)
                self._append_loss_trace(all_loss)

            # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
            if self.deepspeed:
                self.deepspeed.step()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                    # deepspeed does its own clipping

                    if self.use_amp:
                        # AMP: gradients need unscaling
                        self.scaler.unscale_(self.optimizer)

                    if hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(args.max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(args.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer_was_run = True
                if self.deepspeed:
                    pass  # called outside the loop
                elif self.use_amp:
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    self.optimizer.step()

                if optimizer_was_run and not self.deepspeed:
                    self.lr_scheduler.step()
                model.zero_grad()

            self.state.global_step += 1
            self.state.epoch = (step + 1) / max(num_protein_seq_steps_per_epoch, 1) if num_protein_seq_steps_per_epoch > 0 else 0

            if args.save_steps > 0 and self.state.global_step % args.save_steps == 0:
                self._save_checkpoint()

            # print("forward propagation time",time.time()-tempt)
        
        logger.info("\n\nTraining completed.")
        self.is_in_train = False
        self._save_checkpoint()

    def get_learning_rate(self):
        if self.deepspeed:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            last_lr = (
                # backward compatibility for pytorch schedulers
                self.lr_scheduler.get_last_lr()
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()
            )
        return last_lr

    def _save_checkpoint(self):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"

        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        print(f"Saving checkpoint to {output_dir}")
        self._save(output_dir)
        if self.deepspeed:
            self.deepspeed.save_checkpoint(output_dir)
        else:
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
            self.state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))
            self._save_rng_state(output_dir)

        # save loss traces.
        with open(os.path.join(output_dir, 'loss_trace.json'), 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(self.loss_recorder, indent=2, ensure_ascii=False))
        self._write_loss_trace_csv(output_dir)
        # keep latest copies in output_dir for easier visualization
        with open(os.path.join(self.args.output_dir, 'loss_trace.json'), 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(self.loss_recorder, indent=2, ensure_ascii=False))
        self._write_loss_trace_csv(self.args.output_dir)
        self._rotate_checkpoints()

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        # if self.tokenizer is not None:
        #     self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]], inputs_type: str) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state and aligning them to input format of `OntoProteinKELoss` and `OntoProteinMLMLoss`.

        Override `transformers.Trainer._prepare_inputs` method.

        Args:
            inputs: inputs to prepare
            inputs_type: the type of inputs, which could be choosed on {`protein_seq`, `protein_go`, `go_go`}
        """
        def to_device(inputs: Dict[str, torch.Tensor]):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    kwargs = dict(device=self.args.device)
                    if self.deepspeed and inputs[k].dtype != torch.int64:
                        # NLP models inputs are int64 and those get adjusted to the right dtype of the
                        # embedding. Other models such as wav2vec2's inputs are already float and thus
                        # may need special handling to match the dtypes of the model
                        kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))

                    inputs[k] = v.to(**kwargs)
            return inputs

        if inputs_type == 'protein_go':
            # this is needed as "postive" and "negative" are dictionaries of tensors
            postive_inputs = inputs['postive']
            negative_inputs = inputs['negative']
            postive_inputs = to_device(postive_inputs)
            negative_inputs = to_device(negative_inputs)

            inputs = to_device(inputs)
            inputs['positive'] = postive_inputs
            inputs['negative'] = negative_inputs
            return inputs
        elif inputs_type == 'protein_seq':
            return to_device(inputs)
        

    def training_step(
        self, 
        model: nn.Module, 
        protein_seq_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        protein_go_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        go_go_inputs: Dict[str, Union[torch.Tensor, Any]] = None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Args:
            model: The model to train.
            protein_seq_inputs: Inputs for MLM.
            protein_go_inputs: Inputs for KE of Protein-Go.
            go_go_inputs: Inputs for KE of Go-Go.
        """

        model.train()

        protein_seq_inputs = self._prepare_inputs(protein_seq_inputs, inputs_type='protein_seq') if protein_seq_inputs else None
        
        #debug
        # print("protein_seq_inputs",protein_seq_inputs)
        # import ipdb;ipdb.set_trace()

        protein_go_inputs = self._prepare_inputs(protein_go_inputs, inputs_type='protein_go') if protein_go_inputs else None
        go_go_inputs = self._prepare_inputs(go_go_inputs, inputs_type='go_go') if go_go_inputs else None

        #debug
        # print("protein_go_inputs",protein_go_inputs)
        # import ipdb;ipdb.set_trace()

        """
        Protein-GO triplet inputs, refer to dataloader collate_fn for more details

        protein_input_ids : <class 'torch.Tensor'>
        relation_ids : <class 'torch.Tensor'>
        relation_attention_mask : <class 'torch.Tensor'>
        relation_token_type_ids : <class 'torch.Tensor'>

        postive : <class 'dict'>
            {
            'tail_input_ids': all_postive_go_input_ids,
            'tail_attention_mask': all_postive_go_attention_mask,
            'tail_token_type_ids': all_postive_go_token_type_ids
            }
        negative : <class 'dict'>
            {
            'tail_input_ids': all_negative_go_input_ids,
            'tail_attention_mask': all_negative_go_attention_mask,
            'tail_token_type_ids': all_negative_go_token_type_ids
            }
        """


        if self.use_amp:
            logger.info('autocast working')
            with autocast():
                loss, all_loss = self.compute_loss(model, protein_seq_inputs=protein_seq_inputs, protein_go_inputs=protein_go_inputs, go_go_inputs=go_go_inputs)
        else:
            loss, all_loss = self.compute_loss(model, protein_seq_inputs=protein_seq_inputs, protein_go_inputs=protein_go_inputs, go_go_inputs=go_go_inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:

            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            if not self.deepspeed:
                loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach(), all_loss

    def compute_loss(
        self, 
        model: T5ForConditionalGeneration,
        protein_seq_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        protein_go_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        go_go_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
    ):
        """
        Override `transformers.Trainer.compute_loss`.
        """
        total_loss = torch.tensor(0.0).to(self.args.device)
        
        all_loss = collections.defaultdict(float)

        if protein_seq_inputs:
            # head_relation_embed is from postive_protein_go_inputs
            mlm_loss, pos_pfi_loss, neg_pfi_loss  = self.model_loss(model=model, use_desc=self.args.use_desc, global_step=self.state.global_step, use_pfi=self.args.use_pfi,protein_seq_inputs=protein_seq_inputs)
            if self.args.use_pfi:
                pfi_loss = pos_pfi_loss + neg_pfi_loss
                total_loss += pfi_loss + mlm_loss
                all_loss['pfi_positive_loss'] = pos_pfi_loss.item()
                all_loss['pfi_negative_loss'] = neg_pfi_loss.item()
                all_loss['pfi_loss'] = pfi_loss.item()
                all_loss['mlm_loss'] = mlm_loss.item()
            else:
                total_loss += mlm_loss
                all_loss['mlm_loss'] = mlm_loss.item()
        # if protein_go_inputs:
        #     assert ('postive' in protein_go_inputs) & ('negative' in protein_go_inputs), 'Inputs need contain `postive` and `negative` keys.'

        #     # head_relation_embed is from postive_protein_go_inputs
        #     mlm_loss, positive_loss, negative_loss = self.model_loss(model=model, use_desc=self.args.use_desc, global_step=self.state.global_step, use_pfi=self.args.use_pfi,protein_go_inputs=protein_go_inputs)

        #     if self.args.use_pfi:
        #         pfi_loss = positive_loss + negative_loss
        #         total_loss += pfi_loss + mlm_loss
        #         all_loss['pfi_positive_loss'] = positive_loss.item()
        #         all_loss['pfi_negative_loss'] = negative_loss.item()
        #         all_loss['pfi_loss'] = pfi_loss.item()
        #         all_loss['mlm_loss'] = mlm_loss.item()
        #     else:
        #         total_loss += mlm_loss
        #         all_loss['mlm_loss'] = mlm_loss.item()
        
        # Add TM-Vec-supervised global structure loss if enabled.
        if self.triplet_structure_loss is not None and protein_seq_inputs is not None:
            triplet_loss, triplet_metrics = self._compute_triplet_structure_loss(model, protein_seq_inputs)
            total_loss = total_loss + self.args.tmvec_weight * triplet_loss
            all_loss["triplet_loss"] = float(triplet_loss.detach().cpu())
            all_loss.update(triplet_metrics)

        if self.tmvec_loss is not None and protein_seq_inputs is not None:
            if "pair_id" not in protein_seq_inputs:
                raise ValueError("use_tmvec_loss=True requires 'pair_id' in protein_seq_inputs. Use ProteinSeqPairDataset.")
            structure_inputs = {
                "input_ids": protein_seq_inputs["input_ids"],
                "attention_mask": protein_seq_inputs["attention_mask"],
                "token_type_ids": protein_seq_inputs.get("token_type_ids"),
            }
            student_repr = model.get_sequence_embedding(structure_inputs)
            tmv_loss = self.tmvec_loss(
                student_repr=student_repr,
                pair_id=protein_seq_inputs["pair_id"].to(student_repr.device),
                tmvec_emb=protein_seq_inputs.get("tmvec_emb"),
            )

            total_loss = total_loss + self.args.tmvec_weight * tmv_loss
            all_loss["tmvec_loss"] = float(tmv_loss.detach().cpu())
            all_loss.update(self._structure_debug_metrics(
                student_repr=student_repr,
                pair_id=protein_seq_inputs["pair_id"].to(student_repr.device),
                tmvec_emb=protein_seq_inputs.get("tmvec_emb"),
            ))
        return total_loss, all_loss

    def _encode_sequence_embeddings_in_chunks(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        microbatch_size = int(getattr(self.args, 'triplet_microbatch_size', 0) or 0)
        if microbatch_size <= 0 or input_ids.size(0) <= microbatch_size:
            return model.get_sequence_embedding({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            })
        outputs = []
        for start in range(0, input_ids.size(0), microbatch_size):
            end = min(start + microbatch_size, input_ids.size(0))
            outputs.append(model.get_sequence_embedding({
                'input_ids': input_ids[start:end],
                'attention_mask': attention_mask[start:end],
                'token_type_ids': token_type_ids[start:end] if token_type_ids is not None else None,
            }))
        return torch.cat(outputs, dim=0)

    def _compute_triplet_structure_loss(self, model: nn.Module, protein_seq_inputs: Dict[str, Union[torch.Tensor, Any]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        anchor_repr = self._encode_sequence_embeddings_in_chunks(
            model,
            protein_seq_inputs['anchor_input_ids'],
            protein_seq_inputs.get('anchor_attention_mask', protein_seq_inputs['attention_mask']),
            protein_seq_inputs.get('anchor_token_type_ids', protein_seq_inputs.get('token_type_ids')),
        )
        positive_repr = self._encode_sequence_embeddings_in_chunks(
            model,
            protein_seq_inputs['positive_input_ids'],
            protein_seq_inputs['positive_attention_mask'],
            protein_seq_inputs.get('positive_token_type_ids'),
        )
        negative_repr = self._encode_sequence_embeddings_in_chunks(
            model,
            protein_seq_inputs['negative_input_ids'],
            protein_seq_inputs['negative_attention_mask'],
            protein_seq_inputs.get('negative_token_type_ids'),
        )
        triplet_loss = self.triplet_structure_loss(anchor_repr, positive_repr, negative_repr)
        metrics = self._triplet_debug_metrics(anchor_repr, positive_repr, negative_repr)
        return triplet_loss, metrics

    def _triplet_debug_metrics(self, anchor_repr: torch.Tensor, positive_repr: torch.Tensor, negative_repr: torch.Tensor) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        with torch.no_grad():
            if getattr(self.args, 'triplet_distance_type', 'l2') == 'cosine':
                pos_dist = 1.0 - F.cosine_similarity(anchor_repr.float(), positive_repr.float(), dim=1)
                neg_dist = 1.0 - F.cosine_similarity(anchor_repr.float(), negative_repr.float(), dim=1)
            else:
                pos_dist = torch.norm(anchor_repr.float() - positive_repr.float(), p=2, dim=1)
                neg_dist = torch.norm(anchor_repr.float() - negative_repr.float(), p=2, dim=1)
            margin = float(getattr(self.args, 'triplet_margin', 0.2))
            violations = (pos_dist - neg_dist + margin > 0).float()
            metrics['anchor_pos_distance_mean'] = float(pos_dist.mean().detach().cpu())
            metrics['anchor_neg_distance_mean'] = float(neg_dist.mean().detach().cpu())
            metrics['triplet_margin_violation_rate'] = float(violations.mean().detach().cpu())
            metrics['anchor_repr_norm_mean'] = float(anchor_repr.norm(dim=1).mean().detach().cpu())
        return metrics

    def _structure_debug_metrics(self, student_repr: torch.Tensor, pair_id: torch.Tensor, tmvec_emb: Optional[torch.Tensor] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        with torch.no_grad():
            z = F.normalize(student_repr.float(), dim=1)
            sim = z @ z.t()
            eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
            pos_mask = pair_id.unsqueeze(0).eq(pair_id.unsqueeze(1)) & (~eye)
            neg_mask = ~pair_id.unsqueeze(0).eq(pair_id.unsqueeze(1))
            metrics["student_repr_norm_mean"] = float(student_repr.norm(dim=1).mean().detach().cpu())
            if pos_mask.any():
                metrics["student_pos_sim_mean"] = float(sim[pos_mask].mean().detach().cpu())
            if neg_mask.any():
                metrics["student_nonpair_sim_mean"] = float(sim[neg_mask].mean().detach().cpu())
            if tmvec_emb is not None:
                if not torch.is_tensor(tmvec_emb):
                    tmvec_emb = torch.as_tensor(tmvec_emb, dtype=student_repr.dtype, device=student_repr.device)
                tm = F.normalize(tmvec_emb.float().to(student_repr.device), dim=1)
                tsim = tm @ tm.t()
                if pos_mask.any():
                    metrics["teacher_pos_sim_mean"] = float(tsim[pos_mask].mean().detach().cpu())
                if neg_mask.any():
                    metrics["teacher_nonpair_sim_mean"] = float(tsim[neg_mask].mean().detach().cpu())
        return metrics

    def num_examples(self, dataloader: DataLoader) -> int:
        num_examples = 0
        if dataloader:
            num_examples = len(dataloader.dataset)
        return num_examples


    def create_optimizer(self):
        """
        Setup the optimizer.

        Note: It is override from `transformers.Trainer.create_optimizer` for dynamically setting learning rate on different
        parameters.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            all_params = [name for name, p in self.model.named_parameters()]
            encoder_parameters = get_parameter_names(self.model, [KnowledgeDecoder])

            decoder_parameters = list(set(all_params) - set(encoder_parameters))
             # freeze encoder
            if self.args.decoder_only:
                #set only optimizer decoder parameters
                all_params = decoder_parameters
                
                # set requires_grad to False for encoder just in case :)
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                print("##### only optimizer decoder #####")
                logger.info("##### only optimizer decoder #####")
            else:
                print("##### optimizer encoder and decoder #####")
                logger.info("##### optimizer encoder and decoder #####")    

            decay_parameters = list(set(decay_parameters) & set(all_params))
            no_decay_parameters = list(set(all_params) - set(decay_parameters))
            
            # note all parameters in textbert have requires_grad = False and are not optimized
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.lm_learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in no_decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                    'lr': self.args.lm_learning_rate
                }
            ]
            # -----------------------------------------------------
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            # optimizer_kwargs["lr"] = self.args.learning_rate
            # TODO: default choose `sharded_ddp` == `zero_dp_2`
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer=None, **kwargs):
        """
        Setup the scheduler. The optimizer must have been set up before this method is called.
        """
        if self.lr_scheduler is None:
            if self.args.deepspeed:
                num_training_steps = num_training_steps // self.args.gradient_accumulation_steps + int(
                    num_training_steps % self.args.gradient_accumulation_steps > 0
                )

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer if optimizer is not None else self.optimizer,
                num_lm_warmup_steps=self.args.get_lm_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def _get_train_sampler(self) -> Tuple:
        train_protein_seq_sampler = None
        train_protein_go_sampler = None
        train_go_go_sampler = None

        if isinstance(self.protein_seq_dataset, collections.abc.Sized):
            generator = None
            if self.args.world_size <= 1 and _is_torch_generator_available:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    train_protein_seq_sampler = RandomSampler(self.protein_seq_dataset, generator=generator)
                train_protein_seq_sampler = RandomSampler(self.protein_seq_dataset)
            else:
                train_protein_seq_sampler = DistributedSampler(
                    self.protein_seq_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
        
        if isinstance(self.protein_go_dataset, collections.abc.Sized):
            generator = None
            if self.args.world_size <= 1 and _is_torch_generator_available:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    train_protein_go_sampler = RandomSampler(self.protein_go_dataset, generator=generator)
                train_protein_go_sampler = RandomSampler(self.protein_go_dataset)
            else:
                train_protein_go_sampler = DistributedSampler(
                    self.protein_go_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

        return train_protein_seq_sampler, train_protein_go_sampler

    def get_train_dataloader(self) -> Tuple:
        protein_seq_dataloader = None
        protein_go_dataloader = None
        go_go_dataloader = None

        protein_seq_sampler, protein_go_sampler = self._get_train_sampler()

        if self.protein_seq_dataset:
            if self.tmvec_loss is not None and hasattr(self.protein_seq_dataset, "pairs"):
                if self.args.world_size > 1:
                    raise NotImplementedError("Pair-preserving TM-Vec batching is not implemented for distributed training")
                if self.args.train_protein_seq_batch_size % 2 != 0:
                    raise ValueError("use_tmvec_loss=True but per-device protein sequence batch size is not even")
                if self.args.train_protein_seq_batch_size < 4:
                    raise ValueError(
                        "use_tmvec_loss=True requires per_device_train_batch_size >= 4"
                    )
                num_pairs = len(self.protein_seq_dataset) // 2
                pairs_per_batch = self.args.train_protein_seq_batch_size // 2
                remainder_pairs = num_pairs % pairs_per_batch
                if (not self.args.dataloader_drop_last) and remainder_pairs == 1:
                    raise ValueError(
                        "The final batch contains only one pair. Set dataloader_drop_last=True, "
                        "increase protein_seq_sample_limit / mined pairs, or change the batch size."
                    )
                generator = None
                if _is_torch_generator_available:
                    generator = torch.Generator()
                    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
                self.protein_seq_dataset._drop_last_for_pairs = bool(self.args.dataloader_drop_last)
                batch_sampler = PairBatchSampler(
                    self.protein_seq_dataset,
                    pairs_per_batch=self.args.train_protein_seq_batch_size // 2,
                    generator=generator,
                )
                protein_seq_dataloader = DataLoader(
                    dataset=self.protein_seq_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=self.protein_seq_data_collator,
                    pin_memory=self.args.dataloader_pin_memory,
                )
            else:
                batch_sampler = None
                if getattr(self.args, 'max_tokens_per_batch', 0):
                    if self.args.world_size > 1:
                        logger.warning('max_tokens_per_batch is ignored for distributed training, using the default sampler instead')
                    elif hasattr(self.protein_seq_dataset, 'get_example_length'):
                        generator = None
                        if _is_torch_generator_available:
                            generator = torch.Generator()
                            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
                        batch_sampler = TokenBudgetBatchSampler(
                            self.protein_seq_dataset,
                            max_tokens=int(getattr(self.args, 'max_tokens_per_batch', 0)),
                            max_batch_size=self.args.train_protein_seq_batch_size,
                            drop_last=self.args.dataloader_drop_last,
                            generator=generator,
                            bucket_size_multiplier=getattr(self.args, 'length_bucket_size_multiplier', 20),
                        )
                elif getattr(self.args, 'length_bucketed_batches', False):
                    if self.args.world_size > 1:
                        logger.warning('length_bucketed_batches is ignored for distributed training, using the default sampler instead')
                    elif hasattr(self.protein_seq_dataset, 'get_example_length'):
                        generator = None
                        if _is_torch_generator_available:
                            generator = torch.Generator()
                            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
                        batch_sampler = LengthBucketBatchSampler(
                            self.protein_seq_dataset,
                            batch_size=self.args.train_protein_seq_batch_size,
                            drop_last=self.args.dataloader_drop_last,
                            generator=generator,
                            bucket_size_multiplier=getattr(self.args, 'length_bucket_size_multiplier', 20),
                        )
                if batch_sampler is not None:
                    protein_seq_dataloader = DataLoader(
                        dataset=self.protein_seq_dataset,
                        batch_sampler=batch_sampler,
                        collate_fn=self.protein_seq_data_collator,
                        pin_memory=self.args.dataloader_pin_memory,
                    )
                else:
                    protein_seq_dataloader = DataLoader(
                        dataset=self.protein_seq_dataset,
                        batch_size=self.args.train_protein_seq_batch_size,
                        collate_fn=self.protein_seq_data_collator,
                        pin_memory=self.args.dataloader_pin_memory,
                        drop_last=self.args.dataloader_drop_last,
                        sampler=protein_seq_sampler,
                    )

        if self.protein_go_dataset:
            protein_go_dataloader = DataLoader(
                dataset=self.protein_go_dataset,
                batch_size=self.args.train_protein_go_batch_size,
                collate_fn=self.protein_go_data_collator,
                num_workers=self.args.dataloader_protein_go_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
                sampler=protein_go_sampler,
            )


        return protein_seq_dataloader, protein_go_dataloader
