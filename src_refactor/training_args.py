from collections import defaultdict
from json import decoder
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional
from transformers import logging
from transformers.training_args import TrainingArguments

from src_refactor.sampling import negative_sampling_strategy


@dataclass
class KMAEModelArguments:
    encoder_model_file_name: str = field(
        default="Rostlab/prot_bert",
        metadata={"help": "The directory of protein sequence pretrained model."}
    )
    text_model_file_name: str = field(
        default="neuml/pubmedbert-base-embeddings",
        metadata={"help": "The directory of text sequence pretrained model."}
    )
    encoder_model_config_name: str = field(
        default=None,
        metadata={'help': "Protein pretrained config name or path if not the same as protein_model_file_name"}
    )
    text_model_config_name: str = field(
        default=None,
        metadata={"help": "Text pretrained config name or path if not the same as text_model_file_name"}
    )
    protein_tokenizer_name: str = field(
        default=None,
        metadata={"help": "Protein sequence tokenizer name or path if not the same as protein_model_file_name"}
    )
    text_tokenizer_name: str = field(
        default=None,
        metadata={"help": "Text sequence tokenizer name or path if not the same as text_model_file_name"}
    )

    # For decoder
    decoder_model_type: str = field(
        default='bert',
        metadata={"help":"The type of decoder. Currently support ['bert', 'Multimodal_Transformer']"}
    )

    decoder_model_file_name: str = field(
        default="initial_decoder_config/config.json",
        metadata={"help":"The directory of the decoder model"}
    )

    protein_hidden_size: int = field(
        default=1024,
        metadata={"help": "The hidden size for protein encoder output"}
    )

    textbert_hidden_size: int = field(
        default=768,
        metadata={"help": "The hidden size for text encoder output"}
    )

    go_encoder_cls: str = field(
        default='bert',
        metadata={"help": "The class of Go term description encoder"}
    )
    protein_encoder_cls: str = field(
        default='bert',
        metadata={'help': 'The class of protein encoder.'}
    )


@dataclass
class KMAETrainingArguments(TrainingArguments):

    decoder_only: bool = field(
        default=False,
        metadata={"help":"Whether to train only the decoder"}
    )
    
    optimize_memory: bool = field(
        default=False,
        metadata={"help": "Whether or not to optimize memory when computering the loss function of negative samples. "}
    )

    use_seq: bool = field(
        default=True,
        metadata={"help": "Whether or not to use protein sequence, which its pooler output through encoder as protein representation."}
    )

    use_desc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use description of Go term, which its pooler output through encoder as Go term embedding."}
    )
    
    dataloader_protein_go_num_workers: int = field(
        default=1,
        metadata={"help": "Number of workers to collate protein-go dataset."}
    )
    dataloader_go_go_num_workers: int = field(
        default=1,
        metadata={"help": "Number of workers to collate go-go dataset."}
    )
    dataloader_protein_seq_num_workers: int = field(
        default=1,
        metadata={'help': "Number of workers to collate protein sequence dataset."}
    )

    use_pfi: bool = field(
        default=False,
        metadata={"help": "Number of workers to collate protein-go dataset."}
    )

    # number of negative sampling
    num_protein_go_neg_sample: int = field(
        default=1,
        metadata={"help": "Number of negatve sampling for Protein-Go"}
    )
    num_go_go_neg_sample: int = field(
        default=1,
        metadata={"help": "Number of negative sampling for Go-Go"}
    )

    # Weight of KE loss and MLM loss in total loss
    mlm_lambda: float = field(
        default=1.0,
        metadata={"help": "Weight of MLM loss."}
    )
    pfi_lambda: float = field(
        default=1.0,
        metadata={"help": "Weight of Protein Function Inference loss."}
    )

    # Global structure / TM-Vec loss (optional)
    use_tmvec_loss: bool = field(
        default=False,
        metadata={"help": "Whether to add the TM-Vec contrastive loss during pretraining."}
    )
    tmvec_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for TM-Vec loss."}
    )
    tmvec_model_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to TM-Vec checkpoint (.ckpt)."}
    )
    tmvec_model_config_json: Optional[str] = field(
        default=None,
        metadata={"help": "Path to TM-Vec model params JSON."}
    )
    tmvec_prot_t5_name: str = field(
        default="Rostlab/prot_t5_xl_uniref50",
        metadata={"help": "ProtT5 encoder name for TM-Vec."}
    )
    tmvec_device: Optional[str] = field(
        default=None,
        metadata={"help": "Device for TM-Vec ('cuda' or 'cpu')."}
    )
    tmvec_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for the global structure contrastive loss on GLProtein embeddings."}
    )
    tmvec_distill_weight: float = field(
        default=0.0,
        metadata={"help": "Optional weight for TM-Vec similarity distillation. Set 0 to disable."}
    )
    triplet_margin: float = field(
        default=0.2,
        metadata={"help": "Margin for the paper-style global structure triplet loss."}
    )
    triplet_distance_type: str = field(
        default="l2",
        metadata={"help": "Distance type for triplet loss: l2 or cosine."}
    )
    tmvec_freeze: bool = field(
        default=True,
        metadata={"help": "Whether to freeze TM-Vec and ProtT5 encoders."}
    )

    tmvec_use_half: bool = field(
        default=False,
        metadata={"help": "Whether to use half precision for TM-Vec to save memory."}
    )

    triplet_microbatch_size: int = field(
        default=0,
        metadata={"help": "Optional microbatch size for sequential triplet encoding. 0 disables chunking."}
    )
    length_bucketed_batches: bool = field(
        default=False,
        metadata={"help": "Whether to bucket protein sequence batches by sequence length to reduce padding and VRAM spikes."}
    )
    length_bucket_size_multiplier: int = field(
        default=20,
        metadata={"help": "Pool size multiplier for length-bucketed batching. Larger values improve bucketing at the cost of more sorting."}
    )

    max_tokens_per_batch: int = field(
        default=0,
        metadata={"help": "Optional padded-token budget for protein sequence batches. 0 disables token-budget batching."}
    )

    # respectively set learning rate to training of protein language model and knowledge embedding
    lm_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial MLM learning rate for AdamW."}
    )
    ke_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "the initial KE learning rate for AdamW."}
    )

    num_protein_seq_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs of Protein MLM to perform."}
    )
    num_protein_go_epochs: int = field(
        default =3,
        metadata={"help": "Total number of training epochs of Protein-Go KE to perform."}
    )
    num_go_go_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs of Go-Go KE to perform."}
    )

    per_device_train_protein_seq_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training of Protein MLM."}
    )
    per_device_train_protein_go_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training of Protein-Go KE."}
    )
    per_device_train_go_go_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training of Go-Go KE."}
    )

    logging_dir: str = field(
        default=None,
        metadata={"help": "Logging directory"}
    )

    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )

    # distinguish steps of linear warmup on LM and KE.
    lm_warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps for LM."}
    )
    lm_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps for LM."}
    )

    do_train: bool = field(
        default=True,
        metadata={"help": "Whether or not to train the model."}
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint directory to resume training from."}
    )
    auto_resume_from_latest: bool = field(
        default=False,
        metadata={"help": "Automatically resume from the latest checkpoint under output_dir if available."}
    )

    adafactor: bool = field(
        default=False,
        metadata={"help": "Whether or not to use adafactor optimizer."}
    )

    def __post_init__(self):
        super().__post_init__()

        self.per_device_train_protein_seq_batch_size = self.per_device_train_batch_size
        self.per_device_train_go_go_batch_size = self.per_device_train_batch_size
        self.per_device_train_protein_go_batch_size = self.per_device_train_batch_size

        if self.use_tmvec_loss:
            deprecated_tmvec_runtime_args = []
            for name in ["tmvec_model_ckpt", "tmvec_model_config_json", "tmvec_prot_t5_name", "tmvec_device", "tmvec_freeze", "tmvec_use_half"]:
                value = getattr(self, name)
                default_value = type(self).__dataclass_fields__[name].default
                if value != default_value and value is not None:
                    deprecated_tmvec_runtime_args.append(f"{name}={value}")
            if deprecated_tmvec_runtime_args:
                warnings.warn(
                    "TM-Vec runtime encoder arguments are now deprecated. TM-Vec is now offline-only during training. "
                    f"These args will be ignored: {', '.join(deprecated_tmvec_runtime_args)}",
                    UserWarning,
                )
            if self.tmvec_distill_weight < 0:
                raise ValueError("tmvec_distill_weight must be >= 0")
            if self.tmvec_weight < 0:
                raise ValueError("tmvec_weight must be >= 0")

        if self.triplet_microbatch_size < 0:
            raise ValueError("triplet_microbatch_size must be >= 0")
        if self.length_bucket_size_multiplier < 1:
            raise ValueError("length_bucket_size_multiplier must be >= 1")
        if self.max_tokens_per_batch < 0:
            raise ValueError("max_tokens_per_batch must be >= 0")

        if self.deepspeed:
            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            from src.op_deepspeed import KMAETrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = KMAETrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

    @property
    def train_protein_seq_batch_size(self) -> int:
        """
        The actual batch size for training of Protein MLM.
        """
        per_device_batch_size = self.per_device_train_protein_seq_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def train_protein_go_batch_size(self) -> int:
        """
        The actual batch size for training of Protein-Go KE.
        """
        per_device_batch_size = self.per_device_train_protein_go_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    @property
    def global_structure_weight(self) -> float:
        return self.tmvec_weight

    @property
    def global_structure_temperature(self) -> float:
        return self.tmvec_temperature

    def get_lm_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup on LM.
        """
        warmup_steps = (
            self.lm_warmup_steps if self.lm_warmup_steps > 0 else math.ceil(num_training_steps * self.lm_warmup_ratio)
        )
        return warmup_steps


@dataclass
class DataArguments:

    # Dataset use
    # Note: We only consider following combinations of dataset for sevral types of model:
    # ProtBert: protein_seq
    # OntoProtein w/o seq: protein_go + go_go
    # OntoProtein w/ seq: protein_seq + protein_go + go_go
    model_protein_seq_data: bool = field(
        default=True,
        metadata={"help": "Whether or not to model protein sequence data."}
    )
    model_protein_go_data: bool = field(
        default=False,
        metadata={"help": "Whether or not to model triplet data of `Protein-Go`"}
    )
    model_go_go_data: bool = field(
        default=False,
        metadata={"help": "Whether or not to model triplet data of `Go-Go`"}
    )

    # Pretrain data directory and specific file name
    # Note: The directory need contain following file:
    # - {protein sequence data}
    #   - data.mdb
    #   - lock.mdb
    # - go_def.txt
    # - go_type.txt
    # - go_go_triplet.txt
    # - protein_go_triplet.txt
    # - protein_seq.txt
    # - protein2id.txt
    # - go2id.txt
    # - relation2id.txt
    pretrain_data_dir: str = field(
        default='data/pretrain_data',
        metadata={"help": "the directory path of pretrain data."}
    )
    # protein_seq_data_file_name: str = field(
    #     default='swiss_seq',
    #     metadata={"help": "the directory path of specific protein sequence data."}
    # )
    in_memory: bool = field(
        default=False,
        metadata={"help": "Whether or not to save data into memory during sampling"}
    )

    tmvec_triplets_tsv: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the explicit triplet TSV for global structure supervision."}
    )

    tmvec_pairs_tsv: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated pair TSV path. Still accepted for the older TMVecLoss path."}
    )

    tmvec_pairs_emb_npy: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated pair-order teacher embedding NPY for TMVecLoss."}
    )

    protein_seq_sample_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Optional limit on number of protein sequence examples loaded."}
    )

    coordinates_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to pickled AlphaFold alpha-carbon coordinates."}
    )

    aa_vec_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the mol2vec model used for amino-acid molecular encodings."}
    )

    filter_triplets_to_coordinate_coverage: bool = field(
        default=False,
        metadata={"help": "If true, drop triplet rows whose anchor_id is missing from the coordinate PKL instead of failing."}
    )

    filtered_triplets_output_tsv: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to save the filtered triplet TSV actually used for training."}
    )

    min_triplet_retention_ratio: float = field(
        default=0.0,
        metadata={"help": "Abort if filtering triplets by coordinate coverage retains less than this fraction of rows."}
    )

    triplet_filter_report_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to write a JSON report about triplet filtering by coordinate coverage."}
    )

    # negative sampling
    negative_sampling_fn: str = field(
        default="simple_random",
        metadata={"help": f"Strategy of negative sampling. Could choose {', '.join(negative_sampling_strategy.keys())}"}
    )
    protein_go_sample_head: bool = field(
        default=False,
        metadata={"help": "Whether or not to sample head entity in triplet of `protein-go`"}
    )
    protein_go_sample_tail: bool = field(
        default=True,
        metadata={"help": "Whether or not to sample tail entity in triplet of `protein-go`"}
    )
    go_go_sample_head: bool = field(
        default=False,
        metadata={"help": "Whether or not to sample head entity in triplet of `go-go`"}
    )
    go_go_sample_tail: bool = field(
        default=False,
        metadata={"help": "Whether or not to sample tail entity in triplet of `go-go`"}
    )

    # max length of protein sequence and Go term description
    max_protein_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum length of protein sequence."}
    )
    max_text_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum length of Go term description."}
    )

