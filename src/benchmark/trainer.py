import collections
import warnings
from typing import Tuple, Optional, Union, Dict, Any, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset, DataLoader
from transformers import Trainer, EvalPrediction
from transformers.utils import is_torch_xla_available
from transformers.trainer_pt_utils import find_batch_size, nested_numpify
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, PredictionOutput
import numpy as np


class OntoProteinTrainer(Trainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: False,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items())
            else:
                logits = outputs[1:]

        if prediction_loss_only:
            pass

        logit = logits[2]
        prediction_score = {}
        probs = None

        prediction_score['precision_at_l5'] = logits[3]['precision_at_l5']
        prediction_score['precision_at_l2'] = logits[3]['precision_at_l2']
        prediction_score['precision_at_l']  = logits[3]['precision_at_l']
        prediction_score['precision_at_l5_short']  = logits[3]['precision_at_l5_short']
        prediction_score['precision_at_l2_short']  = logits[3]['precision_at_l2_short']
        prediction_score['precision_at_l_short']   = logits[3]['precision_at_l_short']
        prediction_score['precision_at_l5_medium'] = logits[3]['precision_at_l5_medium']
        prediction_score['precision_at_l2_medium'] = logits[3]['precision_at_l2_medium']
        prediction_score['precision_at_l_medium']  = logits[3]['precision_at_l_medium']
        prediction_score['precision_at_l5_long']   = logits[3]['precision_at_l5_long']
        prediction_score['precision_at_l2_long']   = logits[3]['precision_at_l2_long']
        prediction_score['precision_at_l_long']    = logits[3]['precision_at_l_long']

        labels = inputs['labels']
        invalid_mask = inputs['invalid_mask']
        if len(logits) == 1:
            logit = logits[0]

        return (loss, logit, labels, prediction_score, probs, invalid_mask)

    def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None):
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        if self.args.n_gpu > 1:
            model = torch.nn.parallel.DistributedDataParallel(model)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        print("***** Running %s *****", description)
        print("  Num examples = %d", num_examples)
        print("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None

        world_size = 1
        if is_torch_xla_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
        world_size = max(1, world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)

        model.eval()

        if is_torch_xla_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        range_names = ['short', 'medium', 'long']
        cutoff_names = ['l5', 'l2', 'l']
        contact_metrics = {f'{c}_{r}': [] for r in range_names for c in cutoff_names}
        for c in cutoff_names:
            contact_metrics[c] = []

        for step, inputs in enumerate(dataloader):
            loss, logits, labels, prediction_score, probs, invalid_mask = self.prediction_step(model, inputs, prediction_loss_only)

            for r in range_names:
                for c, key in [('l5', f'precision_at_l5_{r}'),
                               ('l2', f'precision_at_l2_{r}'),
                               ('l',  f'precision_at_l_{r}')]:
                    contact_metrics[f'{c}_{r}'].append(torch.mean(prediction_score[key]))
            contact_metrics['l5'].append(torch.mean(prediction_score['precision_at_l5']))
            contact_metrics['l2'].append(torch.mean(prediction_score['precision_at_l2']))
            contact_metrics['l'].append(torch.mean(prediction_score['precision_at_l']))

            if loss is not None:
                effective_batch_size = find_batch_size(inputs)
                effective_batch_size = effective_batch_size if effective_batch_size is not None else 1
                losses = loss.repeat(effective_batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                losses_host = None

        if self.args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))

        metrics = {}
        eval_loss = eval_losses_gatherer.finalize()
        for key, vals in contact_metrics.items():
            if vals:
                metrics[f'accuracy_{key}'] = sum(vals) / len(vals)
        metrics = denumpify_detensorize(metrics)

        return PredictionOutput(predictions=None, label_ids=None, metrics=metrics)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.deepspeed:
            print('deepspeed working')
            raise NotImplementedError

        model = self._wrap_model(self.model, training=False)
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        print(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            print(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            print("  Num examples: Unknown")
        print(f"  Batch size = {batch_size}")

        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = dataloader.dataset

        if is_torch_xla_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        losses_host = None
        all_losses = None

        range_names = ['short', 'medium', 'long']
        cutoff_names = ['l5', 'l2', 'l']
        contact_metrics = {f'{c}_{r}': [] for r in range_names for c in cutoff_names}
        for c in cutoff_names:
            contact_metrics[c] = []

        observed_num_examples = 0

        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            loss, logits, labels, prediction_score, probs, invalid_mask = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            for r in range_names:
                for c, key in [('l5', f'precision_at_l5_{r}'),
                               ('l2', f'precision_at_l2_{r}'),
                               ('l',  f'precision_at_l_{r}')]:
                    contact_metrics[f'{c}_{r}'].append(torch.mean(prediction_score[key]))
            contact_metrics['l5'].append(torch.mean(prediction_score['precision_at_l5']))
            contact_metrics['l2'].append(torch.mean(prediction_score['precision_at_l2']))
            contact_metrics['l'].append(torch.mean(prediction_score['precision_at_l']))

            if loss is not None:
                effective_batch_size = observed_batch_size if observed_batch_size is not None else 1
                losses = self.gather_function(loss.repeat(effective_batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                losses_host = None

        if self.args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)

        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        if all_losses is not None:
            all_losses = all_losses[:num_samples]

        metrics = {}
        for key, vals in contact_metrics.items():
            if vals:
                metrics[f'accuracy_{key}'] = sum(vals) / len(vals)
        metrics = denumpify_detensorize(metrics)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]