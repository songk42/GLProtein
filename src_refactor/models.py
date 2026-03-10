from math import gamma
import os
import json
import copy
from typing import Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.utils.weight_norm import weight_norm
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig,BertPreTrainedModel, BertModel,T5EncoderModel, T5Tokenizer,T5Config,T5PreTrainedModel, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder
# from .tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
# from .tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec, encode

# TM-Vec (global structure component)
# We rely on https://github.com/tymor22/tm-vec installed in the environment.
try:
    from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
    from tm_vec.tm_vec_utils import encode
    _TMVEC_IMPORT_ERROR = None
except Exception as e:
    trans_basic_block = None
    trans_basic_block_Config = None
    encode = None
    _TMVEC_IMPORT_ERROR = e
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.file_utils import ModelOutput
from transformers.utils import logging
# from transformers.deepspeed import is_deepspeed_zero3_enabled
# from deepspeed import DeepSpeedEngine
from transformers import DistilBertConfig, BertForMaskedLM
from transformers import pipeline
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List
# from decoder import KnowledgeBertModel
from src_refactor.decoder import KnowledgeBertModel
import warnings
from transformers import T5PreTrainedModel,T5Config
from transformers.models.t5.modeling_t5 import T5Stack, T5LayerNorm, T5LayerFF, T5Attention, T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF
# from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

# import logging

logger = logging.get_logger('pretrain_log')
# logger = logging.getLogger("pretrain")


DECODER_CONFIG_NAME = "config.json"
PROTEIN_CONFIG_NAME = "config.json"
PROTEIN_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'
DECODER_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'
GLPROTEIN_CONFIG_NAME = 'glprotein_config.json'
GLPROTEIN_CHECKPOINT_VERSION = 1


def _json_default(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    return str(value)


def resolve_glprotein_checkpoint_paths(checkpoint_dir: os.PathLike) -> Dict[str, Any]:
    checkpoint_dir = os.fspath(checkpoint_dir)
    config_path = os.path.join(checkpoint_dir, GLPROTEIN_CONFIG_NAME)
    metadata: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)

    encoder_subdir = metadata.get('encoder_subdir', 'encoder')
    decoder_subdir = metadata.get('decoder_subdir', 'decoder')
    protein_tokenizer_subdir = metadata.get('protein_tokenizer_subdir', 'protein_tokenizer')
    text_tokenizer_subdir = metadata.get('text_tokenizer_subdir', 'text_tokenizer')

    paths = {
        'checkpoint_dir': checkpoint_dir,
        'config_path': config_path if os.path.exists(config_path) else None,
        'metadata': metadata,
        'encoder_dir': os.path.join(checkpoint_dir, encoder_subdir),
        'decoder_dir': os.path.join(checkpoint_dir, decoder_subdir),
        'protein_tokenizer_dir': os.path.join(checkpoint_dir, protein_tokenizer_subdir),
        'text_tokenizer_dir': os.path.join(checkpoint_dir, text_tokenizer_subdir),
    }
    return paths


__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True


    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=2):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

# class Protein3DBias(nn.Module):
#     """
#         Compute 3D attention bias according to the position information for each head.
#         """

#     def __init__(self):
#         super(Protein3DBias, self).__init__()
#         self.num_heads = 8
#         self.num_edges = 2
#         self.num_kernel = 128
#         self.embed_dim = 512


#         rpe_heads = self.num_heads
#         self.gbf = GaussianLayer(self.num_kernel, self.num_edges)
#         self.gbf_proj = NonLinear(self.num_kernel, rpe_heads)

#         if self.num_kernel != self.embed_dim:
#             self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
#         else:
#             self.edge_proj = None

#     def forward(self, batched_data):

#         pos, x, node_type_edge = batched_data['protein_coordinates'], batched_data['protein_input_ids'], batched_data['protein_token_type_ids'] # pos shape: [n_examoles, n_nodes, 3]
#         # pos.requires_grad_(True)

#         padding_mask = x.eq(0).all(dim=-1)
#         n_graph, n_node, _ = pos.shape
#         delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
#         dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
#         delta_pos /= dist.unsqueeze(-1) + 1e-5

#         edge_feature = self.gbf(dist, torch.zeros_like(dist).long() if node_type_edge is None else node_type_edge.long())
#         gbf_result = self.gbf_proj(edge_feature)
#         graph_attn_bias = gbf_result

#         graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
#         graph_attn_bias.masked_fill_(
#             padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
#         )

#         edge_feature = edge_feature.masked_fill(
#             padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
#         )

#         sum_edge_features = edge_feature.sum(dim=-2)
#         merge_edge_features = self.edge_proj(sum_edge_features)

#         return graph_attn_bias, merge_edge_features, delta_pos




# class SimpleMLP(nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  hid_dim: int,
#                  out_dim: int,
#                  dropout: float = 0.):
#         super().__init__()
#         self.main = nn.Sequential(
#             weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
#             nn.ReLU(),
#             nn.Dropout(dropout, inplace=True),
#             weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
#         )

#     def forward(self, x):
#         return self.main(x)

class Protein3DBias(nn.Module):
    """Compute residue-level 3D attention bias for aa-vec cross-attention."""

    def __init__(self, num_heads: int, num_kernel: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel
        self.means = nn.Parameter(torch.linspace(0.0, 20.0, num_kernel))
        self.log_stds = nn.Parameter(torch.zeros(num_kernel))
        self.proj = nn.Sequential(
            nn.Linear(num_kernel, num_kernel),
            nn.GELU(),
            nn.Linear(num_kernel, num_heads),
        )

    def forward(self, coordinates: torch.Tensor, residue_mask: Optional[torch.Tensor], query_len: int) -> torch.Tensor:
        if coordinates is None:
            raise ValueError('coordinates must not be None when computing Protein3DBias')

        safe_coords = coordinates.float().clone()
        mask = None
        if residue_mask is not None:
            mask = residue_mask.to(dtype=torch.bool)
            safe_coords = safe_coords.masked_fill((~mask).unsqueeze(-1), 0.0)

        safe_coords = torch.where(
            torch.isfinite(safe_coords),
            safe_coords,
            torch.zeros_like(safe_coords),
        )

        dist = torch.cdist(safe_coords, safe_coords, p=2)
        std = torch.exp(self.log_stds).clamp_min(1e-2)
        rbf = torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.means) / std) ** 2)
        graph_bias = self.proj(rbf).permute(0, 3, 1, 2).contiguous()
        if mask is not None:
            key_pad = (~mask).unsqueeze(1).unsqueeze(2)
            query_pad = (~mask).unsqueeze(1).unsqueeze(-1)
            graph_bias = graph_bias.masked_fill(key_pad, float('-inf'))
            graph_bias = graph_bias.masked_fill(query_pad, 0.0)
        full_bias = graph_bias.new_zeros(graph_bias.size(0), graph_bias.size(1), query_len, graph_bias.size(-1))
        full_bias[:, :, 1:1 + graph_bias.size(2), :] = graph_bias
        return full_bias


class GLProteinConfig:
    """
    contains configs for the decoder, and configs for the 
    """
    def __init__(self,**kwargs):
        self.use_desc = kwargs.pop('use_desc', True)
        self.num_relations = kwargs.pop('num_relations', None)
        self.num_go_terms = kwargs.pop('num_go_terms', None)
        self.num_proteins = kwargs.pop('num_proteins', None)


        self.protein_encoder_cls = kwargs.pop('protein_encoder_cls', None)
        self.go_encoder_cls = kwargs.pop('go_encoder_cls', None)

        #         config.decoder_config.use_desc = self.use_desc
        # config.decoder_config.use_desc = self.num_relations
        # config.decoder_config.use_desc = self.num_go_terms
        # config.decoder_config.use_desc = self.num_proteins
        # config.decoder_config.use_desc = self.protein_encoder_cls
        # config.decoder_config.use_desc = self.go_encoder_cls


        self.protein_model_config = None
        self.decoder_config = None

    def to_serializable_dict(self) -> Dict[str, Any]:
        return {
            'use_desc': self.use_desc,
            'num_relations': self.num_relations,
            'num_go_terms': self.num_go_terms,
            'num_proteins': self.num_proteins,
            'protein_encoder_cls': self.protein_encoder_cls,
            'go_encoder_cls': self.go_encoder_cls,
        }

    @classmethod
    def from_serializable_dict(cls, payload: Optional[Dict[str, Any]] = None):
        payload = dict(payload or {})
        return cls(**payload)

    def save_to_json_file(self, encoder_save_directory: os.PathLike):
        os.makedirs(encoder_save_directory, exist_ok=True)
        # os.makedirs(decoder_save_directory, exist_ok=True)

        self.protein_model_config.save_pretrained(encoder_save_directory)
        # self.decoder_config.save_pretrained(decoder_save_directory)

        logger.info(f'Encoder Configuration saved in {encoder_save_directory}')
        # logger.info(f'Decoder Configuration saved in {decoder_save_directory}')

    @classmethod
    def from_json_file(cls, encoder_config_path: os.PathLike, decoder_config_path: os.PathLike):
        config = cls()
        config.protein_model_config = BertConfig.from_pretrained(encoder_config_path)
        config.decoder_config = AutoConfig.from_pretrained(decoder_config_path)

        return config

@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None


@dataclass
class MaskedLMAndPFIOutput(ModelOutput):

    mlm_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attention: Optional[Tuple[torch.FloatTensor]] = None
    go_attention: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None
    pos_pfi_logits: Optional[torch.FloatTensor] = None
    neg_pfi_logits: Optional[torch.FloatTensor] = None


# only use the last layer----- we can try using other layers
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # attention_mask = attention_mask.bool()
        # num_batch_size = attention_mask.size(0)
        # pooled_output = torch.stack([hidden_states[i, attention_mask[i, :], :].mean(dim=0) for i in range(num_batch_size)], dim=0)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        return pooled_output





class KnowledgeDecoder(BertPreTrainedModel):
# class KnowledgeDecoder(T5EncoderModel):
    """
    Implementation of the full GLProtein decoder
    """

    def __init__(self,decoder_config=None):
        super().__init__(decoder_config)

        # textbert for relation and GO feature extraction. Avoid nested
        # BertModel.from_pretrained() calls inside __init__, because
        # KnowledgeDecoder.from_pretrained() may construct the module under a
        # meta-device initialization context.
        textbert_config = AutoConfig.from_pretrained(decoder_config.text_model_path)
        textbert_config.output_hidden_states = True
        self.textbert = BertModel(textbert_config)
        for param in self.textbert.parameters():
            param.requires_grad = False


        # decoder
        self.config = decoder_config
        self.decoder = KnowledgeBertModel(decoder_config,add_pooling_layer=False)

        # linear layer to project features into the same dimension
        self.go_project = nn.Linear(textbert_config.hidden_size, self.config.hidden_size)
        self.relation_project = nn.Linear(textbert_config.hidden_size, self.config.hidden_size)
        
        self.coordinate_project = nn.Linear(3, self.config.hidden_size)
        self.aa_vec_project = nn.Linear(300, self.config.hidden_size)

        self.gbf = GaussianLayer(128, 1)
        self.gbf_proj = NonLinear(128, 512, 1024)
        self.protein_3d_bias = Protein3DBias(self.config.num_attention_heads, 128)

        self.text_feat_dim = textbert_config.hidden_size
        self.text_pooler = BertPooler(textbert_config)

        # mlm head and pooler
        self.mlm_cls = BertOnlyMLMHead(self.config)
        self.pooler = BertPooler(self.config)

        # pfi head, requires pooled outputs
        if decoder_config.use_pfi:
            self.pfi_cls = nn.Sequential(nn.Linear(self.config.hidden_size, 2), nn.Softmax(dim=-1))

    def load_textbert_backbone(self, text_model_path: Optional[os.PathLike] = None):
        text_model_path = text_model_path or getattr(self.config, 'text_model_path', None)
        if not text_model_path:
            raise ValueError('KnowledgeDecoder.load_textbert_backbone() requires text_model_path.')
        textbert = BertModel.from_pretrained(text_model_path, output_hidden_states=True)
        for param in textbert.parameters():
            param.requires_grad = False
        self.textbert = textbert
        self.text_feat_dim = self.textbert.config.hidden_size
        self.text_pooler = BertPooler(self.textbert.config)
        return self
      
    def forward(self, 
        # relation_inputs,
        # go_inputs,
        relation_inputs=None,
        go_inputs=None,
        inputs_embeds=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_mlm=True,
        coordinate_inputs=None,
        aa_vec_inputs = None):
        batch, protein_len, protein_embed_size = inputs_embeds.size()

        coordinate_input, residue_mask = coordinate_inputs if coordinate_inputs is not None else (None, None)
        aa_vec_input, aa_vec_attention_mask = aa_vec_inputs if aa_vec_inputs is not None else (None, None)
        graph_attn_bias = None
        aa_vec_feat = None
        if coordinate_input is not None:
            graph_attn_bias = self.protein_3d_bias(coordinate_input, residue_mask, protein_len)
        if aa_vec_input is not None:
            aa_vec_feat = self.aa_vec_project(aa_vec_input)


        # #get the coordinate mask
        # coordinate_attention_mask = torch.mean(coordinate_input,dim=2)
        # coordinate_attention_mask = torch.where(torch.isinf(coordinate_attention_mask),torch.zeros_like(coordinate_attention_mask),coordinate_attention_mask)
        # coordinate_attention_mask = torch.where(torch.isnan(coordinate_attention_mask),torch.zeros_like(coordinate_attention_mask),coordinate_attention_mask)
        # coordinate_attention_mask = coordinate_attention_mask.bool()

        # delta_pos = coordinate_input.unsqueeze(1) - coordinate_input.unsqueeze(2)
        # dist = delta_pos.norm(dim=-1).view(-1, protein_len-2, protein_len-2)
        # delta_pos /= dist.unsqueeze(-1) + 1e-5
        
        # delta_pos = torch.where(torch.isinf(delta_pos),torch.zeros_like(delta_pos),delta_pos)
        # delta_pos = torch.where(torch.isnan(delta_pos),torch.zeros_like(delta_pos),delta_pos)
        # delta_pos = torch.mean(delta_pos,dim=2)
        # coordinate_feat = self.coordinate_project(delta_pos) #(batch,coordinate len, decoder hidden dim)
        
        
        # aa_vec_attention_mask = coordinate_attention_mask


        # aa_vec_feat = self.aa_vec_project(aa_vec_input) #(batch,aa_vec len, decoder hidden dim)

        
        # go_input_ids, go_attention_mask, go_token_type_ids = go_inputs
        

        # go_out = self.textbert(go_input_ids,
        #                             attention_mask=go_attention_mask,
        #                             token_type_ids=go_token_type_ids,
        #                             output_hidden_states=True,
        #                             return_dict=True) # (batch,token len, feat_dim)  

        # # hidden size (b,seqlen, 768)
        # go_feat = torch.cat(tuple([go_out.hidden_states[i].unsqueeze(1) for i in [-4, -3, -2, -1]]), dim=1) # (b ,4, go len, hidden_dim)
        # go_feat = torch.mean(go_feat,dim=1) # (b,go len, hidden_dim)

        # go_feat = self.go_project(go_feat) #(batch,, go len, decoder hidden dim)


        # ### relation feature extraction
        # relation_input_ids, relation_attention_mask, relation_token_type_ids = relation_inputs
        # relation_out = self.textbert(relation_input_ids,
        #                             attention_mask=relation_attention_mask,
        #                             token_type_ids=relation_token_type_ids,
        #                             output_hidden_states=True,
        #                             return_dict=True) # (batch,token len, feat_dim)


        # relation_feat = torch.cat(tuple([relation_out.hidden_states[i].unsqueeze(1) for i in [-4, -3, -2, -1]]), dim=1) # (b ,4,relation len, hidden_dim)

        # relation_feat = torch.mean(relation_feat,dim=1) # (b,relation len, hidden_dim)

        # relation_feat = self.relation_project(relation_feat) #(batch,relation len, decoder hidden dim)'
        


        #HACK
        ## input embedding to decoder, mask stay the same as protbert
        out = self.decoder(inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            # relation_hidden_states=relation_feat,
            # relation_attention_mask=relation_attention_mask,
            # go_hidden_states=go_feat,
            # go_attention_mask=go_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            aa_vec_hidden_states=aa_vec_feat,
            aa_vec_attention_mask=aa_vec_attention_mask,
            attn_bias=graph_attn_bias,
        )


        # # output_seq = out.hidden_states[-1]
        output_seq = out[0] # last hidden layer

        # output_seq = inputs_embeds # pretrain for nothing attention

        mlm_prediction_scores = self.mlm_cls(output_seq)

        # pfi output
        pooler_output = self.pooler(output_seq)
        pfi_prediction=None
        if self.config.use_pfi:
            pfi_prediction = self.pfi_cls(pooler_output)

        out.pooler_output = pooler_output

        return (out,mlm_prediction_scores,pfi_prediction)




class GLProtein(nn.Module):
    """
    Implementation of the GLProtein model
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder_config = config.protein_model_config
        self.decoder_config = config.decoder_config
        self.encoder=BertModel(self.encoder_config, add_pooling_layer=False)
        self.decoder = None

        
        

      
        

    def forward(self,
        protein_inputs: Tuple = None,
        pos_relation_inputs: Union[torch.Tensor, Tuple] = None,
        pos_go_tail_inputs: Union[torch.Tensor, Tuple] = None,
        neg_relation_inputs: Union[torch.Tensor, Tuple] = None,
        neg_go_tail_inputs: Union[torch.Tensor, Tuple] = None,
        use_pfi: bool = True,
        output_attentions: bool = False
        ):

      

        # protein_input_ids, protein_attention_mask, protein_token_type_ids, protein_coordinates,coordinate_attention_mask, aa_vec, aa_vec_attention_mask= protein_inputs
        protein_input_ids = protein_inputs["input_ids"]
        protein_attention_mask = protein_inputs["attention_mask"]
        protein_token_type_ids = protein_inputs["token_type_ids"]

        coordinate_inputs = None
        aa_vec_inputs = None
        if protein_inputs.get("coordinates") is not None:
            residue_mask = protein_attention_mask[:, 1:-1].contiguous() if protein_attention_mask.size(1) >= 2 else None
            coordinate_inputs = (protein_inputs["coordinates"], residue_mask)
        if protein_inputs.get("aa_vec") is not None:
            aa_vec_inputs = (protein_inputs["aa_vec"], protein_inputs.get("aa_vec_attention_mask"))

       

        # protein_outputs = self.encoder(
        #     input_ids=protein_input_ids,
        #     attention_mask=protein_attention_mask,
        #     token_type_ids=protein_token_type_ids,
        #     output_hidden_states=True,
        #     return_dict=True,
        #     output_attentions=output_attentions
        # )

        protein_outputs = self.encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            token_type_ids=protein_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=output_attentions
        )


        prot_seq_embed = protein_outputs[0] 


        out, mlm_prediction_scores, pos_pfi_prediction = self.decoder(inputs_embeds=prot_seq_embed,
            attention_mask=protein_attention_mask,
            token_type_ids=protein_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            coordinate_inputs=coordinate_inputs,
            aa_vec_inputs=aa_vec_inputs,
        )

        
        # out, mlm_prediction_scores, pos_pfi_prediction = self.decoder(pos_relation_inputs, pos_go_tail_inputs,inputs_embeds=prot_seq_embed,
        #     attention_mask=protein_attention_mask,
        #     token_type_ids=protein_token_type_ids,
        #     output_hidden_states=True,
        #     return_dict=True,
        #     output_attentions=output_attentions,
        #     coordinate_inputs = coordinate_inputs,
        #     input_ids = protein_input_ids,
        #     aa_vec_inputs = aa_vec_inputs,
        #     )

        neg_pfi_prediction=None
        if use_pfi:
            out_neg, neg_mlm_prediction_scores, neg_pfi_prediction = self.decoder(neg_relation_inputs, neg_go_tail_inputs,inputs_embeds=prot_seq_embed,
            attention_mask=protein_attention_mask,
            token_type_ids=protein_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=output_attentions,
            coordinate_inputs = coordinate_inputs,
            input_ids = protein_input_ids
            )


        return MaskedLMAndPFIOutput(
            mlm_loss=None,
            mlm_logits=mlm_prediction_scores,
            hidden_states=out.hidden_states,
            encoder_attention=protein_outputs.attentions,
            go_attention=out.attentions,
            pooler_output=out.pooler_output,
            pos_pfi_logits=pos_pfi_prediction,
            neg_pfi_logits=neg_pfi_prediction
        )

    def get_sequence_embedding(self, protein_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one differentiable sequence embedding per protein sequence."""
        protein_input_ids = protein_inputs["input_ids"]
        protein_attention_mask = protein_inputs["attention_mask"]
        protein_token_type_ids = protein_inputs.get("token_type_ids", None)

        protein_outputs = self.encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            token_type_ids=protein_token_type_ids,
            output_hidden_states=False,
            return_dict=True,
            output_attentions=False,
        )
        token_embeddings = protein_outputs.last_hidden_state
        attention_mask = protein_attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * attention_mask).sum(dim=1)
        denom = attention_mask.sum(dim=1).clamp(min=1.0)
        return pooled / denom


    def save_pretrained(self, save_directory: os.PathLike, state_dict: Optional[dict] = None, save_config: bool = True):
        save_directory = os.fspath(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        encoder_save_directory = os.path.join(save_directory, 'encoder')
        decoder_save_directory = os.path.join(save_directory, 'decoder')

        encoder_state_dict = None
        decoder_state_dict = None
        if state_dict is not None:
            encoder_state_dict = {
                name[len('encoder.'):]: tensor
                for name, tensor in state_dict.items()
                if name.startswith('encoder.')
            }
            decoder_state_dict = {
                name[len('decoder.'):]: tensor
                for name, tensor in state_dict.items()
                if name.startswith('decoder.')
            }

        self.encoder.save_pretrained(encoder_save_directory, save_config=save_config, state_dict=encoder_state_dict)
        if self.decoder is None:
            raise ValueError('GLProtein.save_pretrained() expected self.decoder to be initialized.')
        self.decoder.save_pretrained(decoder_save_directory, save_config=save_config, state_dict=decoder_state_dict)

        wrapper_payload = {
            'format_version': GLPROTEIN_CHECKPOINT_VERSION,
            'model_class': self.__class__.__name__,
            'encoder_subdir': 'encoder',
            'decoder_subdir': 'decoder',
            'protein_tokenizer_subdir': 'protein_tokenizer',
            'text_tokenizer_subdir': 'text_tokenizer',
            'wrapper_config': GLProteinConfig(
                use_desc=getattr(self.decoder_config, 'use_desc', None),
                num_relations=getattr(self.decoder_config, 'num_relations', None),
                num_go_terms=getattr(self.decoder_config, 'num_go_terms', None),
                num_proteins=getattr(self.decoder_config, 'num_proteins', None),
                protein_encoder_cls=getattr(self.decoder_config, 'protein_encoder_cls', None),
                go_encoder_cls=getattr(self.decoder_config, 'go_encoder_cls', None),
            ).to_serializable_dict(),
            'decoder_text_model_path': getattr(self.decoder_config, 'text_model_path', None),
            'notes': 'Full pretrained GLProtein checkpoint',
        }
        with open(os.path.join(save_directory, GLPROTEIN_CONFIG_NAME), 'w', encoding='utf-8') as handle:
            json.dump(wrapper_payload, handle, indent=2, ensure_ascii=False, default=_json_default)

        logger.info(f'Encoder Model weights saved in {encoder_save_directory}')
        logger.info(f'Decoder Model weights saved in {decoder_save_directory}')

    @classmethod
    def get_encoder_checkpoint_path(cls, checkpoint_dir: os.PathLike) -> str:
        return resolve_glprotein_checkpoint_paths(checkpoint_dir)['encoder_dir']

    @classmethod
    def load_encoder_from_checkpoint(cls, checkpoint_dir: os.PathLike) -> BertModel:
        encoder_dir = cls.get_encoder_checkpoint_path(checkpoint_dir)
        if not os.path.isdir(encoder_dir):
            raise FileNotFoundError(f'Encoder checkpoint directory not found: {encoder_dir}')
        return BertModel.from_pretrained(encoder_dir)

    @classmethod
    def from_pretrained(
        cls,
        protein_model_path: Optional[os.PathLike] = None,
        text_model_path: Optional[os.PathLike] = None,
        decoder_model_path: Optional[os.PathLike] = None,
        model_args=None,
        training_args=None,
        checkpoint_dir: Optional[os.PathLike] = None,
        **kwargs
    ):

        # Will feed the number of relations and entity.
        num_relations = kwargs.pop('num_relations', None)
        num_go_terms = kwargs.pop('num_go_terms', None)
        num_proteins = kwargs.pop('num_proteins', None)

        candidate_checkpoint = checkpoint_dir or kwargs.pop('model_checkpoint_path', None)
        if candidate_checkpoint is None and protein_model_path is not None:
            candidate_path = Path(os.fspath(protein_model_path))
            if candidate_path.is_dir() and (candidate_path / GLPROTEIN_CONFIG_NAME).exists():
                candidate_checkpoint = candidate_path

        if candidate_checkpoint is not None:
            resolved = resolve_glprotein_checkpoint_paths(candidate_checkpoint)
            encoder_dir = resolved['encoder_dir']
            decoder_dir = resolved['decoder_dir']
            metadata = resolved['metadata'] or {}
            if not os.path.isdir(encoder_dir):
                raise FileNotFoundError(f'GLProtein checkpoint is missing encoder directory: {encoder_dir}')
            if not os.path.isdir(decoder_dir):
                raise FileNotFoundError(
                    f'Checkpoint is encoder-only and cannot restore full GLProtein decoder state: {decoder_dir}'
                )

            kmae_config = GLProteinConfig.from_json_file(encoder_dir, decoder_dir)
            wrapper_cfg = GLProteinConfig.from_serializable_dict(metadata.get('wrapper_config'))
            kmae_config.use_desc = wrapper_cfg.use_desc
            kmae_config.num_relations = wrapper_cfg.num_relations
            kmae_config.num_go_terms = wrapper_cfg.num_go_terms
            kmae_config.num_proteins = wrapper_cfg.num_proteins
            kmae_config.protein_encoder_cls = wrapper_cfg.protein_encoder_cls
            kmae_config.go_encoder_cls = wrapper_cfg.go_encoder_cls

            if text_model_path is None:
                text_model_path = metadata.get('decoder_text_model_path') or getattr(kmae_config.decoder_config, 'text_model_path', None)
            if text_model_path is not None:
                kmae_config.decoder_config.text_model_path = text_model_path
            if training_args is not None:
                kmae_config.decoder_config.use_desc = training_args.use_desc
                kmae_config.decoder_config.use_pfi = training_args.use_pfi

            kmae_model = cls(config=kmae_config)
            kmae_model.encoder = BertModel.from_pretrained(encoder_dir)
            kmae_model.decoder = KnowledgeDecoder.from_pretrained(decoder_dir)
            if text_model_path is not None:
                kmae_model.decoder.config.text_model_path = text_model_path
                kmae_model.decoder.load_textbert_backbone(text_model_path)
            kmae_model.eval()
            return kmae_model

        if protein_model_path is None or decoder_model_path is None or text_model_path is None:
            raise ValueError('Legacy GLProtein.from_pretrained requires protein_model_path, text_model_path, and decoder_model_path.')

        # 1 assign useful configs to decoder config
        kmae_config = GLProteinConfig.from_json_file(protein_model_path, decoder_model_path)
        kmae_config.decoder_config.num_relations = num_relations
        kmae_config.decoder_config.num_go_terms = num_go_terms
        kmae_config.decoder_config.num_proteins = num_proteins
        if training_args:
            kmae_config.decoder_config.use_desc = training_args.use_desc
            kmae_config.decoder_config.use_pfi = training_args.use_pfi
        if model_args:
            kmae_config.decoder_config.go_encoder_cls = model_args.go_encoder_cls
            kmae_config.decoder_config.protein_encoder_cls = model_args.protein_encoder_cls

        kmae_config.decoder_config.text_model_path = text_model_path

        # instantiate model. Note textbert in decoder is initialized in this step
        kmae_model = cls(config=kmae_config)

        # 2 load encoder model
        if kmae_model.decoder_config.protein_encoder_cls == 'bert':
            kmae_model.encoder = BertModel.from_pretrained(protein_model_path)
        else:
            raise NotImplementedError('Currently only support bert for encoder')

        # 3 load decoder model
        if kmae_model.decoder_config.model_type == 'bert':
            if os.path.exists(os.path.join(decoder_model_path, 'pytorch_model.bin')):
                logger.info(f'Loading Decoder Model from {decoder_model_path}')
                kmae_model.decoder = KnowledgeDecoder.from_pretrained(decoder_model_path)
                kmae_model.decoder.config.text_model_path = text_model_path
                kmae_model.decoder.load_textbert_backbone(text_model_path)
            else:
                kmae_model.decoder = KnowledgeDecoder(kmae_config.decoder_config)
                kmae_model.decoder.load_textbert_backbone(text_model_path)
        else:
            raise NotImplementedError('Currently only support bert cls')

        kmae_model.eval()
        return kmae_model

@dataclass
class GLProteinLoss:
    """
     Perform forward propagation and return loss for protein function inference

    for pfi task (default don't use):
        pfi_weight: weight of protein function inference loss
        num_protein_go_neg_sample: number of negative samples per positive sample  
    """
    def __init__(self,pfi_weight=1.0,num_protein_go_neg_sample=1,mlm_lambda=1.0):
        self.pfi_weight = pfi_weight
        self.mlm_lambda = mlm_lambda
        self.num_protein_go_neg_sample = num_protein_go_neg_sample
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(
        self,
        model: BertModel,
        use_desc: bool = False,
        use_seq: bool = True,
        use_pfi: bool = True,
        protein_go_inputs = None,
        protein_seq_inputs = None,
        **kwargs
    ):
        # get protein inputs
        # protein_mlm_input_ids = protein_go_inputs['protein_input_ids']
        # protein_mlm_attention_mask = protein_go_inputs['protein_attention_mask']
        # protein_mlm_token_type_ids = protein_go_inputs['protein_token_type_ids']
        # protein_mlm_pos_embed = protein_go_inputs['protein_coordinates']  # add coordinates here
        # coordinate_attenion_mask = protein_go_inputs['coordinate_attention_mask']
        # protein_mlm_aa_vec_embed = protein_go_inputs['aa_vec']  # add aa_vec here
        # aa_vec_attention_mask= protein_go_inputs['aa_vec_attention_mask']



        # protein_input = (protein_mlm_input_ids,protein_mlm_attention_mask,protein_mlm_token_type_ids, protein_mlm_pos_embed,coordinate_attenion_mask,protein_mlm_aa_vec_embed, aa_vec_attention_mask)  # add coordinates here
        # # coordinate_input = (protein_mlm_pos_embed,coordinate_attenion_mask)

        # protein_mlm_labels = protein_go_inputs['protein_labels']

        # # relation inputs
        # relation_ids = protein_go_inputs['relation_ids']
        # relation_attention_mask = protein_go_inputs['relation_attention_mask']
        # relation_token_type_ids = protein_go_inputs['relation_token_type_ids']
        # relation_inputs = (relation_ids, relation_attention_mask, relation_token_type_ids)
 

        # ## positive inputs
        # positive = protein_go_inputs['positive']

        # # get tail inputs
        # positive_tail_input_ids = positive['tail_input_ids']
        # positive_tail_attention_mask = positive['tail_attention_mask']
        # positive_tail_token_type_ids = positive['tail_token_type_ids']

        # positive_go_tail_inputs = positive_tail_input_ids
        # if use_desc:
        #     positive_go_tail_inputs = (positive_tail_input_ids, positive_tail_attention_mask, positive_tail_token_type_ids)


        # ## negative inputs
        # negative_go_tail_inputs=None
        # if use_pfi:
        #     negative = protein_go_inputs['negative']

        #     # get tail inputs
        #     negative_tail_input_ids = negative['tail_input_ids']
        #     negative_tail_attention_mask = negative['tail_attention_mask']
        #     negative_tail_token_type_ids = negative['tail_token_type_ids']

        #     negative_go_tail_inputs = negative_tail_input_ids
        #     if use_desc:
        #         negative_go_tail_inputs = (negative_tail_input_ids, negative_tail_attention_mask, negative_tail_token_type_ids)




        # model_output = model(protein_mlm_input_ids['input_ids'].to('cuda'), labels =  protein_mlm_labels['input_ids'].to('cuda'))
        model_output = model(protein_inputs=protein_seq_inputs, use_pfi=use_pfi)

        # mlm loss
        mlm_logits = model_output.mlm_logits
        batch, seq_len, vocab_size = mlm_logits.size()
        # mlm_loss = self.loss_fn(mlm_logits.view(-1, vocab_size), protein_mlm_labels.view(-1)) * self.mlm_lambda
        # mlm_loss = model_output.mlm_loss * self.mlm_lambda
        mlm_loss = self.loss_fn(mlm_logits.view(-1, vocab_size), protein_seq_inputs['labels'].view(-1)) * self.mlm_lambda

        # pfi loss
        pos_pfi_loss =0
        neg_pfi_loss =0
        if use_pfi:
            pos_pfi_logits = model_output.pos_pfi_logits #(batch,2)
            neg_pfi_logits = model_output.neg_pfi_logits
   
            pos_pfi_label = protein_go_inputs['pfi_pos'].repeat(pos_pfi_logits.size(0))
            neg_pfi_label = protein_go_inputs['pfi_neg'].repeat(neg_pfi_logits.size(0))

            pos_pfi_loss = self.loss_fn(pos_pfi_logits.view(-1, 2), pos_pfi_label.view(-1)) * self.pfi_weight
            neg_pfi_loss = self.loss_fn(neg_pfi_logits.view(-1, 2), neg_pfi_label.view(-1)) * self.pfi_weight


        # import ipdb; ipdb.set_trace() 

        return(mlm_loss,pos_pfi_loss,neg_pfi_loss)

                
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


# performs pooling that do not considers pads efficiently, supports max,avg and summation
def pool(h, mask, type='max'):
    # h dim (batch,seq len, feat dim); mask dim(batch, seq len,1|feat dim)
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def copy_layers(src_layers, dest_layers, layers_to_copy):
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())


@dataclass
class GlobalStructureTripletLoss:
    """Paper-style margin triplet loss on pooled protein sequence embeddings."""

    margin: float = 0.2
    distance_type: str = "l2"

    def __init__(self, margin: float = 0.2, distance_type: str = "l2"):
        self.margin = float(margin)
        self.distance_type = str(distance_type).lower()
        if self.distance_type not in {"l2", "cosine"}:
            raise ValueError("distance_type must be 'l2' or 'cosine'")

    def _distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.distance_type == "cosine":
            return 1.0 - F.cosine_similarity(x.float(), y.float(), dim=1)
        return torch.norm(x.float() - y.float(), p=2, dim=1)

    def __call__(self, anchor_repr: torch.Tensor, positive_repr: torch.Tensor, negative_repr: torch.Tensor) -> torch.Tensor:
        if anchor_repr.shape != positive_repr.shape or anchor_repr.shape != negative_repr.shape:
            raise ValueError(
                f"Triplet repr shape mismatch: {tuple(anchor_repr.shape)}, {tuple(positive_repr.shape)}, {tuple(negative_repr.shape)}"
            )
        pos_dist = self._distance(anchor_repr, positive_repr)
        neg_dist = self._distance(anchor_repr, negative_repr)
        loss = torch.relu(pos_dist - neg_dist + self.margin).mean()
        if not torch.isfinite(loss):
            raise ValueError("Triplet/global-structure loss became non-finite")
        return loss


@dataclass
class TMVecLoss:
    """
    Global structure loss computed on GLProtein sequence embeddings.

    Optional TM-Vec embeddings are treated as teacher targets
    for similarity distillation. Contrastive loss is computed on
    the embeddings produced by the model being trained.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        distill_weight: float = 0.0,
    ):
        self.temperature = float(temperature)
        self.distill_weight = float(distill_weight)

    def _pairwise_contrastive_loss(self, student_repr: torch.Tensor, pair_id: torch.Tensor) -> torch.Tensor:
        if student_repr.ndim != 2:
            raise ValueError(f"Expected 2D student embeddings, got shape {tuple(student_repr.shape)}")
        if pair_id.ndim != 1 or pair_id.shape[0] != student_repr.shape[0]:
            raise ValueError("pair_id must be a 1D tensor aligned with the batch dimension")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

        z = F.normalize(student_repr.float(), dim=1)
        logits = torch.matmul(z, z.transpose(0, 1)) / self.temperature
        batch_size = logits.shape[0]
        eye = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        valid_mask = ~eye

        positive_mask = pair_id.unsqueeze(0).eq(pair_id.unsqueeze(1))
        positive_mask = positive_mask & valid_mask
        positive_counts = positive_mask.sum(dim=1)
        if torch.any(positive_counts == 0):
            bad = torch.nonzero(positive_counts == 0, as_tuple=False).view(-1).tolist()
            raise ValueError(f"Missing positives at indices {bad[:8]}")

        nonpair_mask = valid_mask & (~pair_id.unsqueeze(0).eq(pair_id.unsqueeze(1)))
        nonpair_counts = nonpair_mask.sum(dim=1)
        if torch.any(nonpair_counts == 0):
            bad = torch.nonzero(nonpair_counts == 0, as_tuple=False).view(-1).tolist()
            raise ValueError(
                f"Rows without negatives: {bad[:8]}. Increase per_device_train_batch_size to at least 4, "
                "increase the number of mined pairs, or enable dataloader_drop_last to avoid one-pair final batches."
            )

        masked_logits = logits.masked_fill(~valid_mask, float('-inf'))
        log_denom = torch.logsumexp(masked_logits, dim=1, keepdim=True)
        log_prob = logits - log_denom
        positive_log_prob_sum = log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
        mean_log_prob_pos = positive_log_prob_sum / positive_counts.to(log_prob.dtype)
        loss = -mean_log_prob_pos.mean()
        if not torch.isfinite(loss):
            raise ValueError("TM-Vec/global-structure loss became non-finite")
        return loss

    def _teacher_distill_loss(self, student_repr: torch.Tensor, teacher_repr: Optional[torch.Tensor]) -> torch.Tensor:
        if teacher_repr is None or self.distill_weight <= 0.0:
            return student_repr.new_zeros(())
        if teacher_repr.shape != student_repr.shape:
            raise ValueError(
                f"Teacher and student representation mismatch: "
                f"{tuple(teacher_repr.shape)} vs {tuple(student_repr.shape)}"
            )
        s = F.normalize(student_repr.float(), dim=1)
        t = F.normalize(teacher_repr.float().to(student_repr.device), dim=1)
        s_sim = torch.matmul(s, s.transpose(0, 1))
        t_sim = torch.matmul(t, t.transpose(0, 1))
        mask = ~torch.eye(s_sim.shape[0], device=s_sim.device, dtype=torch.bool)
        return F.mse_loss(s_sim[mask], t_sim[mask])

    def __call__(self, student_repr: torch.Tensor, pair_id: torch.Tensor, tmvec_emb: Optional[torch.Tensor] = None, **kwargs):
        contrastive_loss = self._pairwise_contrastive_loss(student_repr=student_repr, pair_id=pair_id)
        if tmvec_emb is not None and not torch.is_tensor(tmvec_emb):
            tmvec_emb = torch.as_tensor(tmvec_emb, dtype=student_repr.dtype, device=student_repr.device)
        distill_loss = self._teacher_distill_loss(student_repr=student_repr, teacher_repr=tmvec_emb)
        return contrastive_loss + self.distill_weight * distill_loss
