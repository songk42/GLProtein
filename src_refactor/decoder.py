import math
from math import gamma
import os
import json
import copy
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, BertModel, BertPreTrainedModel
# from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertIntermediate, BertOutput, BertSelfOutput, prune_linear_layer, BaseModelOutputWithPastAndCrossAttentions, BertEmbeddings
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertIntermediate, BertOutput, BertSelfOutput, BaseModelOutputWithPastAndCrossAttentions, BertEmbeddings
from transformers.file_utils import ModelOutput
from transformers.utils import logging
# from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
# from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices
from transformers.pytorch_utils import apply_chunking_to_forward
# from transformers.modeling_utils import find_pruneable_heads_and_indices
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions,Seq2SeqLMOutput,BaseModelOutput
# from deepspeed import DeepSpeedEngine
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List
import torch.nn.functional as F
import warnings
from transformers.modeling_outputs import ModelOutput


# import logging




logger = logging.get_logger('pretrain_log')

_CHECKPOINT_FOR_DOC = "decoder"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


DECODER_CONFIG_NAME = "config.json"
PROTEIN_CONFIG_NAME = "config.json"
PROTEIN_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'
DECODER_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class KMAEConfig:
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

    def save_to_json_file(self, encoder_save_directory: os.PathLike, decoder_save_directory: os.PathLike):
        os.makedirs(encoder_save_directory, exist_ok=True)
        os.makedirs(decoder_save_directory, exist_ok=True)

        self.protein_model_config.save_pretrained(encoder_save_directory)
        self.decoder_config.save_pretrained(decoder_save_directory)

        logger.info(f'Encoder Configuration saved in {encoder_save_directory}')
        logger.info(f'Decoder Configuration saved in {decoder_save_directory}')

    @classmethod
    def from_json_file(cls, encoder_config_path: os.PathLike, decoder_config_path: os.PathLike):
        config = cls()
        config.protein_model_config = AutoConfig.from_pretrained(encoder_config_path)
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
    attentions: Optional[Tuple[torch.FloatTensor]] = None
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
 

class BertSelfAttention(nn.Module):
    """
    performs attention with protein seq embeddings as query and relation/go/coordinate as key, value.

    THIS VERSION TAKES PROTEIN AND TEXT EMBEDDING WITH SAME DIMENSION AS INPUT

    notes: check if text bert output is masked
    """


    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        self.protein_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.text_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        attn_bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:

        """
        hidden_states: protein hidden states
        attention_mask: protein padding mask

        text_hidden_states: aa_vec hidden states (key/value)
        text_attention_mask: aa_vec padding mask
        attn_bias: Phi_distance 3D distance encoding [batch, num_heads, L, L]
        """

        # print(f'hidden states shape: {hidden_states.shape}')
        # print('hidden states: ', hidden_states)
        # import ipdb; ipdb.set_trace()

        hidden_states = self.protein_norm(hidden_states)
        text_hidden_states = self.text_norm(text_hidden_states)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = text_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = text_attention_mask
        # use cross_attention in our method
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(text_hidden_states))
            value_layer = self.transpose_for_scores(self.value(text_hidden_states))
            # only use the text attention_mask
            attention_mask = text_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attn_bias is not None:
            attention_scores = attention_scores + attn_bias
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # original mask shape: (batch,1,1,Seq Len), broadcastable to (batch,#heads,Seq len, Seq len)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs




# BertAttention outputs the result of a multihead self attn, followed by fc, LN, dropout
class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        attn_bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            text_hidden_states,
            text_attention_mask,
            past_key_value,
            output_attentions,
            attn_bias,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    """
    Edited BertLayer module from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    
    This BertLayer implementes one layer of our decoder.

    protein seq embedding -> attention with relation embedding -> attention with go embedding -> mlp

    Note BertAttention, BertIntermediate, BertOutput are imported from modelling_bert.py
    """

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        # cross-attention: protein queries aa_vec (K/V), with 3D distance bias
        self.aa_vec_attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        aa_vec_hidden_states: Optional[torch.FloatTensor] = None,
        aa_vec_attention_mask: Optional[torch.FloatTensor] = None,
        attn_bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # cross-attention: protein (query) attends to aa_vec (key/value) with 3D distance bias
        aa_vec_attention_outputs = self.aa_vec_attention(
            hidden_states,
            attention_mask,
            head_mask,
            text_hidden_states=aa_vec_hidden_states,
            text_attention_mask=aa_vec_attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            attn_bias=attn_bias,
        )
        aa_vec_attention_output = aa_vec_attention_outputs[0]
        outputs = aa_vec_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, aa_vec_attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        aa_vec_hidden_states: Optional[torch.FloatTensor] = None,
        aa_vec_attention_mask: Optional[torch.FloatTensor] = None,
        attn_bias: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                    aa_vec_hidden_states,
                    aa_vec_attention_mask,
                    attn_bias,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                    aa_vec_hidden_states,
                    aa_vec_attention_mask,
                    attn_bias,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class KnowledgeBertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].aa_vec_attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        aa_vec_hidden_states: Optional[torch.FloatTensor] = None,
        aa_vec_attention_mask: Optional[torch.FloatTensor] = None,
        graph_attn_bias: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if aa_vec_hidden_states is None:
            raise ValueError("You have to specify aa_vec_hidden_states")
        if aa_vec_attention_mask is None:
            raise ValueError("You have to specify aa_vec_attention_mask")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # expand aa_vec mask to [batch, 1, 1, aa_vec_len] for broadcasting across heads
        extended_aa_vec_attention_mask = aa_vec_attention_mask[:, None, None, :].to(dtype=self.dtype)
        extended_aa_vec_attention_mask = (1.0 - extended_aa_vec_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            aa_vec_hidden_states=aa_vec_hidden_states,
            aa_vec_attention_mask=extended_aa_vec_attention_mask,
            attn_bias=graph_attn_bias,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # import ipdb; ipdb.set_trace()
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

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


