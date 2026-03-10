import torch
from dataclasses import dataclass
from torch._C import dtype
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Optional, Tuple
import numpy as np
from src_refactor.dataset import ProteinGoInputFeatures, GoGoInputFeatures, ProteinSeqInputFeatures, ProteinSeqPairInputFeatures, ProteinSeqTripletInputFeatures
import random




def _collate_batch_for_protein_seq(
    examples: List[Dict], 
    tokenizer: PreTrainedTokenizerBase,
    are_protein_length_same: bool
):  
    # import ipdb;ipdb.set_trace()
    if isinstance(examples[0], (ProteinSeqInputFeatures, ProteinSeqPairInputFeatures)):
        examples = [torch.tensor(e.input_ids, dtype=torch.long) for e in examples]
    elif isinstance(examples[0], ProteinSeqTripletInputFeatures):
        raise ValueError('ProteinSeqTripletInputFeatures should be collated via anchor/positive/negative fields')
    elif isinstance(examples[0], dict) and 'input_ids' in examples[0]:
        examples = [torch.tensor(e['input_ids'], dtype=torch.long) for e in examples]

    if are_protein_length_same:
        return torch.stack(examples, dim=0)

    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], fill_value=tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == 'right':
            result[i, :example.size(0)] = example
        else:
            result[i, -example.size(0):] = example
    return result

def _collate_batch_for_protein_cor(
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        are_protein_length_same: bool
):
    if isinstance(examples[0], ProteinGoInputFeatures):
        examples = [torch.as_tensor(np.asarray(e.coordinates, dtype=np.float32)) for e in examples]
    elif isinstance(examples[0], ProteinSeqTripletInputFeatures):
        examples = [torch.from_numpy(np.asarray(e.anchor_coordinates, dtype=np.float32)) for e in examples]
    elif isinstance(examples[0], dict) and 'coordinates' in examples[0]:
        examples = [torch.as_tensor(np.asarray(e['coordinates'], dtype=np.float32)) for e in examples]

    if are_protein_length_same:
        return torch.stack(examples, dim=0)

    max_length = max(x.size(0) for x in examples)
    result = torch.full((len(examples), max_length, 3), float('-inf'), dtype=torch.float32)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == 'right':
            result[i, :example.size(0)] = example
        else:
            result[i, -example.size(0):] = example
    return result


def _collate_batch_for_aa_vec(
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        are_protein_length_same: bool
):
    if isinstance(examples[0], ProteinGoInputFeatures):
        examples = [torch.as_tensor(np.asarray(e.aa_vec, dtype=np.float32)) for e in examples]
    elif isinstance(examples[0], ProteinSeqTripletInputFeatures):
        examples = [torch.from_numpy(np.asarray(e.anchor_aa_vec, dtype=np.float32)) for e in examples]
    elif isinstance(examples[0], dict) and 'aa_vec' in examples[0]:
        examples = [torch.as_tensor(np.asarray(e['aa_vec'], dtype=np.float32)) for e in examples]

    if are_protein_length_same:
        return torch.stack(examples, dim=0)

    feature_dim = int(examples[0].size(-1)) if examples else 0
    max_length = max(x.size(0) for x in examples)
    result = torch.zeros((len(examples), max_length, feature_dim), dtype=torch.float32)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == 'right':
            result[i, :example.size(0)] = example
        else:
            result[i, -example.size(0):] = example
    return result


def _collate_batch_for_protein_go(
    examples: List[ProteinGoInputFeatures],
    protein_tokenizer: PreTrainedTokenizerBase,
    text_tokenizer: PreTrainedTokenizerBase,
    are_protein_length_same: bool,
    use_pfi: bool
):  
    assert isinstance(examples[0], ProteinGoInputFeatures), "Only support `ProteinGoInputFeatures`"

    use_seq = False
    use_desc = False
    # positive_go_input_ids is a list of int, each id represents a word in input go sentence
    if not isinstance(examples[0].postive_go_input_ids, int):
        use_desc = True
    # postive_protein_input_ids is list of intc each id represent an amino acid in protein seq
    if not isinstance(examples[0].postive_protein_input_ids, int):
        use_seq = True

    
    # collate postive samples
    # protein
    if use_seq:
        # use sequence
        all_postive_protein_input_ids = [torch.tensor(example.postive_protein_input_ids, dtype=torch.long) for example in examples] # list -> tensor
        if are_protein_length_same:
            all_postive_protein_input_ids = torch.stack(all_postive_protein_input_ids, dim=0)
        # protein sequence padding, pad id = 0
        else:
            max_length = max(x.size(0) for x in all_postive_protein_input_ids)
            all_postive_protein_input_ids_ = all_postive_protein_input_ids[0].new_full([len(all_postive_protein_input_ids), max_length], fill_value=protein_tokenizer.pad_token_id)
            for i, postive_protein_input_ids in enumerate(all_postive_protein_input_ids):
                if protein_tokenizer.padding_side == 'right':
                    all_postive_protein_input_ids_[i, :postive_protein_input_ids.size(0)] = postive_protein_input_ids
                else:
                    all_postive_protein_input_ids_[i, -postive_protein_input_ids.size(0):] = postive_protein_input_ids
            all_postive_protein_input_ids = all_postive_protein_input_ids_
    else:
        # not use sequence
        all_postive_protein_input_ids = torch.tensor([example.postive_protein_input_ids for example in examples], dtype=torch.long)
    # relation
    all_postive_relation_ids = torch.tensor([example.postive_relation_ids for example in examples], dtype=torch.long)
    # go term
    if use_desc:
        all_postive_go_input_ids = [torch.tensor(example.postive_go_input_ids, dtype=torch.long) for example in examples]
        all_postive_go_input_ids = torch.stack(all_postive_go_input_ids, dim=0)
    else:
        all_postive_go_input_ids = torch.tensor([example.postive_go_input_ids for example in examples], dtype=torch.long)
    
    all_negative_protein_input_ids = None
    all_negative_relation_ids = []
    all_negative_go_input_ids = []
    if use_pfi:
        # collate negative samples
        # protein
        for example in examples:
            all_negative_relation_ids.extend(example.negative_relation_ids)
        all_negative_relation_ids = torch.tensor(all_negative_relation_ids, dtype=torch.long)

        # go term
        all_negative_go_input_ids = []
        for example in examples:
            all_negative_go_input_ids.extend(example.negative_go_input_ids)
        all_negative_go_input_ids = torch.tensor(all_negative_go_input_ids, dtype=torch.long)

    all_postive_go_attention_mask = None
    all_postive_go_token_type_ids = None
    all_negative_go_attention_mask = None
    all_negative_go_token_type_ids = None



    if use_desc:
        all_postive_relation_attention_mask = (all_postive_relation_ids != text_tokenizer.pad_token_id).long()
        all_postive_relation_token_type_ids = torch.zeros_like(all_postive_relation_ids, dtype=torch.long)


        all_postive_go_attention_mask = (all_postive_go_input_ids != text_tokenizer.pad_token_id).long()
        all_postive_go_token_type_ids = torch.zeros_like(all_postive_go_input_ids, dtype=torch.long)

        if use_pfi:
            all_negative_go_attention_mask = (all_negative_go_input_ids != text_tokenizer.pad_token_id).long()
            all_negative_go_token_type_ids = torch.zeros_like(all_negative_go_input_ids, dtype=torch.long)

    # note all_negative_protein_input_ids = None, just use all_postive_protein_input_ids to save memory


    return {
        'protein_input_ids': all_postive_protein_input_ids,
        'relation_ids': all_postive_relation_ids,
        'relation_attention_mask': all_postive_relation_attention_mask,
        'relation_token_type_ids': all_postive_relation_token_type_ids,
        'postive': {
            'tail_input_ids': all_postive_go_input_ids,
            'tail_attention_mask': all_postive_go_attention_mask,
            'tail_token_type_ids': all_postive_go_token_type_ids
        },
        'negative': {
            'tail_input_ids': all_negative_go_input_ids,
            'tail_attention_mask': all_negative_go_attention_mask,
            'tail_token_type_ids': all_negative_go_token_type_ids
        }
    }



def _collate_batch_for_go_go(
    examples: List[GoGoInputFeatures],
    tokenizer: PreTrainedTokenizerBase,
):
    assert isinstance(examples[0], GoGoInputFeatures), "Only support `GoGoInputFeatures`"
    
    use_desc = False
    if not isinstance(examples[0].postive_go_head_input_ids, int):
        use_desc = True
    #collate postive samples.

    if use_desc:
        all_postive_go_head_input_ids = [torch.tensor(example.postive_go_head_input_ids, dtype=torch.long) for example in examples]
        all_postive_go_head_input_ids = torch.stack(all_postive_go_head_input_ids, dim=0)
    else:
        all_postive_go_head_input_ids = torch.tensor([example.postive_go_head_input_ids for example in examples], dtype=torch.long)
    # relation
    all_postive_relation_ids = torch.tensor([example.postive_relation_ids for example in examples], dtype=torch.long)
    new_pos_relation_ids = [torch.tensor(example.postive_relation_ids, dtype=torch.long) for example in examples]
    print("original relation id:   ", all_postive_relation_ids)
    print("New relation id:   ", new_pos_relation_ids)


    # Go tail
    if use_desc:
        all_postive_go_tail_input_ids = [torch.tensor(example.postive_go_tail_input_ids, dtype=torch.long) for example in examples]
        all_postive_go_tail_input_ids = torch.stack(all_postive_go_tail_input_ids, dim=0)
    else:
        all_postive_go_tail_input_ids = torch.tensor([example.postive_go_tail_input_ids for example in examples], dtype=torch.long)

    # collate negative samples.
    # Go head
    all_negative_go_head_input_ids = []
    for example in examples:
        all_negative_go_head_input_ids.extend(example.negative_go_head_input_ids)
    all_negative_go_head_input_ids = torch.tensor(all_negative_go_head_input_ids, dtype=torch.long)
    # relation
    all_negative_relation_ids = []
    for example in examples:
        all_negative_relation_ids.extend(example.negative_relation_ids)
    all_negative_relation_ids = torch.tensor(all_negative_relation_ids, dtype=torch.long)
    # Go tail
    all_negative_go_tail_input_ids = []
    for example in examples:
        all_negative_go_tail_input_ids.extend(example.negative_go_tail_input_ids)
    all_negative_go_tail_input_ids = torch.tensor(all_negative_go_tail_input_ids, dtype=torch.long)

    all_postive_go_head_attention_mask = None
    all_postive_go_head_token_type_ids = None
    all_postive_go_tail_attention_mask = None
    all_postive_go_tail_token_type_ids = None
    all_negative_go_head_attention_mask = None
    all_negative_go_head_token_type_ids = None
    all_negative_go_tail_attention_mask = None
    all_negative_go_tail_token_type_ids = None
    if use_desc:
        #mask = true when id NOT equals pad token, note 
        all_postive_go_head_attention_mask = (all_postive_go_head_input_ids != tokenizer.pad_token_id).long()
        all_postive_go_head_token_type_ids = torch.zeros_like(all_postive_go_head_input_ids, dtype=torch.long)
        all_negative_go_head_attention_mask = (all_negative_go_head_input_ids != tokenizer.pad_token_id).long()
        all_negative_go_head_token_type_ids = torch.zeros_like(all_negative_go_head_input_ids, dtype=torch.long)
        all_postive_go_tail_attention_mask = (all_postive_go_tail_input_ids != tokenizer.pad_token_id).long()
        all_postive_go_tail_token_type_ids = torch.zeros_like(all_postive_go_tail_input_ids, dtype=torch.long)
        all_negative_go_tail_attention_mask = (all_negative_go_tail_input_ids != tokenizer.pad_token_id).long()
        all_negative_go_tail_token_type_ids = torch.zeros_like(all_negative_go_tail_input_ids, dtype=torch.long)

    return {
        'postive': {
            'head_input_ids': all_postive_go_head_input_ids,
            'head_attention_mask': all_postive_go_head_attention_mask,
            'head_token_type_ids': all_postive_go_head_token_type_ids,
            'relation_ids': all_postive_relation_ids,
            'tail_input_ids': all_postive_go_tail_input_ids,
            'tail_attention_mask': all_postive_go_tail_attention_mask,
            'tail_token_type_ids': all_postive_go_tail_token_type_ids
        },
        'negative': {
            'head_input_ids': all_negative_go_head_input_ids,
            'head_attention_mask': all_negative_go_head_attention_mask,
            'head_token_type_ids': all_negative_go_head_token_type_ids,
            'relation_ids': all_negative_relation_ids,
            'tail_input_ids': all_negative_go_tail_input_ids,
            'tail_attention_mask': all_negative_go_tail_attention_mask,
            'tail_token_type_ids': all_negative_go_tail_token_type_ids
        }
    }


@dataclass
class DataCollatorForGoGo:
    """
    Data collator used for KE model which the type of dataset is `GoGoDataset`
    """
    tokenizer: PreTrainedTokenizerBase
    

    def __call__(
        self,
        examples: List[GoGoInputFeatures]
    ) -> Dict[str, torch.Tensor]:
        batch = _collate_batch_for_go_go(examples, self.tokenizer)
        return batch


@dataclass
class DataCollatorForProteinGo:
    """
    Data collator used for KE model which the type of dataset is `ProteinGoDataset`

    Args:
        protein_tokenizer: the tokenizer used for encoding protein sequence.
        are_protein_length_same: If the length of proteins in a batch is different, protein sequence will
                                 are dynamically padded to the maximum length in a batch.
    """

    protein_tokenizer: PreTrainedTokenizerBase
    text_tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.20
    are_protein_length_same: bool = False
    use_pfi: bool = False

    # def __post_init__(self):
    #     if self.mlm and self.protein_tokenizer.mask_token is None:
    #         raise ValueError(
    #             "This protein tokenizer does not have a mask token which is necessary for masked language modeling. "
    #             "You should pass `mlm=False` to train on causal language modeling instead."
    #         )

    def __call__(
        self,
        examples: List[ProteinGoInputFeatures]
    ) -> Dict[str, torch.Tensor]:


        batch = _collate_batch_for_protein_go(examples, self.protein_tokenizer, self.text_tokenizer, self.are_protein_length_same,use_pfi=self.use_pfi)
        batch['protein_coordinates']= _collate_batch_for_protein_cor(examples,self.protein_tokenizer, self.are_protein_length_same)
        # special_tokens_mask always None

        batch['aa_vec'] = _collate_batch_for_aa_vec(examples,self.protein_tokenizer, self.are_protein_length_same)
    

        special_tokens_mask = batch.pop('special_tokens_mask', None)
        if self.mlm:
            batch['protein_input_ids'], batch['protein_labels'] = self.mask_tokens(
                batch['protein_input_ids']
            )
        else:
            labels = batch['protein_input_ids'].clone()
            if self.protein_tokenizer.pad_token_id is not None:
                labels[labels == self.protein_tokenizer.pad_token_id] = -100
            batch['protein_labels'] = labels


        batch['protein_attention_mask'] = batch['protein_input_ids']['attention_mask']
        batch['coordinate_attention_mask'] = batch['protein_attention_mask']
        batch['aa_vec_attention_mask'] = batch['protein_attention_mask']
        batch['protein_token_type_ids'] = torch.zeros_like(batch['protein_input_ids']['input_ids'], dtype=torch.long)


        batch['pfi_pos'] = torch.tensor([1],dtype=torch.long)
        batch['pfi_neg'] = torch.tensor([0],dtype=torch.long)



        return batch

    def mask_tokens(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        default: 80% MASK, 10%  random, 10% original
        """
        # labels = inputs.clone()
        # probability_matrix = torch.full(labels.size(), fill_value=self.mlm_probability)
        # # if `special_tokens_mask` is None, generate it by `labels`
        # if special_tokens_mask is None:
        #     special_tokens_mask = [
        #         self.protein_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        #     ]
        #     special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # else:
        #     special_tokens_mask = special_tokens_mask.bool()

        # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # masked_indices = torch.bernoulli(probability_matrix).bool()
        # # only compute loss on masked tokens.
        # labels[~masked_indices] = -100

        # # 80% of the time, replace masked input tokens with tokenizer.mask_token
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, fill_value=0.8)).bool() & masked_indices
        # inputs[indices_replaced] = self.protein_tokenizer.convert_tokens_to_ids(self.protein_tokenizer.mask_token)

        # # 10% of the time, replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, fill_value=0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.protein_tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]


        def racha_detection(lista):
            # It returns a list of lists where each sub-list contains the consecutive tokens in the list
            rachas = []
            racha = []
            for i, element in enumerate(lista):
                if (i<len(lista)-1) and (lista[i+1] == element+1):
                    racha.append(element)
                else:
                    if len(racha)>0:
                        rachas.append(racha + [element])          
                    else:# (i!=len(lista)-1):
                        rachas.append([element])
                    racha = []
            return rachas

        def masking(tokenized_sentence, rachas,tokenizer):
            # Function to mask a tokenized_sentence (token ids) following the rachas described in rachas
            # Only one sentinel_token per racha
            sent_token_id = 0


            # tokenized_sentence = tokenized_sentence.tolist()
            
            enmascared = tokenized_sentence.copy()
            for racha in rachas:
                sent_token = f'<extra_id_{sent_token_id}>'
                sent_id = tokenizer.encode(sent_token)[0]
                for i, idx in enumerate(racha):
                    if i==0:
                        enmascared[idx] = sent_id
                    else:
                        enmascared[idx] = -100
                sent_token_id += 1
            
            enmascared = [t for t in enmascared if t!=-100] 

            return enmascared

        def add_noise(sentence, tokenizer, percent=0.15):
            # Function that takes a sentence, tokenizer and a noise percentage and returns
            # the masked input_ids and masked target_ids accordling with the T5 paper and HuggingFace docs
            # To see the process working uncomment all the prints ;)

            # import ipdb; ipdb.set_trace()
            tokenized_sentences =sentence.tolist()
            # print('PRE-MASKED:')
            # print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(tokenized_sentence)))

            
            enmascared_inputs = []
            enmascared_targets = []
            for tokenized_sentence in tokenized_sentences:
                


                idxs_2_mask = sorted(random.sample(range(len(tokenized_sentence)), 
                                                int(len(tokenized_sentence)*percent)))
                rachas = racha_detection(idxs_2_mask)
                enmascared_input = masking(tokenized_sentence, rachas,tokenizer)
                #print('RACHAS INPUT: {}'.format(rachas))
                idxs_2_mask = [idx for idx in range(len(tokenized_sentence)) if idx not in idxs_2_mask]
                rachas = racha_detection(idxs_2_mask)
                enmascared_target = masking(tokenized_sentence, rachas,tokenizer)
                # print('RACHAS TARGET: {}'.format(rachas))

                # print('POST-MASKED:')
                # print('INPUT: {}'.format(tokenizer.convert_ids_to_tokens(enmascared_input)))
                # print('TARGET: {}'.format(tokenizer.convert_ids_to_tokens(enmascared_target)))
                input = tokenizer.decode(enmascared_input)
                target = tokenizer.decode(enmascared_target)

                enmascared_inputs.append(input)
                enmascared_targets.append(target)
            

            
            input_ids = tokenizer(enmascared_inputs, return_tensors='pt', padding=True, truncation=True)
            labels = tokenizer(enmascared_targets, return_tensors='pt', padding=True, truncation=True)

            return input_ids,labels

        input_ids, labels = add_noise(inputs, self.protein_tokenizer)
        return input_ids, labels

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language model. Inputs are dynamically padded to the maximum length
    of a batch if they are not all of the same length.
    The class is rewrited from 'Transformers.data.data_collator.DataCollatorForLanguageModeling'.
        
    Agrs:
        tokenizer: the tokenizer used for encoding sequence.
        mlm: Whether or not to use masked language modeling. If set to 'False', the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability: the probablity of masking tokens in a sequence.
        are_protein_length_same: If the length of proteins in a batch is different, protein sequence will
                                 are dynamically padded to the maximum length in a batch.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.20
    are_protein_length_same: bool = False

    # def __post_init__(self):
    #     if self.mlm and self.tokenizer.mask_token is None:
    #         raise ValueError(
    #             "This tokenizer does not have a mask token which is necessary for masked language modeling. "
    #             "You should pass `mlm=False` to train on causal language modeling instead."
    #         )
    
    def __call__(
        self,
        examples: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        # example here is a list of ProteinSeqInputFeatures

        if hasattr(examples[0], 'anchor_input_ids'):
            anchor_input_ids = _collate_batch_for_protein_seq([
                {'input_ids': getattr(e, 'anchor_input_ids')} for e in examples
            ], self.tokenizer, self.are_protein_length_same)
            positive_input_ids = _collate_batch_for_protein_seq([
                {'input_ids': getattr(e, 'positive_input_ids')} for e in examples
            ], self.tokenizer, self.are_protein_length_same)
            negative_input_ids = _collate_batch_for_protein_seq([
                {'input_ids': getattr(e, 'negative_input_ids')} for e in examples
            ], self.tokenizer, self.are_protein_length_same)
            batch = {
                'input_ids': anchor_input_ids,
                'anchor_input_ids': anchor_input_ids,
                'positive_input_ids': positive_input_ids,
                'negative_input_ids': negative_input_ids,
                'anchor_attention_mask': (anchor_input_ids != self.tokenizer.pad_token_id).long(),
                'positive_attention_mask': (positive_input_ids != self.tokenizer.pad_token_id).long(),
                'negative_attention_mask': (negative_input_ids != self.tokenizer.pad_token_id).long(),
                'anchor_token_type_ids': torch.zeros_like(anchor_input_ids, dtype=torch.long),
                'positive_token_type_ids': torch.zeros_like(positive_input_ids, dtype=torch.long),
                'negative_token_type_ids': torch.zeros_like(negative_input_ids, dtype=torch.long),
                'anchor_sequence': [getattr(e, 'anchor_sequence') for e in examples],
                'positive_sequence': [getattr(e, 'positive_sequence') for e in examples],
                'negative_sequence': [getattr(e, 'negative_sequence') for e in examples],
            }
            if hasattr(examples[0], 'anchor_id'):
                batch['anchor_id'] = [getattr(e, 'anchor_id') for e in examples]
                batch['positive_id'] = [getattr(e, 'positive_id') for e in examples]
                batch['negative_id'] = [getattr(e, 'negative_id') for e in examples]
            if hasattr(examples[0], 'positive_score') and getattr(examples[0], 'positive_score') is not None:
                batch['positive_score'] = torch.tensor([getattr(e, 'positive_score') for e in examples], dtype=torch.float32)
            if hasattr(examples[0], 'negative_score') and getattr(examples[0], 'negative_score') is not None:
                batch['negative_score'] = torch.tensor([getattr(e, 'negative_score') for e in examples], dtype=torch.float32)
            special_tokens_mask = None
            if hasattr(examples[0], 'anchor_coordinates') and getattr(examples[0], 'anchor_coordinates') is not None:
                batch['coordinates'] = _collate_batch_for_protein_cor(examples, self.tokenizer, self.are_protein_length_same)
            if hasattr(examples[0], 'anchor_aa_vec') and getattr(examples[0], 'anchor_aa_vec') is not None:
                batch['aa_vec'] = _collate_batch_for_aa_vec(examples, self.tokenizer, self.are_protein_length_same)
            if self.mlm:
                batch['input_ids'], batch['labels'] = self.mask_tokens(batch['input_ids'], special_tokens_mask=special_tokens_mask)
                batch['anchor_input_ids'] = batch['input_ids']
                batch['anchor_attention_mask'] = (batch['anchor_input_ids'] != self.tokenizer.pad_token_id).long()
            else:
                labels = batch['input_ids'].clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                batch['labels'] = labels
            batch['attention_mask'] = batch['anchor_attention_mask']
            batch['token_type_ids'] = batch['anchor_token_type_ids']
            if 'aa_vec' in batch:
                batch['aa_vec_attention_mask'] = batch['anchor_attention_mask'][:, 1:-1].contiguous()
            return batch

        batch = {'input_ids': _collate_batch_for_protein_seq(examples, self.tokenizer, self.are_protein_length_same)}

        if hasattr(examples[0], 'sequence'):
            batch['sequence'] = [getattr(e, 'sequence') for e in examples]

        # protein_coordinates
        # batch['protein_coordinates'] = _collate_batch_for_protein_cor(examples, self.tokenizer, self.are_protein_length_same)
        # batch['aa_vec'] = _collate_batch_for_aa_vec(examples, self.tokenizer, self.are_protein_length_same)
        
        if hasattr(examples[0], 'pair_id'):
            batch['pair_id'] = torch.tensor([getattr(e, 'pair_id') for e in examples], dtype=torch.long)
        if hasattr(examples[0], 'view_id'):
            batch['view_id'] = torch.tensor([getattr(e, 'view_id') for e in examples], dtype=torch.long)

        if 'pair_id' in batch and 'view_id' in batch:
            pair_ids = batch['pair_id'].tolist()
            view_ids = batch['view_id'].tolist()
            counts = {}
            for pid, vid in zip(pair_ids, view_ids):
                counts.setdefault(pid, set()).add(vid)
            bad = [pid for pid, vids in counts.items() if len(vids) != len(set(vids)) or len(vids) < 2]
            if bad:
                raise ValueError(f"pair_ids do not contain both views exactly once: {bad[:5]}")

        # Use precomputed TM-Vec embeddings if available as optional teacher targets.
        if hasattr(examples[0], 'tmvec_emb') and getattr(examples[0], 'tmvec_emb') is not None:
            batch['tmvec_emb'] = torch.tensor([e.tmvec_emb for e in examples], dtype=torch.float32)
            if 'pair_id' in batch and batch['tmvec_emb'].shape[0] != batch['pair_id'].shape[0]:
                raise ValueError("tmvec_emb batch dimension must match pair_id batch dimension")

        special_tokens_mask = batch.pop('special_tokens_mask', None)
        if self.mlm:
            batch['input_ids'], batch['labels'] = self.mask_tokens(
                batch['input_ids'], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch['input_ids'].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch['labels'] = labels

        batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).long()
        # batch['coordinate_attention_mask'] = batch['attention_mask']
        # batch['aa_vec_attention_mask'] = batch['attention_mask']
        batch['token_type_ids'] = torch.zeros_like(batch['input_ids'], dtype=torch.long)


        return batch

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        default: 80% MASK, 10%  random, 10% original
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.size(), fill_value=self.mlm_probability)
        # if `special_tokens_mask` is None, generate it by `labels`
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # only compute loss on masked tokens.
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, fill_value=0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, fill_value=0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
