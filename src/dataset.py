import os
from posixpath import join
from sys import path
import time
import lmdb
import torch
import json
import numpy as np
import pickle as pkl
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from scipy.spatial import distance_matrix
import pickle
# from .prediction import prediction
import numpy as np
import math as m
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from rdkit import Chem
import re
from transformers import T5Tokenizer
from Bio import SwissProt
from itertools import islice






## cart2sph
def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
    az = m.atan2(y,x)                           # phi
    return r, elev, az

def cart2sphArray(points):
    sph = []
    for i in range(len(points)):
        r_, elev_, az_ = cart2sph(points[i][0],points[i][1],points[i][2])
        sph.append([r_, elev_, az_])
    return np.array(sph)



def _split_go_by_type(go_types) -> Dict[str, List]:
    component_go = []
    function_go = []
    process_go = []
    for go_id, type_ in go_types.items():
        if type_ == 'Process':
            process_go.append(go_id)
        elif type_ == 'Component':
            component_go.append(go_id)
        elif type_ == 'Function':
            function_go.append(go_id)
        else:
            print(type_,len(type_))
            raise Exception('the type not supported.')

    go_terms_type_dict = {
        'Process': process_go,
        'Component': component_go,
        'Function': function_go
    }

    return go_terms_type_dict


def get_triplet_data(data_path):
    #input is protein_go_train triplet
    heads = []
    relations = []
    tails = []
    true_tail = {}
    true_head = {}

    for line in open(data_path, 'r'):
        head, relation, tail = [int(id) for id in line.rstrip('\n').split()]
        heads.append(head)
        relations.append(relation)
        tails.append(tail)

        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    true_tail = {key: np.array(list(set(val))) for key, val in true_tail.items()}
    true_head = {key: np.array(list(set(val))) for key, val in true_head.items()}
    return heads, relations, tails, true_tail, true_head


@dataclass
class ProteinGoInputFeatures:
    """
    A single set of feature of data for OntoProtein pretrain.
    """
    postive_protein_input_ids: List[int]
    postive_relation_ids: int
    postive_go_input_ids: Union[int, List[int]]
    negative_protein_input_ids: List[List[int]] = None
    negative_protein_attention_mask: Optional[List[int]] = None
    negative_relation_ids: List[int] = None
    negative_go_input_ids: List[Union[int, List[int]]] = None
    coordinates: Optional[List[List[float]]] = None
    aa_vec: Optional[Dict] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass
class GoGoInputFeatures:
    """
    A single set of feature of data for Go-GO triplet in OntoProtein pretrain.
    """
    postive_go_head_input_ids: Union[int, List[int]]
    postive_relation_ids: int
    postive_go_tail_input_ids: Union[int, List[int]]
    negative_go_head_input_ids: List[Union[int, List[int]]] = None
    negative_relation_ids: List[int] = None
    negative_go_tail_input_ids: List[Union[int, List[int]]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass
class ProteinSeqInputFeatures:
    """
    A single set of feature of data for protein sequences.
    """
    input_ids: List[int]
    coordinates: Optional[List[List[float]]] = None
    label: Optional[Union[int, float]] = None
    

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"
    

class ProteinGoDataset(Dataset):
    """
    Dataset for Protein-GO triplet.

    Args:
        data_dir: the diractory need contain pre-train datasets.
        use_seq: Whether or not to use the representation of protein sequence through encoder as entity embedding.
        use_desc: Whether or not to use the representation of Go term' description through encoder as entity embedding. 
                  Otherwise, using the embedding of Go term' entity in KE.
        protein_tokenizer: Tokenizer used to tokenize protein sequence.
        text_tokenizer: Tokenizer used to tokenize text.
        negative_sampling_fn: The strategy of negative sampling.
        num_neg_sample: the number of negative samples on one side. In other words, if set `sample_head` and `sample_tail`
                        to `True`, the total number of negative samples is 2*`num_neg_sample`.
        sample_head: Whether or not to construct negative sample pairs by fixing tail entity.
        sample_tail: Whether or not to construct negative sample pairs by fixing head entity.
        max_protein_seq_length: the max length of sequence. If set `None` to `max_seq_length`, It will dynamically set the max length
                        of sequence in batch to `max_seq_length`.
        max_text_seq_length: It need to set `max_text_seq_length` when using desciption of Go term to represent the Go entity.
    """
    def __init__(
        self,
        data_dir: str,
        use_seq: bool,
        use_desc: bool,
        protein_tokenizer: PreTrainedTokenizerBase = None,
        text_tokenizer: PreTrainedTokenizerBase = None,
        negative_sampling_fn = None,
        num_neg_sample: int = 1,
        sample_head: bool = False,
        sample_tail: bool = True,
        max_protein_seq_length: int = None,
        max_text_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.use_seq = use_seq
        self.use_desc = use_desc
        self._load_data()

        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.negative_sampling_fn = negative_sampling_fn
        self.num_neg_sample = num_neg_sample
        self.sample_head = sample_head
        self.sample_tail = sample_tail
        self.max_protein_seq_length = max_protein_seq_length
        self.max_text_seq_length = max_text_seq_length
    
    def _load_data(self):
        # go2id and relation2id are dictionaries. Keys are go or words, values are id.
        go2id = [line.rstrip('\n').split() for line in open(os.path.join(self.data_dir, 'go2id.txt'), 'r')]
        go2id_dict = {}
        for i in range(len(go2id)):
            go2id_dict[go2id[i][0]] = go2id[i][1]
        self.go2id = go2id_dict


        relation_id = [line.rstrip('\n').split('\t') for line in open(os.path.join(self.data_dir, 'relation2id.txt'), 'r')]
        id2relation_dict={}
        for i in range(len(relation_id)):
            id2relation_dict[relation_id[i][1]] = relation_id[i][0].replace('_',' ').replace('|',' ')
        self.id2relation = id2relation_dict

        self.num_go_terms = len(self.go2id)
        self.num_relations = len(self.id2relation)

        self.go_types = {idx: line.rstrip('\n') for idx, line in enumerate(open(os.path.join(self.data_dir, 'go_type.txt'), 'r'))}
        with open('data/seqs.pkl','rb') as f:
            self.protein_seq = pickle.load(f)
        # self.protein_seq = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'protein_seq.txt'), 'r')] #avg protein len ~360
        def trans_sequence(sequence):
            sequence = " ".join(sequence)
            sequence = re.sub(r"[UZOB]", "X", sequence) 
            return sequence
        self.protein_seq = [trans_sequence(item) for item in self.protein_seq]
        self.num_proteins = len(self.protein_seq)

        #go_descs is a dictionary of go descriptions, id2description
        if self.use_desc:
            self.go_descs = {idx: line.rstrip('\n') for idx, line in enumerate(open(os.path.join(self.data_dir, 'go_def.txt'), 'r'))}
        
        # split go term according to ontology type.
        self.go_terms_type_dict = _split_go_by_type(self.go_types)

        # for negative sample. true_tail is a dict, key is (head,relation) value is list of tails. True head: key is (relation,tail) 
        self.protein_heads, self.pg_relations, self.go_tails, self.true_tail, self.true_head = get_triplet_data(
            data_path=os.path.join(self.data_dir, 'protein_go_train_triplet_v2.txt')
        )
        # self.protein_heads = [trans_sequence(item) for item in self.protein_heads]
        self.protein_cor = pickle.load(open(
            self.data_dir + '/id2cor_dict.pkl',
              'rb'))
        protein_cor = pickle.load(open('/home/yunqing/GLProtein/coordinates.pkl','rb'))
        # print('cor_length:',len(self.protein_cor))
        # print('protein_num:',len(self.protein_seq))
        # # print(self.protein_cor[0])
        # import pdb;pdb.set_trace()

        # aa vec
        aa_smis = ['CC(N)C(=O)O', 'N=C(N)NCCCC(N)C(=O)O', 'NC(=O)CC(N)C(=O)O', 'NC(CC(=O)O)C(=O)O',
            'NC(CS)C(=O)O', 'NC(CCC(=O)O)C(=O)O', 'NC(=O)CCC(N)C(=O)O', 'NCC(=O)O',
            'NC(Cc1cnc[nH]1)C(=O)O', 'CCC(C)C(N)C(=O)O', 'CC(C)CC(N)C(=O)O', 'NCCCCC(N)C(=O)O',
            'CSCCC(N)C(=O)O', 'NC(Cc1ccccc1)C(=O)O', 'O=C(O)C1CCCN1', 'NC(CO)C(=O)O',
            'CC(O)C(N)C(=O)O', 'NC(Cc1c[nH]c2ccccc12)C(=O)O', 'NC(Cc1ccc(O)cc1)C(=O)O',
            'CC(C)C(N)C(=O)O','CC1CC=NC1C(=O)NCCCCC(C(=O)O)N','C(C(C(=O)O)N)[Se]']
        aa_codes = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V','O','U', 'B', 'Z', 'X'] # B = D or N  , Z = E or Q , X 

        aa_idx_codes = dict(zip(aa_codes, range(len(aa_codes))))

        aas = [Chem.MolFromSmiles(x) for x in aa_smis]

        model = word2vec.Word2Vec.load('./src/model_300dim.pkl')
        aa_sentences = [mol2alt_sentence(x, 1) for x in aas]

        aa_vecs = sentences2vec(aa_sentences, model, unseen='UNK')

        B_vec = (aa_vecs[aa_idx_codes['D']] + aa_vecs[aa_idx_codes['N']])/2
        B_vec = B_vec.reshape(1,300)
        Z_vec = (aa_vecs[aa_idx_codes['E']] + aa_vecs[aa_idx_codes['Q']])/2
        Z_vec = Z_vec.reshape(1,300)
        aa_vecs = np.concatenate([aa_vecs,B_vec,Z_vec],axis=0)
        X_vecs = np.mean(aa_vecs,axis=0).reshape(1,300)
        aa_vecs = np.concatenate([aa_vecs,X_vecs],axis=0)

        aa_vocab = dict()

        tmp_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        # tm-vec expects tokenizer.batch_encode_plus in some versions
        if not hasattr(tmp_tokenizer, "batch_encode_plus"):
            tmp_tokenizer.batch_encode_plus = tmp_tokenizer.__call__
        tmp_vocab = tmp_tokenizer.get_vocab()
        keys = list (tmp_vocab.keys())
        signal = keys[5][0]
        tmp_vocab = list(tmp_vocab)
        for i in range(len(tmp_vocab)):
            if tmp_vocab[i][0] == signal:
                aa_vocab[i] = {'aa':keys[i][1],'vec':list(aa_vecs[aa_idx_codes[keys[i][1]]])}

        self.aa_vocab = aa_vocab

        assert len(self.protein_heads) == len(self.pg_relations) and len(self.pg_relations) == len(self.go_tails), "error with dataloading, the number of samples of protein,relation,go do not match"



    def __getitem__(self, index):

        protein_head_id, relation_id, go_tail_id = self.protein_heads[index], self.pg_relations[index], self.go_tails[index]
        relation_str = self.id2relation[str(relation_id)]
        relation_input_ids = self.text_tokenizer.encode(relation_str, max_length=55, truncation=True, padding='max_length')

        protein_input_ids = protein_head_id

        # use sequence.
        if self.use_seq:
            # tokenize protein sequence.
            # protein_head_seq = list(self.protein_seq[protein_head_id])
            protein_head_seq = self.protein_seq[protein_head_id].split(' ')
            if self.max_protein_seq_length is not None:
                protein_head_seq = protein_head_seq[:self.max_protein_seq_length] # remove amino after max seq len
            protein_head_seq = ' '.join(protein_head_seq)

            # import ipdb;ipdb.set_trace()
            protein_input_ids = self.protein_tokenizer.encode(protein_head_seq)

        go_tail_type = self.go_types[go_tail_id]
        go_input_ids = go_tail_id
        if self.use_desc:
            go_desc = self.go_descs[go_tail_id]
            go_input_ids = self.text_tokenizer.encode(go_desc, max_length=self.max_text_seq_length, truncation=True, padding='max_length') #max_text_len 128

        negative_protein_input_ids_list = []
        negative_relation_ids_list = []
        negative_go_input_ids_list = []

        if self.sample_tail:
            # list of negative go tail terms(id)
            tail_negative_samples = self.negative_sampling_fn(
                cur_entity=(protein_head_id, relation_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_tail,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_tail_type]
            )

            for neg_go_id in tail_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length, truncation=True, padding='max_length')

                negative_protein_input_ids_list.append(protein_input_ids)
                negative_relation_ids_list.append(relation_input_ids)
                negative_go_input_ids_list.append(neg_go_input_ids)

        #TODO
        
        protein_sequence = self.protein_seq[protein_head_id]
        protein_sequence = protein_sequence.split(' ')
        protein_sequence = ''.join(protein_sequence)
        cor = self.protein_cor[protein_head_id]
        assert len(cor) == len(protein_sequence)
        cor = []
        try:
            
            with open('../output/predictions/protein_'+ str(protein_head_id) + ".pdb",'r') as f:
                CA = f.readlines()
                for line in CA:
                    line = line.replace('-', ' -')
                    line_split = line.split()
                    if len(line_split)>2:
                        if line_split[2]=='CA':
                            cor.append([float(line_split[6]), float(line_split[7]), float(line_split[8])])
        except:
            cor = np.zeros((len(protein_sequence),3)).tolist()


    
        if self.max_protein_seq_length is not None:
            cor = cor[:self.max_protein_seq_length]
        ### coordinates normalize & padding
        # cor = np.array(cor)-np.array(cor).mean(axis=0)
        cor = np.array(cor)
        if cor.any():
            cor = (cor - cor.mean(axis=0)) / cor.std(axis=0)
        # print(cor)
        # cor = cart2sphArray(cor)
        # print(cor)
        # import ipdb;ipdb.set_trace()
        padding = np.full((1,3),float('-inf'))
        # cor = np.concatenate([padding,cor,padding],axis=0)
        cor = cor.tolist()

        

        # if len(cor) != len(protein_input_ids):
        #     cor = np.zeros((len(protein_input_ids),3)).tolist()

        # print("cor_length:",len(cor))
        # print("seq_length:",len(protein_input_ids))
        # import ipdb;ipdb.set_trace()

        assert len(negative_protein_input_ids_list) == len(negative_relation_ids_list)
        assert len(negative_relation_ids_list) == len(negative_go_input_ids_list)
        
        

        
    
        aa_vec = []
        # import ipdb;ipdb.set_trace()
        aa_vec.append(self.aa_vocab[protein_input_ids[0]]['vec'])
        aa_vec_padding = np.full((1,300),float('-inf'))

        for i in range(1,len(protein_input_ids)-1):
            if protein_input_ids[i] in self.aa_vocab:
                aa_vec.append(list(self.aa_vocab[protein_input_ids[i]]['vec']))
            else:
                aa_vec.append(list(aa_vec_padding))
        # aa_vec  = np.array(aa_vec)


        if self.max_protein_seq_length is not None:
            aa_vec = aa_vec[:self.max_protein_seq_length]
        # import ipdb;ipdb.set_trace()
        # note negative relation ids = [relation id]*neg_sample size
        # negative protein ids = [protein id]*neg_sample size
        return ProteinGoInputFeatures(
            postive_protein_input_ids=protein_input_ids,
            postive_relation_ids=relation_input_ids,
            postive_go_input_ids=go_input_ids,
            negative_protein_input_ids=negative_protein_input_ids_list,
            negative_relation_ids=negative_relation_ids_list,
            negative_go_input_ids=negative_go_input_ids_list,
            coordinates = cor,
            aa_vec = aa_vec
        )


    def __len__(self):
        assert len(self.protein_heads) == len(self.pg_relations)
        assert len(self.pg_relations) == len(self.go_tails)

        return len(self.protein_heads)

    def get_num_go_terms(self):
        return len(self.go_types)

    def get_num_protein_go_relations(self):
        return len(list(set(self.pg_relations)))


class ProteinSeqDataset(Dataset):
    """
    Dataset for Protein sequence.

    Args:
        data_dir: the diractory need contain pre-train datasets.
        seq_data_file_name: path of sequence data, in view of the multiple corpus choices (e.g. Swiss, UniRef50...), 
                            and only support LMDB file.
        tokenizer: tokenizer used for encoding sequence.
        in_memory: Whether or not to save full sequence data to memory. Suggest that set to `False` 
                   when using UniRef50 or larger corpus.
    """

    def __init__(
        self,
        data_dir: str,
        seq_data_path: str = None,
        tokenizer: PreTrainedTokenizerBase = None,
        in_memory: bool=True,
        max_protein_seq_length: int = None,
        protein_seq_sample_limit: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.seq_data_path = seq_data_path

        # self.env = lmdb.open(os.path.join(data_dir, seq_data_path), readonly=True)
        
        # with self.env.begin(write=False) as txn:
        #     self.num_examples = pkl.loads(txn.get(b'num_examples'))

        # self.in_memory = in_memory
        # if in_memory:
        #     cache = [None] * self.num_examples
        #     self.cache = cache

        def trans_sequence(sequence):
            sequence = " ".join(sequence)
            sequence = re.sub(r"[UZOB]", "X", sequence) 
            return sequence
        
        with open(os.path.join(self.data_dir, "uniprot_sprot.dat")) as f:
            records = SwissProt.parse(f)
            if protein_seq_sample_limit is None:
                self.protein_seq = [r.sequence for r in records]
            else:
                self.protein_seq = [r.sequence for r in islice(records, protein_seq_sample_limit)]

        
        # self.protein_seq = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'uniprot_sprot.dat'), 'r')]
        self.protein_seq = [trans_sequence(item) for item in self.protein_seq]

        self.tokenizer = tokenizer
        self.max_protein_seq_length = max_protein_seq_length
        # self.protein_cor = pickle.load(open('./ProteinKG25/id2cor_dict.pkl', 'rb'))
        
    def __getitem__(self, index):
        # if self.in_memory and self.cache[index] is not None:
        #     item = self.cache[index]
        # else:
        #     with self.env.begin(write=False) as txn:
        #         item = pkl.loads(txn.get(str(index).encode()))
        #     if self.in_memory:
        #         self.cache[index] = item
        item = self.protein_seq[index]

        # implement padding of sequences at 'DataCollatorForLanguageModeling'
        # item = list(item)
        if self.max_protein_seq_length is not None:
            tokens = item.split()[:self.max_protein_seq_length]
            item = " ".join(tokens)
        input_ids = self.tokenizer.encode(item, add_special_tokens=True)

        # cor = self.protein_cor[index]
        # if self.max_protein_seq_length is not None:
        #     cor = cor[:self.max_protein_seq_length]
        # ### coordinates normalize & padding
        # cor = np.array(cor)-np.array(cor).mean(axis=0)
        # cor = np.concatenate([np.zeros((1,3)),cor,np.zeros((1,3))],axis=0)
        # cor = cor.tolist()

        return ProteinSeqInputFeatures(
            input_ids=input_ids,
            # coordinates=cor,
        )
        
    def __len__(self):
        # return self.num_examples
        return len(self.protein_seq)
    
    # def get_distance_matrix(self,index):
    #     item = self.protein_cor[index]
    #     if self.max_protein_seq_length is not None:
    #         item = item[:self.max_protein_seq_length]
    #     ### coordinates normalize & padding
    #     item = np.array(item)-np.array(item).mean(axis=0)
    #     # item = np.concatenate([np.zeros((1,3)),item,np.zeros((1,3))],axis=0)

    #     ### distance matrix

    #     distance = distance_matrix(item,item)

    #     return distance


@dataclass
class ProteinSeqPairInputFeatures:
    """
    A set of features for paired protein sequences used by TMVecLoss.
    """
    input_ids: List[int]
    sequence: str


class ProteinSeqPairDataset(Dataset):
    """
    A dataset that yields paired protein sequences used by TMVecLoss.

    Expected TSV format: <anchor_sequence>\t<positive_sequence>

    Note: Sequences should be raw amino acid strings from UniProt.
    """

    def __init__(
        self,
        data_dir: str,
        pairs_tsv: str,
        tokenizer: PreTrainedTokenizerBase,
        max_protein_seq_length: Optional[int] = None,
        protein_seq_sample_limit: Optional[int] = None,
    ):
        def trans_sequence(sequence: str) -> str:
            sequence = " ".join(sequence)
            sequence = re.sub(r"[UZOB]", "X", sequence)
            return sequence

        self.data_dir = data_dir
        self.pairs_tsv = pairs_tsv
        self.tokenizer = tokenizer
        self.max_protein_seq_length = max_protein_seq_length

        if not os.path.isabs(self.pairs_tsv):
            self.pairs_tsv = os.path.join(self.data_dir, self.pairs_tsv)
        if not os.path.exists(self.pairs_tsv):
            raise FileNotFoundError(f"TSV not found: {self.pairs_tsv}")

        self.pairs: List[Tuple[str, str]] = []
        with open(self.pairs_tsv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError("Invalid TSV line")
                anchor, positive = parts[0].strip(), parts[1].strip()
                self.pairs.append((trans_sequence(anchor), trans_sequence(positive)))

        if protein_seq_sample_limit is not None:
            max_pairs = int((protein_seq_sample_limit + 1) // 2)
            self.pairs = self.pairs[:max_pairs]

        if len(self.pairs) == 0:
            raise ValueError("No pairs loaded from TSV")

    def __len__(self) -> int:
        # each pair yields 2 examples (anchor then positive)
        return len(self.pairs) * 2

    def __getitem__(self, index: int) -> ProteinSeqPairInputFeatures:
        pair_idx = index // 2
        anchor_or_positive = index % 2
        seq = self.pairs[pair_idx][anchor_or_positive]

        if self.max_protein_seq_length is not None:
            tokens = seq.split()[: self.max_protein_seq_length]
            seq = " ".join(tokens)

        input_ids = self.tokenizer.encode(seq, add_special_tokens=True)
        return ProteinSeqPairInputFeatures(input_ids=input_ids, sequence=seq)


class GoGoDataset(Dataset):
    """
    Dataset used for Go-Go triplet.

    Args:
        data_dir: the diractory need contain pre-train datasets.
        use_desc: Whether or not to use the representation of Go term' description through encoder as entity embedding. 
                  Otherwise, using the embedding of Go term' entity in KE.
        text_tokenizer: Tokenizer used for tokenize the description of Go term.
        negative_sampling_fn: the strategy of negative sampling.
        num_neg_sample: the number of negative samples on one side. In other words, if set `sample_head` and `sample_tail`
                        to `True`, the total number of negative samples is 2*`num_neg_sample`.
        sample_head: Whether or not to construct negative sample pairs by fixing tail entity.
        sample_tail: Whether or not to construct negative sample pairs by fixing head entity.
        max_text_seq_length: It need to set `max_text_seq_length` when using desciption of Go term to represent the Go entity.
    """

    def __init__(
        self,
        data_dir: str,
        use_desc: bool = False,
        text_tokenizer: PreTrainedTokenizerBase = None,
        negative_sampling_fn = None,
        num_neg_sample: int = 1,
        sample_head: bool = True,
        sample_tail: bool = True,
        max_text_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.use_desc = use_desc
        self.text_tokenizer = text_tokenizer
        self.negative_sampling_fn = negative_sampling_fn
        self.num_neg_sample = num_neg_sample
        self.sample_head = sample_head
        self.sample_tail = sample_tail
        self.max_text_seq_length = max_text_seq_length
        self._load_data()

    def _load_data(self):
        self.go2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'go2id.txt'), 'r')]
        self.relation2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'relation2id.txt'), 'r')]
        self.num_go_terms = len(self.go2id)
        self.num_relations = len(self.relation2id)

        self.go_types = {idx: line.rstrip('\n') for idx, line in enumerate(open(os.path.join(self.data_dir, 'go_type.txt'), 'r'))}
        if self.use_desc:
            self.go_descs = {idx: line.rstrip('\n') for idx, line in enumerate(open(os.path.join(self.data_dir, 'go_def.txt'), 'r'))}

        # split go term according to ontology type.
        # same negative sampling strategy in `ProteinGODataset`
        self.go_terms_type_dict = _split_go_by_type(self.go_types)
        self.go_heads, self.gg_relations, self.go_tails, self.true_tail, self.true_head = get_triplet_data(
            data_path=os.path.join(self.data_dir, 'go_go_triplet.txt')
        )

    def __getitem__(self, index):
        go_head_id, relation_id, go_tail_id = self.go_heads[index], self.gg_relations[index], self.go_tails[index]

        go_head_type = self.go_types[go_head_id]
        go_tail_type = self.go_types[go_tail_id]
        go_head_input_ids = go_head_id
        go_tail_input_ids = go_tail_id
        if self.use_desc:
            go_head_desc = self.go_descs[go_head_id]
            go_tail_desc = self.go_descs[go_tail_id]
            go_head_input_ids = self.text_tokenizer.encode(go_head_desc, padding='max_length', truncation=True, max_length=self.max_text_seq_length)
            go_tail_input_ids = self.text_tokenizer.encode(go_tail_desc, padding='max_length', truncation=True, max_length=self.max_text_seq_length)
        
        negative_go_head_input_ids_list = []
        negative_relation_ids_list = []
        negative_go_tail_input_ids_list = []

        if self.sample_tail:
            tail_negative_samples = self.negative_sampling_fn(
                cur_entity=(go_head_id, relation_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_tail,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_tail_type]
            )

            for neg_go_id in tail_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length, truncation=True, padding='max_length')

                negative_go_head_input_ids_list.append(go_head_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_tail_input_ids_list.append(neg_go_input_ids)

        if self.sample_head:
            head_negative_samples = self.negative_sampling_fn(
                cur_entity=(relation_id, go_tail_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_head,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_head_type]
            )

            for neg_go_id in head_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length, truncation=True, padding='max_length')
                
                negative_go_head_input_ids_list.append(neg_go_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_tail_input_ids_list.append(go_tail_input_ids)

        assert len(negative_go_head_input_ids_list) == len(negative_relation_ids_list)
        assert len(negative_relation_ids_list) == len(negative_go_tail_input_ids_list)

        return GoGoInputFeatures(
            postive_go_head_input_ids=go_head_input_ids,
            postive_relation_ids=relation_id,
            postive_go_tail_input_ids=go_tail_input_ids,
            negative_go_head_input_ids=negative_go_head_input_ids_list,
            negative_relation_ids=negative_relation_ids_list,
            negative_go_tail_input_ids=negative_go_tail_input_ids_list
        )

    def __len__(self):
        assert len(self.go_heads) == len(self.gg_relations)
        assert len(self.gg_relations) == len(self.go_tails)

        return len(self.go_heads)

    def get_num_go_terms(self):
        return len(self.go_types)

    def get_num_go_go_relations(self):
        return len(list(set(self.gg_relations)))


class ProteinCorDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cor_data_path: str = None,
        max_protein_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.cor_data_path = cor_data_path
        
        self.protein_cor = pickle.load(open('../ProteinKG25/id2cor_dict.pkl', 'rb'))
        self.max_protein_seq_length = max_protein_seq_length


    def __getitem__(self, index):
        item = self.protein_cor[index]
        if self.max_protein_seq_length is not None:
            item = item[:self.max_protein_seq_length]
        
        ### coordinates normalize & padding
        item = np.array(item)-np.array(item).mean(axis=0)
        item = np.concatenate([np.zeros((1,3)),item,np.zeros((1,3))],axis=0)

        return item
    
    def __len__(self):
        return len(self.protein_cor)

    def get_distance_matrix(self,index):
        item = self.protein_cor[index]
        if self.max_protein_seq_length is not None:
            item = item[:self.max_protein_seq_length]
        ### coordinates normalize & padding
        item = np.array(item)-np.array(item).mean(axis=0)
        item = np.concatenate([np.zeros((1,3)),item,np.zeros((1,3))],axis=0)

        ### distance matrix

        distance = distance_matrix(item,item)

        return distance
