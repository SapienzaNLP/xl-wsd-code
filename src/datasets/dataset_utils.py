import hashlib
import logging
import os
from collections import Counter
from typing import Dict, List, Set, Tuple, Union

import _pickle as pkl
import torch
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from nlp_tools.allen_data.iterators import get_bucket_iterator
from nlp_tools.allennlp_training_callbacks.callbacks import OutputWriter
from nlp_tools.data_io.data_utils import MultilingualLemma2Synsets
from nlp_tools.data_io.datasets import LabelVocabulary, WSDDataset
from src.datasets import (BABELNET_VOCABULARY,
                          DEFAULT_INVENTORY_DIR,
                          WORDNET_DICT_PATH)
from src.utils.logging import get_info_logger

logger = get_info_logger(__name__)


def offsets_from_wn_sense_index():
    lemmapos2gold = dict()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            lexeme = key.split("%")[0] + "#" + pos
            golds = lemmapos2gold.get(lexeme, set())
            golds.add(offset)
            lemmapos2gold[lexeme] = golds
    return MultilingualLemma2Synsets(**{"en": lemmapos2gold})


def from_bn_mapping(langs=("en"), **kwargs):
    lang2inventory = dict()
    if "inventory_dir" in kwargs and kwargs["inventory_dir"] is not None:
        inventory_dir = kwargs.pop("inventory_dir")
    else:
        inventory_dir = DEFAULT_INVENTORY_DIR
    for lang in langs:
        lemmapos2gold = dict()
        inventory_path = os.path.join(
            inventory_dir, "inventory.{}.withgold.txt".format(lang)
        )
        if not os.path.exists(inventory_path):
            inventory_path = os.path.join(
                inventory_dir, "inventory.{}.txt".format(lang)
            )
        with open(inventory_path) as lines:
            for line in lines:
                fields = line.strip().lower().split("\t")
                if len(fields) < 2:
                    continue
                lemma, pos = fields[0].split("#")
                pos = get_simplified_pos(pos)
                lemmapos = lemma + "#" + pos
                synsets = fields[1:]
                old_synsets = lemmapos2gold.get(lemmapos, set())
                old_synsets.update(synsets)
                lemmapos2gold[lemmapos] = old_synsets
        lang2inventory[lang] = lemmapos2gold
    return MultilingualLemma2Synsets(**lang2inventory)


def sensekey_from_wn_sense_index():
    lemmapos2gold = dict()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            lexeme = key.split("%")[0] + "#" + pos
            golds = lemmapos2gold.get(lexeme, set())
            golds.add(key)
            lemmapos2gold[lexeme] = golds
    return MultilingualLemma2Synsets(**{"en": lemmapos2gold})


def load_bn_offset2bnid_map(path):
    offset2bnid = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            for wnid in fields[1:]:
                offset2bnid[wnid] = bnid
    return offset2bnid


def load_wn_key2id_map(path):
    """
    assume the path points to a file in the same format of index.sense in WordNet dict/ subdirectory
    :param path: path to the file
    :return: dictionary from key to wordnet offsets
    """
    key2id = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            key2id[key] = ("wn:%08d" % int(fields[1])) + pos
    return key2id


def load_bn_key2id_map(path):
    """
    assumes the path points to a file with the following format:
    bnid\twn_key1\twn_key2\t...
    :param path:
    :return: a dictionary from wordnet key to bnid
    """
    key2bn = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bn = fields[0]
            for k in fields[1:]:
                key2bn[k] = bn
    return key2bn


def bnoffset_vocabulary():
    bnids = set()
    with open(BABELNET_VOCABULARY) as lines:
        for line in lines:
            bnids.add(line.strip())
    return LabelVocabulary(Counter(sorted(bnids)), specials=["<pad>", "<unk>"])


def get_mfs_vocab(mfs_file):
    if mfs_file is None:
        return None
    mfs = dict()
    with open(mfs_file) as lines:
        for line in lines:
            fields = line.strip().lower().split("\t")
            if len(fields) < 2:
                continue
            mfs[fields[0].lower()] = fields[1].replace("%5", "%3")
    return mfs


def get_allen_datasets(
    encoder_name: str,
    lemma2synsets: MultilingualLemma2Synsets,
    label_vocab: LabelVocabulary,
    label_mapper: Dict,
    max_segments_in_batch: int,
    lang2paths: Dict[str, List[str]],
    is_trainingset=True,
    device=torch.device("cuda"),
    pos=None,
):
    training_ds = get_dataset(
        encoder_name, lang2paths, lemma2synsets, label_mapper, label_vocab, pos=pos
    )
    # training_ds.index_with(Vocabulary())
    training_iterator = get_bucket_iterator(
        training_ds, max_segments_in_batch, is_trainingset=is_trainingset, device=device
    )
    return training_ds, training_iterator


def get_dataset(
    encoder_name: str,
    paths: Dict[str, List[str]],
    lemma2synsets: MultilingualLemma2Synsets,
    label_mapper: Dict[str, str],
    label_vocab: LabelVocabulary,
    pos: Union[None, Set] = None,
) -> DatasetReader:
    indexer = PretrainedTransformerMismatchedIndexer(encoder_name)
    dataset = WSDDataset(
        paths,
        lemma2synsets=lemma2synsets,
        label_mapper=label_mapper,
        indexer=indexer,
        label_vocab=label_vocab,
        pos=pos,
    )
    return dataset


def get_dev_dataset(dev_lang, dev_name, test_dss: Dict[str, List[str]], test_names):
    dev_ds = None, None
    if dev_name is not None:
        lang_dss = test_dss[dev_lang]
        names = test_names[dev_lang]
        if lang_dss is None:
            return None, None
        try:
            dev_index = names.index(dev_name)
            dev_ds = lang_dss[dev_index]
        except:
            return None, None
    return dev_ds


def get_test_datasets(
    dataset_builder, encoder_name, label_mapper, langs, mfs_file, test_paths
):
    get_cached_dataset_file_name(*test_paths, encoder_name)
    test_dss = [
        dataset_builder(encoder_name, t, label_mapper, langs, mfs_file)[0]
        for t in test_paths
    ]
    for td in test_dss:
        td.index_with(Vocabulary())
    return test_dss


def build_outpath_subdirs(path):
    if not os.path.exists(path):
        os.mkdir(path)
    try:
        os.mkdir(os.path.join(path, "checkpoints"))
    except:
        pass
    try:
        os.mkdir(os.path.join(path, "predictions"))
    except:
        pass


def get_data(langs, mfs_file, **kwargs) -> Tuple[MultilingualLemma2Synsets, Dict, LabelVocabulary]:
    lemma2synsets = from_bn_mapping(langs, **kwargs)
    label_vocab = bnoffset_vocabulary()
    for lang in langs:
        inventory = lemma2synsets.get_inventory(lang)
        for key, synsets in inventory.items():
            inventory[key] = [label_vocab.get_idx(l) for l in synsets]
    mfs_vocab = get_mfs_vocab(mfs_file)
    return lemma2synsets, mfs_vocab, label_vocab

def get_cached_dataset_file_name(*args):
    m = hashlib.sha256()
    for arg in args:
        m.update(bytes(str(arg), "utf8"))
    return m.hexdigest()


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


def get_wnoffset2wnkeys():
    offset2keys = dict()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            keys = offset2keys.get(fields[1], set())
            keys.add(fields[0].replace("%5", "%3"))
            offset2keys[fields[1]] = keys
    return offset2keys


def get_wnkeys2wnoffset():
    key2offset = dict()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0].replace("%5", "%3")
            pos = get_pos_from_key(key)
            key2offset[key] = ["wn:" + fields[1] + pos]
    return key2offset

def get_pos_from_key(key):
    """
    assumes key is in the wordnet key format, i.e., 's_gravenhage%1:15:00:
    :param key: wordnet key
    :return: pos tag corresponding to the key
    """
    numpos = key.split("%")[-1][0]
    if numpos == "1":
        return "n"
    elif numpos == "2":
        return "v"
    elif numpos == "3" or numpos == "5":
        return "a"
    else:
        return "r"


def get_universal_pos(simplified_pos):
    if simplified_pos == "n":
        return "NOUN"
    if simplified_pos == "v":
        return "VERB"
    if simplified_pos == "a":
        return "ADJ"
    if simplified_pos == "r":
        return "ADV"
    return ""


def get_simplified_pos(long_pos):
    long_pos = long_pos.lower()
    if long_pos.startswith("n") or long_pos.startswith("propn"):
        return "n"
    elif long_pos.startswith("adj") or long_pos.startswith("j") or long_pos == "a":
        return "a"
    elif long_pos.startswith("adv") or long_pos.startswith("r") or long_pos == "r":
        return "r"
    elif long_pos.startswith("v"):
        return "v"
    return "o"


class SemEvalOutputWriter(OutputWriter):
    def __init__(self, output_file, labeldict):
        super().__init__(output_file, labeldict)

    def write(self, outs):
        predictions = outs["predictions"]
        labels = outs["labels"]
        ids = outs["ids"]
        if type(predictions) is torch.Tensor:
            predictions = predictions.flatten().tolist()
        else:
            predictions = torch.cat(predictions).tolist()
        if type(labels) is torch.Tensor:
            labels = labels.flatten().tolist()
        else:
            labels = torch.cat(labels).tolist()
        for i, p, l in zip(ids, predictions, labels):
            self.writer.write(
                i + "\t" + self.labeldict[p] + "\t" + self.labeldict[l] + "\n"
            )
