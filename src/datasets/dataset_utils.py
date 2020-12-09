import hashlib
import logging
from collections import Counter
from typing import Dict, Tuple, Union, List, Set

import torch
import os
import _pickle as pkl
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.nn.util import move_to_device
from nlp_tools.allen_data.iterators import get_bucket_iterator
from nlp_tools.allennlp_training_callbacks.callbacks import OutputWriter
from nlp_tools.data_io.data_utils import Lemma2Synsets, MultilingualLemma2Synsets
from nlp_tools.data_io.datasets import LabelVocabulary, WSDDataset

from src.misc.wsdlogging import get_info_logger

logger = get_info_logger(__name__)  # pylint: disable=invalid-name

WORDNET_DICT_PATH = "/opt/WordNet-3.0/dict/index.sense"


def offsets_from_wn_sense_index():
    lemmapos2gold = dict()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
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
        inventory_dir ="resources/evaluation_framework_3.0/inventories/"
    for lang in langs:
        lemmapos2gold = dict()
        inventory_path = os.path.join(inventory_dir, "inventory.{}.withgold.txt".format(lang))
        if not os.path.exists(inventory_path):
            inventory_path = os.path.join(inventory_dir, "inventory.{}.txt".format(lang))
        with open(inventory_path) as lines:
            for line in lines:
                fields = line.strip().lower().split("\t")
                if len(fields) < 2:
                    continue
                lemma, pos = fields[0].split("#")
                pos = get_simplified_pos(pos)
                lemmapos = lemma + "#" + pos  # + "#" + lang
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


def vocabulary_from_gold_key_file(gold_key, key2wnid_path=None, key2bnid_path=None):
    key2id = None
    if key2bnid_path:
        key2id = load_bn_key2id_map(key2bnid_path)
    elif key2wnid_path:
        key2id = load_wn_key2id_map(key2wnid_path)
    labels = Counter()
    with open(gold_key) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            golds = [x.replace("%5", "%3") for x in fields[1:]]
            if key2id is not None:
                golds = [key2id[g] for g in golds]
            labels.update(golds)
    return LabelVocabulary(labels, specials=["<pad>", "<unk>"])


def wnoffset_vocabulary():
    offsets = list()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            offsets.append(offset)
    return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>", "<unk>"])


def bnoffset_vocabulary():
    wn2bn = get_wnoffset2bnoffset()
    offsets = set()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            bnoffset = wn2bn[offset]
            offsets.update(bnoffset)
    return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>", "<unk>"])


def wn_sensekey_vocabulary():
    with open(WORDNET_DICT_PATH) as lines:
        keys = [line.strip().split(" ")[0].replace("%5", "%3") for line in lines]
    return LabelVocabulary(Counter(sorted(keys)), specials=["<pad>", "<unk>"])


def get_label_mapper(target_inventory, labels):
    label_types = set(
        ["wnoffsets" if l.startswith("wn:") else "bnoffsets" if l.startswith("bn:") else "sensekeys" for l in labels if
         l != "<pad>" and l != "<unk>"])
    if target_inventory in label_types:
        label_types.remove(target_inventory)
    if len(label_types) > 1:
        raise RuntimeError(
            "cannot handle the mapping from 2 or more label types ({}) to the target inventory {}".format(
                ",".join(label_types), target_inventory))
    if len(label_types) == 0:
        return None
    label_type = next(iter(label_types))
    if label_type == "wnoffsets":
        if target_inventory == "bnoffsets":
            return get_wnoffset2bnoffset()
        elif target_inventory == "sensekeys":
            return get_wnoffset2wnkeys()
        return None
    elif label_type == "sensekeys":
        if target_inventory == "bnoffsets":
            return get_wnkeys2bnoffset()
        elif target_inventory == "wnoffsets":
            return get_wnkeys2wnoffset()
        else:
            return None
    else:
        if target_inventory == "wnoffsets":
            return get_bnoffset2wnoffset()
        elif target_inventory == "sensekeys":
            return get_bnoffset2wnkeys()
        else:
            raise RuntimeError("Cannot infer label type from {}".format(label_type))


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


def get_allen_datasets(cached_dataset_file_name: str,
                       encoder_name: str,
                       lemma2synsets: MultilingualLemma2Synsets,
                       label_vocab: LabelVocabulary,
                       label_mapper: Dict,
                       max_segments_in_batch: int,
                       lang2paths: Dict[str, List[str]],
                       force_reload: bool,
                       serialize=True,
                       is_trainingset=True,
                       device=torch.device("cuda"),
                       pos=None):
    training_ds = get_dataset(encoder_name, lang2paths, lemma2synsets, label_mapper, label_vocab, pos=pos)
    training_ds.index_with(Vocabulary())
    training_iterator = get_bucket_iterator(training_ds, max_segments_in_batch, is_trainingset=is_trainingset,
                                            device=device)
    return training_ds, training_iterator


def get_dataset(encoder_name: str,
                paths: Dict[str, List[str]],
                lemma2synsets: MultilingualLemma2Synsets,
                label_mapper: Dict[str, str],
                label_vocab: LabelVocabulary, pos:Union[None, Set]=None) \
        -> AllennlpDataset:
    indexer = PretrainedTransformerMismatchedIndexer(encoder_name)
    dataset = WSDDataset(paths, lemma2synsets=lemma2synsets, label_mapper=label_mapper,
                         indexer=indexer, label_vocab=label_vocab, pos=pos)
    return dataset


def get_dev_dataset(dev_lang, dev_name, test_dss: Dict[str,List[str]], test_names):
    dev_ds = None,None
    if dev_name is not None:
        lang_dss = test_dss[dev_lang]
        names = test_names[dev_lang]
        if lang_dss is None:
            return None,None
        try:
            dev_index = names.index(dev_name)
            dev_ds = lang_dss[dev_index]
        except:
            return None,None
    return dev_ds


def get_test_datasets(dataset_builder, encoder_name, label_mapper, langs, mfs_file, test_paths):
    get_cached_dataset_file_name(*test_paths, encoder_name)
    test_dss = [dataset_builder(encoder_name, t, label_mapper, langs, mfs_file)[0] for t in test_paths]
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


def get_data(sense_inventory, langs, mfs_file, **kwargs) -> Tuple[MultilingualLemma2Synsets, Dict, LabelVocabulary]:
    if sense_inventory == "wnoffsets":
        getter = get_wnoffsets_data
    elif sense_inventory == "sensekeys":
        getter = get_sensekey_data
    elif sense_inventory == "bnoffsets":
        getter = get_bnoffsets_data
    else:
        raise RuntimeError(
            "%s sense_inventory has not been recognised, ensure it is one of the following: {wnoffsets, sensekeys, bnoffsets}" % (
                sense_inventory))
    return getter(langs, mfs_file, **kwargs)


def get_mapper(training_paths, sense_inventory):
    paths = [p for v in training_paths.values() for p in v]
    all_labels = list()
    for f in paths:
        with open(f.replace(".data.xml", ".gold.key.txt")) as reader:
            all_labels.extend([l.split(" ")[1] for l in reader if len(l.strip()) > 0])
    label_mapper = get_label_mapper(target_inventory=sense_inventory, labels=all_labels)
    if label_mapper is not None and len(label_mapper) > 0: ## handles the case when training set has a key set and test sets h
        for k, v in list(label_mapper.items()):
            for x in v:
                label_mapper[x] = [x]
    return label_mapper


def get_cached_dataset_file_name(*args):
    m = hashlib.sha256()
    for arg in args:
        m.update(bytes(str(arg), 'utf8'))
    return m.hexdigest()


def get_wnoffsets_data(langs, mfs_file=None, **kwargs) -> Tuple[MultilingualLemma2Synsets, Dict, LabelVocabulary]:
    label_vocab = wnoffset_vocabulary()
    lemma2synsets = offsets_from_wn_sense_index()
    if langs is not None:
        if "en" in langs:
            langs.remove("en")
        if len(langs) > 0:
            bn2wn = get_bnoffset2wnoffset()
            bnlemma2synsets = from_bn_mapping(langs, **kwargs)
            for key, bns in bnlemma2synsets.items():
                wns = [x for y in bns for x in bn2wn[y]]
                if key in lemma2synsets:
                    lemma2synsets[key].update(wns)
                else:
                    lemma2synsets[key] = wns

    for key, synsets in lemma2synsets.items():
        lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]
    mfs_vocab = get_mfs_vocab(mfs_file)

    return lemma2synsets, mfs_vocab, label_vocab


def get_bnoffsets_data(langs=("en"), mfs_file=None, **kwargs) -> Tuple[MultilingualLemma2Synsets, Dict, LabelVocabulary]:
    lemma2synsets = from_bn_mapping(langs, **kwargs)
    label_vocab = bnoffset_vocabulary()
    for lang in langs:
        inventory = lemma2synsets.get_inventory(lang)
        for key, synsets in inventory.items():
            inventory[key] = [label_vocab.get_idx(l) for l in synsets]
    mfs_vocab = get_mfs_vocab(mfs_file)
    return lemma2synsets, mfs_vocab, label_vocab


def get_sensekey_data(label_mapper, langs=None, mfs_file=None, **kwargs) \
        -> Tuple[Lemma2Synsets, Dict, LabelVocabulary]:
    if langs is not None:
        logger.warning(
            "[get_sensekey_dataset]: the argument langs: {} is ignored by this method.".format(",".join(langs)))

    label_vocab = wn_sensekey_vocabulary()
    lemma2synsets = sensekey_from_wn_sense_index()
    for key, synsets in lemma2synsets.items():
        lemma2synsets[key] = [label_mapper.get_idx(l) for l in synsets]
    mfs_vocab = get_mfs_vocab(mfs_file)

    return lemma2synsets, mfs_vocab, label_vocab


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


def get_wnoffset2bnoffset():
    offset2bn = __load_reverse_multimap("resources/mappings/all_bn_wn.txt")
    new_offset2bn = {"wn:" + offset: bns for offset, bns in offset2bn.items()}
    return new_offset2bn


def get_bnoffset2wnoffset():
    return __load_multimap("resources/mappings/all_bn_wn.txt", value_transformer=lambda x: "wn:" + x)


def get_wnkeys2bnoffset():
    return __load_reverse_multimap("resources/mappings/all_bn_wn_keys.txt",
                                   key_transformer=lambda x: x.replace("%5", "%3"))


def get_bnoffset2wnkeys():
    return __load_multimap("resources/mappings/all_bn_wn_key.txt", value_transformer=lambda x: x.replace("%5", "%3"))


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


def __load_reverse_multimap(path, key_transformer=lambda x: x, value_transformer=lambda x: x):
    sensekey2bnoffset = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            for key in fields[1:]:
                offsets = sensekey2bnoffset.get(key, set())
                offsets.add(value_transformer(bnid))
                sensekey2bnoffset[key_transformer(key)] = offsets
    for k, v in sensekey2bnoffset.items():
        sensekey2bnoffset[k] = list(v)
    return sensekey2bnoffset


def __load_multimap(path, key_transformer=lambda x: x, value_transformer=lambda x: x):
    bnoffset2wnkeys = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnoffset = fields[0]
            bnoffset2wnkeys[key_transformer(bnoffset)] = [value_transformer(x) for x in fields[1:]]
    return bnoffset2wnkeys


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
            self.writer.write(i + "\t" + self.labeldict[p] + "\t" + self.labeldict[l] + "\n")


def build_en_bn_lexeme2synsets_mapping(output_path):
    lemmapos2gold = dict()
    wnoffset2bnoffset = get_wnoffset2bnoffset()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            bnoffset = wnoffset2bnoffset[offset]
            lexeme = key.split("%")[0] + "#" + pos
            golds = lemmapos2gold.get(lexeme, set())
            golds.update(bnoffset)
            lemmapos2gold[lexeme] = golds
    with open(output_path, "wt") as writer:
        for lemmapos, bnids in lemmapos2gold.items():
            writer.write(lemmapos + "\t" + "\t".join(bnids) + "\n")