import os
import subprocess
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, List

import torch
import yaml

from allennlp.data import Vocabulary, DataLoader
from allennlp.nn.util import move_to_device
from nlp_tools.data_io.datasets import LabelVocabulary
from pandas import DataFrame
from tabulate import tabulate
from tqdm import tqdm
import pandas

from src.datasets.dataset_utils import get_data, get_allen_datasets
from src.modelling.neural_wsd_models import AllenWSDModel, WSDF1
from src.utils.utils import get_info_logger, get_model


logger = get_info_logger(__name__)
def evaluate(data_loader, model, output_path, label_vocab, device_int, use_mfs=False, mfs_vocab=None,
             verbose=True, debug=False):
    f1_computer = WSDF1(label_vocab, use_mfs, mfs_vocab)
    batch_generator = iter(data_loader)

    with open(output_path, "w") as writer:
        bar = batch_generator
        if verbose:
            bar = tqdm(batch_generator)
        if debug:
            debugwriter = open(output_path + ".debug.txt", "w")
        for batch in bar:
            if device_int >= 0:
                batch = move_to_device(batch, device_int)
            outputs = model(**batch)
            ids = [y for x in batch["ids"] for y in x if y is not None]
            predictions = outputs["predictions"].tolist()
            labels = [y for x in batch["labels"] for y in x if y != ""]
            possible_labels = [y for x in batch["possible_labels"] for y in x]
            lemmapos = [y for x in batch["labeled_lemmapos"] for y in x]
            assert len(ids) == len(predictions) == len(labels) == len(possible_labels) == len(lemmapos)

            f1_computer(lemmapos, predictions, labels, ids=ids)
            f1_computer.get_metric(False)
            writer.write(
                "\n".join(["{} {}".format(id, label_vocab.itos[p]) for id, p in zip(ids, predictions)]))
            writer.write("\n")
            if mfs_vocab is not None:
                mfs_preds = [mfs_vocab.get(lp, "<unk>") if label_vocab.itos[p] == "<unk>" else label_vocab.itos[p]
                             for
                             lp, p in
                             zip(lemmapos, predictions)]
            if debug:
                for id, lp, p, label, possible_labels in zip(ids, lemmapos, predictions, labels,
                                                             [[label_vocab.get_string(y) for y in x] for x in
                                                              possible_labels]):
                    is_mfs = label_vocab.itos[p] == "<unk>"
                    if mfs_vocab is not None:
                        pred = mfs_vocab.get(lp, "<unk>") if label_vocab.itos[p] == "<unk>" else label_vocab.itos[p]
                    else:
                        pred = label_vocab.itos[p]
                    if debug:
                        debugwriter.write(
                            "{}\t{}\t{}\t{}\t{}\n".format(id, lp, pred, label, ", ".join(possible_labels)))
        metric = f1_computer.get_metric(True)
        if debug:
            debugwriter.close()
        return metric


def evaluate_datasets(model: AllenWSDModel,
                      data_loaders: Dict[str, List[DataLoader]],
                      test_names: Dict[str, List[str]],
                      checkpoint_path: str,
                      label_vocab: LabelVocabulary,
                      device_int: int,
                      mfs_dictionary: Dict,
                      output_path,
                      verbose=True, debug=False,
                      ):
    logger.info("loading checkpoint " + checkpoint_path)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu" if device_int < 0 else "cuda:{}".format(device_int)))
    model.eval()
    all_metrics = dict()
    names = list()
    lines = list()
    logger.info("Evaluation starts")
    for lang, datasets in data_loaders.items():
        t_names = test_names[lang]
        for (data_loader, iterator), name in zip(datasets, t_names):
            names.append(name)
            metrics = evaluate(iterator, model, os.path.join(output_path, name + ".predictions.txt"),
                               label_vocab, device_int, mfs_dictionary is not None, mfs_dictionary, verbose=verbose,
                               debug=debug)
            all_metrics[name] = OrderedDict(
                {"precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
                 "f1_mfs": metrics.get("f1_mfs", None)})
            if verbose:
                logger.info(f"{name}: instances: {metrics['total']}, precision: {metrics['precision']}, recall: {metrics['recall']}, f1: {metrics['f1']}")
            lines.append([metrics["total"],metrics["f1"]])
    logger.fino("SUMMARY:")
    d = DataFrame(lines, columns=["instances","F1"], index=names)
    with pandas.option_context('display.max_rows', None, 'display.max_columns',
                               None, 'display.float_format', '{:0.3f}'.format):
        print(d.to_csv(sep="\t"))
        print(tabulate(d))


def main(args):
    with open(args.config) as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    verbose = args.verbose
    debug = args.debug
    test_pos = args.pos
    if test_pos is not None:
        test_pos = set(test_pos)

    data_config = config["data"]
    model_config = config["model"]
    outpath = data_config["outpath"]
    wsd_model_name = model_config["wsd_model_name"]
    test_data_root = data_config["test_data_root"]
    test_lang2name = data_config["test_names"]
    inventory_dir = data_config.get("inventory_dir", None)
    langs = data_config["langs"]
    cpu = args.cpu
    mfs_file = data_config.get("mfs_file", None)
    device = "cpu" if cpu else "cuda"
    encoder_name = model_config["encoder_name"]
    output_path = os.path.join(outpath, wsd_model_name + "_" + encoder_name)
    if "checkpoint_path" in vars(args) and args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    elif "checkpoint_name" in model_config and model_config["checkpoint_name"] is not None:
        checkpoint_path = os.path.join(output_path, "checkpoints", model_config["checkpoint_name"])
    else:
        checkpoint_path = os.path.join(output_path, "checkpoints", "best.th")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("path for checkpoints does not exist", checkpoint_path)
    device_int = 0 if device == "cuda" else -1
    lang2test_paths = {lang: [os.path.join(test_data_root, name, name + ".data.xml") for name in names] for lang, names
                       in test_lang2name.items()}
    
    test_label_mapper = None
    lemma2synsets, mfs_dictionary, label_vocab = get_data(langs, mfs_file, inventory_dir=inventory_dir)
    test_dss = {lang: [get_allen_datasets(encoder_name, lemma2synsets,
                                          label_vocab, test_label_mapper, config["data"]["max_segments_in_batch"],
                                          {lang: [tp]}, force_reload=True, serialize=False,
                                          device = torch.device(device), pos=test_pos) for tp in test_paths]
                for lang, test_paths in lang2test_paths.items()}

    metric = WSDF1(label_vocab, mfs_dictionary is not None, mfs_dictionary)
    dataset = list(test_dss.values())[0][0][0]
    model = get_model(model_config, len(label_vocab), dataset.pad_token_id, label_vocab.stoi["<pad>"],
                      metric=metric, device=device)
    if "output_path" in vars(args) and args.output_path is not None:
        eval_path = args.output_path
    else:
        eval_path = os.path.join(output_path, "evaluation/")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    evaluate_datasets(model, test_dss, test_lang2name, checkpoint_path,
                      label_vocab, device_int,
                      mfs_dictionary,
                      eval_path,
                      verbose=verbose,
                      debug=debug)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--pos", default=None, nargs="+")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
