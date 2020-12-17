import os
import socket
from argparse import ArgumentParser
from logging import warning
from pprint import pprint

import _pickle as pkl
import numpy as np
import torch
import wandb
import yaml
from allennlp.training import Checkpointer, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import PolynomialDecay
from nlp_tools.allennlp_training_callbacks.callbacks import (
    TestAndWrite, WanDBLogger, WanDBTrainingCallback)
from src.datasets.dataset_utils import (build_outpath_subdirs,
                                        get_allen_datasets,
                                        get_data,
                                        get_dev_dataset)
from src.evaluation.evaluate_model import evaluate_datasets
from src.modelling.neural_wsd_models import WSDF1, WSDOutputWriter
from src.utils.utils import get_info_logger, get_model
from transformers import AdamW

def init_seeds(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger = get_info_logger(__name__)

def main(args):
    with open(args.config) as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    seed = config["random_seed"]
    init_seeds(seed)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    outpath = data_config["outpath"]
    test_data_root = data_config["test_data_root"]
    train_lang2paths = data_config["train_data_root"]
    test_lang2name = data_config["test_names"]
    langs = data_config["langs"]
    max_segments_in_batch = data_config["max_segments_in_batch"]
    inventory_dir = data_config.get("inventory_dir", "none")
    dev_lang, dev_name = data_config.get("dev_name", (None, None))
    wandb_config = config["wandb"]
    mfs_file = data_config.get("mfs_file", None)
    device = model_config["device"]
    encoder_name = model_config["encoder_name"]
    wsd_model_name = model_config["wsd_model_name"]
    num_epochs = training_config["num_epochs"]
    patience = training_config["patience"]
    gradient_accumulation = training_config.get("gradient_accumulation", 1)
    if args.cpu:
        device = "cpu"
    if args.gradient_clipping is not None:
        training_config["gradient_clipping"] = float(args.gradient_clipping)
        if training_config["gradient_clipping"] == 0.0:
            training_config["gradient_clipping"] = None
    if args.learning_rate is not None:
        training_config["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        training_config["weight_decay"] = float(args.weight_decay)
    if hasattr(args, 'dropout_1') and args.dropout_1 is not None:
        model_config["dropout_1"] = float(args.dropout_1)
    if hasattr(args, 'dropout_2') and args.dropout_2 is not None:
        model_config["dropout_2"] = float(args.dropout_2)
    print(training_config)
    print(model_config)
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    gradient_clipping = training_config.get("gradient_clipping", None)
    wandb_run_name = wandb_config.get("run_name", wsd_model_name + "_" + encoder_name)
    wandb.init(config=config, project=wandb_config.get("project_name", "xl-wsd-framework"),
            tags=[socket.gethostname(), wsd_model_name, ",".join(langs)],
            name=wandb_run_name, resume=wandb_config.get("resume", False),
            entity=wandb_config.get("entity", None))
    wandb.log({"random_seed": seed})
    logger.info("loading config: " + args.config)
    pprint(config)
    if dev_name is None:
        logger.warning("No dev name set... In this way I won't save in best.th the best model according to the "
                       "development set. best.th will contain the weights of the model at its last epoch")

    device_int = 0 if device == "cuda" else -1

    lang2test_paths = {lang: [os.path.join(test_data_root, name, name + ".data.xml") for name in names] for lang, names
                       in test_lang2name.items()}
    training_paths = train_lang2paths
    outpath = os.path.join(outpath, wsd_model_name + "_" + encoder_name.replace("/", "_"))
    build_outpath_subdirs(outpath)

    lemma2synsets, mfs_dictionary, label_vocab = get_data(langs, mfs_file, inventory_dir=inventory_dir)
    train_label_mapper = None

    logger.info("loading training data")
    training_ds, training_iterator = get_allen_datasets(
        encoder_name, lemma2synsets,
        label_vocab, train_label_mapper, max_segments_in_batch,
        training_paths, is_trainingset=True)

    test_label_mapper = None
    logger.info("loading testing data")
    test_dss = {lang: [get_allen_datasets(encoder_name, lemma2synsets,
                                          label_vocab, test_label_mapper, max_segments_in_batch,
                                          {lang: [tp]}, is_trainingset=False) for tp in test_paths]
                for lang, test_paths in lang2test_paths.items()}

    dev_ds, dev_iterator = get_dev_dataset(dev_lang, dev_name, test_dss, test_lang2name)
    if dev_ds is None:
        dev_path = os.path.join(test_data_root, dev_name, dev_name + ".data.xml")
        dev_label_mapper = None 
        get_allen_datasets(encoder_name, lemma2synsets, label_vocab, dev_label_mapper,
                           max_segments_in_batch, {dev_lang: [dev_path]}, True, serialize=False, is_trainingset=False)
    metric = WSDF1(label_vocab, mfs_dictionary is not None, mfs_dictionary)
    logger.info("loading model")
    model = get_model(model_config, len(label_vocab),
                      training_ds.pad_token_id,
                      label_vocab.stoi["<pad>"],
                      metric=metric, device=device)
    callbacks = list()
    wandb_logger = WanDBLogger(metrics_to_report=config["wandb"]["metrics_to_report"])
    callbacks.append(WanDBTrainingCallback(wandb_logger))
    for lang, lang_tdss in test_dss.items():
        names = test_lang2name[lang]
        for name, (dss, iterator) in zip(names, lang_tdss):
            tandw = TestAndWrite(test_iterator=iterator,
                                 output_writer=None,
                                 name=name,
                                 wandb_logger=wandb_logger,
                                 is_dev=name == dev_name if dev_name is not None else False)
            callbacks.append(tandw)

    
    optim = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    learning_rate_scheduler = None
    if training_config.get("warmup_lr", False):
        total_steps = (len(training_iterator) * num_epochs) // gradient_accumulation
        warmup_steps = total_steps // training_config.get("warmup_steps_perc", 10)
        learning_rate_scheduler = PolynomialDecay(optim, total_steps, warmup_steps=warmup_steps)
        logger.info("Learning rate warmup steps: {}".format(warmup_steps))
    if args.no_checkpoint:
        serialization_dir = None
    else:
        serialization_dir = os.path.join(outpath, "checkpoints")
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optim,
                                     data_loader=training_iterator,
                                     cuda_device=device_int,
                                     grad_clipping=gradient_clipping,
                                     num_epochs=num_epochs,
                                     validation_data_loader=dev_iterator,
                                     num_gradient_accumulation_steps=gradient_accumulation,
                                     validation_metric=training_config.get("validation_metric", "-loss"),
                                     epoch_callbacks=callbacks,
                                     patience=patience,
                                     serialization_dir=serialization_dir,
                                     checkpointer=Checkpointer(serialization_dir,
                                                               num_serialized_models_to_keep=1),
                                     learning_rate_scheduler=learning_rate_scheduler
                                     )

    trainer.train()
    with open(os.path.join(outpath, "checkpoints", "last.th"), "wb") as writer:
        torch.save(model.state_dict(), writer)
    if not os.path.exists(os.path.join(outpath, "evaluation")):
        os.mkdir(os.path.join(outpath, "evaluation"))
    evaluate_datasets(model,
                      test_dss,
                      test_lang2name,
                      os.path.join(outpath, "checkpoints", "best.th"),
                      label_vocab,
                      device_int,
                      mfs_dictionary,
                      os.path.join(outpath, "evaluation"),
                      verbose=True,
                      debug=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dryrun", "--no-wandb-log", action="store_true")
    parser.add_argument("--no_checkpoint", default=False, type=bool)
    parser.add_argument("--reload_checkpoint", action="store_true", default=False)
    parser.add_argument("--weight_decay")
    parser.add_argument("--learning_rate")
    parser.add_argument("--gradient_clipping")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    if args.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"
    print("config {}".format(args.config))
    main(args)
