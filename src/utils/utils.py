from allennlp.data import Vocabulary

from src.misc.wsdlogging import get_info_logger
from src.modelling import POSSIBLE_MODELS
from src.modelling.neural_wsd_models import AllenBatchNormWsdModel, AllenFFWsdModel

logger = get_info_logger(__name__)


def get_model(model_config, out_size, pad_token_id, label_pad_token_id, metric=None, device="cuda"):
    wsd_model_name = model_config["wsd_model_name"]
    assert wsd_model_name in POSSIBLE_MODELS or logger.error(
        "WSD model not recognised: {}. Choose among {}".format(wsd_model_name, ",".join(POSSIBLE_MODELS)))
    if wsd_model_name == "ff_wsd_classifier":
        model_type = AllenFFWsdModel
    if wsd_model_name == "batchnorm_wsd_classifier":
        model_type = AllenBatchNormWsdModel
    if device is not None and "device" in model_config:
        model_config.pop("device")
    model = model_type.get_transformer_based_wsd_model(**model_config,
                                                       out_size=out_size,
                                                       pad_id=pad_token_id,
                                                       label_pad_id=label_pad_token_id,
                                                       vocab=Vocabulary(),
                                                       # model_path=model_config.get("model_path", None),
                                                       metric=metric,
                                                       device=device)
    return model


def get_token_indexer(model_name):
    if model_name.lower() == "nhs":
        model_name = "bert-base-multilingual-cased"
    indexer = PretrainedTransformerIndexer(
        model_name=model_name,
    )
    return indexer, indexer._tokenizer.pad_token_id


import yaml

if __name__ == "__main__":
    config = "/home/tommaso/dev/PycharmProjects/WSDframework/config/config_mulan_bnoffsets.yaml"
    with open(config) as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    seed = config["random_seed"]
    data_config = config["data"]
    model_config = config["model"]
    model = get_model(model_config, 117660,
                      0,
                      0)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    print(pytorch_trainable_params)
