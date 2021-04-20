from allennlp.data import Vocabulary

from src.modelling import POSSIBLE_MODELS
from src.modelling.neural_wsd_models import AllenBatchNormWsdModel




def get_model(model_config, out_size, pad_token_id, label_pad_token_id, metric=None, device="cuda"):
    if device is not None and "device" in model_config:
        model_config.pop("device")
    model = AllenBatchNormWsdModel.get_transformer_based_wsd_model(**model_config,
                                                       out_size=out_size,
                                                       pad_id=pad_token_id,
                                                       label_pad_id=label_pad_token_id,
                                                       vocab=Vocabulary(),
                                                       # model_path=model_config.get("model_path", None),
                                                       metric=metric,
                                                       device=device)
    return model

