import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import Metric
from nlp_tools.allennlp_training_callbacks.callbacks import OutputWriter
from nlp_tools.data_io.datasets import LabelVocabulary
from nlp_tools.nlp_models.multilayer_pretrained_transformer_mismatched_embedder import \
    MultilayerPretrainedTransformerMismatchedEmbedder
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.batchnorm import BatchNorm1d
from transformers.activations import swish

from utils.utils import get_info_logger


class WSDOutputWriter(OutputWriter):
    def __init__(self, output_file, labeldict):
        super().__init__(output_file, labeldict)

    def write(self, outs):
        predictions = outs["predictions"].flatten().tolist()
        golds = [x for y in outs["str_labels"] for x in y if x != ""]
        ids = [x for y in outs["ids"] for x in y]
        assert len(predictions) == len(golds)
        for i in range(len(predictions)):
            p, l = predictions[i], "\t".join(golds[i])
            id = ids[i]
            out_str = (id if ids is not None else "") + "\t" + self.labeldict[p] + "\t" + l + "\n"
            self.writer.write(out_str)
        self.writer.flush()


class WSDF1(Metric):
    def __init__(self, label_vocab: LabelVocabulary, use_mfs: bool = False, mfs_vocab: Dict[str, int] = None):
        assert not use_mfs or mfs_vocab is not None
        self.correct = 0.0
        self.correct_mfs = 0.0
        self.tot = 0.0
        self.tot_mfs = 0.0
        self.answers = 0.0
        self.answers_mfs = 0.0
        self.mfs_vocab = mfs_vocab
        self.use_mfs = use_mfs
        self.unk_id = label_vocab.get_idx("<unk>") if "<unk>" in label_vocab.stoi else label_vocab.get_idx("<pad>")
        assert self.unk_id is not None
        self.label_vocab = label_vocab

    def compute_metrics_no_mfs(self, predictions, labels):
        for p, l in zip(predictions, labels):
            self.tot += 1
            if p == self.unk_id:
                continue
            self.answers += 1
            if self.label_vocab.get_string(p) in l:
                self.correct += 1

    def compute_metrics_mfs(self, lemmapos, predictions, labels):
        for lp, p, l in zip(lemmapos, predictions, labels):
            self.tot_mfs += 1

            if p == self.unk_id:
                p = self.mfs_vocab.get(lp, p)
            else:
                p = self.label_vocab.get_string(p)
            if p != self.unk_id:
                self.answers_mfs += 1
            if p in l:
                self.correct_mfs += 1

    def __call__(self, lemmapos, predictions, gold_labels, mask=None, ids=None):
        """
        :param predictions:
        :param gold_labels: assumes this is a List[List[Set[str]]] containing for each batch a list of Set each
        representing the possible gold labels for each token. This is parallel to predictions
        :param mask:
        :return:
        """
        assert len(predictions) == len(gold_labels)
        self.compute_metrics_no_mfs(predictions, gold_labels)
        if self.mfs_vocab is not None:
            self.compute_metrics_mfs(lemmapos, predictions, gold_labels)

    def get_metric(self, reset: bool):
        if self.answers == 0:
            return {}
        precision = self.correct / self.answers
        recall = self.correct / self.tot
        f1 = 2 * (precision * recall) / ((precision + recall) if (precision + recall) > 0 else 1)
        ret_dict = {"precision": precision, "recall": recall, "f1": f1, "correct": self.correct,
                    "answers": self.answers,
                    "total": self.tot}
        if self.mfs_vocab is not None:
            precision_mfs = self.correct_mfs / self.answers_mfs
            recall_mfs = self.correct_mfs / self.tot_mfs
            f1_mfs = 2 * (precision_mfs * recall_mfs) / (
                (precision_mfs + recall_mfs) if (precision_mfs + recall_mfs) > 0 else 1)
            ret_dict.update({"p_mfs": precision_mfs,
                             "recall_mfs": recall_mfs,
                             "f1_mfs": f1_mfs, "correct_mfs": self.correct_mfs,
                             "answers_mfs": self.answers_mfs, "tot_mfs": self.tot_mfs})

        if reset:
            self.tot = 0.0
            self.tot_mfs = 0.0
            self.answers = 0.0
            self.correct = 0.0
            self.correct_mfs = 0.0
            self.answers_mfs = 0.0
        return ret_dict


@Model.register("wsd_classifier")
class AllenWSDModel(Model, ABC):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 out_sz,
                 vocab=None,
                 return_full_output=False,
                 finetune_embedder=False,
                 cache_instances=False,
                 pad_id=0,
                 label_pad_id=0,
                 metric: Metric = None,
                 embedding_size: int = None, **kwargs):
        vocab = Vocabulary() if vocab is None else vocab
        super().__init__(vocab)
        self.out_size = out_sz
        self.embedding_size = embedding_size
        self.label_pad_id = label_pad_id
        self.finetune_embedder = finetune_embedder
        self.word_embeddings = word_embeddings
        if not finetune_embedder:
            self.word_embeddings.eval()
            for p in self.word_embeddings.parameters():
                p.requires_grad = False
        else:
            self.word_embeddings.train()
        self.loss = nn.CrossEntropyLoss()
        self.pad_id = pad_id
        self.return_full_output = return_full_output
        self.cache_instances = cache_instances
        self.cache = dict()
        self.cache_file = kwargs.get("cache_file", None)
        self.accuracy = metric
        if self.cache_file is not None:
            if os.path.exists(self.cache_file):
                get_info_logger(__name__).info("cache found, loading instances' hidden states.")
                self.cache = self._load_cache(kwargs["cache_file"])
        self.save_cache = kwargs.get("save_cache", False)

    def _load_cache(self, path):
        files = np.load(path)
        ids = files["ids"]
        vectors = files["vectors"]
        self.cache = dict(zip(ids, vectors))

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            if module == self.word_embeddings and not self.finetune_embedder:
                continue
            module.train(mode)
        return self

    def named_parameters(self, prefix: str = ..., recurse: bool = ...) -> Iterator[Tuple[str, Parameter]]:
        params = list()
        if self.finetune_embedder:
            params.extend(self.word_embeddings.named_parameters())
        params.extend(self.classifier.named_parameters())
        yield from params

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.accuracy.get_metric(reset)

    def get_embeddings_from_cache(self, tokens, mask, instance_ids):
        to_compute = [any(x not in self.cache for x in batch_instance_id) for batch_instance_id in instance_ids]
        indices_to_compute = [i for i in range(len(to_compute)) if to_compute[i]]

        if len(indices_to_compute) == 0:
            all_embeddings = list()
            for k, batch_instance_id in enumerate(instance_ids):
                embeddings = [self.cache[instance_id] for instance_id in batch_instance_id]
                all_embeddings.append(torch.stack(embeddings, 0).to(self.classifier.weight.device))
            return torch.cat(all_embeddings, 0)
        mask_to_compute = mask[indices_to_compute]
        embeddings = self.word_embeddings(tokens)

        retrieved_embedding_mask = mask_to_compute != 0
        embeddings = embeddings[retrieved_embedding_mask]
        self.cache.update(dict(zip([x for y in instance_ids for x in y], embeddings.detach().cpu())))
        return embeddings

    def get_embeddings(self, tokens, mask=None, instance_ids=None):

        retrieved_embedding_mask = None
        if mask is not None:
            retrieved_embedding_mask = mask != 0
        if not self.training:
            embeddings = self.word_embeddings(tokens)
        elif not self.finetune_embedder:
            if not self.cache_instances:
                with torch.no_grad():
                    embeddings = self.word_embeddings(tokens)
            else:
                return self.get_embeddings_from_cache(tokens, mask, instance_ids), retrieved_embedding_mask
        else:
            embeddings = self.word_embeddings(tokens)
        if retrieved_embedding_mask is not None:
            masked_embeddings = embeddings[retrieved_embedding_mask]
            embeddings = masked_embeddings
        else:
            embeddings = embeddings.view(-1, embeddings.shape[-1])

        return embeddings, retrieved_embedding_mask

    @abstractmethod
    def wsd_head(self, embeddings):
        pass

    def predict(self, tokens: Dict[str, torch.Tensor], tokens_to_disambiguate: torch.Tensor = None):
        """
        :param tokens:
        :param tokens_to_disambiguate: tensor having 0 for those positions which should not be disambiguated and 1
        in those positions that we want to disambiguate.
        :return: a list containing the logits for those thones that had to be disambiguated.
        """
        embeddings, _ = self.get_embeddings(tokens, tokens_to_disambiguate, None)
        logits = self.wsd_head(embeddings).tolist()
        if tokens_to_disambiguate is not None:
            output = list()
            for mask in tokens_to_disambiguate:
                output.append([logits.pop() for _ in range(sum(mask))])
            return output
        return logits

    def forward(self, tokens: Dict[str, torch.Tensor],
                possible_labels=None,
                ids: Any = None,
                label_ids: torch.Tensor = None,
                labeled_lemmapos=None,
                labels=None,
                cache_instance_ids=None,
                compute_accuracy=True,
                compute_loss=True,
                **kwargs) -> torch.Tensor:
        if label_ids is not None:
            mask = (label_ids != self.label_pad_id).float().to(tokens["tokens"]["token_ids"].device)
        else:
            mask = None

        embeddings, retrieved_embedding_mask = self.get_embeddings(tokens, mask, cache_instance_ids)
        labeled_logits = self.wsd_head(embeddings)

        predictions = None

        if possible_labels is not None:
            possible_labels = [x for y in possible_labels for x in y]
            possible_classes_mask = torch.zeros_like(labeled_logits)
            for i, ith_lp in enumerate(possible_labels):
                possible_classes_mask[i][possible_labels[i]] = 1
            possible_classes_mask[:, 0] = 0
            masked_labeled_logits = labeled_logits * possible_classes_mask
        else:
            masked_labeled_logits = labeled_logits
        if not self.training and compute_accuracy:
            assert labeled_lemmapos is not None and labels is not None
            flatten_labels = [x for y in labels for x in y if x != ""]

            predictions = self.get_predictions(masked_labeled_logits)
            self.accuracy([x for y in labeled_lemmapos for x in y], predictions.tolist(), flatten_labels)
        loss = None
        if compute_loss:
            target_labels = label_ids[retrieved_embedding_mask]
            loss = self.loss(labeled_logits, target_labels)
        output = {"class_logits": masked_labeled_logits,
                  "all_logits": labeled_logits,
                  "predictions": predictions,
                  "labels": labels,
                  "all_labels": label_ids,
                  "str_labels": labels,
                  "ids": [[x for x in i if x is not None] for i in ids] if ids is not None else None,
                  "loss": loss}
        if self.return_full_output:
            full_labeled_logits, full_predictions = self.reconstruct_full_output(retrieved_embedding_mask,
                                                                                 labeled_logits,
                                                                                 predictions)
            output.update({"full_labeled_logits": full_labeled_logits, "full_predictions": full_predictions})

        return output

    def reconstruct_full_output(self, retrieved_embedding_mask, labeled_logits, predictions):
        full_logits = torch.zeros(retrieved_embedding_mask.size(0), retrieved_embedding_mask.size(1),
                                  labeled_logits.size(-1)).to(predictions.device)
        full_predictions = torch.zeros(retrieved_embedding_mask.size(0), retrieved_embedding_mask.size(1)).to(
            predictions.device)
        index = 0
        for i, b_mask in enumerate(retrieved_embedding_mask):
            for j, elem in enumerate(b_mask):
                if elem.item():
                    full_logits[i][j] = labeled_logits[index]
                    full_predictions[i][j] = predictions[index]
                    index += 1

        return full_logits, full_predictions

    def get_predictions(self, labeled_logits):
        predictions = list()
        for ll in labeled_logits:
            mask = (ll != 0).float()
            ll = torch.exp(ll) * mask
            predictions.append(torch.argmax(ll, -1))
        return torch.stack(predictions)

    @classmethod
    def get_transformer_based_wsd_model(cls, encoder_name,
                                        out_size,
                                        device,
                                        pad_id,
                                        label_pad_id,
                                        layers_to_use=(-4, -3, -2, -1),
                                        vocab=None,
                                        return_full_output=False,
                                        finetune_embedder=False,
                                        cache_instances=False,
                                        model_path=None,
                                        metric=None,
                                        **kwargs):
        bpe_combiner = kwargs.get("bpe_combiner", "mean")
        vocab = Vocabulary() if vocab is None else vocab
        text_embedder = MultilayerPretrainedTransformerMismatchedEmbedder(encoder_name, layers_to_use,
                                                                          word_segment_emb_merger=bpe_combiner)
        embedding_size = text_embedder.get_output_dim()

        word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": text_embedder})
        model = cls(word_embeddings=word_embeddings,
                    out_sz=out_size, vocab=vocab,
                    return_full_output=return_full_output,
                    cache_instances=cache_instances,
                    pad_id=pad_id,
                    label_pad_id=label_pad_id,
                    finetune_embedder=finetune_embedder, metric=metric, embedding_size=embedding_size, **kwargs)
        model.to(device)
        return model


@Model.register("batchnorm_wsd_classifier")
class AllenBatchNormWsdModel(AllenWSDModel):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 out_sz, embedding_size, **kwargs):
        super().__init__(word_embeddings, out_sz, **kwargs)
        self.classifier = nn.Linear(embedding_size, out_sz, bias=False)
        self.batchnorm = BatchNorm1d(embedding_size)
        self.linear = nn.Linear(embedding_size, embedding_size)

    def named_parameters(self, prefix: str = ..., recurse: bool = ...) -> Iterator[Tuple[str, Parameter]]:
        params = list()
        if self.finetune_embedder:
            params.extend(self.word_embeddings.named_parameters())
            params.extend(self.linear.named_parameters())
            params.extend(self.batchnorm.named_parameters())
        params.extend(self.classifier.named_parameters())
        yield from params

    def wsd_head(self, embeddings):
        if len(embeddings) > 1:
            embeddings = self.batchnorm(embeddings)

        embeddings = swish(self.linear(embeddings))
        return self.classifier(embeddings)
