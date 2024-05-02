import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from allennlp.common.util import sanitize_wordpiece
from allennlp.modules import TimeDistributed
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn.util import get_token_ids_from_text_field_tensors, get_range_vector, get_device_of
from transformers.models.t5.modeling_t5 import T5Model, T5EncoderModel
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BooleanAccuracy, F1Measure, Average
from torch.nn.functional import cross_entropy

from allennlp_lib.models.utils import (
    get_best_span,
    get_best_k_spans,
    replace_masked_values_with_big_negative_number,
)
from allennlp_lib.training.metrics import (
    SquadEmAndF1, ListCompareEmAndF1, FullRecallRank,
    AnswerWithSufficiency, SupportWithSufficiency
)
from allennlp_lib.nn.utils import pluck_tokens

logger = logging.getLogger(__name__)


@Model.register("transformer_rc")
class TransformerRC(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model: PretrainedTransformerEmbedder,
        supervize_answerability: bool = False,
        supervize_answer: bool = True,
        supervize_support: bool = False,
        has_single_support: bool = False,
        skip_grouped_metrics: bool = False,
        max_answer_length: int = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        if isinstance(transformer_model.transformer_model, T5Model):
        # Take only the encoder
            logger.info("WARNING: Pre-Finetuning of T5 isn't supported.")
            transformer_model_name = transformer_model.transformer_model.config.to_dict()['_name_or_path']
            transformer_model.transformer_model = T5EncoderModel.from_pretrained(transformer_model_name)
            # For future: (following doesn't work yet because of embedding size differences.)
            # encoder.load_state_dict(transformer_model.transformer_model.state_dict(), strict=False)
            # transformer_model.transformer_model = transformer_model.transformer_model.encoder

        self._transformer_model = transformer_model
        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": self._transformer_model}
        )
        # NOTE: Transformer embedding extension is taken care of in init of PretrainedTransformerEmbedder

        self._span_start_accuracy = BooleanAccuracy()
        self._span_end_accuracy = BooleanAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._answer_text_metrics = SquadEmAndF1()
        self._answerability_metrics = F1Measure(positive_label=1) # 1: answerable; 0: unanswerable.
        self._rank_support_metrics = ListCompareEmAndF1()
        self._select_support_metrics = ListCompareEmAndF1()
        self._full_recall_rank_metrics = FullRecallRank()
        self._answer_text_suff_metrics = AnswerWithSufficiency()
        self._rank_support_suff_metrics = SupportWithSufficiency()

        self._supervize_answerability = supervize_answerability
        self._supervize_answer = supervize_answer
        self._supervize_support = supervize_support

        hidden_size = self._text_field_embedder.get_output_dim()
        if self._supervize_answerability:
            self._predict_answerability = nn.Linear(hidden_size, 2)
            self._answerability_loss = torch.nn.CrossEntropyLoss()

        if self._supervize_support:
            self._predict_support_logits = TimeDistributed(torch.nn.Linear(hidden_size, 1))

            self._has_single_support = has_single_support
            if self._has_single_support:
                self._support_loss = torch.nn.CrossEntropyLoss()
            else:
                self._support_loss = torch.nn.BCEWithLogitsLoss()

        if self._supervize_answer:
            self._predict_answer = nn.Linear(hidden_size, 2)

        self._avg_answerability_loss = Average()
        self._avg_answer_span_loss = Average()
        self._avg_support_loss = Average()

        self._max_answer_length = max_answer_length
        self._skip_grouped_metrics = skip_grouped_metrics


    def forward(  # type: ignore
        self,
        document: Dict[str, Dict[str, torch.LongTensor]],
        context_span: torch.IntTensor,
        cls_index: torch.LongTensor,
        global_attention_mask: torch.LongTensor = None,
        answerability_labels: torch.Tensor = None,
        answer_span_labels: torch.IntTensor = None,
        support_positions: torch.Tensor = None,
        support_labels: torch.Tensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        extra_parameters = {}
        if global_attention_mask is not None:
            extra_parameters["global_attention_mask"] = global_attention_mask

        embedded_document = self._text_field_embedder(document, **extra_parameters)
        range_vector = get_range_vector(embedded_document.shape[0], get_device_of(embedded_document))
        output_dict = {"loss": torch.tensor(0.0).to(embedded_document.device)}


        if self._supervize_answerability:
            #### ANSWERABILITY LOGITS

            embedded_cls = embedded_document[range_vector, cls_index, :]
            answerability_logits = self._predict_answerability(embedded_cls)
            answerability_probs = answerability_logits.softmax(dim=1)[:, 1] # 0:unans, 1:ans
            output_dict["predicted_answerability_prob"] = answerability_probs

            #### ANSWERABILITY LOSS

            if answerability_labels is not None:
                answerability_loss = self._answerability_loss(answerability_logits, answerability_labels)
                output_dict["loss"] += answerability_loss
                self._avg_answerability_loss(answerability_loss)

                output_dict["answerability_labels"] = answerability_labels.detach().cpu().numpy()


        if self._supervize_answer:
            #### ANSWER LOGITS

            answer_logits = self._predict_answer(embedded_document)
            span_start_logits, span_end_logits = answer_logits.split(1, dim=-1)
            span_start_logits = span_start_logits.squeeze(-1)
            span_end_logits = span_end_logits.squeeze(-1)

            possible_answer_mask = torch.zeros_like(
                get_token_ids_from_text_field_tensors(document), dtype=torch.bool
            )

            for i, (start, end) in enumerate(context_span):
                # Extra check if answerability_labels is redundant here, since I'm making
                # sure that for unanswerables, answer_labels are -1, -1 and so the loss is ignored.
                # Yet, let me keep this extra check.
                possible_answer_mask[i, start : end + 1] = (
                    answerability_labels[i] if answerability_labels is not None else True
                )

            span_start_logits = replace_masked_values_with_big_negative_number(
                span_start_logits, possible_answer_mask
            )
            span_end_logits = replace_masked_values_with_big_negative_number(
                span_end_logits, possible_answer_mask
            )
            best_spans = get_best_span(
                span_start_logits, span_end_logits, max_length=self._max_answer_length
            )

            span_start_probs = span_start_logits.softmax(dim=-1)
            span_end_probs = span_end_logits.softmax(dim=-1)

            best_span_probs = (
                span_start_probs[range_vector, best_spans[:, 0]] +
                span_end_probs[range_vector, best_spans[:, 1]]
            )/2.0
            output_dict["predicted_best_answer_prob"] = best_span_probs

            #### ANSWER LOSS

            if answer_span_labels is not None:
                answer_span_loss = self._answer_span_loss(
                    span_start_logits, span_end_logits, answer_span_labels
                )
                output_dict["loss"] += answer_span_loss
                self._avg_answer_span_loss(answer_span_loss)

                # TODO: Move these in metrics area?
                self._span_accuracy(best_spans, answer_span_labels)
                self._span_start_accuracy(best_spans[:, 0], answer_span_labels[:, 0])
                self._span_end_accuracy(best_spans[:, 1], answer_span_labels[:, 1])

                output_dict["answer_span_labels"] = answer_span_labels.detach().cpu().numpy()


        if self._supervize_support:
            #### SUPPORT LOGITS

            embedded_paragraphs_vectors, paragraph_tokens_mask = pluck_tokens(embedded_document,
                                                                              support_positions)
            support_logits = self._predict_support_logits(embedded_paragraphs_vectors).squeeze(dim=-1)

            support_logits = replace_masked_values_with_big_negative_number(
                support_logits,
                paragraph_tokens_mask
            )

            support_probs = (support_logits.softmax(dim=-1)
                             if self._has_single_support else support_logits.sigmoid())
            output_dict["predicted_support_probs"] = support_probs

            #### SUPPORT LOSS

            if support_labels is not None and support_labels.numel() > 0: # numel to ignore 0 contexts cases.
                support_labels[~paragraph_tokens_mask] = 0
                mismatching_num_supports = False
                if self._has_single_support:
                    batch_size = support_labels.shape[0]
                    support_labels = support_labels.nonzero(as_tuple=True)[1]
                    mismatching_num_supports = support_labels.shape[0] != batch_size

                if mismatching_num_supports and self.training:
                    raise Exception("Expected single-support but found multiple.")

                if not mismatching_num_supports:
                    support_loss = self._support_loss(support_logits, support_labels)
                    output_dict["loss"] += support_loss
                    self._avg_support_loss(support_loss)

                output_dict["support_labels"] = support_labels.detach().cpu().numpy()


        #### METRICS: ANSWERABILITY, ANSWER STRING and SUPPORT METRICS

        if not self.training:

            if self._supervize_answer:
                best_spans = best_spans.detach().cpu().numpy()
                output_dict["predicted_best_answer_span"] = best_spans
                output_dict["predicted_best_answer"] = []

                best_k_spans = get_best_k_spans(
                    span_start_logits, span_end_logits,
                    num_spans=1, max_length=self._max_answer_length # settin 1 to save time.
                )
                best_k_spans = best_k_spans.detach().cpu()

                best_k_span_probs = []
                for local_best_spans in torch.unbind(best_k_spans, dim=1):

                    local_best_span_probs = (
                        span_start_probs[range_vector, local_best_spans[:, 0]] +
                        span_end_probs[range_vector, local_best_spans[:, 1]]
                    )/2.0
                    best_k_span_probs.append(local_best_span_probs)

                best_k_span_probs = torch.stack(best_k_span_probs, dim=1)
                output_dict["predicted_best_k_answers_probs"] = best_k_span_probs
                output_dict["predicted_best_k_answers"] = []

            output_dict["answer_labels"] = []
            output_dict["aligned_answer_labels"] = []
            output_dict["predicted_answerability"] = []
            output_dict["predicted_select_support_indices"] = []
            output_dict["predicted_rank_support_indices"] = []
            output_dict["predicted_ordered_support_indices"] = []
            output_dict["contexts"] = []
            output_dict["skipped_support_contexts"] = []

            for index, metadatum in enumerate(metadata):
                question_id = metadatum["id"]

                answers = metadatum.get("answers")
                output_dict["answer_labels"].append(answers)

                contexts = metadatum.get("contexts", None)
                output_dict["contexts"].append(contexts)

                skipped_support_contexts = metadatum.get("skipped_support_contexts", [])
                output_dict["skipped_support_contexts"].append(skipped_support_contexts)

                if self._supervize_answerability:
                    threshold = 0.5
                    predicted_answerability = (answerability_probs[index] > threshold).int()
                    answerability_scores = torch.zeros((1, 2), dtype=answerability_probs.dtype,
                                                       device=answerability_probs.device)
                    answerability_scores[0, predicted_answerability] = 1
                    gold_answerability = answerability_labels[index].unsqueeze(0)
                    self._answerability_metrics(answerability_scores, gold_answerability)
                    output_dict["predicted_answerability"].append(predicted_answerability.squeeze())

                if self._supervize_answer:
                    best_span_string = self._answer_span_to_string(
                        best_spans[index], context_span[index], metadatum
                    )
                    output_dict["predicted_best_answer"].append(best_span_string)

                    output_dict["predicted_best_k_answers"].append([
                        self._answer_span_to_string(spans[index], context_span[index], metadatum)
                        for spans in torch.unbind(best_k_spans, dim=1)
                    ])

                    aligned_answer_label_text = self._answer_span_to_string(
                        answer_span_labels[index], context_span[index], metadatum
                    )
                    output_dict["aligned_answer_labels"].append([aligned_answer_label_text])

                if self._supervize_answer and answers:
                    self._answer_text_metrics(best_span_string, answers)

                if self._supervize_answerability and self._supervize_answer:
                    self._answer_text_suff_metrics(
                        best_span_string, answers, predicted_answerability.item(),
                        gold_answerability.item(), question_id
                    )

                if self._supervize_support:
                    # We need to populate the output_dict for all the items in the batch, or else it'll be skipped.
                    threshold = 0.5
                    support_select_predicted_indices = (support_probs[index] > threshold).nonzero(as_tuple=True)[0]
                    output_dict["predicted_select_support_indices"].append(support_select_predicted_indices)

                    support_ordered_predicted_indices = support_probs[index].sort(descending=True)[1]
                    output_dict["predicted_ordered_support_indices"].append(support_ordered_predicted_indices)

                    if support_labels is not None:

                        if self._has_single_support:
                            support_label_indices = support_labels[index].unsqueeze(0)
                        else:
                            support_label_indices = support_labels[index].nonzero(as_tuple=True)[0]

                        support_rank_predicted_indices = support_probs[index].topk(k=len(support_label_indices))[1]
                        output_dict["predicted_rank_support_indices"].append(support_rank_predicted_indices)

                if self._supervize_support and bool(answers) and support_labels is not None:
                    self._select_support_metrics(support_select_predicted_indices, support_label_indices)
                    self._rank_support_metrics(support_rank_predicted_indices, support_label_indices)
                    self._full_recall_rank_metrics(support_ordered_predicted_indices, support_label_indices)

                if self._supervize_answerability and self._supervize_support:
                    self._rank_support_suff_metrics(
                        support_rank_predicted_indices, support_label_indices,
                        predicted_answerability.item(), gold_answerability.item(), question_id
                    )

        return output_dict

    def _answer_span_loss(
        self,
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        answer_span_labels: torch.Tensor,
    ) -> torch.Tensor:

        span_start = answer_span_labels[:, 0]
        span_end = answer_span_labels[:, 1]

        start_loss = cross_entropy(span_start_logits, span_start, ignore_index=-1)
        big_constant = min(torch.finfo(start_loss.dtype).max, 1e9)
        assert not torch.any(start_loss > big_constant), "Start loss too high"

        end_loss = cross_entropy(span_end_logits, span_end, ignore_index=-1)
        assert not torch.any(end_loss > big_constant), "End loss too high"

        return (start_loss + end_loss) / 2

    def _answer_span_to_string(
            self,
            answer_span: torch.Tensor, # TODO: Fix, it can be both numpy.arra or torch.Tensor
            context_span: torch.IntTensor,
            metadatum: Dict[str, Any],
        ) -> str:

        answer_span = copy.deepcopy(answer_span)
        answer_span -= int(context_span[0])
        if answer_span[0] < 0 or answer_span[1] < 0:
            # logger.info(f"WARNING: answer_span {answer_span} not within context_span ({context_span}). "
            #             f"Returning empty answer string; Instance-id {metadatum.get('id', 'N/A')}")
            return ""

        predicted_start, predicted_end = tuple(answer_span)

        while (
            predicted_start >= 0
            and metadatum["context_tokens"][predicted_start].idx is None
        ):
            predicted_start -= 1
        if predicted_start < 0:
            logger.warning(
                f"Could not map the token '{metadatum['context_tokens'][answer_span[0]].text}' at index "
                f"'{answer_span[0]}' to an offset in the original text."
            )
            character_start = 0
        else:
            character_start = metadatum["context_tokens"][predicted_start].idx

        while (
            predicted_end < len(metadatum["context_tokens"])
            and metadatum["context_tokens"][predicted_end].idx is None
        ):
            predicted_end += 1
        if predicted_end >= len(metadatum["context_tokens"]):
            logger.warning(
                f"Could not map the token '{metadatum['context_tokens'][answer_span[1]].text}' at index "
                f"'{answer_span[1]}' to an offset in the original text."
            )
            character_end = len(metadatum["context"])
        else:
            end_token = metadatum["context_tokens"][predicted_end]
            character_end = end_token.idx + len(sanitize_wordpiece(end_token.text))

        answer_span_string = metadatum["context"][character_start:character_end]
        return answer_span_string


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
        }
        if not self.training:
            if self._supervize_answerability:
                metrics_dict = self._answerability_metrics.get_metric(reset)
                output["answerability_precision"] = metrics_dict["precision"]
                output["answerability_recall"] = metrics_dict["recall"]
                output["answerability_f1"] = metrics_dict["f1"]
                output["_avg_answerability_loss"] = self._avg_answerability_loss.get_metric(reset)

            if self._supervize_answer:
                exact_match, f1_score = self._answer_text_metrics.get_metric(reset)
                output["answer_text_em"] = exact_match
                output["answer_text_f1"] = f1_score
                output["_avg_answer_span_loss"] = self._avg_answer_span_loss.get_metric(reset)

            if self._supervize_answerability and self._supervize_answer:
                output["avg_answer_answerability_f1"] = (
                    output["answerability_f1"]+output["answer_text_f1"]
                )/2

            if (
                self._supervize_answerability and self._supervize_answer
                and reset and not self._skip_grouped_metrics
            ):
                answer_sufficiency_scores = self._answer_text_suff_metrics.get_metric(reset)
                output["answer_sufficiency_f1"] = answer_sufficiency_scores["f1"]
                output["answer_sufficiency_em"] = answer_sufficiency_scores["em"]
                output["sufficiency_em"] = answer_sufficiency_scores["suff"]

            if self._supervize_support:
                exact_match, f1_score = self._rank_support_metrics.get_metric(reset)
                output["support_rank_em"] = exact_match
                output["support_rank_f1"] = f1_score

                exact_match, f1_score = self._select_support_metrics.get_metric(reset)
                output["support_select_em"] = exact_match
                output["support_select_f1"] = f1_score

                full_recall_rank = self._full_recall_rank_metrics.get_metric(reset)
                output["full_recall_rank"] = full_recall_rank

                output["_avg_support_loss"] = self._avg_support_loss.get_metric(reset)

            if self._supervize_answerability and self._supervize_support:
                output["avg_select_support_answerability_f1"] = (
                    output["answerability_f1"]+output["support_select_f1"]
                )/2

            if (
                self._supervize_answerability and self._supervize_support
                and reset and not self._skip_grouped_metrics
            ):
                rank_support_sufficiency_scores = self._rank_support_suff_metrics.get_metric(reset)
                output["rank_support_sufficiency_f1"] = rank_support_sufficiency_scores["f1"]
                output["rank_support_sufficiency_em"] = rank_support_sufficiency_scores["em"]

        return output

    default_predictor = "transformer_rc"
