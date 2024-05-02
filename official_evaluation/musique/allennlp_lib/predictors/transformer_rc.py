from typing import List
import logging

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from allennlp_lib.data.dataset_readers.utils import answer_offset_in_context
from utils.constants import PARAGRAPH_MARKER


logger = logging.getLogger(__name__)


@Predictor.register('transformer_rc')
class TransformerRCPredictor(Predictor):
    """
    This is the default transformer_rc predictor.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:

        instance = self._dataset_reader.json_to_instance(json_dict)
        self._dataset_reader.apply_token_indexers(instance)
        return instance

    @overrides
    def predict_json(self, input: JsonDict) -> JsonDict:
        instance = self._json_to_instance(input)
        output = self.predict_instance(instance)
        output["input"] = input
        return output

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self.predict_batch_instance(instances)
        for input, output in zip(inputs, outputs):
            output["input"] = input
        return outputs

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        output = self._model.forward_on_instance(instance)
        prediction = {

            "predicted_answerability": output.get("predicted_answerability", None),
            "predicted_answerability_prob": output.get("predicted_answerability_prob", None),

            "predicted_best_answer": output.get("predicted_best_answer", None),
            "predicted_best_answer_span": output.get("predicted_best_answer_span", None),
            "predicted_best_answer_prob": output.get("predicted_best_answer_prob", None),

            "predicted_best_k_answers": output.get("predicted_best_k_answers", None),
            "predicted_best_k_answers_probs": output.get("predicted_best_k_answers_probs", None),

            "predicted_rank_support_indices": output.get("predicted_rank_support_indices", None),
            "predicted_select_support_indices": output.get("predicted_select_support_indices", None),
            "predicted_ordered_support_indices": output.get("predicted_ordered_support_indices", None),
            "predicted_support_probs": output.get("predicted_support_probs", None),

            "answerability_labels": output.get("answerability_labels", None),
            "answer_labels": output.get("answer_labels", None),
            "aligned_answer_labels": output.get("aligned_answer_labels", None),
            "answer_span_labels": output.get("answer_span_labels", None),
            "support_labels": output.get("support_labels", None),

            "contexts": output.get("contexts", None),
            "skipped_support_contexts": output.get("skipped_support_contexts", [])
        }
        return sanitize(prediction)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        predictions = [{

            "predicted_answerability": output.get("predicted_answerability", None),
            "predicted_answerability_prob": output.get("predicted_answerability_prob", None),

            "predicted_best_answer": output.get("predicted_best_answer", None),
            "predicted_best_answer_span": output.get("predicted_best_answer_span", None),
            "predicted_best_answer_prob": output.get("predicted_best_answer_prob", None),

            "predicted_best_k_answers": output.get("predicted_best_k_answers", None),
            "predicted_best_k_answers_probs": output.get("predicted_best_k_answers_probs", None),

            "predicted_rank_support_indices": output.get("predicted_rank_support_indices", None),
            "predicted_select_support_indices": output.get("predicted_select_support_indices", None),
            "predicted_ordered_support_indices": output.get("predicted_ordered_support_indices", None),
            "predicted_support_probs": output.get("predicted_support_probs", None),

            "answerability_labels": output.get("answerability_labels", None),
            "answer_labels": output.get("answer_labels", None),
            "aligned_answer_labels": output.get("aligned_answer_labels", None),
            "answer_span_labels": output.get("answer_span_labels", None),
            "support_labels": output.get("support_labels", None),

            "contexts": output.get("contexts", None),
            "skipped_support_contexts": output.get("skipped_support_contexts", [])
        } for output in outputs]
        return sanitize(predictions)
