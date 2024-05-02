from typing import List
import logging
from copy import deepcopy

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from allennlp_lib.data.dataset_readers.utils import answer_offset_in_context
from utils.constants import PARAGRAPH_MARKER


logger = logging.getLogger(__name__)


@Predictor.register('question_translator')
class QuestionTranslatorPredictor(Predictor):
    """
    This is the default question_translator predictor.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:

        json_dict = deepcopy(json_dict)
        if self._dataset_reader._translation_type == "compose":
            # To handle special case that predicting on augmented dataset
            # there's not a mix of instances with and without gold outputs.
            json_dict.pop("composed_question_text", None)
        instance = self._dataset_reader.json_to_instance(json_dict)
        self._dataset_reader.apply_token_indexers(instance)
        return instance

    @overrides
    def predict_json(self, input: JsonDict) -> JsonDict:
        instance = self._json_to_instance(input)
        prediction = self.predict_instance(instance)
        output = deepcopy(input)
        if self._dataset_reader._translation_type == "compose":
            output["predicted_composed_question_text"] = prediction["predicted_text"]
        else:
            output["predicted_decomposed_question_text"] = prediction["predicted_text"]
        return output

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        predictions = self.predict_batch_instance(instances)
        outputs = []
        for input, prediction in zip(inputs, predictions):
            output = deepcopy(input)
            if self._dataset_reader._translation_type == "compose":
                output["predicted_composed_question_text"] = prediction["predicted_text"]
            else:
                output["predicted_decomposed_question_text"] = prediction["predicted_text"]
            outputs.append(output)
        return outputs

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        output = self._model.forward_on_instance(instance)
        prediction = {
            "predicted_text": output["predicted_text"],
            "translation_type": self._dataset_reader._translation_type
        }
        return sanitize(prediction)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        predictions = [{
            "predicted_text": output["predicted_text"],
            "translation_type": self._dataset_reader._translation_type
        } for output in outputs]
        return sanitize(predictions)
