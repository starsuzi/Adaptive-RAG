from typing import List
import logging
from copy import deepcopy

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


logger = logging.getLogger(__name__)


@Predictor.register('inplace_text_ranker')
class InplaceTextRankerPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:

        instance = self._dataset_reader.json_to_instance(json_dict)
        self._dataset_reader.apply_token_indexers(instance)
        return instance

    @overrides
    def predict_json(self, input: JsonDict) -> JsonDict:
        instance = self._json_to_instance(input)
        prediction = self.predict_instance(instance)
        output = deepcopy(input)
        output["predicted_ordered_contexts"] = prediction["predicted_ordered_contexts"]
        return output

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        predictions = self.predict_batch_instance(instances)
        outputs = []
        for input, prediction in zip(inputs, predictions):
            output = deepcopy(input)
            output["predicted_ordered_contexts"] = prediction["predicted_ordered_contexts"]
            outputs.append(output)
        return outputs

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        output = self._model.forward_on_instance(instance)
        prediction = {
            "predicted_ordered_contexts": [
                output["contexts"][index]
                for index in output["predicted_ordered_indices"]
                # Because model includes masked ones also, although at the end:
                if index < len(output["contexts"])
            ]
        }
        return sanitize(prediction)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        predictions = [{
            "predicted_ordered_contexts": [
                output["contexts"][index]
                for index in output["predicted_ordered_indices"]
                # Because model includes masked ones also, although at the end:
                if index < len(output["contexts"])
            ]
        } for output in outputs]
        return sanitize(predictions)
