from typing import List, Dict
import copy
import logging

from overrides import overrides
import numpy as np

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from allennlp_lib.data.dataset_readers.utils import answer_offset_in_context
from allennlp_lib.predictors.transformer_rc import TransformerRCPredictor

from utils.common import step_placeholder
from utils.constants import (
    CONSTITUENT_QUESTION_START,
    CONSTITUENT_QUESTION_END,
    REPLACEMENT_QUESTION_START,
    REPLACEMENT_QUESTION_END
)

logger = logging.getLogger(__name__)


def get_answer_substituted_question(question_text: str, constituent_predicted_answers: List[str]) -> str:
    question_text = question_text.strip()

    for index in range(0, 10):
        prefix = step_placeholder(index, is_prefix=True, strip=True)
        if question_text.startswith(prefix):
            question_text = question_text.replace(prefix, "", 1)

    for index in range(0, 10):
        if step_placeholder(index, is_prefix=False, strip=True) in question_text:

            if index >= len(constituent_predicted_answers):
                logger.info("WARNING: Replacement answer not found, using empty string.")
                replacement_text = ""
            else:
                replacement_text = constituent_predicted_answers[index]
            question_text = question_text.replace(
                step_placeholder(index, is_prefix=False, strip=True),
                replacement_text
            )

    # I'm removing (commenting) the following behavior since it's no longer true.
    # # This is because by default SH questions start with [CQS] 0 [CQE] prefix
    # question_text = step_placeholder(0, is_prefix=True, strip=False) + question_text.strip()

    return question_text


def decomposed_question_to_instances(decomposition_question_text: str):

    decomposed_instances = []
    for component_question_text in decomposition_question_text.split(CONSTITUENT_QUESTION_START)[1:]:
        component_question_text = CONSTITUENT_QUESTION_START+component_question_text
        # Single-hop questions do have [CQS] 0 [CQE] prefix by default, so this is needed.

        decomposed_instance = {"question_text": component_question_text}
        decomposed_instances.append(decomposed_instance)

    assert decomposition_question_text.count(CONSTITUENT_QUESTION_START) == len(decomposed_instances)
    return decomposed_instances


def check_decomposition_validity(decomposed_instances) -> bool:
    """Checks if the decomposition defined by decomposed_instances valid or not"""
    # Mainly check if usage of answer occurs only after prediction of the answer.
    for question_index, decomposed_instance in enumerate(decomposed_instances):
        question_text = decomposed_instance["question_text"]
        for entity_index in range(10):
            placeholder = step_placeholder(entity_index, is_prefix=False, strip=True)
            if placeholder in question_text and entity_index >= question_index:
                return False
    return True


class MultiStepTransformerRCPredictor(TransformerRCPredictor):
    """
    Abstract class of MultiStepTransformerRCPredictor. It's used in
    MultiStepEnd2EndTransformerRCPredictor and MultiStepSelectAndAnswerTransformerRCPredictor
    """

    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        use_predicted_decomposition: bool,
        skip_distractor_paragraphs: bool = False,
        predict_answerability: bool = False,
        frozen: bool = True,
        beam_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(model, dataset_reader, frozen, **kwargs)
        self._beam_size = beam_size
        self._use_predicted_decomposition = use_predicted_decomposition
        self._skip_distractor_paragraphs = skip_distractor_paragraphs
        self._predict_answerability = predict_answerability

    def _get_outputs_for_single_step(
        self,
        question_text: str,
        common_contexts: List[Dict]
    ) -> List[Dict]:
        raise NotImplementedError

    def execute_multi_steps(self, decomposed_instances: List[Dict], common_contexts: List[Dict]) -> List[List[Dict]]:
        list_constituent_outputs = []
        self._execute_multi_steps(decomposed_instances, common_contexts, [], list_constituent_outputs)
        return list_constituent_outputs

    def _execute_multi_steps(
        self,
        decomposed_instances: List[Dict],
        common_contexts: List[Dict],
        constituent_outputs: List[Dict],
        list_constituent_outputs: List[List[Dict]]
    ) -> None:

        if not decomposed_instances:
            list_constituent_outputs.append(constituent_outputs)
            return None

        instance = decomposed_instances[0]

        question_text = instance["question_text"]
        constituent_predicted_answers = [output["predicted_answer"] for output in constituent_outputs]
        question_text = get_answer_substituted_question(question_text, constituent_predicted_answers)

        branching_outputs = self._get_outputs_for_single_step(question_text, common_contexts)
        for branching_output in branching_outputs:
            self._execute_multi_steps(
                decomposed_instances[1:], common_contexts,
                constituent_outputs + [branching_output], list_constituent_outputs
            )

    @overrides
    def predict_json(self, input_dict: JsonDict) -> JsonDict:

        question_id = input_dict["id"]
        answer_labels = [input_dict["answer_text"]]
        common_contexts = input_dict["contexts"]

        if not self._use_predicted_decomposition:
            decomposed_instances = input_dict["decomposed_instances"]
        else:
            decomposed_question_text = input_dict["predicted_decomposed_question_text"]
            decomposed_instances = decomposed_question_to_instances(decomposed_question_text)

        if not check_decomposition_validity(decomposed_instances):
            logger.info(f"WARNING: Decomposed question for question_id {question_id} is not valid.")

        if self._skip_distractor_paragraphs:
            common_contexts = [context for context in common_contexts if context["is_supporting"]]

        support_label_indices = [index for index, context in enumerate(common_contexts)
                                 if context["is_supporting"]]

        list_constituent_outputs = self.execute_multi_steps(decomposed_instances, common_contexts)

        # Replace chain_scorer with actual model later. For default beam-size of 1 it doesn't matter.
        chain_scorer = lambda outputs: sum(output['predicted_answer_prob'] for output in outputs)
        best_constituent_outputs = max(list_constituent_outputs, key=chain_scorer)

        predicted_answer = best_constituent_outputs[-1]["predicted_answer"]

        predicted_support_indices = [index for output in best_constituent_outputs
                                     for index in output["predicted_support_indices"]]
        predicted_support_indices = list(set(predicted_support_indices))

        prediction = {
            "id": question_id,
            "input": input_dict,
            "predicted_best_answer": predicted_answer,
            "predicted_rank_support_indices": predicted_support_indices,
            "predicted_select_support_indices": predicted_support_indices,
            "answer_labels": answer_labels,
            "support_label_indices": support_label_indices,
            "best_constituent_outputs": best_constituent_outputs,
            "list_constituent_outputs": list_constituent_outputs
        }

        if self._predict_answerability:
            predicted_answerability = all([
                output["predicted_answerability"] for output in best_constituent_outputs
            ])
            prediction["predicted_answerability"] = predicted_answerability

        return prediction

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        predictions = []
        for _input in inputs:
            prediction = self.predict_json(_input)
            predictions.append(prediction)
        return predictions

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        raise Exception("Not implemented, please use Json IO API.")

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        raise Exception("Not implemented, please use Json IO API.")
