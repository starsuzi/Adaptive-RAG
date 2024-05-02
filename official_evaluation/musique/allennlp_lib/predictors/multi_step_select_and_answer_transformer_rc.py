from typing import List, Dict
import copy
import logging

from allennlp.common.util import sanitize
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from allennlp_lib.predictors.multi_step_transformer_rc import MultiStepTransformerRCPredictor

from utils.constants import (
    CONSTITUENT_QUESTION_START,
    CONSTITUENT_QUESTION_END,
    REPLACEMENT_QUESTION_START,
    REPLACEMENT_QUESTION_END
)

logger = logging.getLogger(__name__)


@Predictor.register('multi_step_select_and_answer_transformer_rc')
class MultiStepSelectAndAnswerTransformerRCPredictor(MultiStepTransformerRCPredictor):
    """
    MultiStep Transformer RC where each step is a Select+Answer model.
    """

    def __init__(
        self,
        model: Model, # Answerer
        dataset_reader: DatasetReader,
        use_predicted_decomposition: bool,
        selector_model_path: str,
        num_select: int,
        skip_distractor_paragraphs: bool = False,
        predict_answerability: bool = False,
        frozen: bool = True,
        beam_size: int = 1,
        **kwargs,
    ) -> None:

        super().__init__(
            model=model,
            dataset_reader=dataset_reader,
            use_predicted_decomposition=use_predicted_decomposition,
            skip_distractor_paragraphs=skip_distractor_paragraphs,
            predict_answerability=predict_answerability,
            frozen=frozen,
            beam_size=beam_size,
            **kwargs
        )
        self._selector = Predictor.from_path(
            selector_model_path,
            predictor_name="inplace_text_ranker",
            cuda_device=self.cuda_device
        )
        self._num_select = num_select

        self._dataset_reader._take_predicted_topk_contexts = num_select
        self._dataset_reader._question_key = "question_text"
        self._dataset_reader._prefer_question_key_if_present = None

        self._selector._dataset_reader._question_key = "question_text"


    def _get_outputs_for_single_step(
        self,
        question_text: str,
        common_contexts: List[Dict]
    ) -> List[Dict]:

        # Use same contexts common for all, but update the supporting
        # para to be one corresponding to gold para of this instance only.
        local_contexts = copy.deepcopy(common_contexts)

        predicted_answers = []
        predicted_answer_probs = []

        # Obtain Model Ans/Support as Prediction

        # Full-Context Executor
        single_step_input = {}
        single_step_input["id"] = "<placeholder_id>"
        single_step_input["answer_text"] = "<placeholder_answer_text>"
        single_step_input["answerable"] = True
        single_step_input["contexts"] = copy.deepcopy(local_contexts)
        single_step_input["question_text"] = question_text
        single_step_input["dataset"] = "<placeholder_dataset>"

        for special_token in [
            CONSTITUENT_QUESTION_START,
            CONSTITUENT_QUESTION_END,
            REPLACEMENT_QUESTION_START,
            REPLACEMENT_QUESTION_END
        ]:
            if special_token in single_step_input["question_text"]:
                logger.info(f"WARNING: Found {special_token} in single-step "
                            f"question_text {single_step_input['question_text']}")

        single_step_output = self._selector.predict_json(single_step_input) # TextRanker Predictor

        single_step_instance = self._json_to_instance(single_step_output) # From TransformerRCPredictor
        single_step_output = self._model.forward_on_instance(single_step_instance)
        single_step_output = sanitize(single_step_output)

        if self._predict_answerability:
            predicted_answerability = single_step_output.get("predicted_answerability", None)
            predicted_answerability_prob = single_step_output.get("predicted_answerability_prob", None)
            assert predicted_answerability is not None or predicted_answerability_prob is not None, \
                "Asked to predict answerability, but base model doesn't produce it."

        predicted_best_k_answers = single_step_output["predicted_best_k_answers"]
        predicted_best_k_answers_probs = single_step_output["predicted_best_k_answers_probs"]

        predicted_support_probs = single_step_output.get("predicted_support_probs", [])
        predicted_ordered_support_indices = single_step_output.get("predicted_ordered_support_indices", [])

        answers_and_probs = [(answer, prob)
                             for answer, prob in zip(predicted_best_k_answers, predicted_best_k_answers_probs)
                             if answer.strip()][:self._beam_size]
        # Note: Don't sort them by answer probabilities, they are based on different masks.
        # The order is already sorted.

        assert answers_and_probs[0][0] == single_step_output["predicted_best_answer"], \
            "The best predicted answer isn't the same as 1st k top answers."

        predicted_answers.extend([answer_prob[0] for answer_prob in answers_and_probs])
        predicted_answer_probs.extend([answer_prob[1] for answer_prob in answers_and_probs])

        local_contexts = copy.deepcopy(single_step_output["contexts"])
        for context in local_contexts:
            for key, value in context.items():
                if value == 'None':
                    context[key] = None

        index_for_keys_match = lambda search_list, search_item, keys: next(
            index for index, item in enumerate(search_list)
            if all(search_item[key] == item[key] for key in keys)
        )
        predicted_rank_support_indices = [
            index_for_keys_match(
                common_contexts,
                local_contexts[predicted_ordered_support_indices[0]],
                ['wikipedia_id', 'paragraph_indices', 'paragraph_text']
            )
        ]
        # TODO: I should get predicted_support_probs indexed into common_contexts. Not needed for now.

        branching_constituent_outputs = [{
            "question_text": question_text,
            "predicted_answer": predicted_answer,
            "predicted_answer_prob": predicted_answer_prob,
            "predicted_support_indices": predicted_rank_support_indices,
        } for predicted_answer, predicted_answer_prob in zip(predicted_answers, predicted_answer_probs)]

        if self._predict_answerability:
            for output in branching_constituent_outputs:
                output["predicted_answerability"] = predicted_answerability
                output["predicted_answerability_prob"] = predicted_answerability_prob

        return branching_constituent_outputs
