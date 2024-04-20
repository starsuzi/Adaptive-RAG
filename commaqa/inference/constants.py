from typing import Dict

from commaqa.inference.dataset_readers import DatasetReader, MultiParaRCReader
from commaqa.inference.participant_qa import LLMQAParticipantModel
from commaqa.inference.ircot import (
    AnswerExtractor,
    CopyQuestionParticipant,
    RetrieveAndResetParagraphsParticipant,
    StepByStepCOTGenParticipant,
    StepByStepLLMTitleGenParticipant,
    StepByStepExitControllerParticipant,
)

MODEL_NAME_CLASS = {
    "answer_extractor": AnswerExtractor,
    "copy_question": CopyQuestionParticipant,
    "llmqa": LLMQAParticipantModel,
    "retrieve_and_reset_paragraphs": RetrieveAndResetParagraphsParticipant,
    "step_by_step_cot_gen": StepByStepCOTGenParticipant,
    "step_by_step_llm_title_gen": StepByStepLLMTitleGenParticipant,
    "step_by_step_exit_controller": StepByStepExitControllerParticipant,
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "multi_para_rc": MultiParaRCReader,
}

PREDICTION_TYPES = {"answer", "titles", "pids"}
