import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead
from transformers.generation_utils import SampleEncoderDecoderOutput
import logging

logger = logging.getLogger(__name__)


class LMGenerator:
    def __init__(self, model_path, device=None, generation_args={}, encoder_args={}, decoder_args={}):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelWithLMHead.from_pretrained(model_path, config=self.config).to(self.device)
        self.generation_args = generation_args
        # always generate output with scores
        self.generation_args["output_scores"] = True
        self.generation_args["return_dict_in_generate"] = True
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

    def generate_text_sequence(self, input_text):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        encoded_prompt = self.tokenizer.encode(input_text, **self.encoder_args)

        encoded_prompt = encoded_prompt.to(self.device)
        generated_dict = self.model.generate(input_ids=encoded_prompt, **self.generation_args)

        generated_seqs = generated_dict.sequences
        if isinstance(generated_dict, SampleEncoderDecoderOutput):
            logger.warning("No scores generated when sampled sequences")
            generated_scores = [0] * len(generated_seqs)
        else:
            generated_scores = generated_dict.sequences_scores.tolist()
        if len(generated_seqs.shape) > 2:
            generated_seqs.squeeze_()

        output_seq_score = []

        for generated_sequence_idx, generated_seq in enumerate(generated_seqs):
            generated_output = generated_seq.tolist()
            text = self.tokenizer.decode(generated_output, **self.decoder_args)
            # flip the negative logit so that sequence with lowest scores is best
            output_seq_score.append((text, -generated_scores[generated_sequence_idx]))

        # Ensure sorted output
        return sorted(output_seq_score, key=lambda x: x[1])
