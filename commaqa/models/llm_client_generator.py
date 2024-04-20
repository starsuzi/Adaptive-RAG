import requests
import os
from typing import Dict

from diskcache import Cache
from commaqa.inference.prompt_reader import fit_prompt_into_given_limit

cache = Cache(os.path.expanduser("~/.cache/llmcalls"))


def non_cached_llm_call(  # kwargs doesn't work with caching.
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
) -> Dict:

    params = {
        "prompt": prompt,
        "max_input": max_input,
        "max_length": max_length,
        "min_length": min_length,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "keep_prompt": keep_prompt,
    }

    host = os.environ.get("LLM_SERVER_HOST", None)
    port = os.environ.get("LLM_SERVER_PORT", None)

    if "/" in model_name:
        assert model_name.count("/", 1)
        model_name = model_name.split("/")[1]

    llm_server_key_suffix = os.environ.get("LLM_SERVER_KEY_SUFFIX", "")
    if model_name.replace("-", "_") + "_LLM_SERVER_HOST" in os.environ:
        host = os.environ[model_name.replace("-", "_") + "_LLM_SERVER_HOST" + llm_server_key_suffix]
    if model_name.replace("-", "_") + "_LLM_SERVER_PORT" in os.environ:
        port = os.environ[model_name.replace("-", "_") + "_LLM_SERVER_PORT" + llm_server_key_suffix]

    response = requests.get(host + ":" + str(port) + "/generate", params=params)

    if response.status_code != 200:
        raise Exception("LLM Generation request failed!")

    result = response.json()

    model_name_ = result.get("model_name", "")  # To assure that response is from the right model.

    if model_name_.replace("-bf16", "").replace("-dsbf16", "").replace("-8bit", "") != model_name:
        raise Exception(f"Looks like incorrect LLM server is ON: {model_name_} != {model_name}.")

    return result


@cache.memoize()
def cached_llm_call(  # kwargs doesn't work with caching.
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
) -> Dict:
    return non_cached_llm_call(
        prompt,
        model_name,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt,
    )


def llm_call(
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
) -> Dict:
    function = cached_llm_call if not do_sample and temperature > 0 else non_cached_llm_call
    return function(
        prompt,
        model_name,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt,
    )


class LLMClientGenerator:

    # Instructions to start the LLM Server are in the README here:
    # https://github.com/harshTrivedi/llm_server

    def __init__(
        self,
        model_name,
        max_input=None,
        max_length=100,
        min_length=1,
        do_sample=False,
        eos_text="\n",
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        num_return_sequences=1,
        repetition_penalty=None,
        length_penalty=None,
        model_tokens_limit=2000,
        remove_method="first",
    ):

        valid_model_names = [
            "gpt-j-6B",
            "opt-66b",
            "gpt-neox-20b",
            "T0pp",
            "flan-t5-base",
            "flan-t5-large",
            "flan-t5-xl",
            "flan-t5-xxl",
            "ul2",
        ]
        model_name_ = model_name
        if "/" in model_name:
            assert model_name.count("/", 1)
            model_name_ = model_name.split("/")[1]
        assert model_name_ in valid_model_names, f"Model name {model_name_} not in {valid_model_names}"

        self.model_name = model_name
        self.max_input = max_input
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_text = eos_text
        self.num_return_sequences = num_return_sequences
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.model_tokens_limit = model_tokens_limit
        self.remove_method = remove_method

    def generate_text_sequence(self, prompt):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        prompt = prompt.rstrip()

        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_length,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name=self.model_name,
            last_is_test_example=True,
        )

        # Note: Don't pass eos_text. Doesn't seem to work right.
        params = {
            "prompt": prompt,
            "model_name": self.model_name,
            "max_input": self.max_input,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_return_sequences": self.num_return_sequences,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "keep_prompt": False,
        }
        result = llm_call(**params)

        generated_texts = result["generated_texts"]
        modified_texts = []
        for text in generated_texts:
            # remove the prompt
            if text.startswith(prompt):
                text = text[len(prompt) :]
            if self.eos_text and self.eos_text in text:
                text = text[: text.index(self.eos_text)]
            modified_texts.append(text)
        generated_texts = modified_texts

        output_seq_score = [(text, 1 / (index + 1)) for index, text in enumerate(generated_texts)]
        # print(prompt)
        # print("------------")
        # print(output_seq_score[0][0])

        # TODO: Deal with output-probabilities if needed.

        return sorted(output_seq_score, key=lambda x: x[1])
