# Set dataset:
local dataset = "nq";
local retrieval_corpus_name = 'wiki';
local add_pinned_paras = if dataset == "iirc" then true else false;
local valid_qids = ["5abb14bd5542992ccd8e7f07", "5ac2ada5554299657fa2900d", "5a758ea55542992db9473680", "5ae0185b55429942ec259c1b", "5a8ed9f355429917b4a5bddd", "5abfb3435542990832d3a1c1", "5ab92dba554299131ca422a2", "5a835abe5542996488c2e426", "5a89c14f5542993b751ca98a", "5a90620755429933b8a20508", "5a7bbc50554299042af8f7d0", "5a8f44ab5542992414482a25", "5add363c5542990dbb2f7dc8", "5a7fc53555429969796c1b55", "5a790e7855429970f5fffe3d"];
local prompt_reader_args = {
    "filter_by_key_values": {
        "qid": valid_qids
    },
    "order_by_key": "qid",
    "estimated_generation_length": 0, # don't drop in reading phase.
    "shuffle": false,
    "model_length_limit": 1000000, # don't drop in reading phase.
    "tokenizer_model_name": "google/flan-t5-xl",
};

# (Potentially) Hyper-parameters:
# null means it's unused.
local llm_retrieval_count = null;
local llm_map_count = null;
local bm25_retrieval_count = 15;
local rc_context_type_ = "gold_with_n_distractors"; # Choices: no, gold, gold_with_n_distractors
local distractor_count = "1"; # Choices: 1, 2, 3
local rc_context_type = (
    if rc_context_type_ == "gold_with_n_distractors"
    then "gold_with_" + distractor_count + "_distractors"  else rc_context_type_
);
local rc_qa_type = "direct"; # Choices: direct, cot
local qa_question_prefix = (
    if std.endsWith(rc_context_type, "cot")
    then "Answer the following question by reasoning step-by-step.\n"
    else "Answer the following question.\n"
);

{
    "start_state": "generate_titles",
    "end_state": "[EOQ]",
    "models": {
        "generate_titles": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "generate_main_question",
            "retrieval_type": "bm25",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": bm25_retrieval_count,
            "global_max_num_paras": 15,
            "query_source": "original_question",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "end_state": "[EOQ]",
        },

        "generate_main_question": {
            "name": "copy_question",
            "next_model": "answer_main_question",
            "eoq_after_n_calls": 1,
            "end_state": "[EOQ]",
        },
        "answer_main_question": {
            "name": "llmqa",
            "next_model": if std.endsWith(rc_qa_type, "cot") then "extract_answer" else null,
            "prompt_file": "prompts/"+dataset+"/"+rc_context_type+"_context_"+rc_qa_type+"_qa_flan_t5.txt",
            "question_prefix": qa_question_prefix,
            "prompt_reader_args": prompt_reader_args,
            "end_state": "[EOQ]",
            "gen_model": "llm_api",
            "model_name": "google/flan-t5-xl",
            "model_tokens_limit": 6000,
            "max_length": 200,
            "add_context": true,
        },
        "extract_answer": {
            "name": "answer_extractor",
            "query_source": "last_answer",
            "regex": ".* answer is:? (.*)\\.?",
            "match_all_on_failure": true,
            "remove_last_fullstop": true,
        }
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false,
        "add_pinned_paras": add_pinned_paras,
    },
    "prediction_type": "answer"
}