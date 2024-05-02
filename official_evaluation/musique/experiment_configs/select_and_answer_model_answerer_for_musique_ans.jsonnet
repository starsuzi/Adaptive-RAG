#########################################################
local setting = std.extVar("setting"); #options: full, fixture
local biggest_batch_first = false; # only for debugging.
#########################################################

local seed = 100;

local transformer_model_name = "allenai/longformer-large-4096";

local batch_size = 16;
local accumulation_steps = 1;

local question_key = "composed_question_text";
local max_document_tokens = 4000;
local max_paragraph_tokens = 400;
local max_question_tokens = 150;
local take_predicted_topk_contexts = 7;

local supervize_answerability = false;
local supervize_answer = true;
local supervize_support = true;
local has_single_support = false;

local num_workers = if setting == 'full' then 5 else 2;

local num_epochs =
    if setting == 'full'
        then 3
    else if setting == 'fixture'
        then 30
    else 2;

local fixture_path = 'fixtures/datasets/multi_hop_all_retrieved_para_data.jsonl';
local train_data_path =
    if setting == 'full'
        then 'serialization_dir/select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_train.jsonl'
    else if setting == 'fixture'
        then fixture_path
    else '-';
local validation_data_path =
    if setting == 'full'
        then 'serialization_dir/select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_dev.jsonl'
    else if setting == 'fixture'
        then fixture_path
    else '-';

local patience =
    if setting == 'full'
        then 3
    else if setting == 'fixture'
        then num_epochs
    else num_epochs;


{
    "dataset_reader": {
        "type": "transformer_rc",
        "transformer_model_name": transformer_model_name,
        "question_key": question_key,
        "max_document_tokens": max_document_tokens,
        "max_paragraph_tokens": max_paragraph_tokens,
        "max_question_tokens": max_question_tokens,
        "take_predicted_topk_contexts": take_predicted_topk_contexts
        // "max_instances": 1000,  // debug setting
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "transformer_rc",
        "transformer_model": {
            "model_name": transformer_model_name,
            "gradient_checkpointing": true
        },
        "supervize_answerability": supervize_answerability,
        "supervize_answer": supervize_answer,
        "supervize_support": supervize_support,
        "has_single_support": has_single_support
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size,
        },
        "num_workers": num_workers,
        "max_instances_in_memory": batch_size*50
    },
    "trainer": {
        "cuda_device": 0,
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": 0.0,
            "parameter_groups": [
                [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}],
            ],
            "lr": 2e-5,
            "eps": 1e-8
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": num_epochs,
            "cut_frac": 0.1,
        },
        "callbacks": [
            {"type": "tensorboard"}
        ],
        "patience": patience,
        "grad_clipping": 1.0,
        "num_gradient_accumulation_steps": accumulation_steps,
        "num_epochs": num_epochs,
        "validation_metric": "+answer_text_f1",
    },
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed,
}
