local setting = std.extVar("setting");
local model_name = "facebook/bart-large";

local seed = 100;

local num_epochs =
    if setting == 'full'
        then 3
    else if setting == 'fixture'
        then 10
    else 2;

local patience = 10;

local fixture_path = 'fixtures/datasets/2wikimultihopqa.jsonl';
local train_data_path =
    if setting == 'full'
        then 'raw_data/2wikimultihopqa_train_20k.jsonl'
    else if setting == 'fixture'
        then fixture_path
    else '-';
local validation_data_path =
    if setting == 'full'
        then 'raw_data/2wikimultihopqa_dev_20k.jsonl'
    else if setting == 'fixture'
        then fixture_path
    else '-';

{
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "dataset_reader": {
        "type": "question_translator",
        "transformer_model_name": model_name,
        "max_source_tokens": 100,
        "max_target_tokens": 100,
        "translation_type": "decompose",
        "composed_question_key": "question_text",
        // "max_instances": 1000 // DEBUG setting
    },
    "model": {
        "type": "bart",
        "model_name": model_name,
        "beam_size": 10,
        "max_steps": 100
    },
    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
        },
        "callbacks": [
            {"type": "tensorboard"}
        ],
        "grad_norm": 1.0,
        "patience": patience,
        "validation_metric": "+BLEU",
    },
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed,
}
