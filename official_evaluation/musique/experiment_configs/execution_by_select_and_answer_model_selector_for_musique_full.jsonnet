local setting = std.extVar("setting"); #options: full, fixture

local seed = 100;

local fixture_path = 'fixtures/datasets/single_hop_all_retrieved_mixed_answerable_para_data.jsonl';
local train_data_path =
    if setting == 'full'
        then 'raw_data/musique_full_single_hop_version_train.jsonl'
    else if setting == 'fixture'
        then fixture_path
    else '-';
local validation_data_path =
    if setting == 'full'
        then 'raw_data/musique_full_single_hop_version_dev.jsonl'
    else if setting == 'fixture'
        then fixture_path
    else '-';

local batch_size = 12;
local accumulation_steps = 1;

local num_epochs =
    if setting == 'full'
        then 3
    else if setting == 'fixture'
        then 30
    else 2;

local patience =
    if setting == 'full'
        then 2
    else if setting == 'fixture'
        then num_epochs
    else num_epochs;

local num_workers = if setting == 'full' then 5 else 2;

local question_key = "question_text";
local add_question_info = true;

{
    "dataset_reader": {
        "type": "text_ranker",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "roberta-large",
            "add_special_tokens": false
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": "roberta-large"
            }
        },
        "max_tokens": 300,
        "add_question_info": add_question_info,
        "question_key": question_key,
        "balance": true
    },
    "validation_dataset_reader": self.dataset_reader + {
        "balance": false
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        },
        "num_workers": num_workers,
        "max_instances_in_memory": batch_size*50
    },
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "text_ranker",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": "roberta-large",
                    "gradient_checkpointing": true
                }
            }
        },
        "has_single_positive": false,
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
        "validation_metric": "+rank_f1",
    },

    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed,
}
