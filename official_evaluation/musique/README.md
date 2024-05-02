# <h2 align="center"> :musical_note: MuSiQue: Multi-hop Questions via Single-hop Question Composition </h2>

Repository for our TACL 2022 paper "[MuSiQue: Multi-hop Questions via Single-hop Question Composition](https://arxiv.org/pdf/2108.00573.pdf)"

# Data

MuSiQue is distributed under a [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

**Usage Caution:** If you're using any of our seed single-hop datasets ([SQuAD](https://arxiv.org/abs/1606.05250), [T-REx](https://hadyelsahar.github.io/t-rex/paper.pdf), [Natural Questions](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf), [MLQA](https://arxiv.org/pdf/1910.07475.pdf), [Zero Shot RE](https://arxiv.org/pdf/1706.04115.pdf)) in any way (e.g., pretraining on them), please note that MuSiQue was created by composing questions from these seed datasets. Therefore, single-hop questions used in MuSiQue's dev/test sets may occur in the training sets of these seed datasets. To help avoid information leakage, we are releasing the IDs of single-hop questions that are used in MuSiQue dev/test sets. Once you download the data below, these IDs and corresponding questions will be in `data/dev_test_singlehop_questions_v1.0.json`. If you use our seed single-hop datasets in any way in your model, please be sure to **avoid using any single-hop question IDs present in this file**

To download MuSiQue, either run the following script or download it manually from [here](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing).

```
bash download_data.sh
```

The result will be stored in `data/` directory. It contains (i) train, dev and test sets of `MuSiQue-Ans` and `MuSiQue-Full`, (ii) single-hop questions and ids from source datasets (squad, natural questions, trex, mlqa, zerore) that are part of dev or test of MuSiQue.


# Predictions

We're releasing the model predictions (in official format) for 4 models on dev sets of `MuSiQue-Ans` and `MuSiQue-Full`. To get it, you can run the following script or download it manually from [here](https://drive.google.com/file/d/1XZocqLOTAu4y_1EeAj1JM4Xc1JxGJtx6/view?usp=sharing).

```
bash download_predictions.sh
```


# Evaluation

You can use `evaluate_v1.0.py` to evaluate your predictions against ground-truths. For eg.:

```
python evaluate_v1.0.py predictions/musique_ans_v1.0_dev_end2end_model_predictions.jsonl data/musique_ans_v1.0_dev.jsonl
```

These are the results you would get for MuSiQue-Answerable and MuSiQue-Full validation sets and for each of the four models (End2End Model, Select+Answer Model, Execution by End2End Model, Execution by Select+Answer Model).

```bash
# MuSiQue-Answerable
python evaluate_v1.0.py predictions/musique_ans_v1.0_dev_end2end_model_predictions.jsonl data/musique_ans_v1.0_dev.jsonl
# => {"answer_f1": 0.423, "support_f1": 0.676}

python evaluate_v1.0.py predictions/musique_ans_v1.0_dev_select_answer_model_predictions.jsonl data/musique_ans_v1.0_dev.jsonl
# => {"answer_f1": 0.473, "support_f1": 0.723}

python evaluate_v1.0.py predictions/musique_ans_v1.0_dev_step_execution_by_end2end_model_predictions.jsonl data/musique_ans_v1.0_dev.jsonl
# => {"answer_f1": 0.456, "support_f1": 0.778}

python evaluate_v1.0.py predictions/musique_ans_v1.0_dev_step_execution_by_select_answer_model_predictions.jsonl data/musique_ans_v1.0_dev.jsonl
# => {"answer_f1": 0.497, "support_f1": 0.792}

# MuSiQue-Full
python evaluate_v1.0.py predictions/musique_full_v1.0_dev_end2end_model_predictions.jsonl data/musique_full_v1.0_dev.jsonl
# => {"answer_f1": 0.406, "support_f1": 0.325, "group_answer_sufficiency_f1": 0.22, "group_support_sufficiency_f1": 0.252}

python evaluate_v1.0.py predictions/musique_full_v1.0_dev_select_answer_model_predictions.jsonl data/musique_full_v1.0_dev.jsonl
# => {"answer_f1": 0.486, "support_f1": 0.522, "group_answer_sufficiency_f1": 0.344, "group_support_sufficiency_f1": 0.42}

python evaluate_v1.0.py predictions/musique_full_v1.0_dev_step_execution_by_end2end_model_predictions.jsonl data/musique_full_v1.0_dev.jsonl
# => {"answer_f1": 0.463, "support_f1": 0.75, "group_answer_sufficiency_f1": 0.321, "group_support_sufficiency_f1": 0.447}

python evaluate_v1.0.py predictions/musique_full_v1.0_dev_step_execution_by_select_answer_model_predictions.jsonl data/musique_full_v1.0_dev.jsonl
# => {"answer_f1": 0.498, "support_f1": 0.777, "group_answer_sufficiency_f1": 0.328, "group_support_sufficiency_f1": 0.431}
```

# Leaderboard

We've two leaderboards for MuSiQue: [MuSiQue-Answerable](https://leaderboard.allenai.org/musique_ans) and [MuSiQue-Full](https://leaderboard.allenai.org/musique_full).

Once you've the test set predictions in the official format, it's just about uploading the files in the above leadboards! Feel free to contact me (Harsh) in case you've any questions.


# Models and Experiments

We've relased the code that we used for experiments in the paper. If you're interested in trying our trained models, training them from sratch, viewing their predictions or generating their predictions from your trained model, follow the steps below. 

## Installations

```bash

# Set env.
conda create -n musique python=3.8 -y && conda activate musique

# Set allennlp in root directory
git clone https://github.com/allenai/allennlp
cd allennlp
git checkout v2.1.0
git apply ../allennlp.diff # small diff to get longformer global attention to work correctly.
cd ..

pip install allennlp==2.1.0 # we only need dependencies of allennlp
pip uninstall -y allennlp

pip install gdown==v4.5.1
python -m nltk.downloader stopwords

pip uninstall -y transformers
pip install transformers==4.7.0 # we used this version of transformers
```

## Download Raw Data

Our models were developed using a different (non-official) format of the dataset files. So to run our code, you'll first need to download the dataset files in the raw format. 

```bash
python download_raw_data.py
```

Note that officially released data and what we've used here are only different in the format (e.g. uses different names for json fields), and are not qualitatively different. Take a look at `raw_data_to_official_format.py` if you're interested.

-----

We've done experiments on 4 datasets (MuSiQue-Ans, MuSiQue-Full, HotpotQA-20K, 2WikiMultihopQA-20K) with 4 multihop models (End2End Model, Select+Answer Model, Execution by End2End Model, Execution by Select+Answer Model) where possible. See Table 1. You can explore each combination using the instruction toggle below.

For each combination, you'll see instructions on how (i) download trained model (ii) train a model from scratch
(iii) download model prediction/s (iv) generate predictions with a trained or a downloaded model.

Our models are implemented in `allennlp`. If you're familiar with it, using the code should be pretty straightforward. The only difference is that instead of using `allennlp` command, we're using `run.py` as an entrypoint, which mainly loads `allennlp_lib` to load our allennlp code (readers, models, predictors, etc).




## MuSiQue-Answerable

<details> <summary>
<strong>End2End Model [EE]</strong>
</summary>


----

### Experiment Name

```bash
end2end_model_for_musique_ans_dataset
```

### Download model

```bash
python download_models.py end2end_model_for_musique_ans_dataset
```

### Train from scratch

```bash
python run.py train experiment_configs/end2end_model_for_musique_ans_dataset.jsonnet \
                    --serialization-dir serialization_dir/end2end_model_for_musique_ans_dataset
```

### Download prediction/s

```bash
python download_raw_predictions.py end2end_model_for_musique_ans_dataset
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/end2end_model_for_musique_ans_dataset/model.tar.gz \
                      raw_data/musique_ans_dev.jsonl \
                      --output-file serialization_dir/end2end_model_for_musique_ans_dataset/predictions/musique_ans_dev.jsonl \
                      --predictor transformer_rc --batch-size 16 --cuda-device 0 --silent

# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/end2end_model_for_musique_ans_dataset/predictions/musique_ans_dev.jsonl

```

</details>
    
<details> <summary>
<strong>Select+Answer Model [SA]</strong>
</summary>

<br>The system has 2 parts given below: (i) Selector Model (ii) Answerer Model


----

### Experiment Name

```bash
# Selector Model
select_and_answer_model_selector_for_musique_ans
```

### Download model

```bash
python download_models.py select_and_answer_model_selector_for_musique_ans
```

### Train from scratch

```bash
python run.py train experiment_configs/select_and_answer_model_selector_for_musique_ans.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_selector_for_musique_ans
```

### Download prediction/s

```bash
python download_raw_predictions.py select_and_answer_model_selector_for_musique_ans
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/select_and_answer_model_selector_for_musique_ans/model.tar.gz \
                      raw_data/musique_ans_train.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_train.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

python run.py predict serialization_dir/select_and_answer_model_selector_for_musique_ans/model.tar.gz \
                      raw_data/musique_ans_dev.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_dev.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Answerer Model
select_and_answer_model_answerer_for_musique_ans
```

### Download model

```bash
python download_models.py select_and_answer_model_answerer_for_musique_ans
```

### Train from scratch

```bash
python run.py train experiment_configs/select_and_answer_model_answerer_for_musique_ans.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_answerer_for_musique_ans
```

### Download prediction/s

```bash
python download_raw_predictions.py select_and_answer_model_answerer_for_musique_ans
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/select_and_answer_model_answerer_for_musique_ans/model.tar.gz \
                      serialization_dir/select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_dev.jsonl \
                      --output-file serialization_dir/select_and_answer_model_answerer_for_musique_ans/predictions/serialization_dir__select_and_answer_model_selector_for_musique_ans__predictions__musique_ans_dev.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/select_and_answer_model_answerer_for_musique_ans/predictions/serialization_dir__select_and_answer_model_selector_for_musique_ans__predictions__musique_ans_dev.jsonl

```

</details>
    
<details> <summary>
<strong>Execution by End2End Model [EX(EE)]</strong>
</summary>

<br>The system has 2 parts given below: (i) Decomposer Model (ii) Executor Model.


----

### Experiment Name

```bash
# Decomposer Model
execution_model_decomposer_for_musique_ans_and_full
```

### Download model

```bash
python download_models.py execution_model_decomposer_for_musique_ans_and_full
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_model_decomposer_for_musique_ans_and_full.jsonnet \
                    --serialization-dir serialization_dir/execution_model_decomposer_for_musique_ans_and_full
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_model_decomposer_for_musique_ans_and_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_model_decomposer_for_musique_ans_and_full/model.tar.gz \
                      raw_data/musique_ans_dev.jsonl \
                      --output-file serialization_dir/execution_model_decomposer_for_musique_ans_and_full/predictions/musique_ans_dev.jsonl \
                      --predictor question_translator --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Executor Model
execution_by_end2end_model_for_musique_ans
```

### Download model

```bash
python download_models.py execution_by_end2end_model_for_musique_ans
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_end2end_model_for_musique_ans.jsonnet \
                    --serialization-dir serialization_dir/execution_by_end2end_model_for_musique_ans
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_end2end_model_for_musique_ans
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_end2end_model_for_musique_ans/model.tar.gz \
                      serialization_dir/execution_model_decomposer_for_musique_ans_and_full/predictions/musique_ans_dev.jsonl \
                      --output-file serialization_dir/execution_by_end2end_model_for_musique_ans/predictions/serialization_dir__execution_model_decomposer_for_musique_ans_and_full__predictions__musique_ans_dev.jsonl \
                      --predictor multi_step_end2end_transformer_rc --batch-size 16 --cuda-device 0 --silent \
                      --predictor-args '{"predict_answerability":false,"skip_distractor_paragraphs":false,"use_predicted_decomposition":true}'


# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/execution_by_end2end_model_for_musique_ans/predictions/serialization_dir__execution_model_decomposer_for_musique_ans_and_full__predictions__musique_ans_dev.jsonl

```

</details>
    
<details> <summary>
<strong>Execution by Select+Answer Model [EX(SA)]</strong>
</summary>

<br>The system has 3 parts given below: (i) Decomposer Model (ii) Selector of Executor Model (iii) Answerer of Executor Model.


----

### Experiment Name

```bash
# Decomposer Model
execution_model_decomposer_for_musique_ans_and_full
```

### Download model

```bash
python download_models.py execution_model_decomposer_for_musique_ans_and_full
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_model_decomposer_for_musique_ans_and_full.jsonnet \
                    --serialization-dir serialization_dir/execution_model_decomposer_for_musique_ans_and_full
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_model_decomposer_for_musique_ans_and_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_model_decomposer_for_musique_ans_and_full/model.tar.gz \
                      raw_data/musique_ans_dev.jsonl \
                      --output-file serialization_dir/execution_model_decomposer_for_musique_ans_and_full/predictions/musique_ans_dev.jsonl \
                      --predictor question_translator --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Selector of Executor Model
execution_by_select_and_answer_model_selector_for_musique_ans
```

### Download model

```bash
python download_models.py execution_by_select_and_answer_model_selector_for_musique_ans
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_select_and_answer_model_selector_for_musique_ans.jsonnet \
                    --serialization-dir serialization_dir/execution_by_select_and_answer_model_selector_for_musique_ans
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_select_and_answer_model_selector_for_musique_ans
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_select_and_answer_model_selector_for_musique_ans/model.tar.gz \
                      raw_data/musique_ans_single_hop_version_train.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_single_hop_version_train.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

python run.py predict serialization_dir/execution_by_select_and_answer_model_selector_for_musique_ans/model.tar.gz \
                      raw_data/musique_ans_single_hop_version_dev.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_selector_for_musique_ans/predictions/musique_ans_single_hop_version_dev.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Answerer of Executor Model
execution_by_select_and_answer_model_answerer_for_musique_ans
```

### Download model

```bash
python download_models.py execution_by_select_and_answer_model_answerer_for_musique_ans
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_select_and_answer_model_answerer_for_musique_ans.jsonnet \
                    --serialization-dir serialization_dir/execution_by_select_and_answer_model_answerer_for_musique_ans
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_select_and_answer_model_answerer_for_musique_ans
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_select_and_answer_model_answerer_for_musique_ans/model.tar.gz \
                      serialization_dir/execution_model_decomposer_for_musique_ans_and_full/predictions/musique_ans_dev.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_answerer_for_musique_ans/predictions/serialization_dir__execution_model_decomposer_for_musique_ans_and_full__predictions__musique_ans_dev.jsonl \
                      --predictor multi_step_select_and_answer_transformer_rc --batch-size 16 --cuda-device 0 --silent \
                      --predictor-args '{"predict_answerability":false,"skip_distractor_paragraphs":false,"use_predicted_decomposition":true,"selector_model_path":"serialization_dir/execution_by_select_and_answer_model_selector_for_musique_ans/model.tar.gz","num_select":3}'


# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/execution_by_select_and_answer_model_answerer_for_musique_ans/predictions/serialization_dir__execution_model_decomposer_for_musique_ans_and_full__predictions__musique_ans_dev.jsonl

```

</details>
    

## MuSiQue-Full

<details> <summary>
<strong>End2End Model [EE]</strong>
</summary>


----

### Experiment Name

```bash
end2end_model_for_musique_full_dataset
```

### Download model

```bash
python download_models.py end2end_model_for_musique_full_dataset
```

### Train from scratch

```bash
python run.py train experiment_configs/end2end_model_for_musique_full_dataset.jsonnet \
                    --serialization-dir serialization_dir/end2end_model_for_musique_full_dataset
```

### Download prediction/s

```bash
python download_raw_predictions.py end2end_model_for_musique_full_dataset
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/end2end_model_for_musique_full_dataset/model.tar.gz \
                      raw_data/musique_full_dev.jsonl \
                      --output-file serialization_dir/end2end_model_for_musique_full_dataset/predictions/musique_full_dev.jsonl \
                      --predictor transformer_rc --batch-size 16 --cuda-device 0 --silent

# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/end2end_model_for_musique_full_dataset/predictions/musique_full_dev.jsonl

```

</details>
    
<details> <summary>
<strong>Select+Answer Model [SA]</strong>
</summary>

<br>The system has 2 parts given below: (i) Selector Model (ii) Answerer Model.


----

### Experiment Name

```bash
# Selector Model
select_and_answer_model_selector_for_musique_full
```

### Download model

```bash
python download_models.py select_and_answer_model_selector_for_musique_full
```

### Train from scratch

```bash
python run.py train experiment_configs/select_and_answer_model_selector_for_musique_full.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_selector_for_musique_full
```

### Download prediction/s

```bash
python download_raw_predictions.py select_and_answer_model_selector_for_musique_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/select_and_answer_model_selector_for_musique_full/model.tar.gz \
                      raw_data/musique_full_train.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_musique_full/predictions/musique_full_train.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

python run.py predict serialization_dir/select_and_answer_model_selector_for_musique_full/model.tar.gz \
                      raw_data/musique_full_dev.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_musique_full/predictions/musique_full_dev.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Answerer Model
select_and_answer_model_answerer_for_musique_full
```

### Download model

```bash
python download_models.py select_and_answer_model_answerer_for_musique_full
```

### Train from scratch

```bash
python run.py train experiment_configs/select_and_answer_model_answerer_for_musique_full.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_answerer_for_musique_full
```

### Download prediction/s

```bash
python download_raw_predictions.py select_and_answer_model_answerer_for_musique_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/select_and_answer_model_answerer_for_musique_full/model.tar.gz \
                      serialization_dir/select_and_answer_model_selector_for_musique_full/predictions/musique_full_dev.jsonl \
                      --output-file serialization_dir/select_and_answer_model_answerer_for_musique_full/predictions/serialization_dir__select_and_answer_model_selector_for_musique_full__predictions__musique_full_dev.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/select_and_answer_model_answerer_for_musique_full/predictions/serialization_dir__select_and_answer_model_selector_for_musique_full__predictions__musique_full_dev.jsonl

```

</details>
    
<details> <summary>
<strong>Execution by End2End Model [EX(EE)]</strong>
</summary>

<br>The system has 2 parts given below: (i) Decomposer Model (ii) Executor Model.


----

### Experiment Name

```bash
# Decomposer Model
execution_model_decomposer_for_musique_ans_and_full
```

### Download model

```bash
python download_models.py execution_model_decomposer_for_musique_ans_and_full
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_model_decomposer_for_musique_ans_and_full.jsonnet \
                    --serialization-dir serialization_dir/execution_model_decomposer_for_musique_ans_and_full
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_model_decomposer_for_musique_ans_and_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_model_decomposer_for_musique_ans_and_full/model.tar.gz \
                      raw_data/musique_full_dev.jsonl \
                      --output-file serialization_dir/execution_model_decomposer_for_musique_ans_and_full/predictions/musique_full_dev.jsonl \
                      --predictor question_translator --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Executor Model
execution_by_end2end_model_for_musique_full
```

### Download model

```bash
python download_models.py execution_by_end2end_model_for_musique_full
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_end2end_model_for_musique_full.jsonnet \
                    --serialization-dir serialization_dir/execution_by_end2end_model_for_musique_full
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_end2end_model_for_musique_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_end2end_model_for_musique_full/model.tar.gz \
                      serialization_dir/execution_model_decomposer_for_musique_ans_and_full/predictions/musique_full_dev.jsonl \
                      --output-file serialization_dir/execution_by_end2end_model_for_musique_full/predictions/serialization_dir__execution_model_decomposer_for_musique_ans_and_full__predictions__musique_full_dev.jsonl \
                      --predictor multi_step_end2end_transformer_rc --batch-size 16 --cuda-device 0 --silent \
                      --predictor-args '{"predict_answerability":true,"skip_distractor_paragraphs":false,"use_predicted_decomposition":true}'


# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/execution_by_end2end_model_for_musique_full/predictions/serialization_dir__execution_model_decomposer_for_musique_ans_and_full__predictions__musique_full_dev.jsonl

```

</details>
    
<details> <summary>
<strong>Execution by Select+Answer Model [EX(SA)]</strong>
</summary>

<br>The system has 3 parts given below: (i) Decomposer Model (ii) Selector of Executor Model (iii) Answerer of Executor Model.


----

### Experiment Name

```bash
# Decomposer Model
execution_model_decomposer_for_musique_ans_and_full
```

### Download model

```bash
python download_models.py execution_model_decomposer_for_musique_ans_and_full
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_model_decomposer_for_musique_ans_and_full.jsonnet \
                    --serialization-dir serialization_dir/execution_model_decomposer_for_musique_ans_and_full
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_model_decomposer_for_musique_ans_and_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_model_decomposer_for_musique_ans_and_full/model.tar.gz \
                      raw_data/musique_full_dev.jsonl \
                      --output-file serialization_dir/execution_model_decomposer_for_musique_ans_and_full/predictions/musique_full_dev.jsonl \
                      --predictor question_translator --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Selector of Executor Model
execution_by_select_and_answer_model_selector_for_musique_full
```

### Download model

```bash
python download_models.py execution_by_select_and_answer_model_selector_for_musique_full
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_select_and_answer_model_selector_for_musique_full.jsonnet \
                    --serialization-dir serialization_dir/execution_by_select_and_answer_model_selector_for_musique_full
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_select_and_answer_model_selector_for_musique_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_select_and_answer_model_selector_for_musique_full/model.tar.gz \
                      raw_data/musique_full_single_hop_version_train.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_selector_for_musique_full/predictions/musique_full_single_hop_version_train.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

python run.py predict serialization_dir/execution_by_select_and_answer_model_selector_for_musique_full/model.tar.gz \
                      raw_data/musique_full_single_hop_version_dev.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_selector_for_musique_full/predictions/musique_full_single_hop_version_dev.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Answerer of Executor Model
execution_by_select_and_answer_model_answerer_for_musique_full
```

### Download model

```bash
python download_models.py execution_by_select_and_answer_model_answerer_for_musique_full
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_select_and_answer_model_answerer_for_musique_full.jsonnet \
                    --serialization-dir serialization_dir/execution_by_select_and_answer_model_answerer_for_musique_full
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_select_and_answer_model_answerer_for_musique_full
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_select_and_answer_model_answerer_for_musique_full/model.tar.gz \
                      serialization_dir/execution_model_decomposer_for_musique_ans_and_full/predictions/musique_full_dev.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_answerer_for_musique_full/predictions/serialization_dir__execution_model_decomposer_for_musique_ans_and_full__predictions__musique_full_dev.jsonl \
                      --predictor multi_step_select_and_answer_transformer_rc --batch-size 16 --cuda-device 0 --silent \
                      --predictor-args '{"predict_answerability":true,"skip_distractor_paragraphs":false,"use_predicted_decomposition":true,"selector_model_path":"serialization_dir/execution_by_select_and_answer_model_selector_for_musique_full/model.tar.gz","num_select":3}'


# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/execution_by_select_and_answer_model_answerer_for_musique_full/predictions/serialization_dir__execution_model_decomposer_for_musique_ans_and_full__predictions__musique_full_dev.jsonl

```

</details>
    

## HotpotQA

<details> <summary>
<strong>End2End Model [EE]</strong>
</summary>


----

### Experiment Name

```bash
end2end_model_for_hotpotqa_20k_dataset
```

### Download model

```bash
python download_models.py end2end_model_for_hotpotqa_20k_dataset
```

### Train from scratch

```bash
python run.py train experiment_configs/end2end_model_for_hotpotqa_20k_dataset.jsonnet \
                    --serialization-dir serialization_dir/end2end_model_for_hotpotqa_20k_dataset
```

### Download prediction/s

```bash
python download_raw_predictions.py end2end_model_for_hotpotqa_20k_dataset
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/end2end_model_for_hotpotqa_20k_dataset/model.tar.gz \
                      raw_data/hotpotqa_dev_20k.jsonl \
                      --output-file serialization_dir/end2end_model_for_hotpotqa_20k_dataset/predictions/hotpotqa_dev_20k.jsonl \
                      --predictor transformer_rc --batch-size 16 --cuda-device 0 --silent

# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/end2end_model_for_hotpotqa_20k_dataset/predictions/hotpotqa_dev_20k.jsonl

```

</details>
    
<details> <summary>
<strong>Select+Answer Model [SA]</strong>
</summary>

<br>The system has 2 parts given below: (i) Selector Model (ii) Answerer Model.


----

### Experiment Name

```bash
# Selector Model
select_and_answer_model_selector_for_hotpotqa_20k
```

### Download model

```bash
python download_models.py select_and_answer_model_selector_for_hotpotqa_20k
```

### Train from scratch

```bash
python run.py train experiment_configs/select_and_answer_model_selector_for_hotpotqa_20k.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_selector_for_hotpotqa_20k
```

### Download prediction/s

```bash
python download_raw_predictions.py select_and_answer_model_selector_for_hotpotqa_20k
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/select_and_answer_model_selector_for_hotpotqa_20k/model.tar.gz \
                      raw_data/hotpotqa_train_20k.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_hotpotqa_20k/predictions/hotpotqa_train_20k.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

python run.py predict serialization_dir/select_and_answer_model_selector_for_hotpotqa_20k/model.tar.gz \
                      raw_data/hotpotqa_dev_20k.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_hotpotqa_20k/predictions/hotpotqa_dev_20k.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Answerer Model
select_and_answer_model_answerer_for_hotpotqa_20k
```

### Download model

```bash
python download_models.py select_and_answer_model_answerer_for_hotpotqa_20k
```

### Train from scratch

```bash
python run.py train experiment_configs/select_and_answer_model_answerer_for_hotpotqa_20k.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_answerer_for_hotpotqa_20k
```

### Download prediction/s

```bash
python download_raw_predictions.py select_and_answer_model_answerer_for_hotpotqa_20k
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/select_and_answer_model_answerer_for_hotpotqa_20k/model.tar.gz \
                      serialization_dir/select_and_answer_model_selector_for_hotpotqa_20k/predictions/hotpotqa_dev_20k.jsonl \
                      --output-file serialization_dir/select_and_answer_model_answerer_for_hotpotqa_20k/predictions/serialization_dir__select_and_answer_model_selector_for_hotpotqa_20k__predictions__hotpotqa_dev_20k.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/select_and_answer_model_answerer_for_hotpotqa_20k/predictions/serialization_dir__select_and_answer_model_selector_for_hotpotqa_20k__predictions__hotpotqa_dev_20k.jsonl

```

</details>
    

## 2WikiMultihopQA

<details> <summary>
<strong>End2End Model [EE]</strong>
</summary>


----

### Experiment Name

```bash
end2end_model_for_2wikimultihopqa_20k_dataset
```

### Download model

```bash
python download_models.py end2end_model_for_2wikimultihopqa_20k_dataset
```

### Train from scratch

```bash
python run.py train experiment_configs/end2end_model_for_2wikimultihopqa_20k_dataset.jsonnet \
                    --serialization-dir serialization_dir/end2end_model_for_2wikimultihopqa_20k_dataset
```

### Download prediction/s

```bash
python download_raw_predictions.py end2end_model_for_2wikimultihopqa_20k_dataset
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/end2end_model_for_2wikimultihopqa_20k_dataset/model.tar.gz \
                      raw_data/2wikimultihopqa_dev_20k.jsonl \
                      --output-file serialization_dir/end2end_model_for_2wikimultihopqa_20k_dataset/predictions/2wikimultihopqa_dev_20k.jsonl \
                      --predictor transformer_rc --batch-size 16 --cuda-device 0 --silent

# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/end2end_model_for_2wikimultihopqa_20k_dataset/predictions/2wikimultihopqa_dev_20k.jsonl

```

</details>
    
<details> <summary>
<strong>Select+Answer Model [SA]</strong>
</summary>

<br>The system has 2 parts given below: (i) Selector Model (ii) Answerer Model.


----

### Experiment Name

```bash
# Selector Model
select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset
```

### Download model

```bash
python download_models.py select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset
```

### Train from scratch

```bash
python run.py train experiment_configs/select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset
```

### Download prediction/s

```bash
python download_raw_predictions.py select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset/model.tar.gz \
                      raw_data/2wikimultihopqa_train_20k.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset/predictions/2wikimultihopqa_train_20k.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

python run.py predict serialization_dir/select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset/model.tar.gz \
                      raw_data/2wikimultihopqa_dev_20k.jsonl \
                      --output-file serialization_dir/select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset/predictions/2wikimultihopqa_dev_20k.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Answerer Model
select_and_answer_model_answerer_for_2wikimultihopqa_20k_dataset
```

### Download model

```bash
python download_models.py select_and_answer_model_answerer_for_2wikimultihopqa_20k_dataset
```

### Train from scratch

```bash
python run.py train experiment_configs/select_and_answer_model_answerer_for_2wikimultihopqa_20k_dataset.jsonnet \
                    --serialization-dir serialization_dir/select_and_answer_model_answerer_for_2wikimultihopqa_20k_dataset
```

### Download prediction/s

```bash
python download_raw_predictions.py select_and_answer_model_answerer_for_2wikimultihopqa_20k_dataset
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/select_and_answer_model_answerer_for_2wikimultihopqa_20k_dataset/model.tar.gz \
                      serialization_dir/select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset/predictions/2wikimultihopqa_dev_20k.jsonl \
                      --output-file serialization_dir/select_and_answer_model_answerer_for_2wikimultihopqa_20k_dataset/predictions/serialization_dir__select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset__predictions__2wikimultihopqa_dev_20k.jsonl \
                      --predictor transformer_rc --batch-size 16 --cuda-device 0 --silent

# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/select_and_answer_model_answerer_for_2wikimultihopqa_20k_dataset/predictions/serialization_dir__select_and_answer_model_selector_for_2wikimultihopqa_20k_dataset__predictions__2wikimultihopqa_dev_20k.jsonl

```

</details>
    
<details> <summary>
<strong>Execution by End2End Model [EX(EE)]</strong>
</summary>

<br>The system has 2 parts given below: (i) Decomposer Model (ii) Executor Model.


----

### Experiment Name

```bash
# Decomposer Model
execution_model_decomposer_for_2wikimultihopqa
```

### Download model

```bash
python download_models.py execution_model_decomposer_for_2wikimultihopqa
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_model_decomposer_for_2wikimultihopqa.jsonnet \
                    --serialization-dir serialization_dir/execution_model_decomposer_for_2wikimultihopqa
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_model_decomposer_for_2wikimultihopqa
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_model_decomposer_for_2wikimultihopqa/model.tar.gz \
                      raw_data/2wikimultihopqa_dev_20k.jsonl \
                      --output-file serialization_dir/execution_model_decomposer_for_2wikimultihopqa/predictions/2wikimultihopqa_dev_20k.jsonl \
                      --predictor question_translator --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Executor Model
execution_by_end2end_model_for_2wikimultihopqa
```

### Download model

```bash
python download_models.py execution_by_end2end_model_for_2wikimultihopqa
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_end2end_model_for_2wikimultihopqa.jsonnet \
                    --serialization-dir serialization_dir/execution_by_end2end_model_for_2wikimultihopqa
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_end2end_model_for_2wikimultihopqa
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_end2end_model_for_2wikimultihopqa/model.tar.gz \
                      serialization_dir/execution_model_decomposer_for_2wikimultihopqa/predictions/2wikimultihopqa_dev_20k.jsonl \
                      --output-file serialization_dir/execution_by_end2end_model_for_2wikimultihopqa/predictions/serialization_dir__execution_model_decomposer_for_2wikimultihopqa__predictions__2wikimultihopqa_dev_20k.jsonl \
                      --predictor multi_step_end2end_transformer_rc --batch-size 16 --cuda-device 0 --silent \
                      --predictor-args '{"predict_answerability":false,"skip_distractor_paragraphs":false,"use_predicted_decomposition":true}'


# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/execution_by_end2end_model_for_2wikimultihopqa/predictions/serialization_dir__execution_model_decomposer_for_2wikimultihopqa__predictions__2wikimultihopqa_dev_20k.jsonl

```

</details>
    
<details> <summary>
<strong>Execution by Select+Answer Model [EX(SA)]</strong>
</summary>

<br>The system has 3 parts given below: (i) Decomposer Model (ii) Selector of Executor Model (iii) Answerer of Executor Model.


----

### Experiment Name

```bash
# Decomposer Model
execution_model_decomposer_for_2wikimultihopqa
```

### Download model

```bash
python download_models.py execution_model_decomposer_for_2wikimultihopqa
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_model_decomposer_for_2wikimultihopqa.jsonnet \
                    --serialization-dir serialization_dir/execution_model_decomposer_for_2wikimultihopqa
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_model_decomposer_for_2wikimultihopqa
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_model_decomposer_for_2wikimultihopqa/model.tar.gz \
                      raw_data/2wikimultihopqa_dev_20k.jsonl \
                      --output-file serialization_dir/execution_model_decomposer_for_2wikimultihopqa/predictions/2wikimultihopqa_dev_20k.jsonl \
                      --predictor question_translator --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Selector of Executor Model
execution_by_select_and_answer_model_selector_for_2wikimultihopqa
```

### Download model

```bash
python download_models.py execution_by_select_and_answer_model_selector_for_2wikimultihopqa
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_select_and_answer_model_selector_for_2wikimultihopqa.jsonnet \
                    --serialization-dir serialization_dir/execution_by_select_and_answer_model_selector_for_2wikimultihopqa
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_select_and_answer_model_selector_for_2wikimultihopqa
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_select_and_answer_model_selector_for_2wikimultihopqa/model.tar.gz \
                      raw_data/2wikimultihopqa_single_hop_version_train_20k.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_selector_for_2wikimultihopqa/predictions/2wikimultihopqa_single_hop_version_train_20k.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

python run.py predict serialization_dir/execution_by_select_and_answer_model_selector_for_2wikimultihopqa/model.tar.gz \
                      raw_data/2wikimultihopqa_single_hop_version_dev.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_selector_for_2wikimultihopqa/predictions/2wikimultihopqa_single_hop_version_dev.jsonl \
                      --predictor inplace_text_ranker --batch-size 16 --cuda-device 0 --silent

```


----

### Experiment Name

```bash
# Answerer of Executor Model
execution_by_select_and_answer_model_answerer_for_2wikimultihopqa
```

### Download model

```bash
python download_models.py execution_by_select_and_answer_model_answerer_for_2wikimultihopqa
```

### Train from scratch

```bash
python run.py train experiment_configs/execution_by_select_and_answer_model_answerer_for_2wikimultihopqa.jsonnet \
                    --serialization-dir serialization_dir/execution_by_select_and_answer_model_answerer_for_2wikimultihopqa
```

### Download prediction/s

```bash
python download_raw_predictions.py execution_by_select_and_answer_model_answerer_for_2wikimultihopqa
```

### Predict with a trained or a downloaded model


```bash
python run.py predict serialization_dir/execution_by_select_and_answer_model_answerer_for_2wikimultihopqa/model.tar.gz \
                      serialization_dir/execution_model_decomposer_for_2wikimultihopqa/predictions/2wikimultihopqa_dev_20k.jsonl \
                      --output-file serialization_dir/execution_by_select_and_answer_model_answerer_for_2wikimultihopqa/predictions/serialization_dir__execution_model_decomposer_for_2wikimultihopqa__predictions__2wikimultihopqa_dev_20k.jsonl \
                      --predictor multi_step_select_and_answer_transformer_rc --batch-size 16 --cuda-device 0 --silent \
                      --predictor-args '{"predict_answerability":false,"skip_distractor_paragraphs":false,"use_predicted_decomposition":true,"selector_model_path":"serialization_dir/execution_by_select_and_answer_model_selector_for_2wikimultihopqa/model.tar.gz","num_select":3}'


# If you want to convert predictions to the official format, run:
python raw_predictions_to_official_format.py serialization_dir/execution_by_select_and_answer_model_answerer_for_2wikimultihopqa/predictions/serialization_dir__execution_model_decomposer_for_2wikimultihopqa__predictions__2wikimultihopqa_dev_20k.jsonl

```

</details>
    



# Citation

If you use this in your work, please cite use:

```
@article{trivedi2021musique,
  title={{M}u{S}i{Q}ue: Multihop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022}
  publisher={MIT Press}
}
```
