# 2WikiMultihopQA: A Dataset for Comprehensive Evaluation of Reasoning Steps

This is the repository for the paper: [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps](https://www.aclweb.org/anthology/2020.coling-main.580/) (COLING 2020).



### New Update (April 7, 2021)
- We release the [para_with_hyperlink.zip](https://www.dropbox.com/s/wlhw26kik59wbh8/para_with_hyperlink.zip) file that contains all articles/paragraphs with hyperlink information (except for some error paragraphs). Each paragraph has the following information:
  * ```id```: Wikipedia id of an article
  * ```title```
  * ```sentences```: a list of sentence
  * ```mentions```: a list of hyperlink information, each element has the following information: ```id```, ```start```, ```end```, ```ref_url```, ```ref_ids```, ```sent_idx```. 
- [Here](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip) is the link of the dataset that has fixed the inconsistency of sentence segmentation in paragraphs. 
 

### Update on December 8, 2020
- Due to the multiple names of an entity in Wikidata, we add ```evidences_id``` and ```answer_id``` to our dataset. Here are the details:
  * For Inference and Compositional questions: we add to all questions.
  * For Comparison and Bridge_comparison questions: we add to questions that have relations: ```country```, ```country of origin```, and ```country of citizenship```.
- We update the new evaluation script [2wikimultihop_evaluation_v1.1.py](https://github.com/Alab-NII/2wikimultihop/blob/main/2wikimultihop_evaluate_v1.1.py). We can use this evaluation script to evaluate the dataset with ```evidences_id``` and ```answer_id```.
- We also update the results of the baseline model by using the new evaluation script and the dataset with ```evidences_id``` and ```answer_id```. The updated results of tables 5, 6, and 7 in the paper are in the folder [update_results](https://github.com/Alab-NII/2wikimultihop/tree/main/update_results).
- [Here](https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=0) is the link of the dataset with ```evidences_id``` and ```answer_id```.  File ```id_aliases.json``` is used for evaluation.


### Leaderboard 

| Date | Model| Ans <br> EM | Ans <br> F1 | Sup <br> EM | Sup <br> F1 | Evi <br> EM | Evi <br> F1 | Joint <br> EM | Joint <br> F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oct 29, 2021 | [NA-Reviewer](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.12411) | 76.73 |  81.91 |  89.61 |  94.31 | 53.66 | 70.83 |  52.75 | 65.23 |
| Oct 26, 2021 | [CRERC](https://arxiv.org/abs/2110.13472) | 69.58 |  72.33 |  82.86 |  90.68 | 54.86 | 68.83 |  49.80 | 58.99 |
| Jan 12, 2022 | BigBird-base model - Weighted (Anonymous) | 73.04 |  78.90 |  76.92 |  91.95 | 45.05 | 76.13 |  38.72 | 62.33 |
| Jan 12, 2022 | BigBird-base model - Unweighted (Anonymous) | 72.38 |  77.98 |  75.68 |  91.56 | 35.07 | 71.09 |  29.86 | 57.74 |
| June 14, 2021 | BigBird-base model (Anonymous) | 71.42 |  77.64 |  73.84 |  90.68 | 24.64 | 63.69 |  21.37 | 51.44 |
| Dec 11, 2021 |RoBERTa-base (Anonymous)  | 32.24 | 40.90 | 40.91 | 71.85 | 13.80 | 41.37 | 6.92 |  20.54 |
| Oct 25, 2020 | [Baseline model](https://www.aclweb.org/anthology/2020.coling-main.580.pdf) | 36.53 | 43.93 | 24.99 | 65.26 | 1.07 | 14.94 | 0.35 | 5.41 |
| July 30, 2021 | HGN-revise model (Anonymous) | 71.20 |  75.69 |  69.35 |  89.07 | x | x |  x | x |
  


### Submission guide

To evaluate your model on the test data, please contact us.
Please prepare the following information: 


1. Your prediction file (follow format in file: [prediction_format.json](https://github.com/Alab-NII/2wikimultihop/blob/main/prediction_format.json))
2. The name of your model
3. Public repository of your model (optional)
4. Reference to your publication (optional)



### Dataset Contents
The full dataset is in [here](https://www.dropbox.com/s/npidmtadreo6df2/data.zip).

Our dataset follows the format of HotpotQA.
Each sample has the following keys:
- ```_id```: a unique id for each sample
- ```question```: a string
- ```answer```: an answer to the question. The test data does not have this information.
- ```supporting_facts```: a list, each element is a list that contains: ```[title, sent_id]```, ```title``` is the title of the paragraph, ```sent_id``` is the sentence index (start from 0) of the sentence that the model uses. The test data does not have this information.
- ```context```: a list, each element is a list that contains ```[title, setences]```, ```sentences``` is a list of sentences.
- ```evidences```: a list, each element is a triple that contains ```[subject entity, relation, object entity]```. The test data does not have this information.
- ```type```: a string, there are four types of questions in our dataset: comparison, inference, compositional, and bridge-comparison.
- ```entity_ids```: a string that contains the two Wikidata ids (four for bridge_comparison question) of the gold paragraphs, e.g., 'Q7320430_Q51759'.





### Baseline model 
Our baseline model is based on the baseline model in HotpotQA. The process to train and test is quite similar to HotpotQA.

- Process train data
```
python3 main.py --mode prepro --data_file wikimultihop/train.json --para_limit 2250 --data_split train

```

- Process dev data
```
python3 main.py --mode prepro --data_file wikimultihop/dev.json --para_limit 2250 --data_split dev

```

- Train a model
```
python3 -u main.py --mode train --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0
```

- Evaluation on dev (Local Evaluation)

```
python3 main.py --mode test --data_split dev --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 --save WikiMultiHop-20201024-023745 --prediction_file predictions/wikimultihop_dev_pred.json

```

```
python3 2wikimultihop_evaluate.py predictions/wikimultihop_dev_pred.json data/dev.json
```

- Use new evaluation script

```
python3 2wikimultihop_evaluate_v1.1.py predictions/wikimultihop_dev_pred.json data_ids/dev.json id_aliases.json
```



### Citation
If you plan to use the dataset, please cite our paper:

```
@inproceedings{xanh2020_2wikimultihop,
    title = "Constructing A Multi-hop {QA} Dataset for Comprehensive Evaluation of Reasoning Steps",
    author = "Ho, Xanh  and
      Duong Nguyen, Anh-Khoa  and
      Sugawara, Saku  and
      Aizawa, Akiko",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.580",
    pages = "6609--6625",
}
```


### References
The baseline model and the evaluation script are adapted from https://github.com/hotpotqa/hotpot

