#!/bin/bash

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

mkdir -p .temp/
mkdir -p raw_data

echo "\n\nDownloading raw hotpotqa data\n"
mkdir -p raw_data/hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O raw_data/hotpotqa/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O raw_data/hotpotqa/hotpot_dev_distractor_v1.json

echo "\n\nDownloading raw 2wikimultihopqa data\n"
mkdir -p raw_data/2wikimultihopqa
wget https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=0 -O .temp/2wikimultihopqa.zip
unzip -jo .temp/2wikimultihopqa.zip -d raw_data/2wikimultihopqa -x "*.DS_Store"
rm data_ids.zip*

echo "\n\nDownloading raw musique data\n"
mkdir -p raw_data/musique
# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
gdown "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&confirm=t" -O .temp/musique_v1.0.zip
unzip -jo .temp/musique_v1.0.zip -d raw_data/musique -x "*.DS_Store"

echo "\n\nDownloading raw iirc data\n"
mkdir -p raw_data/iirc
wget https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz -O .temp/iirc_train_dev.tgz
tar -xzvf .temp/iirc_train_dev.tgz -C .temp/
mv .temp/iirc_train_dev/train.json raw_data/iirc/train.json
mv .temp/iirc_train_dev/dev.json raw_data/iirc/dev.json

echo "\n\nDownloading iirc wikipedia corpus (this will take 2-3 mins)\n"
wget https://iirc-dataset.s3.us-west-2.amazonaws.com/context_articles.tar.gz -O .temp/context_articles.tar.gz
tar -xzvf .temp/context_articles.tar.gz -C raw_data/iirc

echo "\n\nDownloading hotpotqa wikipedia corpus (this will take ~5 mins)\n"
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -O .temp/wikpedia-paragraphs.tar.bz2
tar -xvf .temp/wikpedia-paragraphs.tar.bz2 -C raw_data/hotpotqa
mv raw_data/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts raw_data/hotpotqa/wikpedia-paragraphs

rm -rf .temp/

# The resulting raw_data/ directory should look like:
# ── 2wikimultihopqa
# │   ├── dev.jsonl
# │   ├── id_aliases.json
# │   ├── test.json
# │   └── train.json
# ├── hotpotqa
# │   ├── dev_random_20_single_hop_annotations.txt
# │   ├── wikpedia-paragraphs/
# │   ├──  ├── ...
# │   ├── hotpot_dev_distractor_v1.json
# │   └── train_random_20_single_hop_annotations.txt
# ├── iirc
# │   ├── context_articles.json
# │   ├── dev.jsonl
# │   └── train.json
# └── musique
#     ├── dev_test_singlehop_questions_v1.0.json
#     ├── musique_ans_v1.0_dev.jsonl
#     ├── musique_ans_v1.0_test.jsonl
#     ├── musique_ans_v1.0_train.jsonl
#     ├── musique_full_v1.0_dev.jsonl
#     ├── musique_full_v1.0_test.jsonl
#     └── musique_full_v1.0_train.jsonl
