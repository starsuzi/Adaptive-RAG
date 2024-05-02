#!/bin/bash

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

ZIP_NAME="musique_v1.0.zip"

# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h --output $ZIP_NAME
unzip $(basename $ZIP_NAME)
rm $ZIP_NAME

# TODO: prevent these from zipping in.
rm -rf __MACOSX
