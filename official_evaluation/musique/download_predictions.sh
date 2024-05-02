#!/bin/bash

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

ZIP_NAME="musique_predictions_v1.0.zip"

# URL: https://drive.google.com/file/d/1XZocqLOTAu4y_1EeAj1JM4Xc1JxGJtx6/view?usp=sharing
gdown --id 1XZocqLOTAu4y_1EeAj1JM4Xc1JxGJtx6 --output $ZIP_NAME
unzip $(basename $ZIP_NAME)
rm $ZIP_NAME

# TODO: prevent these from zipping in.
rm -rf __MACOSX
