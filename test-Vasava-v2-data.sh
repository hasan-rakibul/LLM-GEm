#!/bin/bash

python ./Vasava-2022-Transformer-modified/test.py

# zip for submission
cd Vasava-2022-Transformer-modified
zip predictions.zip predictions_EMP.tsv
echo "predictions.zip created"