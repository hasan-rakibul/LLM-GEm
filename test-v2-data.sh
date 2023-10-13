#!/bin/bash

python3 LLM-GEm-test.py \
--anno_diff 0.0 \
--test_file ./data/PREPROCESSED-WS22-test.tsv \
--save_id ws22

# zip for submission
cd tmp
zip predictions.zip predictions_EMP.tsv
echo "predictions.zip created"