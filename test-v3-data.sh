#!/bin/bash

python3 LLM-GEm-test.py \
--anno_diff 0.0 \

# zip for submission
cd tmp
zip predictions.zip predictions_EMP.tsv
echo "predictions.zip created"