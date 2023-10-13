#!/bin/bash

python3 ./core/LLM-GEm-train.py \
--dev_file ./data/WS22-dev-gpt.tsv \
--dev_label_crowd ./data/WASSA22/goldstandard_dev_2022.tsv \
--save_trained \
--save_id ws22