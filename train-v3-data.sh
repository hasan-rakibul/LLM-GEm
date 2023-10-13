#!/bin/bash

python3 ./core/LLM-GEm-train.py \
# --train_file ./data/WS22-WS23-sep-from-aug-train-gpt.tsv \ # enabled --> w/o augmentation
--save_trained \
--save_id ws23