# LLM-GEm

# Useful files and folders
- `LLM-GEm.ipynb`: Select between LLM and crowdsource annotation and predict empathy levels using the RoBERTa-MLP model
	- By default, the code is ready for NewsEmpathy v3 (WASSA 2023) dataset. 
	- To execute for the NewsEmpathy v2 (WASSA 2022) dataset, we just need to provide the correct files names corresponding to the v2 dataset. To do this, values of the `dev_file`, `dev_label_crowd` and `dev_label_gpt` variables can be changed  easily done by uncommenting and commenting within the code.   
- `annotation-by-LLM.ipynb`: access LLM (GPT-3.5) to annotate the essays
	- requires `openai-api.txt` file, consisting of the api
- `data-preprocessing.ipynb`: all preprocessing, including using LLM to convert numerical demographic to text, rephrase texts, etc.
	- requires `openai-api.txt` file, consisting of the api
- `kfold.ipynb`: 10-fold cross-validation-based empathy level prediction on the NewsEmpathy v1 dataset
- `roberta-basic.ipynb`: RoBERTa without any MLP to predict empathy levels without any demographic numbers. Used in ablation study.
- `evaluation.py`: official evaluation file from WASSA workshop, consisting of pearson r calculation
- `analysis-and-plots.ipynb`: analysis on LLM annotation consistency and producing other plots
- `./utils/`: classes, methods and other functions used in above notebooks
- `./data/`: datasets (raw and processed)
- `*.tsv`: results from 10-fold cross-validation for the v1 dataset and different seed value for the v2 and v3 datasets

# Glossary
- `anno-diff`: The annotation selection threshold (alpha)
- `dev_alpha`: False (default): not to change validation annotation, True: change validation annotation using threshold
- `mode`: 0 for LLM-GEM, 1 for LLM only and -1 for crowdsource annotation only

