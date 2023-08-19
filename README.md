# LLM-GEm: Large Language Model-Guided Prediction of People's Empathy Levels towards Newspaper Article

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
- `./data/`: processed datasets
	- raw dataset (only required if you want to pre-process from scratch) can be downloaded from the following places: NewsEmpathy v1 from [wwbp/empathic\_reactions](https://github.com/wwbp/empathic_reactions), NewsEmpathy v2 from [WASSA 2022](https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-datasets) and NewsEmpathy v3 from [WASSA 2023](https://codalab.lisn.upsaclay.fr/competitions/11167#learn_the_details-datasets)
	- `./intermediate-files/`: generated during intermediate processing of the data but not required in the final training/validation/testing
- `*.tsv`: results from 10-fold cross-validation for the v1 dataset and different seed value for the v2 and v3 datasets

# Glossary
- `anno-diff`: The annotation selection threshold (alpha)
- `dev_alpha`: False (default) means not to change validation annotation; True means change validation annotation using the threshold
- `mode`: 0 for LLM-GEm, 1 for LLM only and -1 for crowdsource annotation only

