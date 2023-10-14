# LLM-GEm: Large Language Model-Guided Prediction of People's Empathy Levels towards Newspaper Article

# Useful files and folders
- `*.sh`: bash scripts to train and test in all three datasets. **These are the primary scripts to train and test the proposed model.**
	- `train-test-v1-data.sh` are for the NewsEmpathy v1 dataset
	- `train-v2-data.sh` and `test-v2-data.sh` are for the NewsEmpathy v2 (WASSA 2022) dataset
	- `train-v3-data.sh` and `test-v3-data.sh` are for the NewsEmpathy v3 (WASSA 2023) dataset
- `./core/`: classes, methods and other functions used in above notebooks
	- `LLM-GEm-train.py`: train the model
		- By default, the code is ready for NewsEmpathy v3 (WASSA 2023) dataset. 
		- To execute for the NewsEmpathy v2 (WASSA 2022) dataset, `train-v2-data.sh` includes the changed arguments.
	- `LLM-GEm-test.py`: test the model. It select between LLM and crowdsource annotation and predict empathy levels using the RoBERTa-MLP model
		- By default, the code is ready for NewsEmpathy v3 (WASSA 2023) dataset. 
		- To execute for the NewsEmpathy v2 (WASSA 2022) dataset, `test-v2-data.sh` includes the changed arguments.
	- `evaluation.py`: official evaluation file from WASSA workshop, consisting of pearson r calculation
	- `kfold.py` and `kfold-v1-data.py`: 10-fold cross-validation-based empathy level prediction on the NewsEmpathy v1 dataset
- `annotation-by-LLM.ipynb`: access LLM (GPT-3.5) to annotate the essays
	- requires `openai-api.txt` file, consisting of the api
- `data-preprocessing.ipynb`: all preprocessing, including using LLM to convert numerical demographic to text, rephrase texts, etc.
	- requires `openai-api.txt` file, consisting of the api
- `roberta-basic.ipynb`: RoBERTa without any MLP to predict empathy levels without any demographic numbers. Used in ablation study.
- `analysis-and-plots.ipynb`: analysis on LLM annotation consistency and producing other plots
- `./data/`: processed datasets
	- raw dataset (only required if you want to pre-process from scratch) can be downloaded from the following places: NewsEmpathy v1 from [wwbp/empathic\_reactions](https://github.com/wwbp/empathic_reactions), NewsEmpathy v2 from [WASSA 2022](https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-datasets) and NewsEmpathy v3 from [WASSA 2023](https://codalab.lisn.upsaclay.fr/competitions/11167#learn_the_details-datasets)
	- `./intermediate-files/`: generated during intermediate processing of the data but not required in the final training/validation/testing
- `*.tsv`: results from 10-fold cross-validation for the v1 dataset and different seed value for the v2 and v3 datasets
- Other folders in gitignore
	- `./ws22ckp/` and `./ws23ckp/`: to save checkpoints for the v2 and v3 datasets
	- `./tmp/`: temporary files of test results and zip file for submission to WASSA 2022 and 2023

# License
- The [v1 dataset](https://github.com/wwbp/empathic_reactions) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- The terms and conditions of [v2 dataset](https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-datasets) and [v3 dataset](https://codalab.lisn.upsaclay.fr/competitions/11167#learn_the_details-datasets) includes the following statement: "The dataset should only be used for scientific or research purposes. Any other use is explicitly prohibited."
- Our processed datasets are released under the above same licenses and conditions of the original datasets
- Our codes are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license

# Glossary
- `anno-diff`: The annotation selection threshold (alpha)
- `dev_alpha`: False (default) means not to change validation annotation; True means change validation annotation using the threshold
- `mode`: 0 for LLM-GEm, 1 for LLM only and -1 for crowdsource annotation only