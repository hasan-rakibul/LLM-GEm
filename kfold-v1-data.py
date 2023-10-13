import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from core.utils import set_all_seeds
from core.kfold import KFoldDataModule, KFoldTrainer

def main():
    checkpoint = 'roberta-base'
    task = ['empathy', 'wrong_empathy']
    feature_to_tokenise=['demographic_essay']
    seed = 0
    anno_diff_range = np.arange(0, 6.5, 0.5)

    #################### v1 dataset ###################
    filename = './data/v1-90-percent.tsv'

    kfold_results = pd.DataFrame()

    set_all_seeds(seed)

    data_module = KFoldDataModule(
        task=task,
        checkpoint=checkpoint,
        batch_size=16,
        feature_to_tokenise=feature_to_tokenise,
        seed=seed
    )

    data = data_module.get_data(file=filename)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
        print('Fold:', fold)
        
        train_loader = data_module.kfold_dataloader(
            file=filename, idx=train_idx
        )
        dev_loader = data_module.kfold_dataloader(
            file=filename, idx=test_idx
        )

        for anno_diff in anno_diff_range:
            trainer = KFoldTrainer(
                task=task,
                checkpoint=checkpoint,
                lr=1e-5,
                n_epochs=10,
                train_loader=train_loader,
                dev_loader=dev_loader,
                dev_label_gpt=filename,
                dev_label_crowd=None,
                device_id=0,
                anno_diff=anno_diff,
                mode=0 # -1: crowd, 1: gpt, 0: crowd-gpt
            )
        
            val_pearson_r = trainer.fit(dev_alpha=True)
        
            # save as seed in index and anno_diff in columns
            print(f'\n----Pearson r: {val_pearson_r}----\n')
            kfold_results.loc[fold, anno_diff] = val_pearson_r

        # # Saving in each fold to be cautious
        kfold_results.to_csv('./v1-kfold_results_anno_diff_dev_alpha_True.tsv', sep='\t')


if __name__ == '__main__':
    main()