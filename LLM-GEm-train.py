import numpy as np
import pandas as pd
import argparse

from core.utils import set_all_seeds
from core.common import DataModule, Trainer

def main():
    checkpoint = 'roberta-base'
    task = ['empathy', 'wrong_empathy'] # empathy: LLM annotation, wrong_empathy: crowdsource annotation
    # feature_to_tokenise=['demographic_essay', 'article']
    # feature_to_tokenise=['demographic', 'essay']
    feature_to_tokenise=['demographic_essay']

    # parse arguments. Defaults are for v3 dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='0: crowd-gpt, 1: gpt, -1: crowd')
    parser.add_argument('--train_file', type=str, default='./data/WS22-WS23-augmented-train-gpt.tsv', help='train file')
    parser.add_argument('--dev_file', type=str, default='./data/WS23-dev-gpt.tsv', help='dev file w/ gpt label')
    parser.add_argument('--dev_label_crowd', type=str, default='./data/WASSA23/goldstandard_dev.tsv', help='dev label crowd')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--save_trained', type=bool, default=False, help='save trained model', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_id', type=str, default='ws23', help='save id')

    args = parser.parse_args()
    
    mode = args.mode
    train_file = args.train_file
    dev_file = args.dev_file
    dev_label_crowd = args.dev_label_crowd
    gpu_id = args.gpu_id
    save_trained = args.save_trained
    save_id = args.save_id

    # print the arguments passed in
    print('Arguments:')
    print('\tmode: ', mode)
    print('\ttrain_file: ', train_file)
    print('\tdev_file: ', dev_file)
    print('\tdev_label_crowd: ', dev_label_crowd)
    print('\tgpu_id: ', gpu_id)
    print('\tsave_trained: ', save_trained)
    print('\tsave_id: ', save_id)
    print('')

    seed_range = [0, 42, 100, 999, 1234]
    anno_diff_range = np.arange(0, 6.5, 0.5)

    val_results = pd.DataFrame()

    for seed in seed_range:

        set_all_seeds(seed)
        
        data_module = DataModule(
            task=task,
            checkpoint=checkpoint,
            batch_size=16,
            feature_to_tokenise=feature_to_tokenise,
            seed=seed
        )
        
        train_loader = data_module.dataloader(file=train_file, send_label=True, shuffle=True)
        dev_loader = data_module.dataloader(file=dev_file, send_label=False, shuffle=False)

        for anno_diff in anno_diff_range:
            trainer = Trainer(
                task=task,
                checkpoint=checkpoint,
                lr=1e-5,
                n_epochs=10,
                train_loader=train_loader,
                dev_loader=dev_loader,
                dev_label_gpt=dev_file,
                dev_label_crowd=dev_label_crowd,
                device_id=gpu_id,
                anno_diff=anno_diff,
                mode=mode
            )

            ## If we want to save model to use while testing
            if save_trained:
                save_as_loss = './' + save_id + 'ckp/loss-llm-roberta-seed-' + str(seed) + '-anno_diff-' + str(anno_diff) + '.pth'
                save_as_pearson = './' + save_id + 'ckp/pearson-llm-roberta-seed-' + str(seed) + '-anno_diff-' + str(anno_diff) + '.pth'
            else:
                save_as_loss = None
                save_as_pearson = None

            val_pearson_r = trainer.fit(save_as_loss=save_as_loss, save_as_pearson=save_as_pearson, dev_alpha=True)

            # save as seed in index and anno_diff in columns
            print(f'\n----Seed {seed}, anno_diff {anno_diff}: {val_pearson_r}----\n')
            val_results.loc[seed, anno_diff] = val_pearson_r

        # Saving in each seed to be cautious
        val_results.to_csv(save_id + '-val_results_diff_seed_anno_diff.tsv', sep='\t')

if __name__ == '__main__':
    main()