import argparse
import pandas as pd

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
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--anno_diff', type=float, default=4.5, help='anno_diff')
    parser.add_argument('--test_file', type=str, default='./data/PREPROCESSED-WS23-test.tsv', help='test file')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--save_id', type=str, default='ws23', help='save id')

    args = parser.parse_args()

    seed = args.seed
    anno_diff = args.anno_diff
    test_file = args.test_file
    gpu_id = args.gpu_id
    save_id = args.save_id

    # print the arguments passed in
    print('Arguments:')
    print('\tseed: ', seed)
    print('\tanno_diff: ', anno_diff)
    print('\ttest_file: ', test_file)
    print('\tgpu_id: ', gpu_id)
    print('\tsave_id: ', save_id)
    print('')

    load_model = './' + save_id + 'ckp/pearson-llm-roberta-seed-' + str(seed) + '-anno_diff-' + str(anno_diff) + '.pth'

    set_all_seeds(seed)

    data_module = DataModule(
        task=task,
        checkpoint=checkpoint,
        batch_size=16,
        feature_to_tokenise=feature_to_tokenise,
        seed=seed
    )

    print('Working with', test_file)
    test_loader = data_module.dataloader(file=test_file, send_label=False, shuffle=False)

    trainer = Trainer(
        task=task,
        checkpoint=checkpoint,
        lr=1e-5,
        n_epochs=10,
        train_loader=None,
        dev_loader=None,
        dev_label_gpt=None,
        dev_label_crowd=None,
        device_id=0,
        anno_diff=anno_diff,
        mode=0 # doesn't matter
    )

    print('Working with', load_model)
    pred = trainer.evaluate(dataloader=test_loader, load_model=load_model)
    pred_df = pd.DataFrame({'emp': pred, 'dis': pred}) # we're not predicting distress, just aligning with submission system
    pred_df.to_csv('./tmp/predictions_EMP.tsv', sep='\t', index=None, header=None)

if __name__ == '__main__':
    main()