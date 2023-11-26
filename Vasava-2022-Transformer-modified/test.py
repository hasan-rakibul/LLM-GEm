# This code is taken from https://github.com/notprameghuikey0913/WASSA-2022-Empathy-detection-and-Emotion-Classification
# and modified to test the performance of their model on our improved WASSA-2022 dataset.

import torch
import torch.nn as nn
from torch import cuda
import math
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

device = 'cuda' if cuda.is_available() else 'cpu'

DEV = True # set to False to generate predictions for submission

if not DEV:
    testing_data = pd.read_csv("./data/PREPROCESSED-WS22-test.tsv", sep='\t')
if DEV:
    testing_data = pd.read_csv("./data/WS22-dev-gpt.tsv", sep='\t')
    crowd_label_file = pd.read_csv("./data/WASSA22/goldstandard_dev_2022.tsv", sep='\t', header=None)
    data3 = crowd_label_file[[0]]
    empathy_crowd = data3.reset_index(drop = True)
    targets_crowd = torch.tensor(empathy_crowd.to_numpy(), dtype=torch.float32)
    
    targets_gpt = torch.tensor(testing_data[['empathy']].to_numpy(), dtype=torch.float32)

data1 = testing_data[['demographic_essay']]
data2 = testing_data[['gender', 'education', 'race', 'age', 'income']]


data2 = pd.DataFrame(scaler.fit_transform(data2), columns=['gender', 'education', 'race', 'age', 'income'])
data1 = pd.concat([data1, data2], axis=1)
to_round = 4

MAX_LEN = 256
TEST_BATCH_SIZE = 8

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation = True, do_lower_case = True)

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.demographic_essay
        self.gender = dataframe.gender
        self.education = dataframe.education
        self.age = dataframe.age
        self.race = dataframe.race
        self.income = dataframe.income
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'gender': torch.tensor(self.gender[index], dtype=torch.float),
            'education': torch.tensor(self.education[index], dtype=torch.float),
            'race': torch.tensor(self.race[index], dtype=torch.float),
            'age': torch.tensor(self.age[index], dtype=torch.float),
            'income': torch.tensor(self.income[index], dtype=torch.float),
        }


test_data = data1.reset_index(drop=True)

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }


test_set = SentimentData(test_data, tokenizer, MAX_LEN)
test_loader = DataLoader(test_set, **test_params)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, 512)
        self.pre_final = torch.nn.Linear(517, 256)
        self.final = torch.nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, gender, education, race, age, income):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = nn.ReLU()(output)
        extra_inputs = torch.cat([output, gender, education, race, age, income], 1)
        output = self.pre_final(extra_inputs)
        output = nn.ReLU()(output)
        output = self.final(output)
        return output
    
model = RobertaClass()
model.to(device)
final = torch.empty((test_data.shape[0], 1))

def pearsonr(preds, targets):
    print(preds.shape)
    x = [float(k) for k in preds]
    y = [float(k) for k in targets]

    xm = sum(x) / len(x)
    ym = sum(y) / len(y)

    xn = [k - xm for k in x]
    yn = [k - ym for k in y]

    r = 0
    r_den_x = 0
    r_den_y = 0
    for xn_val, yn_val in zip(xn, yn):
        r += xn_val * yn_val
        r_den_x += xn_val * xn_val
        r_den_y += yn_val * yn_val

    r_den = math.sqrt(r_den_x * r_den_y)

    if r_den:
        r = r / r_den
    else:
        r = 0

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)

    return round(r, to_round)

def _label_fix(crowd, gpt, anno_diff):
    condition = torch.abs(crowd - gpt) > anno_diff
    crowd[condition] = gpt[condition]
    return crowd

def test(anno_diff=None):
    model.eval()
    tr_steps = 0

    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            gender = data['gender'].to(device, dtype=torch.float).view(-1, 1)
            education = data['education'].to(device, dtype=torch.float).view(-1, 1)
            race = data['race'].to(device, dtype=torch.float).view(-1, 1)
            age = data['age'].to(device, dtype=torch.float).view(-1, 1)
            income = data['income'].to(device, dtype=torch.float).view(-1, 1)

            outputs = model(ids, mask, token_type_ids, gender, education, race, age, income)
            print(tr_steps*TEST_BATCH_SIZE, tr_steps*TEST_BATCH_SIZE + TEST_BATCH_SIZE)
            if(tr_steps*TEST_BATCH_SIZE + TEST_BATCH_SIZE <= test_data.shape[0]):
                final[tr_steps*TEST_BATCH_SIZE : tr_steps*TEST_BATCH_SIZE + TEST_BATCH_SIZE, :] = outputs
            else:
                final[tr_steps*TEST_BATCH_SIZE:, :] = outputs
            tr_steps += 1

    if DEV:
        targets = _label_fix(crowd=targets_crowd, gpt=targets_gpt, anno_diff=anno_diff)
        corr = pearsonr(final, targets)
        return corr
    if not DEV:
        return

if DEV:
    dev_results = pd.DataFrame()
    anno_diff_range = np.arange(0, 6.5, 0.5)

    for anno_diff in anno_diff_range:
        print('anno_diff: ', anno_diff)
        PATH = 'Vasava-model-ws22-' + str(anno_diff) + '.pth'
        model.load_state_dict(torch.load(PATH))
        pearson_r = test(anno_diff=anno_diff)
        print('pearson_r:', pearson_r)
        dev_results.loc[anno_diff, 'pearson_r'] = pearson_r
    dev_results.to_csv("./Vasava-2022-Transformer-modified/dev_results.csv")

if not DEV:
    PATH = 'Vasava-model-ws22-0.0.pth'
    print('\nLoading', PATH, '\n')
    model.load_state_dict(torch.load(PATH))

    test()

    final_np = final.squeeze().numpy() # squeeze to remove extra dimension
    final_df = pd.DataFrame({'emp': final_np, 'dis': final_np}) # we're not predicting distress, just aligning with submission system
    final_df.to_csv("./Vasava-2022-Transformer-modified/predictions_EMP.tsv", sep='\t', index=None, header=None)

    print('predictions_EMP saved')