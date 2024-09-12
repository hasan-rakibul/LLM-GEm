import numpy as np
import pandas as pd
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import Dataset

from evaluation import pearsonr
from utils import plot, get_device, set_all_seeds
from common import EarlyStopper

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # due to huggingface warning
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

class RobertaRegressor(nn.Module):
    def __init__(self, checkpoint):
        super(RobertaRegressor, self).__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):

        output = self.transformer(
            input_ids= input_ids,
            attention_mask=attention_mask,
        )
        return output


class DataModule:
    def __init__(self, task, checkpoint, batch_size, feature_to_tokenise, seed):

        self.task = task
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.tokeniser = AutoTokenizer.from_pretrained(
            self.checkpoint,
            use_fast=True
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokeniser)
        self.feature_to_tokenise = feature_to_tokenise # to tokenise function
        self.seed = seed

        assert len(self.task) == 1, 'task must be a list with one element'
    
    def _process_raw(self, path, send_label):
        data = pd.read_csv(path, sep='\t')
    
        if send_label:
            text = data[self.feature_to_tokenise + self.task]
        else:
            text = data[self.feature_to_tokenise]

        return text

    def _tokeniser_fn(self, sentence):
        if len(self.feature_to_tokenise) == 1: # only one feature
            return self.tokeniser(sentence[self.feature_to_tokenise[0]], truncation=True)
        # otherwise tokenise a pair of sentence
        return self.tokeniser(sentence[self.feature_to_tokenise[0]], sentence[self.feature_to_tokenise[1]], truncation=True)

    def _process_input(self, file, send_label):
        data = self._process_raw(path=file, send_label=send_label)
        data = Dataset.from_pandas(data, preserve_index=False) # convert to huggingface dataset
        data = data.map(self._tokeniser_fn, batched=True, remove_columns=self.feature_to_tokenise) # tokenise
        data = data.with_format('torch')
        return data


    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)     

    def dataloader(self, file, send_label, shuffle):
        data = self._process_input(file=file, send_label=send_label)

        # making sure the shuffling is reproducible
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=24,
            worker_init_fn=self._seed_worker,
            generator=g
        )

class Trainer:
    def __init__(self, task, model, lr, n_epochs, train_loader,
                 dev_loader, dev_label_file, device_id=0):
        self.device = get_device(device_id)
        self.task = task
        self.model = model.to(self.device)
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.dev_label_file = dev_label_file
        
        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-06,
            weight_decay=0.1
        )

        n_training_step = self.n_epochs*len(self.train_loader)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimiser,
            num_warmup_steps=0.06*n_training_step,
            num_training_steps=n_training_step
        )
        
        self.best_pearson_r = -1.0 # initiliasation
        self.early_stopper = EarlyStopper(patience=3, min_delta=0.01)
        
        assert len(self.task) == 1, 'task must be a list with one element'

    def _training_step(self):
        tr_loss = 0.0
        self.model.train()
    
        for data in self.train_loader:
            input_ids = data['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
            targets = data[self.task[0]].to(self.device, dtype=torch.float).view(-1, 1)
    
            outputs = self.model(input_ids, attention_mask)
            loss = self.loss_fn(outputs.logits, targets)
            tr_loss += loss.item()
    
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.lr_scheduler.step()

        epoch_loss = tr_loss / len(train_loader)
        print(f'Train loss: {epoch_loss}')

    def fit(self, save_model=False):
        dev_label = pd.read_csv(self.dev_label_file, sep='\t', header=None)
        if self.task[0] == 'empathy':
            true = dev_label.iloc[:, 0].tolist()
        if self.task[0] == 'distress':
            true = dev_label.iloc[:, 1].tolist()
            
        for epoch in range(self.n_epochs):
            print(f'Epoch: {epoch+1}')
            self._training_step()

            preds = self.evaluate(dataloader=self.dev_loader, load_model=False)
            
            pearson_r = pearsonr(true, preds)
            print(f'Pearson r: {pearson_r}')
            
            val_loss = self.loss_fn(torch.tensor(preds), torch.tensor(true))
            print('Validation loss:', val_loss.item())
            
            if self.early_stopper.early_stop(val_loss, model=self.model, save_as_loss=None):
                break
                
            if (pearson_r > self.best_pearson_r):
                self.best_pearson_r = pearson_r            
                if save_model:
                    torch.save(self.model.state_dict(), 'roberta-baseline.pth')
                    print("Saved the model in epoch " + str(epoch+1))
            
            print(f'Best dev set Pearson r: {self.best_pearson_r}\n')

    def evaluate(self, dataloader, load_model=False):
        if load_model:
            self.model.load_state_dict(torch.load('roberta-baseline.pth'))
    
        pred = torch.empty((len(dataloader.dataset), 1), device=self.device) # len(self.dev_loader.dataset) --> # of samples
        self.model.eval()
    
        with torch.no_grad():
            idx = 0
            for data in dataloader:
                input_ids = data['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
        
                outputs = model(input_ids, attention_mask)

                batch_size = outputs.logits.shape[0]
                pred[idx:idx+batch_size, :] = outputs.logits
                idx += batch_size
            
        return [float(k) for k in pred]   


checkpoint = 'roberta-base'
task = ['empathy']
# feature_to_tokenise=['demographic_essay', 'article']
feature_to_tokenise=['demographic', 'essay']
seed = 0

train_file = './data/not_final_but_important/PREPROCESSED-WS22-WS23-train.tsv'

# WASSA 2022
# dev_file = './data/PREPROCESSED-WS22-dev.tsv'
# dev_label_file = './data/WASSA22/goldstandard_dev_2022.tsv'
# test_file = './data/PREPROCESSED-WS22-test.tsv'

# WASSA 2023
dev_file = './data/not_final_but_important/PREPROCESSED-WS23-dev.tsv'
dev_label_file = './data/WASSA23/goldstandard_dev.tsv'
test_file = './data/not_final_but_important/PREPROCESSED-WS23-test.tsv'

set_all_seeds(seed)

data_module = DataModule(
    task=task,
    checkpoint=checkpoint,
    batch_size=16,
    feature_to_tokenise=feature_to_tokenise,
    seed=0
)

train_loader = data_module.dataloader(file=train_file, send_label=True, shuffle=True)
dev_loader = data_module.dataloader(file=dev_file, send_label=False, shuffle=False)
test_loader = data_module.dataloader(file=test_file, send_label=False, shuffle=False)

model = RobertaRegressor(checkpoint=checkpoint)

trainer = Trainer(
    task=task,
    model=model,
    lr=1e-5,
    n_epochs=10,
    train_loader=train_loader,
    dev_loader=dev_loader,
    dev_label_file=dev_label_file,
    device_id=0
)

trainer.fit(save_model=True)

pred = trainer.evaluate(dataloader=test_loader, load_model=True)
pred_df = pd.DataFrame({'emp': pred, 'dis': pred}) # we're not predicting distress, just aligning with submission system
pred_df.to_csv('./tmp/predictions_EMP.tsv', sep='\t', index=None, header=None)
