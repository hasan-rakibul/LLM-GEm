import numpy as np
import pandas as pd
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

from sklearn.preprocessing import MinMaxScaler

import os
os.chdir("/g/data/jr19/rh2942/text-empathy/")
from evaluation import pearsonr
from utils.utils import plot, get_device

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # due to huggingface warning
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

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

        assert len(self.task) == 2, 'task must be a list with two elements'
    
    def _process_raw(self, path, send_label):
        data = pd.read_csv(path, sep='\t')
    
        if send_label:
            text = data[self.feature_to_tokenise + self.task]
        else:
            text = data[self.feature_to_tokenise]

        demog = ['gender', 'education', 'race', 'age', 'income']
        data_demog = data[demog]
        scaler = MinMaxScaler()
        data_demog = pd.DataFrame(
            scaler.fit_transform(data_demog),
            columns=demog
        )
        data = pd.concat([text, data_demog], axis=1) 
        return data

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
            num_workers=12,
            worker_init_fn=self._seed_worker,
            generator=g
        )
        

class RobertaRegressor(nn.Module):
    def __init__(self, checkpoint):
        super(RobertaRegressor, self).__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=768)
        self.fc1 = nn.Sequential(
            nn.Linear(768, 512), nn.Tanh(), nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512+5, 256), nn.Tanh(), nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        gender=None,
        education=None,
        race=None,
        age=None,
        income=None
    ):

        output = self.transformer(
            input_ids= input_ids,
            attention_mask=attention_mask,
        )

        output = self.fc1(output.logits)
        output = torch.cat([output, gender, education, race, age, income], 1)
        output = self.fc2(output)
        return output

# Written as per https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience # no. of times to allow for no improvement
        self.min_delta = min_delta # the min change to be counted as improvement
        self.counter = 0 # count number of not-improvement
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss, model, save_as_loss):
        if (val_loss + self.min_delta) < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0 # reset counter when val_loss decreased at least by min_delta

            # Saving the model
            if save_as_loss is not None:
                torch.save(model.state_dict(), save_as_loss)
                print('Saved the finetuned model (loss): ', save_as_loss)
            
        elif (val_loss + self.min_delta) > self.min_val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:
    def __init__(self, task, checkpoint, lr, n_epochs, train_loader,
                 dev_loader, dev_label_crowd, dev_label_gpt, device_id,
                 anno_diff, mode):
        self.device = get_device(device_id)
        self.task = task
        self.checkpoint = checkpoint
        self.mode = mode 
        assert self.mode in [-1, 0, 1], 'mode must be --> -1: crowd, 1: gpt, 0: crowd-gpt'
        
        self.model = RobertaRegressor(checkpoint=self.checkpoint).to(self.device)
    
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.dev_label_crowd = dev_label_crowd
        self.dev_label_gpt = dev_label_gpt
        
        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-06,
            weight_decay=0.1
        )

        # train_loader can be None during test time.
        if self.train_loader is not None: 
            n_training_step = self.n_epochs * len(self.train_loader)
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimiser,
                num_warmup_steps=0.06*n_training_step,
                num_training_steps=n_training_step
            )
        
        self.best_pearson_r = -1.0 # initiliasation
        self.early_stopper = EarlyStopper(patience=3, min_delta=0.01)
        
        self.anno_diff = anno_diff
        
        assert len(self.task) == 2, 'task must be a list with two elements'

    def _label_fix(self, crowd, gpt):
        condition = torch.abs(crowd.detach() - gpt.detach()) > self.anno_diff
        label = crowd.clone()
        label[condition] = gpt[condition]
        return label

    def _training_step(self):
        tr_loss = 0.0
        self.model.train()
    
        for data in self.train_loader:
            input_ids = data['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
            
            gpt = data[self.task[0]].to(self.device, dtype=torch.float).view(-1, 1)
            targets = data[self.task[1]].to(self.device, dtype=torch.float).view(-1, 1)
            
            gender = data['gender'].to(self.device, dtype=torch.float).view(-1, 1)
            education = data['education'].to(self.device, dtype=torch.float).view(-1, 1)
            race = data['race'].to(self.device, dtype=torch.float).view(-1, 1)
            age = data['age'].to(self.device, dtype=torch.float).view(-1, 1)
            income = data['income'].to(self.device, dtype=torch.float).view(-1, 1)
    
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gender=gender,
                education=education,
                race=race,
                age=age,
                income=income
            )

            if self.mode == 0:
                targets = self._label_fix(crowd=targets, gpt=gpt)
            if self.mode == 1:
                targets = gpt
            
            loss = self.loss_fn(outputs, targets)
            tr_loss += loss.item()
    
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.lr_scheduler.step()

        epoch_loss = tr_loss / len(self.train_loader)
        print(f'Train loss: {epoch_loss}')

    def fit(self, save_as_loss, save_as_pearson):
        dev_label_anno = pd.read_csv(self.dev_label_gpt, sep='\t')
        true_gpt = dev_label_anno.loc[:, 'empathy'].tolist()
        dev_label_crowd = pd.read_csv(self.dev_label_crowd, sep='\t', header=None)
        true = dev_label_crowd.iloc[:, 0].tolist()

        if self.mode == 0:
            # gpt-label on the second fine tune 
            true = self._label_fix(crowd=torch.tensor(true), gpt=torch.tensor(true_gpt))
            true = true.tolist()
        if self.mode == 1:
            true = true_gpt

        for epoch in range(self.n_epochs):
            print(f'Epoch: {epoch+1}')
            self._training_step()

            preds = self.evaluate(dataloader=self.dev_loader, load_model=None)
            
            pearson_r = pearsonr(true, preds)
            print(f'Pearson r: {pearson_r}')
            
            val_loss = self.loss_fn(torch.tensor(preds), torch.tensor(true))
            print('Validation loss:', val_loss.item())
            
            if (pearson_r > self.best_pearson_r):
                self.best_pearson_r = pearson_r
            
            if self.early_stopper.early_stop(val_loss, model=self.model, save_as_loss=save_as_loss):
                break

            # The following will ensure saving the model on best pearson r if not early stopped due to bad val loss
            if (pearson_r == self.best_pearson_r) and save_as_pearson is not None:
                torch.save(self.model.state_dict(), save_as_pearson)
                print("Saved the finetuned model (Pearson): ", save_as_pearson)
            
            print(f'Best dev set Pearson r: {self.best_pearson_r}\n')
        return self.best_pearson_r

    def evaluate(self, dataloader, load_model=None):
        if load_model is not None:
            self.model.load_state_dict(torch.load(load_model))
    
        pred = torch.empty((len(dataloader.dataset), 1), device=self.device) # len(self.dev_loader.dataset) --> # of samples
        self.model.eval()
    
        with torch.no_grad():
            idx = 0
            for data in dataloader:
                input_ids = data['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
                gender = data['gender'].to(self.device, dtype=torch.float).view(-1, 1)
                education = data['education'].to(self.device, dtype=torch.float).view(-1, 1)
                race = data['race'].to(self.device, dtype=torch.float).view(-1, 1)
                age = data['age'].to(self.device, dtype=torch.float).view(-1, 1)
                income = data['income'].to(self.device, dtype=torch.float).view(-1, 1)
        
                outputs = self.model(input_ids, attention_mask, gender, education, race, age, income)

                batch_size = outputs.shape[0]
                pred[idx:idx+batch_size, :] = outputs
                idx += batch_size
            
        return [float(k) for k in pred]   