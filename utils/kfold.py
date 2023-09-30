import torch
from torch.utils.data import DataLoader

from evaluation import pearsonr
from utils.common import DataModule, Trainer, RobertaRegressor

class KFoldDataModule(DataModule):
    def __init__(self, task, checkpoint, batch_size, feature_to_tokenise, seed):
        super(KFoldDataModule, self).__init__(task, checkpoint, batch_size, feature_to_tokenise, seed)

    def get_data(self, file):
        return self._process_input(file=file, send_label=True)
    
    def kfold_dataloader(self, file, idx):
        '''dataloader for k-fold cross-validation'''
        data = self.get_data(file=file)
        
        # Sample elements randomly from a given list of ids, no replacement.
        subsampler = torch.utils.data.SubsetRandomSampler(idx)
        
        return DataLoader(
            data,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=12,
            sampler=subsampler
        )

class KFoldTrainer(Trainer):
    def __init__(self, task, checkpoint, lr, n_epochs, train_loader,
                 dev_loader, dev_label_crowd, dev_label_gpt, device_id,
                 anno_diff, mode):
        
        super(KFoldTrainer, self).__init__(task, checkpoint, lr, n_epochs, train_loader,
                 dev_loader, dev_label_crowd, dev_label_gpt, device_id, anno_diff, mode)

    def fit(self, dev_alpha=False): 
        '''dev_alpha: whether we want to change the dev annotation'''
        
        # Initialise the model because we want to train from scratch in each fold
        self.model = RobertaRegressor(checkpoint=self.checkpoint).to(self.device)
        
        for epoch in range(self.n_epochs):
            print(f'Epoch: {epoch+1}')
            self._training_step()
    
            pearson_r, val_loss = self.evaluate(dataloader=self.dev_loader, dev_alpha=dev_alpha)
            
            print(f'Pearson r: {pearson_r}')  
            print('Validation loss:', val_loss.item())
            
            if (pearson_r > self.best_pearson_r):
                self.best_pearson_r = pearson_r
            
            if self.early_stopper.early_stop(val_loss, model=self.model, save_as_loss=None):
                break
            
            print(f'Best Pearson r: {self.best_pearson_r}\n')
        return self.best_pearson_r

    def evaluate(self, dataloader, dev_alpha):
        '''dev_alpha: whether we want to change the dev annotation'''
    
        pred = torch.empty((len(dataloader.dataset), 1), device=self.device) # len(self.dev_loader.dataset) --> # of samples
        if dev_alpha:
            true_gpt = torch.empty((len(dataloader.dataset), 1), device=self.device) # len(self.dev_loader.dataset) --> # of samples
        true = torch.empty((len(dataloader.dataset), 1), device=self.device) # len(self.dev_loader.dataset) --> # of samples
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
                # true_gpt[idx:idx+batch_size, :] = data[self.task[0]].to(self.device, dtype=torch.float).view(-1, 1)
                true[idx:idx+batch_size, :] = data[self.task[1]].to(self.device, dtype=torch.float).view(-1, 1)
                idx += batch_size

            if dev_alpha:
                if self.mode == 0:
                    # gpt-label on the second fine tune 
                    true = self._label_fix(crowd=true, gpt=true_gpt)
                if self.mode == 1:
                    true = true_gpt

            true = [float(k) for k in true]
            pred = [float(k) for k in pred]
            
            pearson_r = pearsonr(true, pred)
            val_loss = self.loss_fn(torch.tensor(pred), torch.tensor(true))
        return pearson_r, val_loss