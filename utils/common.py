
# Written as per https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience # no. of times to allow for no improvement
        self.min_delta = min_delta # the min change to be counted as improvement
        self.counter = 0 # count number of not-improvement
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if (val_loss + self.min_delta) < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0 # reset counter when val_loss decreased at least by min_delta
        elif (val_loss + self.min_delta) > self.min_val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False