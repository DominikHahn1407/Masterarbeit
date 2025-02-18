class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        #Stops training if validation loss doesn't improve after a set number of epochs.
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss  # Initialize best loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1  # Increment if no significant improvement
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping
        else:
            self.best_loss = val_loss  # Reset if improvement occurs
            self.counter = 0
