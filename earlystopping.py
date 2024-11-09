import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved!")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs of no improvement.")