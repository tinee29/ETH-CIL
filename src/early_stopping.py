import torch
import torch.nn as nn
import torch.optim as optim
import os


from logger import logger

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, model_name = "model", checkpoint_dir = "checkpoints"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = float('inf')
        self.f1_score_max = 0
        self.model_name = model_name
        self.best_checkpoint = None
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, f1_score, model):
        score = f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
            self.counter = 0

    def clear_checkpoint(self):
        if self.best_checkpoint is not None:
            os.remove(self.best_checkpoint)
            self.best_checkpoint = None

    def save_checkpoint(self, f1_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Best F1 Score increased ({self.f1_score_max:.6f} --> {f1_score:.6f}).')
        self.f1_score_max = f1_score
        if f1_score > 0.45:
            logger.info(f'Saving model to {self.checkpoint_dir}/{self.model_name}-{f1_score:.6f}.pt')
            # remove previous best model
            self.clear_checkpoint()
            torch.save(model.state_dict(), f"{self.checkpoint_dir}/{self.model_name}-{f1_score:.6f}.pt")
            self.best_checkpoint = f"{self.checkpoint_dir}/{self.model_name}-{f1_score:.6f}.pt"