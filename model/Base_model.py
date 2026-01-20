import os
import sys
sys.path.append(".")
import numpy as np 
from tqdm import tqdm
from torchmetrics.functional import average_precision, accuracy
from torchmetrics import AUROC, Precision, Accuracy, Recall, Specificity, F1Score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class Base_model(nn.Module): 
    def __init__(self, args, model_args, name="Baseline_model"): 
        super(Base_model, self).__init__()  
        self._name = name
        self.args = args
        self.model_args = model_args
        pass
    
    @property
    def name(self): 
        return self._name

    def pretrain(self, train_loader, val_loader, epochs=1, batch_size=4): 
        pass

    def train(self, train_loader, val_loader, epochs=1, batch_size=4): 
        pass

    def save(self, best_model_save_path): 
        torch.save(self.net.state_dict(), best_model_save_path)
        print('Save bip model')
        print(f"Timestamp: {self.args['timestamp']}")
    
    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.args['device'])
        self.net.load_state_dict(checkpoint, strict=False)  
        print('Load bip model')
    
    def evaluate(self, args, data_loader, best_model_save_path, epoch_id=0): 
        pass
