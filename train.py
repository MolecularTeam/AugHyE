import os
import random

import numpy as np
import torch

from config import parseArgs
from data_loader import training_dataset
from model.AugHyE import AugHyE_model

args = parseArgs()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(args):
    return AugHyE_model(args=args)


if __name__ == "__main__":
    set_seed(args['random_seed'])

    model = create_model(args)
    if torch.cuda.is_available():
        model.to(args['device'])
    print(f"Number of parameters = {model.size():,}")

    train_dataloader, val_dataloader = training_dataset(args)

    save_dir = os.path.join(args['save_path'], f"AugHyE_{args['timestamp']}")
    os.makedirs(save_dir, exist_ok=True)
    best_model_save_path = os.path.join(save_dir, f"AugHyE_best_{args['timestamp']}.pt")
    metrics_path = os.path.join(save_dir, 'metrics.jsonl')

    print(f"Timestamp: {args['timestamp']}")
    print(f"args: {str(args)}")
    print("Training start!!")
    model.train(train_dataloader, val_dataloader, best_model_save_path, last_metric_1=0.0,
                metrics_path=metrics_path)
    print(f"Best checkpoint: {best_model_save_path}")
