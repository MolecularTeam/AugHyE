import os
from data_loader import training_dataset
import torch 
from model.AugHyE import *
from config import parseArgs

args = parseArgs()  
def create_model(args):
    return AugHyE_model(args=args)


if __name__ == "__main__":

    model = create_model(args)
    if torch.cuda.is_available():
        model.to(args['device'])

    train_dataloader, val_dataloader = training_dataset(args)
    
    # model save
    save_dir = f"save/"
    os.makedirs(save_dir, exist_ok=True)
    best_model_save_path = save_dir + f"bsp_best_{args['timestamp']}.pt"
        
    print(f"Timestamp: {args['timestamp']}")
    print(f"args: {str(args)}")
    print("Training start!!")
    model.train(train_dataloader, val_dataloader, best_model_save_path, last_metric_1=0.0)
