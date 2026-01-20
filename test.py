import os
from data_loader import dataset_selection
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

    _, _, test_dataloader_native_bound, test_dataloader_unbound, test_dataloader_native_unbound = dataset_selection(args)

    # model load
    load_dir = f"model_weight/"  # local dir
    os.makedirs(load_dir, exist_ok=True)
    best_model_save_path = load_dir + f"AugHyE_best.pt"
        
    print(f"Timestamp: {args['timestamp']}")
    print("Test start!!")
    print("Native bound")
    test_loss, test_AP, test_AUC_median = \
        model.evaluate(args, test_dataloader_native_bound, best_model_save_path, bound_type="native_bound") 
    print(f"test_AP: {test_AP}, test_AUC: {test_AUC_median}")   

    print("Unbound")
    test_loss, test_AP, test_AUC_median = \
        model.evaluate(args, test_dataloader_unbound, best_model_save_path, bound_type="unbound")
    print(f"test_AP: {test_AP}, test_AUC: {test_AUC_median}")

    print("Native unbound")
    test_loss, test_AP, test_AUC_median = \
        model.evaluate(args, test_dataloader_native_unbound, best_model_save_path, bound_type="native_unbound")
    print(f"test_AP: {test_AP}, test_AUC: {test_AUC_median}")
