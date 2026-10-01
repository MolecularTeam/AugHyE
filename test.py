import json
import os

import torch

from config import parseArgs
from data_loader import test_dataset
from model.AugHyE import AugHyE_model

args = parseArgs()


def create_model(args):
    return AugHyE_model(args=args)


if __name__ == "__main__":
    model = create_model(args)
    if torch.cuda.is_available():
        model.to(args['device'])

    test_native_bound, test_generated, test_native_unbound = test_dataset(args)

    ckpt = args['ckpt']
    print(f"Checkpoint: {ckpt}")
    print("Test start!!")

    results = {}
    for structure_type, loader in (('native_bound', test_native_bound),
                                   ('generated', test_generated),
                                   ('native_unbound', test_native_unbound)):
        _, test_AP, test_AUC_median, out = model.evaluate(args, loader, ckpt, structure_type=structure_type)
        results[structure_type] = {'auc_median': test_AUC_median,
                                   'auc_mean': out['AUC_mean'],
                                   'ap_mean': test_AP}
        print(f"[{structure_type}] median AUC {test_AUC_median:.4f} | "
              f"mean AUC {out['AUC_mean']:.4f} | mean AP {test_AP:.4f}")

    out_path = os.path.join(args['save_path'], f"test_{args['timestamp']}.json")
    os.makedirs(args['save_path'], exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'checkpoint': ckpt, 'results': results}, f, indent=2)
    print(f"Saved: {out_path}")
