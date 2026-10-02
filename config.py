import argparse
import datetime
import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)


def parseArgs(argv=None):
    parser = argparse.ArgumentParser(description='AugHyE')
    parser.add_argument('-device', type=str, default='0')
    parser.add_argument('-timestamp', type=str,
                        default=datetime.datetime.today().strftime("%Y%m%d%H%M%S"))

    parser.add_argument('-dataset', default='db5', type=str, required=False)
    parser.add_argument('-data_path', default='data/', type=str, required=False)
    parser.add_argument('-bsp_threshold', default=8.0, type=float, required=False)
    parser.add_argument('-translation_interval', default=5.0, type=float, required=False)

    parser.add_argument('-n_epochs', default=30, type=int, required=False)
    parser.add_argument('-random_seed', default=123, type=int, required=False)
    parser.add_argument('-bs', type=int, default=4, required=False)
    parser.add_argument('-lr', type=float, default=1e-4, required=False)
    parser.add_argument('-wd', type=float, default=1e-4, required=False)
    parser.add_argument('-sr_loss_ratio', type=float, default=0.35, required=False)

    parser.add_argument('-h_dim', type=int, default=32, required=False)
    parser.add_argument('-hidden_dim', type=int, default=16, required=False)
    parser.add_argument('-SEGCN_layer', type=int, default=3, required=False)
    parser.add_argument('-bern_k', type=int, default=10, required=False)
    parser.add_argument('-max_radius', type=float, default=8.0, required=False)
    
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-dp_encoder', type=float, default=0.3, required=False)
    parser.add_argument('-dp_cls', type=float, default=0.2, required=False)
    parser.add_argument('-nonlin', type=str, default='lkyrelu', choices=['swish', 'lkyrelu'])
    parser.add_argument('-leakyrelu_neg_slope', type=float, default=1.0, required=False)

    parser.add_argument('-ckpt', type=str,
                        default='model_weight/AugHyE_model.pt', required=False)
    parser.add_argument('-alignment_ckpt', type=str,
                        default='model_weight/alignment_model.pth', required=False)

    parser.add_argument('-save_path', type=str, default='save/', required=False)

    args = parser.parse_args(argv).__dict__
    args['device'] = torch.device(f"cuda:{args['device']}"
                                  if torch.cuda.is_available() and args['device'] != 'cpu'
                                  else 'cpu')
    print(f"Available GPUs: {torch.cuda.device_count()}, using {args['device']}")
    return args
