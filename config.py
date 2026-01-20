import sys
import warnings
import datetime
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import torch

print('Parsing args')

def parseArgs():
    parser = argparse.ArgumentParser(description='PBI_prediction_AugHyE')    
    parser.add_argument('-device', type=str, default='0', help='device (e.g., "cuda:0", "cuda:1", or "cpu")')
    parser.add_argument('-timestamp', type=str, default=datetime.datetime.today().strftime("%Y%m%d%H%M%S"))
    # Data
    parser.add_argument('-bsp_threshold', default=8.0, type=float, required=False)
    parser.add_argument('-dataset', default='db5', type=str, required=False) # db5
    parser.add_argument('-data_path', default='data/', type=str, required=False)
    parser.add_argument('-bound_type', default='bound', type=str, required=False)  

    # Optim and Scheduler
    parser.add_argument('-n_epochs', default=30, type=int, required=False)
    parser.add_argument('-random_seed', default=123, type=int, required=False)
    parser.add_argument('-bs', type=int, default=4, required=False, help="Batch size")
    parser.add_argument('-lr', type=float, default=1e-4, required=False)
    parser.add_argument('-wd', type=float, default=1e-4, required=False)
    
    ### AugHyE
    parser.add_argument('-graph_nodes', type=str, default='residues', required=False, choices=['residues'])
    parser.add_argument('-SEGCN_layer', type=int, default=3, required=False)
    parser.add_argument('-h_dim', type=int, default=32, required=False) 
    parser.add_argument('-e_dim', type=int, default=32, required=False)
    parser.add_argument('-bern_k', type=int, default=10, required=False) 
    parser.add_argument('-sr_loss_ratio', type=float, default=0.35, required=False, help='0.35 default')
    parser.add_argument('-mu_r_norm', default=True, action='store_true')
    parser.add_argument('-res_feat', default=True, action='store_true')
    parser.add_argument('-atten_head', type=int, default=4, required=False) 
    parser.add_argument('-translation_interval', default=10.0, type=float, required=False, help='translation interval')
    parser.add_argument('-nonlin', type=str, default='lkyrelu', choices=['swish', 'lkyrelu'])
    parser.add_argument('-dp_encoder', type=float, default=0.2, required=False)  
    parser.add_argument('-dp_cls', type=float, default=0., required=False)
    parser.add_argument('-dropout', type=float, default=0.2)


    ## alignment equidock
    parser.add_argument('-rot_model', type=str, default='kb_att', choices=['kb_att'])
    parser.add_argument('-num_att_heads', type=int, default=50, required=False)
    parser.add_argument('-an_dropout', type=float, default=0., required=False)
    parser.add_argument('-layer_norm', type=str, default='LN', choices=['0', 'BN', 'LN'])
    parser.add_argument('-layer_norm_coors', type=str, default='0', choices=['0', 'LN'])
    parser.add_argument('-final_h_layer_norm', type=str, default='0', choices=['0', 'GN', 'BN', 'LN'])
    parser.add_argument('-iegmn_lay_hid_dim', type=int, default=64, required=False)
    parser.add_argument('-iegmn_n_lays', type=int, default=5, required=False)
    parser.add_argument('-residue_emb_dim', type=int, default=64, required=False, help='embedding') 
    parser.add_argument('-shared_layers', default=False, action='store_true')
    parser.add_argument('-cross_msgs', default=False, action='store_true')
    parser.add_argument('-divide_coors_dist', default=False, action='store_true')
    parser.add_argument('-use_dist_in_layers', default=False, action='store_true')
    parser.add_argument('-use_edge_features_in_gmn', default=False, action='store_true')
    parser.add_argument('-noise_decay_rate', type=float, default=0., required=False)
    parser.add_argument('-noise_initial', type=float, default=0., required=False)
    parser.add_argument('-use_mean_node_features', default=False, action='store_true')
    parser.add_argument('-skip_weight_h', type=float, default=0.75, required=False)
    parser.add_argument('-leakyrelu_neg_slope', type=float, default=0.5, required=False)
    parser.add_argument('-x_connection_init', type=float, default=0., required=False)
    parser.add_argument('-fine_tune', default=False, required=False)
    parser.add_argument('-input_edge_feats_dim', type=int, default=27, required=False)

    
    args = parser.parse_args().__dict__  
    device = torch.device(f"cuda:{args['device']}" if torch.cuda.is_available() else 'cpu')
    args['device'] = device
    print(f"Available GPUS:{torch.cuda.device_count()}")
    __all__ = ['args']
    return args
