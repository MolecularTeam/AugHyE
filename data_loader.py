import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import os, random, dill, gzip, shutil, pickle
from scipy.spatial.transform import Rotation
from sklearn.neighbors import BallTree
from biopandas.pdb import PandasPdb
import torch
from torch.utils.data import DataLoader
# from data_atprot_graph_construction import DockingDataset_aug, DockingDataset_aug_align, batchify_and_create_respective_graphs
# import SASNet.data_loader
import dgl
import math

from alignment_network.rigid_docking_model import *

def create_model_equidock(args, log=None):
    # assert 'input_edge_feats_dim' in args.keys(), 'get_loader has to be called before create_model.'
    return Rigid_Body_Docking_Net(args=args, log=log)


## atprot만 사용
class DockingDataset_aug_align(Dataset):

    def __init__(self, args, alignment_model, data_set, reload_mode, raw_data_path, load_from_cache=True, bound_type='bound'):
        self.args = args
        self.dataset = data_set
        self.alignment_model = alignment_model
        self.reload_mode = reload_mode

        if load_from_cache: 
            if reload_mode == "train":                                  
                self.data_unbound = pickle.load(open(os.path.join(raw_data_path, reload_mode + f'_unbound_esm3.pkl'), 'rb'))
                self.data_bound = pickle.load(open(os.path.join(raw_data_path, reload_mode + f'.pkl'), 'rb'))
                
                ### augmented unbound structure alignment
                # alignment network           
                if args['u_b_align'] == "equidock":
                    self.unbound_indices = np.arange(len(self.data_unbound))    
                    with torch.inference_mode():  
                        for i in range(len(self.data_unbound)):
                            
                            rot_T_ligand, rot_b_ligand = UniformRotation_Translation(translation_interval=self.args['translation_interval'])
                            rot_T_receptor, rot_b_receptor = UniformRotation_Translation(translation_interval=self.args['translation_interval'])

                            ligand_original_loc = self.data_unbound[i]['lig_graph'].ndata['x'].detach().numpy()
                            receptor_original_loc = self.data_unbound[i]['rec_graph'].ndata['x'].detach().numpy()
                            bound_ligand_original_loc = self.data_bound[i]['lig_graph'].ndata['x'].detach().numpy()
                            bound_receptor_original_loc = self.data_bound[i]['rec_graph'].ndata['x'].detach().numpy()
                            
                            ligand_mean_to_remove = ligand_original_loc.mean(axis=0, keepdims=True)
                            receptor_mean_to_remove = receptor_original_loc.mean(axis=0, keepdims=True)
                            bound_ligand_mean_to_remove = bound_ligand_original_loc.mean(axis=0, keepdims=True)
                            bound_receptor_mean_to_remove = bound_receptor_original_loc.mean(axis=0, keepdims=True)

                            ligand_new_loc = (rot_T_ligand @ (ligand_original_loc - ligand_mean_to_remove).T).T + rot_b_ligand
                            receptor_new_loc = (rot_T_receptor @ (receptor_original_loc - receptor_mean_to_remove).T).T + rot_b_receptor

                            bound_ligand_new_loc = (rot_T_ligand @ (bound_ligand_original_loc - bound_ligand_mean_to_remove).T).T + rot_b_ligand
                            bound_receptor_new_loc = (rot_T_receptor @ (bound_receptor_original_loc - bound_receptor_mean_to_remove).T).T + rot_b_receptor
                            self.data_unbound[i]['lig_graph'].ndata['new_x'] = zerocopy_from_numpy(ligand_new_loc.astype(np.float32))
                            self.data_unbound[i]['rec_graph'].ndata['new_x'] = zerocopy_from_numpy(receptor_new_loc.astype(np.float32)) 
                            self.data_bound[i]['lig_graph'].ndata['new_x'] = zerocopy_from_numpy(bound_ligand_new_loc.astype(np.float32)) 
                            self.data_bound[i]['rec_graph'].ndata['new_x'] = zerocopy_from_numpy(bound_receptor_new_loc.astype(np.float32)) 
                            
                            hetero_graph = hetero_graph_from_sg_l_r_pair(self.data_unbound[i]['lig_graph'], 
                                                                            self.data_unbound[i]['rec_graph']) 
                            hetero_graph = hetero_graph.to(self.args['device'])
                            ## aligned unbound ligand, receptor
                            ligand_pred, receptor_pred, \
                            _, _, \
                            _, _, _, _,  = self.alignment_model(hetero_graph, epoch=0)
                                    
                            self.data_unbound[i]['lig_pos'] = ligand_pred[0].detach().cpu().numpy()
                            self.data_unbound[i]['rec_pos'] = receptor_pred[0].detach().cpu().numpy()
                            
                        print("total pairs: ", len(self.unbound_indices))
                        print("Unbound to bound Alignment completed")
                            
            elif reload_mode == "val":
                self.data = pickle.load(open(os.path.join(raw_data_path, reload_mode + f'.pkl'), 'rb'))
                
            elif reload_mode == "test":
                if bound_type == "unbound":
                    print("bound type: ", bound_type)
                    self.data = pickle.load(open(os.path.join(raw_data_path, reload_mode + f'_unbound_esm3.pkl'), 'rb'))
                    ## alignment network
                    if args['u_b_align'] == "equidock":
                        with torch.inference_mode():  
                            for i in range(len(self.data)):
                                rot_T_ligand, rot_b_ligand = UniformRotation_Translation(translation_interval=self.args['translation_interval'])
                                rot_T_receptor, rot_b_receptor = UniformRotation_Translation(translation_interval=self.args['translation_interval'])

                                ligand_original_loc = self.data[i]['lig_graph'].ndata['x'].detach().numpy()
                                receptor_original_loc = self.data[i]['rec_graph'].ndata['x'].detach().numpy()
                                ligand_mean_to_remove = ligand_original_loc.mean(axis=0, keepdims=True)
                                receptor_mean_to_remove = receptor_original_loc.mean(axis=0, keepdims=True)

                                ligand_new_loc = (rot_T_ligand @ (ligand_original_loc - ligand_mean_to_remove).T).T + rot_b_ligand
                                receptor_new_loc = (rot_T_receptor @ (receptor_original_loc - receptor_mean_to_remove).T).T + rot_b_receptor

                                self.data[i]['lig_graph'].ndata['new_x'] = zerocopy_from_numpy(ligand_new_loc.astype(np.float32))
                                self.data[i]['rec_graph'].ndata['new_x'] = zerocopy_from_numpy(receptor_new_loc.astype(np.float32))
                                
                                hetero_graph = hetero_graph_from_sg_l_r_pair(self.data[i]['lig_graph'], 
                                                                             self.data[i]['rec_graph'])
                                hetero_graph = hetero_graph.to(self.args['device'])
                                ligand_pred, receptor_pred, \
                                _, _, \
                                _, _, _, _,  = self.alignment_model(hetero_graph, epoch=0)
                                        
                                self.data[i]['lig_pos'] = ligand_pred[0].detach().cpu().numpy()
                                self.data[i]['rec_pos'] = receptor_pred[0].detach().cpu().numpy()
                            print("total pairs: ", len(self.data))
                            print("Unbound to bound Alignment completed")
                    
                elif bound_type == "native_bound":  
                    print("bound type: ", bound_type)
                    print("file path: ", os.path.join(raw_data_path, reload_mode + '_native_bound.pkl'))
                    self.data = pickle.load(open(os.path.join(raw_data_path, reload_mode + '_native_bound.pkl'), 'rb'))
                elif bound_type == "native_unbound": 
                    print("bound type: ", bound_type)
                    print("file path: ", os.path.join(raw_data_path, reload_mode + '_native_unbound.pkl'))
                    self.data = pickle.load(open(os.path.join(raw_data_path, reload_mode + '_native_unbound.pkl'), 'rb'))

    
    def __len__(self):
        if self.reload_mode == "train":
            return len(self.data_bound) + len(self.data_unbound)
        elif self.reload_mode != "train":
            return len(self.data)

    def __getitem__(self, idx):
        if self.reload_mode == "train":
            if idx < len(self.data_bound):
                data = self.data_bound[idx] 
            else:
                idx_unbound = idx - len(self.data_bound)
                data = self.data_unbound[idx_unbound]
            rec_pos = data['rec_pos']
            lig_pos = data['lig_pos']
            rec_atom = data['rec_atom']
            lig_atom = data['lig_atom']
            bsp_lig = data['bsp_lig']
            bsp_rec = data['bsp_rec']
            lig_seq = data['lig_seq']
            rec_seq = data['rec_seq']
            lig_graph = data['lig_graph']
            rec_graph = data['rec_graph']
            file_name = data['filename']
                        
        elif self.reload_mode != "train":
            rec_pos = self.data[idx]['rec_pos']
            lig_pos = self.data[idx]['lig_pos']
            rec_atom = self.data[idx]['rec_atom']
            lig_atom = self.data[idx]['lig_atom']
            bsp_lig = self.data[idx]['bsp_lig']
            bsp_rec = self.data[idx]['bsp_rec']
            lig_seq = self.data[idx]['lig_seq']
            rec_seq = self.data[idx]['rec_seq']
            lig_graph = self.data[idx]['lig_graph']
            rec_graph = self.data[idx]['rec_graph']
            file_name = self.data[idx]['filename']
            
        # Randomly rotate and translate the ligand.
        rot_T, rot_b = UniformRotation_Translation(translation_interval=self.args['translation_interval'])
        ligand_original_loc = lig_graph.ndata['x'].detach().numpy()
        mean_to_remove = ligand_original_loc.mean(axis=0, keepdims=True)
        ligand_new_loc = (rot_T @ (ligand_original_loc - mean_to_remove).T).T + rot_b
        lig_graph.ndata['new_x_flex'] = zerocopy_from_numpy(ligand_new_loc.astype(np.float32))
        rec_graph.ndata['new_x_flex'] = rec_graph.ndata['x']

        if 'new_x' not in lig_graph.ndata:
            lig_graph.ndata['new_x'] = lig_graph.ndata['x']
        if 'new_x' not in rec_graph.ndata:
            rec_graph.ndata['new_x'] = rec_graph.ndata['x']

        data_item = {'rec_pos': zerocopy_from_numpy(rec_pos.astype(np.float32)),  # x,y,z coordinate
                     'lig_pos': zerocopy_from_numpy(lig_pos.astype(np.float32)),  # x,y,z coordinate
                     'rec_atom': zerocopy_from_numpy(rec_atom.astype(np.int64)),  # amino acid number
                     'lig_atom': zerocopy_from_numpy(lig_atom.astype(np.int64)),  # amino acid number
                     'rec_seq': rec_seq,  # amino acid number
                     'lig_seq': lig_seq,  # amino acid number
                     'bsp_rec':bsp_rec,
                     'bsp_lig':bsp_lig,
                     'lig_graph':lig_graph,
                     'rec_graph':rec_graph,
                     'file_name': file_name}  
        
        return data_item


def hetero_graph_from_sg_l_r_pair(ligand_graph, receptor_graph):
    ll = [('ligand', 'll', 'ligand'), (ligand_graph.edges()[0], ligand_graph.edges()[1])]
    rr = [('receptor', 'rr', 'receptor'), (receptor_graph.edges()[0], receptor_graph.edges()[1])]
    rl = [('receptor', 'cross', 'ligand'),
          (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    lr = [('ligand', 'cross', 'receptor'),
          (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    num_nodes = {'ligand': ligand_graph.num_nodes(), 'receptor': receptor_graph.num_nodes()}
    hetero_graph = dgl.heterograph({ll[0]: ll[1], rr[0]: rr[1], rl[0]: rl[1], lr[0]: lr[1]}, num_nodes_dict=num_nodes)

    hetero_graph.nodes['ligand'].data['res_feat'] = ligand_graph.ndata['res_feat']
    hetero_graph.nodes['ligand'].data['x'] = ligand_graph.ndata['x']
    hetero_graph.nodes['ligand'].data['new_x'] = ligand_graph.ndata['new_x']
    hetero_graph.nodes['ligand'].data['mu_r_norm'] = ligand_graph.ndata['mu_r_norm']

    hetero_graph.edges['ll'].data['he'] = ligand_graph.edata['he']

    hetero_graph.nodes['receptor'].data['res_feat'] = receptor_graph.ndata['res_feat']
    hetero_graph.nodes['receptor'].data['x'] = receptor_graph.ndata['x']
    hetero_graph.nodes['receptor'].data['new_x'] = receptor_graph.ndata['new_x']
    hetero_graph.nodes['receptor'].data['mu_r_norm'] = receptor_graph.ndata['mu_r_norm']

    hetero_graph.edges['rr'].data['he'] = receptor_graph.edata['he']
    return hetero_graph


def zerocopy_from_numpy(x):
    return torch.from_numpy(x)


def UniformRotation_Translation(translation_interval):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt( np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_interval)
    t = t * length
    return rotation_matrix.astype(np.float32), t.astype(np.float32)

def zerocopy_from_numpy(x):
    return torch.from_numpy(x)


def batchify_and_create_respective_graphs(data):

    lig_pos_list = []
    rec_pos_list = []

    lig_atom_list = []
    rec_atom_list = []
    lig_seq_list = []
    rec_seq_list = []
    bsp_lig_list = []
    bsp_rec_list = []

    lig_graph_list = []
    rec_graph_list = []
    batch_lig_list = []
    batch_rec_list = []
    file_name_list = []
    
    for id, item in enumerate(data):
        lig_pos_list.append(item['lig_pos'])
        rec_pos_list.append(item['rec_pos'])

        lig_atom_list.append(item['lig_atom'])
        rec_atom_list.append( item['rec_atom'])
        lig_seq_list.append(item['lig_seq'])
        rec_seq_list.append(item['rec_seq'])
        bsp_lig_list.append(item['bsp_lig'])
        bsp_rec_list.append(item['bsp_rec'])

        lig_graph_list.append(item['lig_graph'])
        rec_graph_list.append(item['rec_graph'])
        batch_lig_list.append(torch.full((item['lig_pos'].shape[0],), id))
        batch_rec_list.append(torch.full((item['rec_pos'].shape[0],), id))
        file_name_list.append(item['file_name'])
        
    
    batch_ligand_graph_list = dgl.batch(lig_graph_list)
    batch_receptor_graph_list = dgl.batch(rec_graph_list)

    batch_lig_ids = torch.cat([
        torch.full((g.num_nodes(),), i, dtype=torch.int32) for i, g in enumerate(lig_graph_list)
    ])
    batch_ligand_graph_list.ndata['batch'] = batch_lig_ids
    batch_rec_ids = torch.cat([
        torch.full((g.num_nodes(),), i, dtype=torch.int32) for i, g in enumerate(rec_graph_list)
    ])
    batch_receptor_graph_list.ndata['batch'] = batch_rec_ids
    
        
    return lig_pos_list, rec_pos_list, lig_atom_list, rec_atom_list, lig_seq_list, rec_seq_list, bsp_lig_list, bsp_rec_list,  \
           batch_ligand_graph_list, batch_receptor_graph_list, batch_lig_list, batch_rec_list, file_name_list



def dataset_selection(args):

    args['u_b_align'] = "equidock"
    alignment_model = create_model_equidock(args).to(args['device'])
    aligment_path = 'model_weight/' + f"alignment_model_best.pth"  # 파일명 수정 
    print("alignment network load :", aligment_path)
    alignment_model.load_state_dict(torch.load(aligment_path, map_location=args['device']), strict=False)
    print("alignment network load completed")
    
    test_dataset_native_bound = DockingDataset_aug_align(args, alignment_model=alignment_model, data_set=args['dataset'], reload_mode='test', load_from_cache=True, raw_data_path=args['data_path'], bound_type='native_bound')
    test_dataloader_native_bound = DataLoader(test_dataset_native_bound, batch_size=1, shuffle=False, collate_fn=batchify_and_create_respective_graphs)

    test_dataset_unbound = DockingDataset_aug_align(args, alignment_model=alignment_model, data_set=args['dataset'], reload_mode='test', load_from_cache=True, raw_data_path=args['data_path'], bound_type='unbound')
    test_dataloader_unbound = DataLoader(test_dataset_unbound, batch_size=1, shuffle=False, collate_fn=batchify_and_create_respective_graphs)
    
    test_dataset_native_unbound = DockingDataset_aug_align(args, alignment_model=alignment_model, data_set=args['dataset'], reload_mode='test', load_from_cache=True, raw_data_path=args['data_path'], bound_type='native_unbound')
    test_dataloader_native_unbound = DataLoader(test_dataset_native_unbound, batch_size=1, shuffle=False, collate_fn=batchify_and_create_respective_graphs)

    return test_dataloader_native_bound, test_dataloader_unbound, test_dataloader_native_unbound
