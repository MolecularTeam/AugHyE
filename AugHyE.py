import math
import os
import sys
import random
import copy
import numpy as np
from typing import List
import torch
from torch import nn
from torch import Tensor
from torch.nn import init
from torch.nn import Parameter
from torch_geometric.utils import degree
import torch_geometric.nn as pygnn
import torch.nn.functional as F
import dgl

from dgl import function as fn


from model.bernnet import Bern_prop
from mamba_ssm import Mamba



from model.Base_model import *
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score



class GraphNorm(nn.Module):
    """
        Param: []
    """
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim = 0, keepdim = True)
        var = x.std(dim = 0, keepdim = True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h, node_type):
        graph_size  = g.batch_num_nodes(node_type) if self.is_node else g.batch_num_edges(node_type)
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x


def get_non_lin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def lexsort(
        keys: List[Tensor],
        dim: int = -1,
        descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out

def compute_cross_attention(queries, keys, values, mask, cross_msgs):
    """Compute cross attention.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    Args:
      queries: NxD float tensor --> queries
      keys: MxD float tensor --> keys
      values: Mxd
      mask: NxM
    Returns:
      attention_x: Nxd float tensor.
    """
    if not cross_msgs:
        return queries * 0.
    a = mask * torch.mm(queries, torch.transpose(keys, 1, 0)) - 1000. * (1. - mask)
    a_x = torch.softmax(a, dim=1)  
    attention_x = torch.mm(a_x, values) 
    return attention_x


def dgl_to_dense_batch(x, batch):
    batch_size = batch.max().item() + 1
    num_features = x.size(1)
    graph_sizes = torch.bincount(batch)
    max_nodes = graph_sizes.max().item()

    dense_x = torch.zeros((batch_size, max_nodes, num_features), device=x.device)
    mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=x.device)

    for graph_id in range(batch_size):
        node_indices = (batch == graph_id).nonzero(as_tuple=True)[0]
        num_nodes = node_indices.size(0)
        dense_x[graph_id, :num_nodes] = x[node_indices]
        mask[graph_id, :num_nodes] = True

    return dense_x, mask

def get_mask(ligand_batch, receptor_batch, device):
    ligand_batch_num_nodes = torch.bincount(ligand_batch).tolist()  
    receptor_batch_num_nodes = torch.bincount(receptor_batch).tolist() 
    
    rows = sum(ligand_batch_num_nodes)
    cols = sum(receptor_batch_num_nodes)
    mask = torch.zeros(rows, cols).to(device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask


class SEGCN_mamba_Layer(nn.Module):
    def __init__(self, args, layer_norm=False, batch_norm=True):

        super(SEGCN_mamba_Layer, self).__init__()
        self.args = args
        hidden = args['h_dim']
        coe = 5 + 27
        self.device = args['device']
        # EDGES
        self.all_sigmas_dist = [10 ** x for x in range(5)]

        # Embedding Network for Edge Features MLP_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(coe, hidden),
            nn.Dropout(args['dp_encoder']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', hidden),
            nn.Linear(hidden, 1),
        )
        
        # local BernNet
        self.bern_1 = Bern_prop(K=args['bern_k'])

        # global SSM
        self.cross_msgs = True 
        self.self_attn = Mamba(d_model=args['h_dim'],  # Model dimension d_model
                               d_state=16,  # SSM state expansion factor
                               d_conv=4,  # Local convolution width
                               expand=1,  # Block expansion factor
                               )
        
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        if self.layer_norm:
            self.norm1_local = pygnn.norm.GraphNorm(args['h_dim'])
            self.norm1_attn = pygnn.norm.GraphNorm(args['h_dim'])
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(args['h_dim'])
            self.norm1_attn = nn.BatchNorm1d(args['h_dim'])
        self.dropout_attn = nn.Dropout(args['dropout'])

        self.att_mlp_Q = nn.Sequential(
            nn.Linear(args['h_dim'], args['h_dim'], bias=False),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(args['h_dim'], args['h_dim'], bias=False),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(args['h_dim'], args['h_dim'], bias=False),
        )

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(args['h_dim'], args['h_dim'] * 2)
        self.ff_linear2 = nn.Linear(args['h_dim'] * 2, args['h_dim'])
        if self.layer_norm:
            self.norm2 = pygnn.norm.GraphNorm(args['h_dim'])
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(args['h_dim'])
        self.ff_dropout1 = nn.Dropout(args['dropout'])
        self.ff_dropout2 = nn.Dropout(args['dropout'])


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges1(self, edges):
        return {'cat_feat': torch.cat([edges.src['pro_h'], edges.dst['pro_h']], dim=1)}

    def forward(self, batch_1_graph, batch_2_graph):
        h_lig = batch_1_graph.ndata['pro_h']
        h_lig_in1 = h_lig  # for first residual connection
        h_lig_out_list = []
        h_rec = batch_2_graph.ndata['pro_h']
        h_rec_in1 = h_rec  # for first residual connection
        h_rec_out_list = []

        batch_1_graph.apply_edges(fn.u_sub_v('new_x_flex', 'new_x_flex', 'x_dis'))  ## x_i - x_j
        batch_2_graph.apply_edges(fn.u_sub_v('new_x_flex', 'new_x_flex', 'x_dis'))
        edge_tmp_1 = batch_1_graph.edata['x_dis'] ** 2
        edge_tmp_2 = batch_2_graph.edata['x_dis'] ** 2

        edge_weight_1 = torch.sum(edge_tmp_1, dim=1, keepdim=True)  ## ||x_i - x_j||^2 : (N_res, 1)
        edge_weight_1 = torch.cat([torch.exp(-edge_weight_1 / sigma) for sigma in self.all_sigmas_dist], dim=-1)
        edge_weight_2 = torch.sum(edge_tmp_2, dim=1, keepdim=True)  ## ||x_i - x_j||^2 : (N_res, 1)
        edge_weight_2 = torch.cat([torch.exp(-edge_weight_2 / sigma) for sigma in self.all_sigmas_dist], dim=-1)

        ## Embedding Network for edge features: MLP_e (self.edge_mlp)
        weight_lap_1 = F.relu(self.edge_mlp(torch.cat((edge_weight_1, batch_1_graph.edata['he']), 1)))
        weight_lap_2 = F.relu(self.edge_mlp(torch.cat((edge_weight_2, batch_2_graph.edata['he']), 1)))

        edge_index_1 = torch.stack(batch_1_graph.edges())
        edge_index_2 = torch.stack(batch_2_graph.edges())

        ## local BernNet
        h_lig_local, TEMP_1 = self.bern_1(batch_1_graph.ndata['pro_h'], edge_index_1.long(), weight_lap_1.T.squeeze(0))
        h_rec_local, TEMP_2 = self.bern_1(batch_2_graph.ndata['pro_h'], edge_index_2.long(), weight_lap_2.T.squeeze(0))

        h_lig_out_list.append(h_lig_local)
        h_rec_out_list.append(h_rec_local)

        ## global SSM (Mamba)
        if self.args['train_mode'] == 'train': 
            ## ligand
            lig_deg = degree(edge_index_1.long()[0], batch_1_graph.ndata['pro_h'].shape[0]).to(torch.float)
            lig_deg_noise = torch.rand_like(lig_deg).to(lig_deg.device)
            h_lig_ind_perm = lexsort([lig_deg + lig_deg_noise, batch_1_graph.ndata['batch']])
            h_lig_dense, lig_mask = dgl_to_dense_batch(h_lig[h_lig_ind_perm], batch_1_graph.ndata['batch'][h_lig_ind_perm])
            h_lig_ind_perm_reverse = torch.argsort(h_lig_ind_perm)
            h_lig_attn = self.self_attn(h_lig_dense)[lig_mask][h_lig_ind_perm_reverse]

            ## receptor
            rec_deg = degree(edge_index_2.long()[0], batch_2_graph.ndata['pro_h'].shape[0]).to(torch.float)
            rec_deg_noise = torch.rand_like(rec_deg).to(rec_deg.device)
            h_rec_ind_perm = lexsort([rec_deg + rec_deg_noise, batch_2_graph.ndata['batch']])
            h_rec_dense, rec_mask = dgl_to_dense_batch(h_rec[h_rec_ind_perm], batch_2_graph.ndata['batch'][h_rec_ind_perm])
            h_rec_ind_perm_reverse = torch.argsort(h_rec_ind_perm)
            h_rec_attn = self.self_attn(h_rec_dense)[rec_mask][h_rec_ind_perm_reverse]  # Mamba

        else:
            # ligand
            lig_mamba_arr = []
            lig_deg = degree(edge_index_1.long()[0], batch_1_graph.ndata['pro_h'].shape[0]).to(torch.float)
            for i in range(5):
                lig_deg_noise = torch.rand_like(lig_deg).to(lig_deg.device)
                h_lig_ind_perm = lexsort([lig_deg + lig_deg_noise, batch_1_graph.ndata['batch']])
                h_lig_dense, lig_mask = dgl_to_dense_batch(h_lig[h_lig_ind_perm], batch_1_graph.ndata['batch'][h_lig_ind_perm])
                h_lig_ind_perm_reverse = torch.argsort(h_lig_ind_perm)
                h_lig_attn = self.self_attn(h_lig_dense)[lig_mask][h_lig_ind_perm_reverse]  # Mamba
                lig_mamba_arr.append(h_lig_attn)
            h_lig_attn = sum(lig_mamba_arr) / 5

            # receptor
            rec_mamba_arr = []
            rec_deg = degree(edge_index_2.long()[0], batch_2_graph.ndata['pro_h'].shape[0]).to(torch.float)
            for i in range(5):
                rec_deg_noise = torch.rand_like(rec_deg).to(rec_deg.device)
                h_rec_ind_perm = lexsort([rec_deg + rec_deg_noise, batch_2_graph.ndata['batch']])
                h_rec_dense, rec_mask = dgl_to_dense_batch(h_rec[h_rec_ind_perm], batch_2_graph.ndata['batch'][h_rec_ind_perm])
                h_rec_ind_perm_reverse = torch.argsort(h_rec_ind_perm)
                h_rec_attn = self.self_attn(h_rec_dense)[rec_mask][h_rec_ind_perm_reverse]  # Mamba
                rec_mamba_arr.append(h_rec_attn)
            h_rec_attn = sum(rec_mamba_arr) / 5
            
        ## cross attention
        mask = get_mask(batch_1_graph.ndata['batch'], batch_2_graph.ndata['batch'], self.args['device'])
        h_lig_attn = compute_cross_attention(self.att_mlp_Q(h_lig_attn),
                                             self.att_mlp_K(h_rec_attn),
                                             self.att_mlp_V(h_rec_attn),
                                             mask,
                                             self.cross_msgs)
        h_rec_attn = compute_cross_attention(self.att_mlp_Q(h_rec_attn),
                                             self.att_mlp_K(h_lig_attn),
                                             self.att_mlp_V(h_lig_attn),
                                             mask.transpose(0, 1),
                                             cross_msgs=True)

        # ligand
        h_lig_attn = self.dropout_attn(h_lig_attn)
        h_lig_attn = h_lig_in1 + h_lig_attn  # Residual connection. (h_lig_in1)
        if self.layer_norm:
            h_lig_attn = self.norm1_attn(h_lig_attn, batch_1_graph.batch)
        if self.batch_norm:
            h_lig_attn = self.norm1_attn(h_lig_attn)
        h_lig_out_list.append(h_lig_attn)
        # receptor
        h_rec_attn = self.dropout_attn(h_rec_attn)
        h_rec_attn = h_rec_in1 + h_rec_attn  # Residual connection. (h_rec_in1)
        if self.layer_norm:
            h_rec_attn = self.norm1_attn(h_rec_attn, batch_2_graph.batch)
        if self.batch_norm:
            h_rec_attn = self.norm1_attn(h_rec_attn)
        h_rec_out_list.append(h_rec_attn)

        # Combine local and global outputs.
        h_lig = sum(h_lig_out_list)

        # Feed Forward block.
        h_lig = h_lig + self._ff_block(h_lig)
        if self.layer_norm:
            h_lig = self.norm2(h_lig, batch_1_graph.batch)
        if self.batch_norm:
            h_lig = self.norm2(h_lig)
        batch_1_graph.ndata['pro_h'] = h_lig

        h_rec = sum(h_rec_out_list)
        h_rec = h_rec + self._ff_block(h_rec)
        if self.layer_norm:
            h_rec = self.norm2(h_rec, batch_2_graph.batch)
        if self.batch_norm:
            h_rec = self.norm2(h_rec)

        batch_2_graph.ndata['pro_h'] = h_rec

        return batch_1_graph.ndata['new_x_flex'], batch_1_graph.ndata['pro_h'], \
               batch_2_graph.ndata['new_x_flex'], batch_2_graph.ndata['pro_h'], TEMP_1, TEMP_2


    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


    def __repr__(self):
        return "SEGCN mamba Layer " + str(self.__dict__)
    
    
    
class CrossAttentionLayer(nn.Module):
    def __init__(self,args):
        # super().__init__()
        super(CrossAttentionLayer, self).__init__()
        self.h_dim = args['h_dim']
        
        self.h_dim_div = self.h_dim // 1
        self.num_heads = args['atten_head']
        self.head_dim = self.h_dim_div // self.num_heads
        self.merge = nn.Conv1d(self.h_dim_div, self.h_dim_div, kernel_size=1)
        self.proj = nn.ModuleList([nn.Conv1d(self.h_dim, self.h_dim_div, kernel_size=1) for _ in range(3)])
        dropout = args['dp_encoder']

        self.mlp = nn.Sequential(
            nn.Linear(self.h_dim+self.h_dim_div, self.h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.h_dim)
        )
    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, src_h, dst_h, src_num_verts, dst_num_verts):
        h = dst_h
        src_h_list = torch.split(src_h, src_num_verts)
        dst_h_list = torch.split(dst_h, dst_num_verts)
        h_msg = []
        for idx in range(len(src_num_verts)):
            src_hh = src_h_list[idx].unsqueeze(0).transpose(1, 2)
            dst_hh = dst_h_list[idx].unsqueeze(0).transpose(1, 2)
            query, key, value = [hh.view(1, self.head_dim, self.num_heads, -1) \
                for ll, hh in zip(self.proj, (dst_hh, src_hh, src_hh))]
            dim = query.shape[1]
            scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / (dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h_dst = torch.einsum('bhnm,bdhm->bdhn', attn, value) 
            h_dst = h_dst.contiguous().view(1, self.h_dim_div, -1)
            h_msg.append(h_dst.squeeze(0).transpose(0, 1))
        h_msg = torch.cat(h_msg, dim=0)

        # skip connection
        h_out = h + self.mlp(torch.cat((h, h_msg), dim=-1))

        return h_out


# =================================================================================================================
class SEGCN(nn.Module):

    def __init__(self, args, n_lays, fine_tune, log=None):

        super(SEGCN, self).__init__()
        self.args = args
        self.log=log

        self.device = args['device']
        self.graph_nodes = args['graph_nodes']

        self.rot_model = args['rot_model']
        self.noise_decay_rate = args['noise_decay_rate']
        self.noise_initial = args['noise_initial']

        # 21 types of amino-acid types
        self.residue_emb_layer = nn.Embedding(num_embeddings=21, embedding_dim=args['residue_emb_dim'])

        assert self.graph_nodes == 'residues'

        self.c_a_layer = CrossAttentionLayer(args)
        self.n_layer = args['SEGCN_layer']

        # BernNet + SSM
        self.segcn_layers = nn.ModuleList()
        self.segcn_layers.append(SEGCN_mamba_Layer(args))

        if self.n_layer > 1:
            interm_lay = SEGCN_mamba_Layer(args)
            for _ in range(1, self.n_layer):
                self.segcn_layers.append(interm_lay)

        assert args['rot_model'] == 'kb_att'
        input_n_dim = 1280
        if self.args['res_feat']:  
            input_n_dim += 64
        if self.args['mu_r_norm']:
            input_n_dim += 5
            
        ## Embedding Network for Node Features MLP_h
        self.fea_norm_mlp = nn.Sequential(
            nn.Linear(input_n_dim, args['h_dim']),
            nn.Dropout(args['dp_encoder']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['h_dim'])
        )
        
        ## Classifier
        self.clsf1 = nn.Sequential(
            nn.Linear(args['h_dim'], args['h_dim']),
            nn.Dropout(args['dp_cls']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['h_dim']),
            nn.Linear(args['h_dim'], 1),
            nn.Sigmoid()
        )
        self.clsf2 = nn.Sequential(
            nn.Linear(args['h_dim'], args['h_dim']),
            nn.Dropout(args['dp_cls']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['h_dim']),
            nn.Linear(args['h_dim'], 1),
            nn.Sigmoid()
        )
        self.clsf3 = nn.Sequential(
            nn.Linear(args['h_dim'], args['h_dim']),
            nn.Dropout(args['dp_cls']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['h_dim']),
            nn.Linear(args['h_dim'], 1),
            nn.Sigmoid()
        )
        

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)


    def forward(self, batch_1_graph, batch_2_graph):
        node_feat_1 = batch_1_graph.ndata['esm']  
        node_feat_2 = batch_2_graph.ndata['esm']

        ## Embed residue types with a lookup table.
        h_feats_1 = self.residue_emb_layer(
            torch.argmax(batch_1_graph.ndata['res_feat'], dim=1).unsqueeze(1).view(-1).long())  # (N_res, emb_dim)
        h_feats_2 = self.residue_emb_layer(
            torch.argmax(batch_2_graph.ndata['res_feat'], dim=1).unsqueeze(1).view(-1).long())  # (N_res, emb_dim)
        
        if self.args['res_feat']: 
            node_feat_1 = torch.cat([node_feat_1, h_feats_1], dim=1)
            node_feat_2 = torch.cat([node_feat_2, h_feats_2], dim=1)

        if self.args['mu_r_norm']:
            node_feat_1 = torch.cat([node_feat_1, torch.log(batch_1_graph.ndata['mu_r_norm'])], dim=1)
            node_feat_2 = torch.cat([node_feat_2, torch.log(batch_2_graph.ndata['mu_r_norm'])], dim=1)
        
        batch_1_graph.ndata['pro_h'] = self.fea_norm_mlp(node_feat_1)  
        batch_2_graph.ndata['pro_h'] = self.fea_norm_mlp(node_feat_2)  
        
        for i, layer in enumerate(self.segcn_layers):
            coors_ligand, h_feats_ligand, coors_receptor, h_feats_receptor, TEMP_1, TEMP_2 = layer(batch_1_graph, batch_2_graph)

        batch_1_graph.ndata['hv_segcn_out'] = h_feats_ligand
        batch_2_graph.ndata['hv_segcn_out'] = h_feats_receptor
        pre_interface_batch = []
        list_graph_1 = dgl.unbatch(batch_1_graph)
        list_graph_2 = dgl.unbatch(batch_2_graph)
        for ii in range(len(list_graph_1)):  
            h_1 = list_graph_1[ii].ndata['hv_segcn_out']
            h_2 = list_graph_2[ii].ndata['hv_segcn_out']            
            h_2_ca = self.c_a_layer(h_1,h_2,[h_1.size(0)],[h_2.size(0)])
            h_1_ca = self.c_a_layer(h_2,h_1,[h_2.size(0)],[h_1.size(0)])
            
            pred_proxy1 = torch.cat((self.clsf1(h_1_ca), self.clsf1(h_2_ca)), dim=0)
            pred_proxy2 = torch.cat((self.clsf2(h_1_ca), self.clsf2(h_2_ca)), dim=0)
            pred_proxy3 = torch.cat((self.clsf3(h_1_ca), self.clsf3(h_2_ca)), dim=0)
            pre_interface = (pred_proxy1 + pred_proxy2 + pred_proxy3)/3
                
            pre_interface_batch.append(pre_interface)        

        return [TEMP_1, TEMP_2, pre_interface_batch, batch_1_graph, batch_2_graph]


    def __repr__(self):
        return "SEGCN " + str(self.__dict__)


# =================================================================================================================
class AugHyE(nn.Module):

    def __init__(self, args, log=None):

        super(AugHyE, self).__init__()

        self.log=log
        self.args = args
        self.device = args['device']
        
        # Hybrid Encoder
        self.segcn_original = SEGCN(args, n_lays=args['SEGCN_layer'], fine_tune=False, log=log)
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)
                
    def forward(self, batch_ligand_graph, batch_receptor_graph):

        outputs = self.segcn_original(batch_ligand_graph, batch_receptor_graph)

        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


    def __repr__(self):
        return "AugHyE " + str(self.__dict__)


class CELosswithFocal(torch.nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(CELosswithFocal, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma
        self.bce_loss = torch.nn.BCELoss(reduction='none')  

    def forward(self, bsp, iface_label):
        bce_loss = self.bce_loss(bsp, iface_label.float())
        at = self.alpha.to(bsp.device).gather(0, iface_label.data.view(-1))
        pt = torch.exp(-bce_loss)
        focal_loss = at * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


# =================================================================================================================
class AugHyE_model(Base_model):
    def __init__(self, args):
        super(AugHyE_model, self).__init__(args=args, model_args=None, name='AugHyE')
        self.args = args
        self.net = AugHyE(args)
        
    @property
    def size(self): 
        return self._size
    @property
    def name(self): 
        return self._name

    def train(self, train_loader, val_loader, best_model_save_path, last_metric_1=0.0):

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args['lr'], weight_decay=self.args['wd'])
        int_criterion = CELosswithFocal()
        
        for epoch_id in range(self.args['n_epochs']):
            # train
            self.net.train()
            avg_loss, total_loss, total_ap = 0., 0., 0
            total_interface_loss, total_stable_loss = 0., 0.
            auc_list = []
  
            self.args['train_mode'] = 'train'
            tqdm.write(f"[Epoch {epoch_id}] Training")
            loop = tqdm(train_loader, total=len(train_loader), leave=True, dynamic_ncols=True)

            for id, batch in enumerate(loop):
                optimizer.zero_grad()
                
                lig_pos, rec_pos, lig_atom, rec_atom, lig_seq, rec_seq, bsp_lig, bsp_rec,  \
                batch_ligand_graph, batch_receptor_graph, batch_lig, batch_rec, file_name = batch
                
                batch_ligand_graph = batch_ligand_graph.to(self.args['device'])
                batch_receptor_graph = batch_receptor_graph.to(self.args['device'])
                
                TEMP_1, TEMP_2, pre_interface_list, _, _ = self.net(batch_ligand_graph, batch_receptor_graph)     
                
                # SR loss
                batch_stable_loss = torch.max(torch.abs(torch.diff(TEMP_2))).to(self.args['device'])  
                
                batch_interface_loss = torch.zeros([]).to(self.args['device'])       
                batch_ap = torch.zeros([]).to(self.args['device'])
                batch_auc_list = []
                
                for i in range(len(pre_interface_list)):
                    bsp_pred = pre_interface_list[i].squeeze() 
                    label = torch.cat([bsp_lig[i], bsp_rec[i]], dim=0).to(self.args['device'])
                    
                    bsp_loss = int_criterion(bsp_pred, label)
                    batch_interface_loss+=bsp_loss
                        
                    label_np = np.array(label.detach().cpu())
                    bsp_pred_np = np.array(bsp_pred.detach().cpu())
                    
                    batch_ap += average_precision_score(label_np, bsp_pred_np)
                    batch_auc_list.append(roc_auc_score(label_np, bsp_pred_np))
                    auc_list.append(roc_auc_score(label_np, bsp_pred_np))
                
                batch_num = len(pre_interface_list)
                ## total loss
                loss = batch_interface_loss/batch_num + self.args['sr_loss_ratio'] * batch_stable_loss 

                loop.set_postfix(loss=loss.item(), bsp_loss=(batch_interface_loss/batch_num).item(), stable_loss=batch_stable_loss.item(), 
                                AP=batch_ap.item()/batch_num, AUC=np.median(np.array(batch_auc_list)))

                loss.backward()
                optimizer.step()
            
                total_loss += loss.detach().item()
                total_interface_loss += batch_interface_loss.detach().item()/batch_num
                total_stable_loss += batch_stable_loss.detach().item()
                total_ap += batch_ap/batch_num
                auc_median = np.median(np.array(auc_list))
            
            loss_epoch = total_loss / len(train_loader)
            interface_loss_epoch = total_interface_loss / len(train_loader)
            stable_loss_epoch = total_stable_loss / len(train_loader)
            Train_AP = total_ap.item() / len(train_loader)
            Train_AUC_median = auc_median.item()
            
            # validation
            self.net.eval()
            self.args['train_mode'] = 'val'
            avg_loss, total_loss, total_ap, total_auc = 0., 0., 0, 0
            total_interface_loss, total_stable_loss = 0., 0.
            auc_list = []
            tqdm.write(f"[Epoch {epoch_id}] Validation")
            with torch.inference_mode():
                # progress = tqdm(val_loader)
                loop = tqdm(val_loader, total=len(val_loader), leave=True, dynamic_ncols=True)
                for id, batch in enumerate(loop):    
                    
                    lig_pos, rec_pos, lig_atom, rec_atom, lig_seq, rec_seq, bsp_lig, bsp_rec,  \
                    batch_ligand_graph, batch_receptor_graph, batch_lig, batch_rec, file_name = batch
                    
                    batch_ligand_graph = batch_ligand_graph.to(self.args['device'])
                    batch_receptor_graph = batch_receptor_graph.to(self.args['device'])
                    
                    TEMP_1, TEMP_2, pre_interface_list, _, _ = self.net(batch_ligand_graph, batch_receptor_graph)              
                    # SR loss   
                    batch_stable_loss = torch.max(torch.abs(torch.diff(TEMP_2))).to(self.args['device']) 
                    
                    batch_interface_loss = torch.zeros([]).to(self.args['device'])       
                    batch_ap = torch.zeros([]).to(self.args['device'])
                    batch_auc = torch.zeros([]).to(self.args['device'])
                    batch_auc_list = []
                    
                    for i in range(len(pre_interface_list)):
                        bsp_pred = pre_interface_list[i].squeeze()  
                        label = torch.cat([bsp_lig[i], bsp_rec[i]], dim=0).to(self.args['device'])

                        bsp_loss = int_criterion(bsp_pred, label)
                        batch_interface_loss+=bsp_loss
                            
                        label_np = np.array(label.detach().cpu())
                        bsp_pred_np = np.array(bsp_pred.detach().cpu())
                        
                        batch_auc += roc_auc_score(label_np, bsp_pred_np)
                        batch_ap += average_precision_score(label_np, bsp_pred_np)
                        batch_auc_list.append(roc_auc_score(label_np, bsp_pred_np))
                        auc_list.append(roc_auc_score(label_np, bsp_pred_np))
                    
                    batch_num = len(pre_interface_list)
                    ## total loss
                    loss = batch_interface_loss/batch_num + self.args['sr_loss_ratio'] * batch_stable_loss 
                    
                    loop.set_postfix(loss=loss.item(), bsp_loss=(batch_interface_loss/batch_num).item(), stable_loss=batch_stable_loss.item(), 
                                    AP=batch_ap.item()/batch_num, AUC=np.median(np.array(batch_auc_list)))
                
                    total_loss += loss.detach().item()
                    total_interface_loss += batch_interface_loss.detach().item()/batch_num
                    total_stable_loss += batch_stable_loss.detach().item()
                    total_ap += batch_ap/batch_num
                    total_auc += batch_auc/batch_num
                    auc_median = np.median(np.array(auc_list))

                Val_loss_epoch = total_loss / len(val_loader)
                Val_interface_loss_epoch = total_interface_loss / len(val_loader)
                Val_stable_loss_epoch = total_stable_loss / len(val_loader)
                Val_AP = total_ap.item() / len(val_loader)
                Val_metric = total_auc.item() / len(val_loader)
                Val_AUC_median = auc_median.item()
                
                print('Epoch [TRAIN]: ', epoch_id,
                'AP:', f'{Train_AP:.3f}',
                'AUC MEDIAN:', f'{Train_AUC_median:.3f}')    
            
                print('Epoch [VALID]: ', epoch_id,
                'AP:', f'{Val_AP:.3f}',
                'AUC:', f'{Val_metric:.3f}',
                'AUC MEDIAN:', f'{Val_AUC_median:.3f}')    
            
            # model save
            if Val_metric > last_metric_1:
                last_metric_1 = Val_metric
                self.save(best_model_save_path)
                print('Save pretrain model')
            print(f"Timestamp: {self.args['timestamp']}")
            print(f"Best Metric Valid: {last_metric_1}")
    

    def evaluate(self, args, data_loader, best_model_save_path, bound_type="bound"): 
        
        self.load(best_model_save_path)
        self.net.eval()
        self.args['train_mode'] = 'val'
        
        total_loss, total_ap = 0., 0.
        total_interface_loss, total_stable_loss = 0., 0.
        auc_list = []
        int_criterion = CELosswithFocal()
        with torch.inference_mode():
            
            loop = tqdm(data_loader, total=len(data_loader), leave=True, dynamic_ncols=True)
            for id, batch in enumerate(loop):    
                
                lig_pos, rec_pos, lig_atom, rec_atom, lig_seq, rec_seq, bsp_lig, bsp_rec,  \
                batch_ligand_graph, batch_receptor_graph, batch_lig, batch_rec, file_name = batch
                
                batch_ligand_graph = batch_ligand_graph.to(self.args['device'])
                batch_receptor_graph = batch_receptor_graph.to(self.args['device'])
                
                TEMP_1, TEMP_2, pre_interface_list, _, _ = self.net(batch_ligand_graph, batch_receptor_graph) 
                # SR loss
                batch_stable_loss = torch.max(torch.abs(torch.diff(TEMP_2))).to(self.args['device']) 
                
                batch_interface_loss = torch.zeros([]).to(self.args['device'])       
                batch_ap = torch.zeros([]).to(self.args['device'])
                batch_auc_list = []
                        
                for i in range(len(pre_interface_list)):
                    bsp_pred = pre_interface_list[i].squeeze() 
                    label = torch.cat([bsp_lig[i], bsp_rec[i]], dim=0).to(self.args['device'])  
                    
                    bsp_loss = int_criterion(bsp_pred, label)
                    batch_interface_loss+=bsp_loss
                        
                    label_np = np.array(label.detach().cpu())
                    bsp_pred_np = np.array(bsp_pred.detach().cpu())
                
                    batch_ap += average_precision_score(label_np, bsp_pred_np)
                    batch_auc_list.append(roc_auc_score(label_np, bsp_pred_np))
                    auc_list.append(roc_auc_score(label_np, bsp_pred_np))
                
                batch_num = len(pre_interface_list)
                ## total loss
                loss = batch_interface_loss/batch_num + self.args['sr_loss_ratio'] * batch_stable_loss 
                
                loop.set_postfix(loss=loss.item(), bsp_loss=(batch_interface_loss/batch_num).item(), stable_loss=batch_stable_loss.item(), 
                                AP=batch_ap.item()/batch_num, AUC=np.median(np.array(batch_auc_list)))
            
                total_loss += loss.detach().item()
                total_interface_loss += batch_interface_loss.detach().item()/batch_num
                total_stable_loss += batch_stable_loss.detach().item()
                total_ap += batch_ap/batch_num
                auc_median = np.median(np.array(auc_list))

            loss_epoch = total_loss / len(data_loader)
            interface_loss_epoch = total_interface_loss / len(data_loader)
            stable_loss_epoch = total_stable_loss / len(data_loader)
            AP = total_ap.item() / len(data_loader)
            AUC_median = auc_median.item()

            print(f'Test LOSS:', f'{loss:.3f}',
            f'Test AP:', f'{AP:.3f}',
            f'Test AUC_MEDIAN:', f'{AUC_median:.3f}')       
             
            return loss_epoch, AP, AUC_median

