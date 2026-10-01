import os
import sys
import math
from torch.nn import init
import numpy as np
import dgl
import torch
from torch import nn
from torch_geometric.utils import degree
import torch_geometric.nn as pygnn
import sys
import random
from .bernnet import Bern_prop
from mamba_ssm import Mamba
import torch.nn.functional as F
from torch import Tensor
from typing import List
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

from e3nn import o3
from torch_cluster import radius_graph
from .base_code import (EdgeDegreeEmbeddingNetwork, GaussianRadialBasisLayer,
                         NodeEmbeddingNetwork)

_MAX_ATOM_TYPE = 21
_RESCALE = True
_USE_BIAS = True
_AVG_DEGREE = 500


import torch.nn as nn
import torch


class GraphNorm(nn.Module):
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
    a_x = torch.softmax(a, dim=1)  # i->j, NxM, a_x.sum(dim=1) = torch.ones(N)
    attention_x = torch.mm(a_x, values)  # (N,d)
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
        self.device = args['device']

        # local 
        self.bern_1 = Bern_prop(K=args['bern_k'])

        # global mamba
        self.cross_msgs = True 
        self.self_attn = Mamba(d_model=args['h_dim'],  # Model dimension d_model
                               d_state=16,  # SSM state expansion factor
                               d_conv=4,  # Local convolution width
                               expand=1,  # Block expansion factor
                               )
        
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        if self.layer_norm:
            self.norm1_attn = pygnn.norm.GraphNorm(args['h_dim'])
        if self.batch_norm:
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


    def forward(self, batch_1_graph, batch_2_graph, edge_index_1, edge_index_2, weight_lap_1, weight_lap_2):
        h_lig = batch_1_graph.ndata['pro_h']
        h_lig_in1 = h_lig  # for skip connection
        h_lig_out_list = []
        h_rec = batch_2_graph.ndata['pro_h']
        h_rec_in1 = h_rec  # for skip connection
        h_rec_out_list = []

        ## local Bernpro
        h_lig_local, TEMP_1 = self.bern_1(batch_1_graph.ndata['pro_h'], edge_index_1.long(), weight_lap_1.T.squeeze(0))
        h_rec_local, TEMP_2 = self.bern_1(batch_2_graph.ndata['pro_h'], edge_index_2.long(), weight_lap_2.T.squeeze(0))
           
        h_lig_out_list.append(h_lig_local)
        h_rec_out_list.append(h_rec_local)
        
        ## global Mamba
        if self.training:
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

        # cross attn
        mask = get_mask(batch_1_graph.ndata['batch'], batch_2_graph.ndata['batch'], self.args['device'])
        q_lig, k_rec = self.att_mlp_Q(h_lig_attn), self.att_mlp_K(h_rec_attn)
        q_rec, k_lig = self.att_mlp_Q(h_rec_attn), self.att_mlp_K(h_lig_attn)
        h_lig_ca = compute_cross_attention(q_lig,
                                            k_rec,
                                            self.att_mlp_V(h_rec_attn),
                                            mask,
                                            self.cross_msgs)
        h_rec_ca = compute_cross_attention(q_rec,
                                            k_lig,
                                            self.att_mlp_V(h_lig_attn),
                                            mask.transpose(0, 1),
                                            cross_msgs=True)

        # ligand
        h_lig_attn = self.dropout_attn(h_lig_ca)
        h_lig_attn = h_lig_in1 + h_lig_attn  # skip connection.
        if self.layer_norm:
            h_lig_attn = self.norm1_attn(h_lig_attn, batch_1_graph.batch)
        if self.batch_norm:
            h_lig_attn = self.norm1_attn(h_lig_attn)
        h_lig_out_list.append(h_lig_attn)
        
        # receptor
        h_rec_attn = self.dropout_attn(h_rec_ca)
        h_rec_attn = h_rec_in1 + h_rec_attn  # skip connection.
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


        return batch_1_graph.ndata['pro_h'], batch_2_graph.ndata['pro_h'], TEMP_1, TEMP_2

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def __repr__(self):
        return "SEGCN mamba Layer " + str(self.__dict__)


# =================================================================================================================
class SEGCN(nn.Module):

    def __init__(self, args, max_radius=8.0,
                 irreps_node_embedding='64x0e+8x1e+8x2e', irreps_sh='1x0e+1x1e+1x2e',
                 number_of_basis=64, fc_neurons=[64, 64]):

        super(SEGCN, self).__init__()
        self.args = args
        self.device = args['device']
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_edge_attr = o3.Irreps(irreps_sh)
        self.fc_neurons = [number_of_basis] + fc_neurons
        self.lmax = self.irreps_node_embedding.lmax
        
        self.n_layer = args['SEGCN_layer']
        self.segcn_layers = nn.ModuleList()
        self.segcn_layers.append(SEGCN_mamba_Layer(args))

        if self.n_layer > 1:
            interm_lay = SEGCN_mamba_Layer(args)
            for layer_idx in range(1, self.n_layer):
                self.segcn_layers.append(interm_lay)

        input_n_dim = self.irreps_node_embedding.dim + 1280

        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)

        coe = 5  
        self.all_sigmas_dist = [10 ** x for x in range(5)]
        
        self.fea_norm_mlp = nn.Sequential(
            nn.Linear(input_n_dim, args['h_dim']),
            nn.Dropout(args['dp_encoder']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['h_dim']),
        )
        
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding,
                                                         self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        self.edge_mlp = nn.Sequential(
            nn.Linear(coe, args['h_dim']),
            nn.Dropout(args['dp_encoder']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', args['h_dim']),
            nn.Linear(args['h_dim'], 1),
        )
        

        self.clsf1 = nn.Sequential(
            nn.Linear(args['h_dim'], args['hidden_dim']),
            nn.Dropout(args['dp_cls']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['hidden_dim']),
            nn.Linear(args['hidden_dim'], 1),
            nn.Sigmoid()
        )
        self.clsf2 = nn.Sequential(
            nn.Linear(args['h_dim'], args['hidden_dim']),
            nn.Dropout(args['dp_cls']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['hidden_dim']),
            nn.Linear(args['hidden_dim'], 1),
            nn.Sigmoid()
        )
        self.clsf3 = nn.Sequential(
            nn.Linear(args['h_dim'], args['hidden_dim']),
            nn.Dropout(args['dp_cls']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['hidden_dim']),
            nn.Linear(args['hidden_dim'], 1),
            nn.Sigmoid()
        )
                

    def _encode_features(self, batch_graph):
        
        node_feat = batch_graph.ndata['esm'].float()

        edge_index = radius_graph(batch_graph.ndata['pos'], r=self.max_radius, batch=batch_graph.ndata['batch'],
                                  max_num_neighbors=1000)  
        edge_src, edge_dst = edge_index  
        edge_vec = batch_graph.ndata['pos'].index_select(0, edge_src) - batch_graph.ndata['pos'].index_select(0, edge_dst)

        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
                                         x=edge_vec, normalize=True, normalization='component')
        atom_embedding, _, _ = self.atom_embed(batch_graph.ndata['atom'])
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length) 
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh,
                                                    edge_length_embedding, edge_src, edge_dst, batch_graph.ndata['batch'])

        edge_distance_squared = edge_length.unsqueeze(1) ** 2
        edge_weight = torch.cat([
            torch.exp(-edge_distance_squared / sigma)
            for sigma in self.all_sigmas_dist
        ], dim=-1)

        atom_feat = edge_degree_embedding + atom_embedding  
        node_feat = torch.cat([node_feat, atom_feat], dim=1)  

        return edge_index, edge_weight, node_feat


    def forward(self, batch_1_graph, batch_2_graph):
        ## ligand
        edge_index_1, edge_feat_1, node_feat_1 = self._encode_features(batch_1_graph)
        ## receptor
        edge_index_2, edge_feat_2, node_feat_2 = self._encode_features(batch_2_graph)

        batch_1_graph.ndata['pro_h'] = self.fea_norm_mlp(node_feat_1)  
        batch_2_graph.ndata['pro_h'] = self.fea_norm_mlp(node_feat_2)  

        weight_lap_1 = F.relu(self.edge_mlp(edge_feat_1))
        weight_lap_2 = F.relu(self.edge_mlp(edge_feat_2))
        
        for i, layer in enumerate(self.segcn_layers):
            h_feats_ligand, h_feats_receptor, TEMP_1, TEMP_2 = layer(batch_1_graph, batch_2_graph, edge_index_1, edge_index_2, weight_lap_1, weight_lap_2)

        batch_1_graph.ndata['hv_segcn_out'] = h_feats_ligand
        batch_2_graph.ndata['hv_segcn_out'] = h_feats_receptor
        pre_interface_batch = []
        list_graph_1 = dgl.unbatch(batch_1_graph)
        list_graph_2 = dgl.unbatch(batch_2_graph)
        for ii in range(len(list_graph_1)):  
            h_1 = list_graph_1[ii].ndata['hv_segcn_out']
            h_2 = list_graph_2[ii].ndata['hv_segcn_out']            

            pred_proxy1 = torch.cat((self.clsf1(h_1), self.clsf1(h_2)), dim=0)
            pred_proxy2 = torch.cat((self.clsf2(h_1), self.clsf2(h_2)), dim=0)
            pred_proxy3 = torch.cat((self.clsf3(h_1), self.clsf3(h_2)), dim=0)
            pre_interface = (pred_proxy1 + pred_proxy2 + pred_proxy3)/3

            pre_interface_batch.append(pre_interface)        

        return [TEMP_1, TEMP_2, pre_interface_batch, batch_1_graph, batch_2_graph]


    def __repr__(self):
        return "SEGCN " + str(self.__dict__)


# =================================================================================================================
class AugHyE(nn.Module):

    def __init__(self, args):

        super(AugHyE, self).__init__()
        self.args = args
        self.device = args['device']
        self.segcn_original = SEGCN(args, max_radius=args['max_radius'])

    def forward(self, batch_ligand_graph, batch_receptor_graph):
        outputs = self.segcn_original(batch_ligand_graph, batch_receptor_graph)

        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


    def __repr__(self):
        return "AugHyE " + str(self.__dict__)



class FocalLoss_bsp(torch.nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss_bsp, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma
        self.bce_loss = torch.nn.BCELoss(reduction='none')  

    def forward(self, bsp, iface_label):
        at = self.alpha.to(bsp.device).gather(0, iface_label.long().view(-1))
        bce_loss = self.bce_loss(bsp, iface_label.float())
        pt = torch.exp(-bce_loss)
        focal_loss = at * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class AugHyE_model(nn.Module):

    def __init__(self, args):
        super(AugHyE_model, self).__init__()
        self._name = "AugHyE"
        self.args = args
        self.net = AugHyE(args)
        self.bsp_loss_func = FocalLoss_bsp()

    @property
    def name(self):
        return self._name

    def size(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def run_a_generic_epoch(self, ep_type, epoch_id, data_loader, optimizer=None):
        args = self.args
        if ep_type == 'train':
            self.net.train()
        else:
            self.net.eval()

        total_loss, total_interface_loss, total_stable_loss = 0., 0., 0.
        total_auc, total_ap = 0., 0.
        auc_list = []
        int_criterion = self.bsp_loss_func

        loop = tqdm(data_loader, total=len(data_loader),
                    desc=f'{ep_type.upper()} Epoch {epoch_id + 1}/{args["n_epochs"]}',
                    leave=True, dynamic_ncols=True)
        for batch in loop:
            if ep_type == 'train':
                optimizer.zero_grad()

            batch_ligand_graph = batch['ligand'].to(args['device'])
            batch_receptor_graph = batch['receptor'].to(args['device'])
            bsp_lig, bsp_rec = batch['bsp_lig'], batch['bsp_rec']

            TEMP_1, TEMP_2, pre_interface_list, _, _ = self.net(batch_ligand_graph,
                                                               batch_receptor_graph)
            batch_stable_loss = torch.max(torch.abs(torch.diff(TEMP_2))).to(args['device'])  # eq(14)
            batch_interface_loss = torch.zeros([]).to(args['device'])
            batch_auc = batch_ap = 0.

            for i in range(len(pre_interface_list)):
                bsp_pred = pre_interface_list[i].squeeze()
                label = torch.cat([bsp_lig[i], bsp_rec[i]], dim=0).to(args['device'])

                batch_interface_loss = batch_interface_loss + int_criterion(bsp_pred, label)

                label_np = label.detach().cpu().numpy()
                pred_np = bsp_pred.detach().cpu().numpy()

                auc = roc_auc_score(label_np, pred_np)
                ap = average_precision_score(label_np, pred_np)

                batch_auc += auc
                batch_ap += ap
                auc_list.append(auc)

            batch_num = len(pre_interface_list)

            loss = batch_interface_loss / batch_num + args['sr_loss_ratio'] * batch_stable_loss
            if ep_type == 'train':
                loss.backward()
                optimizer.step()

            loop.set_postfix(loss=loss.item(), AP=batch_ap / batch_num, AUC=batch_auc / batch_num)

            total_loss += loss.detach().item()
            total_interface_loss += batch_interface_loss.detach().item() / batch_num
            total_stable_loss += batch_stable_loss.detach().item()
            total_auc += batch_auc / batch_num
            total_ap += batch_ap / batch_num

        n_batch = len(data_loader)
        out = {
            'loss': total_loss / n_batch,
            'interface_loss': total_interface_loss / n_batch,
            'stable_loss': total_stable_loss / n_batch,
            'AUC_mean': total_auc / n_batch,
            'AP': total_ap / n_batch,
            'AUC_median': float(np.median(np.array(auc_list))),
        }
        print(f"Epoch {epoch_id} {ep_type.upper()} "
              f"Loss {out['loss']:.3f} AP {out['AP']:.3f} "
              f"AUC {out['AUC_mean']:.3f} AUC_Median {out['AUC_median']:.3f}")
        return out

    def train(self, train_loader, val_loader, best_model_save_path, last_metric_1=0.0,
              metrics_path=None):
        args = self.args
        optimizer = torch.optim.Adam(self.net.parameters(), lr=args['lr'],
                                        weight_decay=args['wd'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args['n_epochs'], eta_min=0.0)

        for epoch_id in range(args['n_epochs']):
            train_out = self.run_a_generic_epoch('train', epoch_id, train_loader, optimizer)
            with torch.no_grad():
                valid_out = self.run_a_generic_epoch('valid', epoch_id, val_loader)
            scheduler.step()

            record = {'epoch': epoch_id, 'timestamp': args['timestamp'],
                      'train_loss': train_out['loss'], 'valid_loss': valid_out['loss'],
                      'valid_auc': valid_out['AUC_mean'], 'valid_auc_median': valid_out['AUC_median'],
                      'valid_ap': valid_out['AP']}
            
            if metrics_path:
                with open(metrics_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')

            torch.save(self.net.state_dict(),
                       best_model_save_path.replace('_best_', f'_{epoch_id}_'))

            if valid_out['AUC_mean'] > last_metric_1 + 1e-4:
                last_metric_1 = valid_out['AUC_mean']
                torch.save(self.net.state_dict(), best_model_save_path)
                print(f'Save AugHyE model: {best_model_save_path}')
            print(f"Timestamp: {args['timestamp']} | best valid AUC {last_metric_1:.4f}")
            
        return last_metric_1

    def evaluate(self, args, data_loader, best_model_save_path, structure_type="bound"):
        self.load(best_model_save_path)
        with torch.no_grad():
            out = self.run_a_generic_epoch(f'test:{structure_type}', 0, data_loader)
        return out['loss'], out['AP'], out['AUC_median'], out

    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.args['device'])
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        self.net.load_state_dict(checkpoint, strict=True)
        print(f'Load AugHyE model: {load_path}')
