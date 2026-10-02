import argparse
import math
import os
import pickle

import dgl
import numpy as np
import scipy.spatial as spa
import torch
from biopandas.pdb import PandasPdb
from esm import pretrained
from numpy import linalg as LA
from scipy.special import softmax
from sklearn.neighbors import BallTree

SIGMA = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))

dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
       'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
       'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
       'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}


def zerocopy_from_numpy(x):
    return torch.from_numpy(x)


def seq3to1(residue):
    return dit.get(residue, 'X')


def residue_type_one_hot_dips_not_one_hot(residue):
    indicator = {'Y': 0, 'R': 1, 'F': 2, 'G': 3, 'I': 4, 'V': 5,
                 'A': 6, 'W': 7, 'E': 8, 'H': 9, 'C': 10, 'N': 11,
                 'M': 12, 'D': 13, 'T': 14, 'S': 15, 'K': 16, 'L': 17, 'Q': 18, 'P': 19}
    if residue not in dit.keys():
        return 20
    return indicator[dit[residue]]


def residue_type_one_hot_dips(residue):
    allowable_set = ['Y', 'R', 'F', 'G', 'I', 'V', 'A', 'W', 'E', 'H',
                     'C', 'N', 'M', 'D', 'T', 'S', 'K', 'L', 'Q', 'P', None]
    res_name = dit[residue] if residue in dit.keys() else None
    return [res_name == s for s in allowable_set]


def residue_list_featurizer_dips_one_hot(residues):
    residue_list = [term[1]['resname'].iloc[0] for term in residues]
    feature_list = np.stack([residue_type_one_hot_dips(residue) for residue in residue_list])
    return {'res_feat': zerocopy_from_numpy(feature_list.astype(np.float32))}


def distance_list_featurizer(dist_list):
    length_scale_list = [1.5 ** x for x in range(15)]
    center_list = [0. for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)
    transformed_dist = [np.exp(- ((dist_list - center) ** 2) / float(length_scale))
                        for length_scale, center in zip(length_scale_list, center_list)]
    transformed_dist = np.array(transformed_dist).T.reshape((num_edge, -1))
    return {'he': zerocopy_from_numpy(transformed_dist.astype(np.float32))}


def sequence_list_featurizer_esm(residues, esm_model, batch_converter):
    seq_list = [term[1]['resname'].iloc[0] for term in residues]
    seq = [seq3to1(s) for s in seq_list]
    with torch.no_grad():
        _, _, batch_tokens = batch_converter([("protein", ''.join(seq))])
        embedding = esm_model(batch_tokens, repr_layers=[33])["representations"][33]
    embedding = embedding.squeeze(0)[1:-1]
    return {'esm': zerocopy_from_numpy(embedding.numpy().astype(np.float32))}


def read_pdb_residues(pdb_path):
    df = PandasPdb().read_pdb(pdb_path).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue',
                       'residue_name': 'resname', 'x_coord': 'x', 'y_coord': 'y',
                       'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    return list(df.groupby(['chain', 'residue', 'resname']))


def get_residues_db5(pdb_filename, data_type):
    if data_type == 'native_bound':
        ligand_name = pdb_filename + '_l_b.pdb'
        receptor_name = pdb_filename + '_r_b.pdb'
    elif data_type == 'esm3':
        ligand_name = pdb_filename + '_l_u_esm3.pdb'
        receptor_name = pdb_filename + '_r_u_esm3.pdb'
    elif data_type == 'native_unbound':
        ligand_name = pdb_filename + '_l_u.pdb'
        receptor_name = pdb_filename + '_r_u.pdb'
    else:
        raise ValueError(f"unknown data_type {data_type!r}")

    if not os.path.exists(ligand_name) or not os.path.exists(receptor_name):
        return None, None, None
    return read_pdb_residues(ligand_name), read_pdb_residues(receptor_name), pdb_filename


def filter_residues(residues):
    residues_filtered = []
    for residue in residues:
        df = residue[1]
        Natom = df[df['atom_name'] == 'N']
        alphaCatom = df[df['atom_name'] == 'CA']
        Catom = df[df['atom_name'] == 'C']
        if Natom.shape[0] == 1 and alphaCatom.shape[0] == 1 and Catom.shape[0] == 1:
            residues_filtered.append(residue)
    return residues_filtered


def get_alphaC_loc_array(residues):
    alphaC_loc_clean_list = []
    seq_atom = []
    for residue in residues:
        df = residue[1]
        alphaCatom = df[df['atom_name'] == 'CA']
        alphaC_loc = alphaCatom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
        assert alphaC_loc.shape == (3,), \
            f"alphac loc shape problem, shape: {alphaC_loc.shape} residue {df} resid {df['residue']}"
        alphaC_loc_clean_list.append(alphaC_loc)
        seq_atom.append(residue_type_one_hot_dips_not_one_hot(alphaCatom['resname'].tolist()[0]))

    if len(alphaC_loc_clean_list) <= 1:
        return None, None, None

    seq_list = [term[1]['resname'].iloc[0] for term in residues]
    seq = [seq3to1(s) for s in seq_list]
    return (np.stack(alphaC_loc_clean_list, axis=0),
            np.stack(seq_atom, axis=0),
            np.stack(seq, axis=0))


def extract_3d_coord_and_n_u_v_vecs(residues, residue_loc_is_alphaC):
    all_atom_coords_in_residue_list = []
    residue_representatives_loc_list = []
    n_i_list = []
    u_i_list = []
    v_i_list = []

    for residue in residues:
        df = residue[1]
        all_atom_coords_in_residue_list.append(df[['x', 'y', 'z']].to_numpy().astype(np.float32))

        Natom = df[df['atom_name'] == 'N']
        alphaCatom = df[df['atom_name'] == 'CA']
        Catom = df[df['atom_name'] == 'C']

        N_loc = Natom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
        if N_loc.shape != (3,):
            N_loc = N_loc.mean(0)
        alphaC_loc = alphaCatom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
        if alphaC_loc.shape != (3,):
            alphaC_loc = alphaC_loc.mean(0)
        C_loc = Catom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
        if C_loc.shape != (3,):
            C_loc = C_loc.mean(0)

        u_i = (N_loc - alphaC_loc) / LA.norm(N_loc - alphaC_loc)
        t_i = (C_loc - alphaC_loc) / LA.norm(C_loc - alphaC_loc)
        n_i = np.cross(u_i, t_i) / LA.norm(np.cross(u_i, t_i))
        v_i = np.cross(n_i, u_i)
        assert (math.fabs(LA.norm(v_i) - 1.) < 1e-5), \
            "protein utils protein_to_graph_dips, v_i norm larger than 1"

        n_i_list.append(n_i)
        u_i_list.append(u_i)
        v_i_list.append(v_i)

        if residue_loc_is_alphaC:
            residue_representatives_loc_list.append(alphaC_loc)
        else:
            heavy_df = df[df['element'] != 'H']
            residue_representatives_loc_list.append(
                heavy_df[['x', 'y', 'z']].mean(axis=0).to_numpy().astype(np.float32))

    num_residues = len(residues)
    if num_residues <= 1:
        raise ValueError("protein contains only 1 residue!")

    return (all_atom_coords_in_residue_list,
            np.stack(residue_representatives_loc_list, axis=0),
            np.stack(n_i_list, axis=0),
            np.stack(u_i_list, axis=0),
            np.stack(v_i_list, axis=0),
            num_residues)


def compute_dig_kNN_graph(args, num_residues, all_atom_coords_in_residue_list, residues,
                          loc_feat, n_i_feat, u_i_feat, v_i_feat,
                          esm_model, batch_converter):
    assert num_residues == loc_feat.shape[0]
    assert loc_feat.shape[1] == 3

    distance = np.full((num_residues, num_residues), np.inf)
    for i in range(num_residues - 1):
        for j in range((i + 1), num_residues):
            pairwise_dis = spa.distance.cdist(all_atom_coords_in_residue_list[i],
                                              all_atom_coords_in_residue_list[j])
            distance[i, j] = np.mean(pairwise_dis)
            distance[j, i] = np.mean(pairwise_dis)

    protein_graph = dgl.graph(([], []), idtype=torch.int32)
    protein_graph.add_nodes(num_residues)

    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []

    for i in range(num_residues):
        valid_src = list(np.where(distance[i, :] < args['graph_cutoff'])[0])
        assert i not in valid_src

        if len(valid_src) == 0:
            print(f"(dummy edge generation) Warning: Residue {i} has no neighbors "
                  f"within cutoff {args['graph_cutoff']}")
            src_list.append(i)
            dst_list.append(i)
            dist_list.append(args['graph_cutoff'])
            mean_norm_list.append(np.zeros(len(SIGMA)))
            continue

        if len(valid_src) > args['graph_max_neighbor']:
            valid_src = list(np.argsort(distance[i, :]))[0: args['graph_max_neighbor']]
        valid_dst = [i] * len(valid_src)
        dst_list.extend(valid_dst)
        src_list.extend(valid_src)

        valid_dist = list(distance[i, valid_src])
        dist_list.extend(valid_dist)

        valid_dist_np = distance[i, valid_src]
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / SIGMA, axis=1)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = loc_feat[valid_dst, :] - loc_feat[valid_src, :]
        mean_vec = weights.dot(diff_vecs)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))
        mean_norm_list.append(np.linalg.norm(mean_vec, axis=1) / denominator)

    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    protein_graph.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

    protein_graph.ndata.update(residue_list_featurizer_dips_one_hot(residues))
    protein_graph.ndata.update(sequence_list_featurizer_esm(residues, esm_model, batch_converter))
    protein_graph.edata.update(distance_list_featurizer(dist_list))

    edge_feat_ori_list = []
    for i in range(len(dist_list)):
        src = src_list[i]
        dst = dst_list[i]
        basis_matrix = np.stack((n_i_feat[dst, :], u_i_feat[dst, :], v_i_feat[dst, :]), axis=0)
        p_ij = np.matmul(basis_matrix, loc_feat[src, :] - loc_feat[dst, :])
        q_ij = np.matmul(basis_matrix, n_i_feat[src, :])
        k_ij = np.matmul(basis_matrix, u_i_feat[src, :])
        t_ij = np.matmul(basis_matrix, v_i_feat[src, :])
        edge_feat_ori_list.append(np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0))

    edge_feat_ori_feat = zerocopy_from_numpy(np.stack(edge_feat_ori_list, axis=0).astype(np.float32))
    # for alignment feature
    protein_graph.edata['he'] = torch.cat((protein_graph.edata['he'], edge_feat_ori_feat), axis=1)
    protein_graph.ndata['x'] = zerocopy_from_numpy(loc_feat.astype(np.float32))
    protein_graph.ndata['mu_r_norm'] = zerocopy_from_numpy(np.array(mean_norm_list).astype(np.float32))
    return protein_graph


def compute_bsp(bsp_threshold, lig_gt, rec_gt):
    """Interface labels from the native bound coordinates: a residue is positive when the closest
    residue of the partner chain is within bsp_threshold."""
    dist_rec, _ = BallTree(lig_gt).query(rec_gt, k=1)
    bsp_rec = np.zeros(rec_gt.shape[0], dtype=np.int64)
    bsp_rec[np.where(dist_rec <= bsp_threshold)[0]] = 1

    dist_lig, _ = BallTree(rec_gt).query(lig_gt, k=1)
    bsp_lig = np.zeros(lig_gt.shape[0], dtype=np.int64)
    bsp_lig[np.where(dist_lig <= bsp_threshold)[0]] = 1
    return bsp_lig, bsp_rec


def build_data_item(filename, lig_pos, rec_pos, lig_atom, rec_atom,
                    bsp_lig, bsp_rec, lig_graph, rec_graph):
    return {'lig_pos': lig_pos,
            'rec_pos': rec_pos,
            'lig_atom': lig_atom,
            'rec_atom': rec_atom,
            'bsp_lig': zerocopy_from_numpy(bsp_lig.astype(np.int64)),
            'bsp_rec': zerocopy_from_numpy(bsp_rec.astype(np.int64)),
            'lig_graph': lig_graph,
            'rec_graph': rec_graph,
            'filename': filename}


def read_split(raw_data_path, split):
    with open(os.path.join(raw_data_path, split + '.txt'), 'r') as f:
        complex_ids = [line.rstrip() for line in f.readlines()]
    print('Num of pairs in ', split, ' = ', len(complex_ids))
    return complex_ids


def lcs_residues(X, Y):
    """Longest common subsequence over residue names. Mirrors the reference
    implementation: strict > on the DP table and j -= 1 on ties."""
    name_X = [r[1]['resname'].iloc[0] for r in X]
    name_Y = [r[1]['resname'].iloc[0] for r in Y]
    m, n = len(name_X), len(name_Y)

    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if name_X[i - 1] == name_Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    kept_X_idx = []
    kept_Y_idx = []
    i, j = m, n
    while i > 0 and j > 0:
        if name_X[i - 1] == name_Y[j - 1]:
            kept_X_idx.insert(0, i - 1)
            kept_Y_idx.insert(0, j - 1)
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return kept_X_idx, kept_Y_idx


def build_train_val(args, split, esm_model, batch_converter, with_esm3=False):
    structures_path = os.path.join(args['data_path'], 'structures')
    data_bound = []
    data_esm3 = []

    for i, complex_id in enumerate(read_split(args['data_path'], split)):
        pdb_filename = os.path.join(structures_path, complex_id)
        print(f'num: {i}  file: {pdb_filename}')

        b_residues0, b_residues1, _ = get_residues_db5(pdb_filename, 'native_bound')
        if b_residues0 is None:
            print('Skipping this pair')
            continue
        bound_lig_residues = filter_residues(b_residues0)
        bound_rec_residues = filter_residues(b_residues1)

        bound_ligand_pos, lig_atom, _ = get_alphaC_loc_array(bound_lig_residues)
        bound_receptor_pos, rec_atom, _ = get_alphaC_loc_array(bound_rec_residues)
        if bound_ligand_pos is None or bound_receptor_pos is None:
            print('Skipping this pair')
            continue

        graph_inputs = [bound_lig_residues, bound_rec_residues]
        if with_esm3:
            e_residues0, e_residues1, _ = get_residues_db5(pdb_filename, 'esm3')
            if e_residues0 is None:
                print('Skipping this pair')
                continue
            esm3_lig_residues = filter_residues(e_residues0)
            esm3_rec_residues = filter_residues(e_residues1)
            if (len(esm3_lig_residues) != len(bound_lig_residues)
                    or len(esm3_rec_residues) != len(bound_rec_residues)):
                print('Skipping this pair')
                continue

            esm3_ligand_pos, e_lig_atom, _ = get_alphaC_loc_array(esm3_lig_residues)
            esm3_receptor_pos, e_rec_atom, _ = get_alphaC_loc_array(esm3_rec_residues)
            if esm3_ligand_pos is None or esm3_receptor_pos is None:
                print('Skipping this pair')
                continue
            graph_inputs += [esm3_lig_residues, esm3_rec_residues]

        ## label from native bound
        bsp_lig, bsp_rec = compute_bsp(args['bsp_threshold'], bound_ligand_pos, bound_receptor_pos)

        graphs = []
        for residues in graph_inputs:
            coords, loc_feat, n_i, u_i, v_i, num_res = extract_3d_coord_and_n_u_v_vecs(
                residues, args['graph_residue_loc_is_alphaC'])
            graphs.append(compute_dig_kNN_graph(args, num_res, coords, residues,
                                                loc_feat, n_i, u_i, v_i,
                                                esm_model, batch_converter))

        data_bound.append(build_data_item(pdb_filename, bound_ligand_pos, bound_receptor_pos,
                                          lig_atom, rec_atom, bsp_lig, bsp_rec,
                                          graphs[0], graphs[1]))
        if with_esm3:
            data_esm3.append(build_data_item(pdb_filename, esm3_ligand_pos, esm3_receptor_pos,
                                             e_lig_atom, e_rec_atom, bsp_lig, bsp_rec,
                                             graphs[2], graphs[3]))

    return data_bound, data_esm3


def matched_residues(pdb_filename):
    b_residues0, b_residues1, _ = get_residues_db5(pdb_filename, 'native_bound')
    if b_residues0 is None:
        return None
    u_residues0, u_residues1, _ = get_residues_db5(pdb_filename, 'native_unbound')
    if u_residues0 is None:
        return None

    b_lig = filter_residues(b_residues0)
    b_rec = filter_residues(b_residues1)
    u_lig = filter_residues(u_residues0)
    u_rec = filter_residues(u_residues1)
    if min(len(b_lig), len(b_rec), len(u_lig), len(u_rec)) <= 1:
        return None

    kept_b_lig, kept_u_lig = lcs_residues(b_lig, u_lig)
    kept_b_rec, kept_u_rec = lcs_residues(b_rec, u_rec)
    if min(len(kept_b_lig), len(kept_b_rec)) <= 1:
        return None
    return ([b_lig[k] for k in kept_b_lig], [b_rec[k] for k in kept_b_rec],
            [u_lig[k] for k in kept_u_lig], [u_rec[k] for k in kept_u_rec])


def build_test(args, split, esm_model, batch_converter):
    structures_path = os.path.join(args['data_path'], 'structures')
    data_bound = []
    data_unbound = []
    data_esm3 = []

    for i, complex_id in enumerate(read_split(args['data_path'], split)):
        pdb_filename = os.path.join(structures_path, complex_id)
        print(f'num: {i}  file: {pdb_filename}')

        matched = matched_residues(pdb_filename)
        if matched is None:
            print('Skipping the native pair')
        else:
            bound_lig_residues, bound_rec_residues, unbound_lig_residues, unbound_rec_residues = matched

            bound_ligand_pos, lig_atom, _ = get_alphaC_loc_array(bound_lig_residues)
            bound_receptor_pos, rec_atom, _ = get_alphaC_loc_array(bound_rec_residues)
            unbound_ligand_pos, u_lig_atom, _ = get_alphaC_loc_array(unbound_lig_residues)
            unbound_receptor_pos, u_rec_atom, _ = get_alphaC_loc_array(unbound_rec_residues)
            if (bound_ligand_pos is None or bound_receptor_pos is None
                    or unbound_ligand_pos is None or unbound_receptor_pos is None):
                print('Skipping the native pair')
            else:
                ## label from native bound
                bsp_lig, bsp_rec = compute_bsp(args['bsp_threshold'],
                                               bound_ligand_pos, bound_receptor_pos)

                graphs = []
                for residues in (bound_lig_residues, bound_rec_residues,
                                 unbound_lig_residues, unbound_rec_residues):
                    coords, loc_feat, n_i, u_i, v_i, num_res = extract_3d_coord_and_n_u_v_vecs(
                        residues, args['graph_residue_loc_is_alphaC'])
                    graphs.append(compute_dig_kNN_graph(args, num_res, coords, residues,
                                                        loc_feat, n_i, u_i, v_i,
                                                        esm_model, batch_converter))
                bound_lig_graph, bound_rec_graph, unbound_lig_graph, unbound_rec_graph = graphs

                data_bound.append(build_data_item(pdb_filename, bound_ligand_pos, bound_receptor_pos,
                                                  lig_atom, rec_atom, bsp_lig, bsp_rec,
                                                  bound_lig_graph, bound_rec_graph))
                data_unbound.append(build_data_item(pdb_filename, unbound_ligand_pos, unbound_receptor_pos,
                                                    u_lig_atom, u_rec_atom, bsp_lig, bsp_rec,
                                                    unbound_lig_graph, unbound_rec_graph))

        b_residues0, b_residues1, _ = get_residues_db5(pdb_filename, 'native_bound')
        e_residues0, e_residues1, _ = get_residues_db5(pdb_filename, 'esm3')
        if b_residues0 is None or e_residues0 is None:
            print('Skipping the ESM3 pair')
            continue
        raw_bound_lig_residues = filter_residues(b_residues0)
        raw_bound_rec_residues = filter_residues(b_residues1)
        esm3_lig_residues = filter_residues(e_residues0)
        esm3_rec_residues = filter_residues(e_residues1)
        if (len(esm3_lig_residues) != len(raw_bound_lig_residues)
                or len(esm3_rec_residues) != len(raw_bound_rec_residues)):
            print('Skipping the ESM3 pair')
            continue

        raw_bound_ligand_pos, _, _ = get_alphaC_loc_array(raw_bound_lig_residues)
        raw_bound_receptor_pos, _, _ = get_alphaC_loc_array(raw_bound_rec_residues)
        esm3_ligand_pos, e_lig_atom, _ = get_alphaC_loc_array(esm3_lig_residues)
        esm3_receptor_pos, e_rec_atom, _ = get_alphaC_loc_array(esm3_rec_residues)
        if (raw_bound_ligand_pos is None or raw_bound_receptor_pos is None
                or esm3_ligand_pos is None or esm3_receptor_pos is None):
            print('Skipping the ESM3 pair')
            continue

        ## label from native bound
        bsp_lig, bsp_rec = compute_bsp(args['bsp_threshold'],
                                           raw_bound_ligand_pos, raw_bound_receptor_pos)

        esm3_graphs = []
        for residues in (esm3_lig_residues, esm3_rec_residues):
            coords, loc_feat, n_i, u_i, v_i, num_res = extract_3d_coord_and_n_u_v_vecs(
                residues, args['graph_residue_loc_is_alphaC'])
            esm3_graphs.append(compute_dig_kNN_graph(args, num_res, coords, residues,
                                                     loc_feat, n_i, u_i, v_i,
                                                     esm_model, batch_converter))

        data_esm3.append(build_data_item(pdb_filename, esm3_ligand_pos, esm3_receptor_pos,
                                         e_lig_atom, e_rec_atom, bsp_lig, bsp_rec,
                                         esm3_graphs[0], esm3_graphs[1]))

    return data_bound, data_unbound, data_esm3


def build_data(args, split, esm_model, batch_converter):
    if split == 'test':
        bound, unbound, generated = build_test(args, split, esm_model, batch_converter)
        return {f'{split}_native_bound.pkl': bound,
                f'{split}_native_unbound.pkl': unbound,
                f'{split}_esm3.pkl': generated}
    if split == 'val':
        bound, _ = build_train_val(args, split, esm_model, batch_converter)
        return {f'{split}_native_bound.pkl': bound}
    bound, generated = build_train_val(args, split, esm_model, batch_converter, with_esm3=True)
    return {f'{split}.pkl': bound,
            f'{split}_esm3.pkl': generated}


def parseArgs(argv=None):
    parser = argparse.ArgumentParser(description='AugHyE DB5.5 preprocessing')
    parser.add_argument('-data_path', type=str, default='data/DB5')
    parser.add_argument('-out_dir', type=str, default='data')
    parser.add_argument('-split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('-bsp_threshold', type=float, default=8.0)
    parser.add_argument('-graph_cutoff', type=float, default=30)
    parser.add_argument('-graph_max_neighbor', type=int, default=10)
    parser.add_argument('-graph_residue_loc_is_alphaC', type=int, default=1)
    return parser.parse_args(argv).__dict__


if __name__ == '__main__':
    args = parseArgs()
    print(f"split: {args['split']}  data_path: {args['data_path']}")

    ## for esm2_embedding_feature
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    pickles = build_data(args, args['split'], esm_model, batch_converter)

    os.makedirs(args['out_dir'], exist_ok=True)
    for name, data in pickles.items():
        out_path = os.path.join(args['out_dir'], name)
        pickle.dump(data, open(out_path, 'wb'))
        print(f"Saved: {out_path}  ({len(data)} complexes)")
