import os
import pickle

import dgl
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

from alignment_network.rigid_docking_model import Rigid_Body_Docking_Net


def create_model_equidock(args, log=print):
    return Rigid_Body_Docking_Net(args=args, log=log)


def load_alignment_model(args):
    ckpt_path = args['alignment_ckpt']
    ckpt = torch.load(ckpt_path, map_location=args['device'])
    if not (isinstance(ckpt, dict) and 'state_dict' in ckpt):
        raise ValueError(f"{ckpt_path} is not an EquiDock checkpoint dict")

    align_args = dict(args)
    align_args.update({k: v for k, v in ckpt['args'].items()
                       if k != 'device'})
    align_args['an_dropout'] = ckpt['args'].get('dropout', 0.)

    model = create_model_equidock(align_args).to(args['device'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print(f"alignment network loaded: {ckpt_path}")
    return model


def zerocopy_from_numpy(x):
    return torch.from_numpy(x)


def UniformRotation_Translation(translation_interval):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt(np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_interval)
    t = t * length
    return rotation_matrix.astype(np.float32), t.astype(np.float32)


def hetero_graph_from_sg_l_r_pair(ligand_graph, receptor_graph):
    ll = [('ligand', 'll', 'ligand'), (ligand_graph.edges()[0], ligand_graph.edges()[1])]
    rr = [('receptor', 'rr', 'receptor'), (receptor_graph.edges()[0], receptor_graph.edges()[1])]
    rl = [('receptor', 'cross', 'ligand'),
          (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    lr = [('ligand', 'cross', 'receptor'),
          (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    num_nodes = {'ligand': ligand_graph.num_nodes(), 'receptor': receptor_graph.num_nodes()}
    hetero_graph = dgl.heterograph({ll[0]: ll[1], rr[0]: rr[1], rl[0]: rl[1], lr[0]: lr[1]},
                                   num_nodes_dict=num_nodes)

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


def _load_pkl(data_path, name):
    path = os.path.join(data_path, name)
    print("file path: ", path)
    return pickle.load(open(path, 'rb'))


class DockingDataset(Dataset):

    def __init__(self, args, alignment_model, reload_mode, structure_type='native_bound'):
        self.args = args
        self.reload_mode = reload_mode
        self.alignment_model = alignment_model
        data_path = args['data_path']

        if reload_mode == 'train':
            self.data_bound = _load_pkl(data_path, f'{reload_mode}.pkl')
            self.data_generated = _load_pkl(data_path, f'{reload_mode}_esm3.pkl')
            self._alignment(self.data_generated)
            print(f"train pairs: bound {len(self.data_bound)} + generated {len(self.data_generated)}")
        
        elif reload_mode != 'train':
            if structure_type in ('native_bound', 'native_unbound'):
                self.data = _load_pkl(data_path, f'{reload_mode}_{structure_type}.pkl')
            elif structure_type == 'generated':
                self.data = _load_pkl(data_path, f'{reload_mode}_esm3.pkl')
                self._alignment(self.data)
            else:
                raise ValueError(f"unknown structure_type {structure_type!r}")

    def _alignment(self, generated):

        interval = self.args['translation_interval']
        with torch.inference_mode():
            for i in range(len(generated)):
                rot_T_lig, rot_b_lig = UniformRotation_Translation(translation_interval=interval)
                rot_T_rec, rot_b_rec = UniformRotation_Translation(translation_interval=interval)

                lig_g = generated[i]['lig_graph']
                rec_g = generated[i]['rec_graph']

                def _place(graph, rot_T, rot_b):
                    loc = graph.ndata['x'].detach().numpy()
                    loc = (rot_T @ (loc - loc.mean(axis=0, keepdims=True)).T).T + rot_b
                    graph.ndata['new_x'] = zerocopy_from_numpy(loc.astype(np.float32))

                _place(lig_g, rot_T_lig, rot_b_lig)
                _place(rec_g, rot_T_rec, rot_b_rec)

                hetero_graph = hetero_graph_from_sg_l_r_pair(lig_g, rec_g).to(self.args['device'])
                ligand_pred, receptor_pred, _, _, _, _, _, _ = self.alignment_model(hetero_graph, epoch=0)
                generated[i]['lig_pos'] = ligand_pred[0].detach().cpu().numpy()
                generated[i]['rec_pos'] = receptor_pred[0].detach().cpu().numpy()
                del lig_g.ndata['new_x'], rec_g.ndata['new_x']
        print("alignment completed")

    def __len__(self):
        if self.reload_mode == 'train':
            return len(self.data_bound) + len(self.data_generated)
        return len(self.data)

    def __getitem__(self, idx):
        if self.reload_mode == 'train':
            if idx < len(self.data_bound):
                data = self.data_bound[idx]
            else:
                data = self.data_generated[idx - len(self.data_bound)]
        else:
            data = self.data[idx]

        return {
            'lig_pos': zerocopy_from_numpy(data['lig_pos'].astype(np.float32)),
            'rec_pos': zerocopy_from_numpy(data['rec_pos'].astype(np.float32)),
            'lig_atom': zerocopy_from_numpy(data['lig_atom'].astype(np.int64)),
            'rec_atom': zerocopy_from_numpy(data['rec_atom'].astype(np.int64)),
            'bsp_lig': data['bsp_lig'],
            'bsp_rec': data['bsp_rec'],
            'lig_graph': data['lig_graph'],
            'rec_graph': data['rec_graph'],
            'file_name': data['filename'],
        }


def batchify_and_create_respective_graphs(batch):
    lig_graph_list, rec_graph_list = [], []
    bsp_lig_list, bsp_rec_list, file_name_list = [], [], []

    for batch_id, item in enumerate(batch):
        lig_graph = item['lig_graph'].clone()
        rec_graph = item['rec_graph'].clone()

        lig_graph.ndata['pos'] = item['lig_pos']
        lig_graph.ndata['atom'] = item['lig_atom']
        lig_graph.ndata['batch'] = torch.full((lig_graph.num_nodes(),), batch_id, dtype=torch.long)

        rec_graph.ndata['pos'] = item['rec_pos']
        rec_graph.ndata['atom'] = item['rec_atom']
        rec_graph.ndata['batch'] = torch.full((rec_graph.num_nodes(),), batch_id, dtype=torch.long)

        lig_graph_list.append(lig_graph)
        rec_graph_list.append(rec_graph)
        bsp_lig_list.append(item['bsp_lig'])
        bsp_rec_list.append(item['bsp_rec'])
        file_name_list.append(item['file_name'])

    return {
        'ligand': dgl.batch(lig_graph_list),
        'receptor': dgl.batch(rec_graph_list),
        'bsp_lig': bsp_lig_list,
        'bsp_rec': bsp_rec_list,
        'file_name': file_name_list,
    }


def training_dataset(args):
    alignment_model = load_alignment_model(args) 

    train_dataset = DockingDataset(args, alignment_model, reload_mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args['bs'], shuffle=True,
                                  collate_fn=batchify_and_create_respective_graphs)

    val_dataset = DockingDataset(args, alignment_model, reload_mode='val', structure_type='native_bound')
    val_dataloader = DataLoader(val_dataset, batch_size=args['bs'], shuffle=False,
                                collate_fn=batchify_and_create_respective_graphs)
    return train_dataloader, val_dataloader


def test_dataset(args):
    alignment_model = load_alignment_model(args)
    loaders = []
    for structure_type in ('native_bound', 'generated', 'native_unbound'):
        dataset = DockingDataset(args, alignment_model, reload_mode='test',
                                 structure_type=structure_type)
        loaders.append(DataLoader(dataset, batch_size=1, shuffle=False,
                                  collate_fn=batchify_and_create_respective_graphs))
    return tuple(loaders)
