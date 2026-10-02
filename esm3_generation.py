import argparse
import os

import attr
import numpy as np
import torch
from biopandas.pdb import PandasPdb
from huggingface_hub import login

from esmv3.esm3.models.esm3 import ESM3
from esmv3.esm3.sdk.api import ESMProtein, GenerationConfig

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
    'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
    'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G',
}


def seq3to1(residue):
    return THREE_TO_ONE.get(residue, 'X')


def read_pdb_residues(pdb_path):
    df = PandasPdb().read_pdb(pdb_path).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue',
                       'residue_name': 'resname', 'x_coord': 'x', 'y_coord': 'y',
                       'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    return list(df.groupby(['chain', 'residue', 'resname']))


def get_residues_db5(pdb_filename):
    return (read_pdb_residues(pdb_filename + '_l_b.pdb'),
            read_pdb_residues(pdb_filename + '_r_b.pdb'))


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
    alphaC_loc_list = []
    for residue in residues:
        df = residue[1]
        alphaCatom = df[df['atom_name'] == 'CA']
        alphaC_loc = alphaCatom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
        assert alphaC_loc.shape == (3,), \
            f"alphac loc shape problem, shape: {alphaC_loc.shape} residue {df} resid {df['residue']}"
        alphaC_loc_list.append(alphaC_loc)

    if len(alphaC_loc_list) <= 1:
        return None, None

    seq = [seq3to1(residue[1]['resname'].iloc[0]) for residue in residues]
    return np.stack(alphaC_loc_list, axis=0), np.stack(seq, axis=0)


def save_esm3_output_pdb(output, pdb_path, plddt_scale=100.0):
    pdb_path = os.path.abspath(pdb_path)
    base_no_ext = os.path.splitext(pdb_path)[0]
    if output.plddt is not None:
        out = attr.evolve(output, plddt=output.plddt * float(plddt_scale))
    else:
        out = output
    with open(pdb_path, 'w') as f:
        f.write(out.to_protein_chain().to_pdb_string())  # pdb file save
    if output.ptm is not None:
        ptm_val = float(output.ptm.detach().cpu().reshape(-1)[0])
        with open(base_no_ext + '_ptm.txt', 'w') as f:
            f.write(f"{ptm_val:.10f}\n")


def esm3_generation(model_esm3, raw_data_path, split, protein_complex,
                    decoding_step_sizes=8, temperature_rate=0.7):
    with open(os.path.join(raw_data_path, split + '.txt'), 'r') as f:
        complex_ids = [line.rstrip() for line in f.readlines()]
    print('Num of pairs in ', split, ' = ', len(complex_ids))

    structures_path = os.path.join(raw_data_path, 'structures')
    suffix = '_l_esm3.pdb' if protein_complex == 'ligand' else '_r_esm3.pdb'

    for i, complex_id in enumerate(complex_ids):
        pdb_filename = os.path.join(structures_path, complex_id)
        print(f'num: {i}  file: {pdb_filename}')

        residues0, residues1 = get_residues_db5(pdb_filename)
        _, lig_seq = get_alphaC_loc_array(filter_residues(residues0))
        _, rec_seq = get_alphaC_loc_array(filter_residues(residues1))
        if lig_seq is None or rec_seq is None:
            print('Skipping this pair')
            continue

        seq = lig_seq if protein_complex == 'ligand' else rec_seq
        generation_config = GenerationConfig(track='structure',
                                             num_steps=len(seq) // decoding_step_sizes,
                                             temperature=temperature_rate)
        with torch.inference_mode():
            output = model_esm3.generate(ESMProtein(sequence=''.join(seq)),
                                         generation_config)

        if output.coordinates is None:
            print('=====esm3 output coordinate is None!!!!=====')
            print('Skipping this pair')
            continue

        print(pdb_filename + suffix)
        save_esm3_output_pdb(output, pdb_filename + suffix)


def parseArgs(argv=None):
    parser = argparse.ArgumentParser(description='AugHyE ESM3 structure generation')
    parser.add_argument('-device', type=str, default='0')
    parser.add_argument('-data_path', type=str, default='data/DB5')
    parser.add_argument('-split', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('-decoding_steps', type=int, default=8)
    parser.add_argument('-temperature_rate', type=float, default=0.7)

    args = parser.parse_args(argv).__dict__
    args['device'] = torch.device(f"cuda:{args['device']}"
                                  if torch.cuda.is_available() and args['device'] != 'cpu'
                                  else 'cpu')
    return args


if __name__ == '__main__':
    args = parseArgs()
    print(f"split: {args['split']}  decoding_steps: {args['decoding_steps']}  "
          f"temperature_rate: {args['temperature_rate']}  device: {args['device']}")
    
    ## Require huggingface login
    hugginface_token = os.environ.get('HF_TOKEN')
    if hugginface_token:
        login(token=hugginface_token)

    model_esm3 = ESM3.from_pretrained('esm3_sm_open_v1', device=args['device'])
    for protein_complex in ('ligand', 'receptor'):
        esm3_generation(model_esm3, args['data_path'], args['split'], protein_complex,
                        decoding_step_sizes=args['decoding_steps'],
                        temperature_rate=args['temperature_rate'])
