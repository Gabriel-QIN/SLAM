#!/bin/python
# -*- coding:utf-8 -*- 

import os
import torch
import random
import re
import numpy as np
import pandas as pd
import torch.nn.functional as F
from Bio import SeqIO
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

# Thanks for StructTrans and PiFold
# https://github.com/jingraham/neurips19-graph-protein-design
# https://github.com/A4Bio/PiFold

atoms = ['N', 'CA', 'C', 'O', 'R', 'CB']
n_atoms = len(atoms)
atom_idx = {atom:atoms.index(atom) for atom in atoms}

def random_split(datalist, ratio, seed=SEED):
    """Randomly split dataset list into train and test."""
    random.seed(seed)
    random.shuffle(datalist)
    # index_list = [i for i in range(num_samples)]
    # random.shuffle(index_list)
    num_samples = len(datalist)
    split_num = int(num_samples * float(ratio))
    large_list = datalist[split_num:]
    small_list = datalist[:split_num]
    return large_list, small_list

def BLOSUM62(fastas, **kw):
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # X
    }
    encodings = []
    for sequence in fastas:
        code = []
        for aa in sequence:
            code = blosum62[aa]
            encodings.append(code)
    arr = np.array(encodings)
    # scaler = StandardScaler().fit(arr)
    # arr = scaler.transform(arr)
    return arr

def BINA(fastas, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYVX'
    encodings = []
    for sequence in fastas:
        for aa in sequence:
            if aa not in AA:
                aa = 'X'
            if aa == 'X':
                code = [0 for _ in range(len(AA))]
                encodings.append(code)
                continue
            code = []
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
            encodings.append(code)
    arr = np.array(encodings)
    # scaler = StandardScaler().fit(arr)
    # arr = scaler.transform(arr)
    return arr

def get_cb(n, ca, c):
    b = ca - n
    c = c - ca
    a = np.cross(b, c)
    cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + ca
    return cb

def calculate_distances(atom_coordinates, query_position):
    distances = np.linalg.norm(atom_coordinates - query_position, axis=1)
    return distances

def parse_pdb(pdb_file, pos=None, atom_type='CA', nneighbor=32, cal_cb=True):
    """
    ########## Process PDB file ##########
    """
    current_pos = -1000
    X = []
    current_aa = {} # N, CA, C, O, R
    with open(pdb_file, 'r') as pdb_f:
        for line in pdb_f:
            if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
                if current_aa != {}:
                    R_group = []
                    for atom in current_aa:
                        if atom not in ["N", "CA", "C", "O"]:
                            R_group.append(current_aa[atom])
                    if R_group == []:
                        R_group = [current_aa["CA"]]
                    R_group = np.array(R_group).mean(0)
                    X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                    current_aa = {}
                if line[0:4].strip() != "TER":
                    current_pos = int(line[22:26].strip())

            if line[0:4].strip() == "ATOM":
                atom = line[13:16].strip()
                if atom != "H":
                    xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                    current_aa[atom] = xyz
    X = np.array(X)
    if cal_cb:
        X = np.concatenate([X, get_cb(X[:,0], X[:,1], X[:,2])[:, None]], 1)
    if pos is not None:
        atom_ind = atom_idx[atom_type] # CA atom
        if pos >= X.shape[0]:
            pos = X.shape[0] - 1
        query_coord = X[pos,atom_ind]
        distances = calculate_distances(X[:,atom_ind,:], query_coord)
        closest_indices = np.argsort(distances)[:nneighbor]
        X = X[closest_indices]
    return X # array shape: [Length, 6, 3] N, CA, C, O, R, CB


def get_geo_feat(X, edge_index):
    """
    ##############  Geometric Featurizer  ##############
    """
    pos_embeddings = _positional_encodings(edge_index)
    node_angles = _get_angle(X) # 12D
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    node = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    edge = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    return node, edge


def _positional_encodings(edge_index, num_embeddings=16):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE

def _get_angle(X, eps=1e-7):
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles # dim = 12

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _get_distance(X, edge_index):
    atom_N = X[:,atom_idx['N']]  # [L, 3]
    atom_CA = X[:,atom_idx['CA']]
    atom_C = X[:,atom_idx['C']]
    atom_O = X[:,atom_idx['O']]
    atom_R = X[:,atom_idx['R']]
    atom_CB = X[:,atom_idx['CB']]

    node_list = ['N-CA', 'N-C', 'N-O', 'N-R', 'N-CB', 'CA-C', 'CA-O', 'CA-R', 'CA-CB', 'C-O', 'C-R', 'C-CB', 'O-R', 'O-CB', 'R-CB']
    # node_list = ['N-CA', 'N-C', 'N-O', 'N-R', 'CA-C', 'CA-O', 'CA-R',  'C-O', 'C-R',  'O-R']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1) # shape = [N, 15 * 16]
    atom_list = ["N", "CA", "C", "O", "R", "CB"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1) # shape = [E, 36 * 16]
    return node_dist, edge_dist # 240D node features + 576D edge features when dim of rbf is set to 16

def _get_direction_orientation(X, edge_index): # N, CA, C, O, R, CB
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    Q = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index

    t = F.normalize(X[:, [0,2,3,4,5]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    node_direction = torch.matmul(t, Q).reshape(t.shape[0], -1) # [L, 4 * 3]

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, Q[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, Q[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3]
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(Q[node_i].transpose(-1,-2), Q[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]
    return node_direction, edge_direction, edge_orientation

def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [N,3,3]
        Q [N,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q

def get_graph_fea(pdb_path, pos, nneighbor=32, radius=10, atom_type='CA', cal_cb=True):
    X = torch.tensor(parse_pdb(pdb_path, pos=pos, atom_type=atom_type, nneighbor=nneighbor, cal_cb=cal_cb)).float()
    query_atom = X[:, atom_idx[atom_type]]
    edge_index = radius_graph(query_atom, r=radius, loop=False, max_num_neighbors=nneighbor, num_workers = 4)
    node, edge = get_geo_feat(X, edge_index)
    return Data(x=node, edge_index=edge_index, edge_attr=edge, name=os.path.basename(pdb_path).split('.')[0])

class SLAMDataset(object):
    def __init__(self, seqlist, tokenizer, pdb_dir=None, feature=None, nneighbor=32, atom_type='CA'):
        self.seq_list = []
        self.label_list = []
        self.feature_list = []
        self.pdb_list = []
        self.pdb_entry_list = []
        self.node = []
        self.edge = []
        self.relpos = []
        self.cons = []
        self.pdb_dir = pdb_dir
        self.tokenizer = tokenizer
        self.feature = feature
        self.max_edge_num = 1000
        ind = 0
        for record in tqdm(seqlist):
            seq = str(record.seq)
            desc = record.id.split('|')
            name,label = desc[0],int(desc[1])
            if len(desc) == 3:
                pos,length = 0, 0
            else:
                pos,length = int(desc[3]),int(desc[4])
            pdb_path = os.path.join(pdb_dir, f'{name}.pdb')
            if not os.path.exists(pdb_path):
                continue
            else:
                data = get_graph_fea(pdb_path, pos, nneighbor=nneighbor, atom_type=atom_type, cal_cb=True)
                self.pdb_entry_list.append(data)
            fea = self._get_encoding(seq, feature)
            self.feature_list.append(fea)
            self.label_list.append(int(label))
            self.seq_list.append(seq)
            ind += 1
            self.win_size = len(seq)

    def __getitem__(self, index):
        seq = self.seq_list[index]
        seq = [token for token in re.sub(r"[UZOB*]", "X", seq.rstrip('*'))]
        max_len = len(seq)
        encoded = self.tokenizer.encode_plus(' '.join(seq), add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True,truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = encoded['input_ids'].flatten()
        attention_mask = encoded['attention_mask'].flatten()
        return self.pdb_entry_list[index], input_ids, attention_mask, torch.tensor(self.feature_list[index], dtype=torch.float), torch.tensor(self.label_list[index])
                
    def __len__(self):
        return len(self.pdb_entry_list)
    
    def _get_encoding(self, seq, feature=[BLOSUM62, BINA]):
        alphabet = 'ARNDCQEGHILKMFPSTWYVX'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        sample = ''.join([re.sub(r"[UZOB*]", "X", token) for token in seq])
        # seq = [char_to_int[char] for char in sample]
        max_len = len(sample)
        all_fea = []
        for encoder in feature:
            fea = encoder([sample])
            assert fea.shape[0] == max_len
            all_fea.append(fea)
        return np.hstack(all_fea)