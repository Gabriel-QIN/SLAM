#!/bin/python
# -*- coding:utf-8 -*- 

import os
import os.path as osp
import time
import argparse
import torch
import random
import re
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, TransformerConv, global_mean_pool
from metrics import eval_metrics
from torch_scatter import scatter_mean

def print_results(data, desc=['Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']):
    print('\t'.join(desc))
    print('\t'.join([f'{a:.3f}' if isinstance(a, float) else f'{a}' for a in data]))

atoms = ['N', 'CA', 'C', 'O', 'R', 'CB']
n_atoms = len(atoms)
atom_idx = {atom:atoms.index(atom) for atom in atoms}

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


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def detach(x):
    return x.cpu().detach().numpy().squeeze()

class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V

class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input_x, axis=1):
        input_size = input_x.size()
        trans_input = input_x.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input_x):
        x = torch.tanh(self.fc1(input_x))
        x = self.fc2(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		
        return attention

class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        h_E = self.edge_update(h_V, edge_index, h_E)
        h_V = self.context(h_V, batch_id)
        return h_V, h_E

class GraphEncoder(nn.Module):
    """
    # GraphEncoder = GraphEncoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=16, num_layers=3, dropout=0.9):
        super(GraphEncoder, self).__init__()
        
        self.node_project = nn.Linear(node_in_dim, 64, bias=True)
        self.edge_project = nn.Linear(edge_in_dim, 64, bias=True)
        self.bn_node = nn.BatchNorm1d(64)
        self.bn_edge = nn.BatchNorm1d(64)
        
        self.W_v = nn.Linear(64, hidden_dim, bias=True)
        self.W_e = nn.Linear(64, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=dropout, num_heads=4)
            for _ in range(num_layers))


    def forward(self, g):
        h_V, edge_index, h_E, batch_id = g.x, g.edge_index, g.edge_attr, g.batch
        h_V = self.W_v(self.bn_node(self.node_project(h_V)))
        h_E = self.W_e(self.bn_node(self.edge_project(h_E)))
        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
            # print(h_V.shape, h_E.shape)
        h_V = global_mean_pool(x=h_V,batch=batch_id).unsqueeze(1)
        # print(h_V.shape)
        return h_V

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, win_size, out_dim=64, kernel_size=3, strides=1, dropout=0.2):
        super(CNNEncoder, self).__init__()
        if kernel_size == 9:
            if win_size < (kernel_size - 1) * 2:
                kernel_size = 7
        self.kernel_size = kernel_size
        self.strides = strides
        self.emd = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = torch.nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(3, stride=strides)
        self.flat = nn.Flatten()
        # out_channels - (kernel_size - 1) * 2 - 2
        self.lin1 = nn.Linear(hidden_dim * (win_size - (kernel_size - 1) * 2 - 2), out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.emd(x.long())
        x = torch.permute(x, (0,2,1))
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(F.relu(self.lin1(x)))
        return x

class PLMEncoder(nn.Module):
    def __init__(self, BERT_encoder, out_dim, PLM_dim=1024, dropout=0.2):
        super(PLMEncoder, self).__init__()
        self.bert = BERT_encoder # BertModel.from_pretrained("Rostlab/prot_bert")
        for param in self.bert.base_model.parameters():
            param.requires_grad = False
        self.conv1 = nn.Conv1d(PLM_dim, out_dim, kernel_size=3, stride=1, padding='same')
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(out_dim)

    def forward(self, input_ids, attention_mask):
        pooled_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        self.bertout = pooled_output
        imput = pooled_output.permute(0, 2, 1) # shape: (Batch, 1024, length)
        conv1_output = F.relu(self.bn1(self.conv1(imput)))  # shape: (Batch, out_channel, length)
        output = self.dropout(conv1_output)
        prot_out = torch.mean(output, axis=2, keepdim=True) # shape: (Batch, out_channel, 1)
        prot_out = prot_out.permute(0, 2, 1)  # shape: (Batch, 1, out_channel)
        return prot_out

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, n_layers, dropout, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        self.emd_layer = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim*2, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        if self.bidirectional:
            self.lstm2 = nn.LSTM(hidden_dim * 4, out_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.lstm2 = nn.LSTM(hidden_dim * 2, out_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        emd = self.emd_layer(x.long())
        self.raw_emd = emd
        output, (final_hidden_state, final_cell_state) = self.lstm1(emd.float()) # shape: (Batch, length, 128)
        output = self.dropout1(output)
        lstmout2, (_, _) = self.lstm2(output) # shape: (Batch, length, 64)
        bi_lstm_output = self.dropout2(lstmout2)
        bi_lstm_output = torch.mean(bi_lstm_output, axis=1, keepdim=True) # shape: (Batch, 1, 64)
        return bi_lstm_output

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, win_size, kernel_size=3, strides=1, dropout=0.2):
        super(FeatureEncoder, self).__init__()
        self.hidden_channels = hidden_dim
        if win_size < (kernel_size - 1) * 2:
            kernel_size = 7
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv1 = torch.nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(3, stride=strides)
        self.flat = nn.Flatten()
        # out_channels - (kernel_size - 1) * 2 - 2
        self.lin1 = nn.Linear(hidden_dim * (win_size - (kernel_size - 1) * 2 - 2), out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.dropout1(self.bn1(self.conv1(x.permute(0, 2, 1)))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(F.relu(self.lin1(x)))
        return x

class MetaDecoder(nn.Module):
    def __init__(self, combined_dim, dropout=0.5):
        super(MetaDecoder, self).__init__()
        self.fc1 = nn.Linear(combined_dim, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 5)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(5, 1)
        self.w_omega = nn.Parameter(torch.Tensor(combined_dim, combined_dim))
        self.u_omega = nn.Parameter(torch.Tensor(combined_dim, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))

        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        self.att_score = att_score
        scored_x = x * att_score

        context = torch.sum(scored_x, dim=1)
        return context
    
    def forward(self,fused_x):
        fusion_output = torch.cat(fused_x, axis=2) # shape: (Batch, 1, 144=64+16+64)
        self.fusion_out = fusion_output
        attn_output = self.attention_net(fusion_output)  # shape: (Batch, 80)
        self.attn_out = attn_output
        x = F.relu(self.dropout1(self.fc1(attn_output)))
        x = F.relu(self.dropout2(self.fc2(x)))
        self.final_out = x
        logit = self.fc(x) # shape: (Batch, 1)
        return logit

class SLAMNet(nn.Module):
    """
    Parameter details.
    BERT_encoder: huggingface language model instance loaded by AutoModel function. e.g., 
    vocab_size: size of the dictionary of embeddings. [int]
    encoder_list: encoder used in combined model. ['cnn','lstm','plm','mpnn']. [list]
    PLM_dim: dimension of last hidden output for PLM.
    win_size: window size for modification-centered peptide. [int]
    embedding_dim: the size of each embedding vector.
    hidden_dim: hidden dimension for each encoder.
    out_dim: output dimension for each encoder.
    n_layers: number of BiLSTM layers.
    dropout: dropout rate.
    """
    def __init__(self, BERT_encoder, vocab_size, encoder_list=['cnn','lstm','plm'], PLM_dim=1024, win_size=51, embedding_dim=32, fea_dim=41, hidden_dim=64, out_dim=32, node_dim=96, edge_dim=495, gnn_layers=3, noise=0., n_layers=1, dropout=0.2, bidirectional=True):
        super(SLAMNet, self).__init__()
        dim_list = []
        self.encoder_list = encoder_list
        if 'cnn' in self.encoder_list:
            self.cnn_encoder = CNNEncoder(vocab_size, embed_dim=embedding_dim, hidden_dim=hidden_dim, win_size=win_size, out_dim=out_dim, dropout=dropout)
            dim_list.append(out_dim)
        if 'plm' in self.encoder_list:
            self.plm_encoder = PLMEncoder(BERT_encoder=BERT_encoder, out_dim=16, PLM_dim=PLM_dim, dropout=dropout)
            dim_list.append(16)
        if 'lstm' in self.encoder_list:
            self.lstm_encoder = BiLSTMEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
            if bidirectional:
                dim_list.append(out_dim*2)
            else:
                dim_list.append(out_dim)
        if 'fea' in self.encoder_list:
            self.fea_encoder = FeatureEncoder(input_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim, win_size=win_size, dropout=dropout)
            dim_list.append(out_dim)
        if 'gnn' in self.encoder_list:
            self.gnn_encoder = GraphEncoder(node_in_dim=node_dim, edge_in_dim=edge_dim, hidden_dim=16, num_layers=gnn_layers, dropout=0.9)
            gnn_dim = hidden_dim
            dim_list.append(16)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        combined_dim = sum(dim_list)
        self.decoder = MetaDecoder(combined_dim)

    def forward(self, input_ids, attention_mask, feature=None, g_data=None):
        fuse_x = []
        self.model_emd = {}
        if 'cnn' in self.encoder_list:
            # Encoder track 1 : CNN embeding layer
            cnn_out = self.cnn_encoder(input_ids).unsqueeze(1)
            fuse_x.append(cnn_out)
            self.model_emd['cnn_out'] = detach(cnn_out)

        if 'plm' in self.encoder_list:
            # Encoder track 2: PLM layer. shape: (Batch, length, 1024)
            prot_out = self.plm_encoder(input_ids, attention_mask)
            fuse_x.append(prot_out)
            self.model_emd['bert_out'] = detach(self.plm_encoder.bertout)
            self.model_emd['plm_out'] = detach(prot_out)

        if 'lstm' in self.encoder_list:
            # Encoder track 3 : LSTM layer.
            bi_lstm_output = self.lstm_encoder(input_ids)
            fuse_x.append(bi_lstm_output)
            self.model_emd['raw'] = detach(self.lstm_encoder.raw_emd)
            self.model_emd['lstm_out'] = detach(bi_lstm_output)

        if 'fea' in self.encoder_list and feature is not None:
            fea_out = self.fea_encoder(feature).unsqueeze(1)
            fuse_x.append(fea_out)
            self.model_emd['fea_in'] = detach(feature)
            self.model_emd['fea_out'] = detach(fea_out)
            
        if 'gnn' in self.encoder_list and g_data is not None:
            gnn_out = self.gnn_encoder(g_data)
            fuse_x.append(gnn_out)
            self.model_emd['gnn_out'] = detach(gnn_out)
        logit = self.decoder(fuse_x)
        self.model_emd['fusion_out'] = detach(self.decoder.fusion_out)
        self.model_emd['attn_out'] = detach(self.decoder.attn_out)
        self.model_emd['final_out'] = detach(self.decoder.final_out)
        return nn.Sigmoid()(logit)

    def _extract_embedding(self):
        print(f'Extract embedding from', list(model_emd.keys()))
        return self.model_emd

def detach(x):
    return x.cpu().detach().numpy().squeeze()

def random_run(SEED=2024):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"Random seed initialization: {SEED}!")

def train_one_epoch(loader, model, device, optimizer, criterion):
    model.train()
    train_step_loss = []
    train_total_acc = 0
    step = 1
    train_total_loss = 0
    for ind,(data) in enumerate(loader):
        g, input_ids, attention_mask, feature, label = data
        feature = feature.to(device)
        g = g.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, g_data=g)
        logits = pred.squeeze()
        loss = criterion(logits, label.float())
        acc = (logits.round() == label).float().mean()
        # print(f"Training ... Step:{step} | Loss:{loss.item():.4f} | Acc:{acc:.4f}")
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
        train_step_loss.append(loss.item())
        train_total_acc += acc
        step += 1
    avg_train_acc = train_total_acc / step
    avg_train_loss = train_total_loss / step
    return train_step_loss, avg_train_acc, avg_train_loss, step

def test_binary(model, loader, criterion, device):
    model.eval()
    criterion.to(device)
    test_probs = []
    test_targets = []
    valid_total_acc = 0
    valid_total_loss = 0
    valid_step = 1
    for ind,(data) in enumerate(loader):
        g, input_ids, attention_mask, feature, label = data
        feature = feature.to(device)
        g = g.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, g_data=g)
        logits = pred.squeeze()
        loss = criterion(logits, label.float())
        acc = (logits.round() == label).float().mean()
        # print(f"Valid step:{valid_step} | Loss:{loss.item():.4f} | Acc:{acc:.4f}")
        valid_total_loss += loss.item()
        valid_total_acc += acc.item()
        test_probs.extend(logits.cpu().detach().numpy())
        test_targets.extend(label.cpu().detach().numpy())
        valid_step += 1

    avg_valid_loss = valid_total_loss / valid_step
    avg_valid_acc = valid_total_acc / valid_step
    # print(f"Avg Valid Loss: {avg_valid_loss:.4f} | Avg Valid Acc: {avg_valid_acc:.4f}")
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    return test_probs, test_targets, avg_valid_loss, avg_valid_acc

def graph_collate_fn(batch):
    import torch_geometric
    # create mini-batched graph
    graph_batch = torch_geometric.data.Batch.from_data_list([item[0] for item in batch])  # Batch graph data
    # concat batched samples in the first dimension
    token = torch.stack([item[1] for item in batch], dim=0)  # Stack tensors
    attn = torch.stack([item[2] for item in batch], dim=0)  # Stack tensors
    fea  = torch.stack([item[3] for item in batch], dim=0)
    y = torch.stack([item[4] for item in batch], dim=0)
    return graph_batch, token, attn, fea, y

def arg_parse():
    # argument parser
    parser = argparse.ArgumentParser()
    # directory and file settings
    root_dir = 'Datasets'
    parser.add_argument("--project_name", default='SLAM_cv', type=str,
                            help="Project name for saving model checkpoints and best model. Default:`SLAM_cv`.")
    parser.add_argument("--train", default=osp.join(root_dir, 'general_train.fa'), type=str,
                            help="Data directory. Default:`'Datasets/general_train.fa'`.")
    parser.add_argument("--test", default=osp.join(root_dir, 'general_test.fa'), type=str,
                            help="Data directory. Default:`'Datasets/general_test.fa'`.")
    parser.add_argument("--model", default='Models/SLAM_cv', type=str,
                        help="Directory for model storage and logits. Default:`Models/SLAM_cv`.")
    parser.add_argument("--result", default='result/SLAM_cv', type=str,
                        help="Result directory for model training and evaluation. Default:`result/SLAM_cv`.")
    parser.add_argument("--PLM", default='Rostlab/prot_bert', type=str,
                        help="PLM directory. Default:`Rostlab/prot_bert`.")
    parser.add_argument("--pdb_dir", default='Datasets/Structure/PDB', type=str,
                        help="PDB directory. Default:`Datasets/Structure/PDB`.")
    # Experiement settings
    parser.add_argument('--epoch', type=int, default=500, metavar='[Int]',
                        help='Number of training epochs. (default:500)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='[Float]',
                        help='Learning rate. (default:1e-4)')
    parser.add_argument('--batch', type=int, default=128, metavar='[Int]',
                        help='Batch size cutting threshold for Dataloader.(default:128)')
    parser.add_argument('--cpu', '-cpu', type=int, default=4, metavar='[Int]',
                        help='CPU processors for data loading.(default:4).')
    parser.add_argument('--gpu', '-gpu', type=int, default=0, metavar='[Int]',
                        help='GPU id.(default:0).')
    parser.add_argument('--emd_dim', '-ed', type=int, default=32, metavar='[Int]',
                        help='Word embedding dimension.(default:32).')
    parser.add_argument('--hidden_dim', '-hd', type=int, default=64, metavar='[Int]',
                        help='Hidden dimension.(default:64).')
    parser.add_argument('--out_dim', '-od', type=int, default=32, metavar='[Int]',
                        help='Out dimension for each track.(default:32).')
    parser.add_argument('--gnn_layers', '-gl', type=int, default=5, metavar='[Int]',
                        help='Number of GNN layer.(default:5).') 
    parser.add_argument('--nneighbor', '-nb', type=int, default=32, metavar='[Int]',
                        help='Number of residue neighbors in protein graph.(default:32).') 
    parser.add_argument('--atom_type', '-at', type=str, default='CA', metavar='[Str]',
                        help='Atom type for construting the protein graph.(default:`CA`).') 
    parser.add_argument('--dropout', '-dp', type=float, default=0.5, metavar='[Float]',
                        help='Dropout rate.(default:0.5).')
    parser.add_argument('--encoder', type=str, default='cnn,lstm,fea,gnn,plm', metavar='[Str]',
                        help='Encoder list separated by comma chosen from cnn,lstm,fea,plm,gnn. (default:`cnn,lstm,fea,gnn`)')
    parser.add_argument('--seed', type=int, default=2024, metavar='[Int]',
                        help='Random seed. (default:2024)')
    parser.add_argument('--patience', type=int, default=50, metavar='[Int]',
                        help='Early stopping patience. (default:50)')
    return parser.parse_args()

def K_fold_split(datalist, ratio=0.2, k=5, seed=2024):
    """Return K-fold split in a list format, including a tuple with (train_set, test_set)."""
    # random.seed(seed)
    random.shuffle(datalist)
    num_samples = len(datalist)
    split_num = int(num_samples * float(ratio))
    fold_list = []
    for fold in range(k):
        start_idx = fold * split_num
        end_idx = (fold + 1) * split_num  if fold < k -1 else num_samples
        test_set = datalist[start_idx:end_idx]
        train_set = datalist[:start_idx] + datalist[end_idx:]
        fold_list.append((train_set, test_set))
        # print(start_idx, end_idx)
    return fold_list

if __name__=='__main__':
    save_model=False
    args = arg_parse()
    print(args)
    project = args.project_name
    SEED = args.seed
    random_run(SEED)
    embedding_dim = args.emd_dim
    hidden_dim = args.hidden_dim
    out_dim = args.out_dim
    lr = args.learning_rate
    num_epochs = args.epoch
    batch_size = args.batch
    cpu = args.cpu
    gpu = args.gpu
    model_dir = osp.join(args.model, f'{project}')
    os.makedirs(model_dir, exist_ok=True)
    result_dir = osp.join(args.result, f'{project}')
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    pretrained_model = args.PLM # 'pretrained_LM/prot_bert/'
    local_files_only = False
    encoder_list = args.encoder.split(',') # ['cnn','lstm','fea']
    n_layers = 1
    dropout = args.dropout
    patience = num_epochs # do not use early stopping for cross-validation in case model cannot converge
    if 'bert' in pretrained_model:
        PLM_dim = 1024
    elif 'esm' in pretrained_model:
        PLM_dim = 1280

    manual_fea = [BLOSUM62, BINA]
    fea_dim = 41

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False, use_fast=False)
    if 'plm' in encoder_list:
        BERT_encoder = AutoModel.from_pretrained(pretrained_model, local_files_only=local_files_only, output_attentions=False).to(device)
    else:
        BERT_encoder = None

    # graph and GNN parameters
    node_dim = 267
    edge_dim = 632
    nneighbor = args.nneighbor
    atom_type = args.atom_type #'CA' # CB, R, C, N, O
    gnn_layers = args.gnn_layers
    pdb_dir = args.pdb_dir

    train_file = args.train
    datalist = [record for record in SeqIO.parse(train_file, "fasta")]
    
    k = 5
    split_ratio = 1/k
    k_fold_list = K_fold_split(datalist,split_ratio,k)
    performance_list = []
    loss_dict = {}
    # test_file = args.test
    # test_list = [record for record in SeqIO.parse(test_file, "fasta")]
    # test_ds = SLAMDataset(test_list, tokenizer, pdb_dir=pdb_dir, feature=manual_fea, nneighbor=nneighbor, atom_type=atom_type)

    for fold, data in enumerate(k_fold_list):
        seed = random.randint(1,10000)
        random_run(seed)
        train_list, valid_list = data
        train_ds = SLAMDataset(train_list, tokenizer, pdb_dir=pdb_dir, feature=manual_fea, nneighbor=nneighbor, atom_type=atom_type)
        valid_ds = SLAMDataset(valid_list, tokenizer, pdb_dir=pdb_dir, feature=manual_fea, nneighbor=nneighbor, atom_type=atom_type)

        window_size = valid_ds.win_size
        print(f"Fold {fold} | Training dataset: {len(train_ds)}   |Testing dataset: {len(valid_ds)}")

        train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=cpu, collate_fn=graph_collate_fn, prefetch_factor=2)
        valid_loader = DataLoader(valid_ds,batch_size=batch_size,shuffle=True,num_workers=cpu, collate_fn=graph_collate_fn, prefetch_factor=2)

        model = SLAMNet(BERT_encoder=BERT_encoder, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,PLM_dim=PLM_dim,win_size=window_size,embedding_dim=embedding_dim, fea_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim,node_dim=node_dim, edge_dim=edge_dim, gnn_layers=gnn_layers,n_layers=n_layers,dropout=dropout).to(device)
        # model.apply(weight_init)
        params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        print("Model Trainable Parameter: "+ str(params/1024/1024) + 'Mb' + "\n")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.BCELoss().to(device)
        result_list = []
        all_train_loss_list = []
        best_auc = 0
        best_loss = 100
        best_epoch = 0
        patience = 0
        max_patience = args.patience
        desc=['Fold', 'Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']
        for epoch in range(num_epochs):
            # Training
            start = time.perf_counter()
            train_step_loss, train_acc, train_loss, step = train_one_epoch(train_loader, model, device, optimizer, criterion)
            all_train_loss_list.extend(train_step_loss)
            end = time.perf_counter()
            print(f"Fold {fold} | Epoch {epoch+1} | {(end - start):.4f}s | Train | Loss: {train_loss: .6f}| Train acc: {train_acc:.4f}")
            start = time.perf_counter()
            valid_probs, valid_labels, valid_loss, valid_acc = test_binary(model, valid_loader, criterion, device)
            end = time.perf_counter()
            print(f"Fold {fold} | Epoch {epoch+1} | {(end - start):.4f}s | Test | Test loss: {valid_loss:.6f}| Test acc: {valid_acc:.4f}")
            acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(valid_probs, valid_labels)
            result_info = [fold, epoch, (tn+tp)/(tn+tp+fp+fn), th_, rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
            result_list.append(result_info)
            print_results(result_info, desc)
            if best_loss < valid_loss:
                patience += 1
            if patience > max_patience:
                break
            if best_loss < valid_loss:
                best_loss = valid_loss
                best_auc = auc_
                best_acc = acc_
                best_epoch = epoch
                best_test_probs = valid_probs
                best_test_labels = valid_labels
                best_result = result_info
                if save_model:
                    save_path = osp.join(model_dir, f'best_{project}_model_epoch_fold_{fold}.pt')
                    torch.save(model.state_dict(), save_path)
        print(f'\n Fold {fold} best result:\n')
        print_results(best_result, desc)
        # Save training step loss
        loss_df = pd.DataFrame(all_train_loss_list)
        loss_df.columns = ['Loss']
        loss_df.to_csv(osp.join(result_dir, f'fold{fold}_train_step_loss.txt'), sep='\t')
        
        # Save predicted logits and labels
        logit_df = pd.DataFrame([best_test_labels, best_test_probs]).transpose()
        logit_df.columns = ['Label', 'Logit']
        logit_df.to_csv(osp.join(result_dir, f"fold{fold}_logits_results.txt"), sep='\t', index=False)

        epoch_df = pd.DataFrame(result_list)
        epoch_df.columns = desc
        epoch_df.to_csv(osp.join(result_dir, f'fold{fold}_epoch_result.csv'), sep='\t', index=False)

        epoch_df = pd.DataFrame([best_result])
        epoch_df.columns = desc
        epoch_df.to_csv(osp.join(result_dir, f'fold{fold}_best_result.csv'), sep='\t', index=False)