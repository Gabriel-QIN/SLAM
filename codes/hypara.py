import dataclasses
import torch

##  Hyper Parameters  ##
@dataclasses.dataclass
class HyperParam_SLAM:
    gpu:              int = 0
    n_layers:         int = 1
    dropout:          float = 0.5
    embedding_dim:    int = 32
    hidden_dim:       int = 64
    out_dim:          int = 32
    node_dim:         int = 267
    edge_dim:         int = 632
    nneighbor:        int = 64
    atom_type:        str = 'CA' # CB, R, C, N, O
    gnn_layers:       int = 5
    encoder_list:     str = ','.join(['cnn', 'lstm', 'fea', 'gnn', 'plm'])
    fea_dim:          int = 41
    PLM_dim:          int = 1024
    window_size:      int = 51
    pretrained_model: str = '/public/softwares/SLAM/01.SLAM/prot_bert/'
    model_dir:        str = '/public/softwares/SLAM/01.SLAM/Models/SLAM'

class HyperParam_seq:
    gpu:              int = 0
    n_layers:         int = 1
    dropout:          float = 0.5
    embedding_dim:    int = 32
    hidden_dim:       int = 64
    out_dim:          int = 32
    node_dim:         int = 267
    edge_dim:         int = 632
    nneighbor:        int = 64
    atom_type:        str = 'CA' # CB, R, C, N, O
    gnn_layers:       int = 5
    encoder_list:     str = ','.join(['cnn', 'lstm', 'fea'])
    fea_dim:          int = 41
    PLM_dim:          int = 1024
    window_size:      int = 51
    pretrained_model: str = '/public/softwares/SLAM/01.SLAM/prot_bert/'
    model_dir:        str = '/public/softwares/SLAM/01.SLAM/Models/SLAM_seq'
