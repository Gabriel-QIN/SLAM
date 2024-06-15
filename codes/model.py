#!/bin/python
# -*- coding:utf-8 -*- 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_scatter import scatter_mean

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
    def __init__(self, input_dim, hidden_dim, out_dim, win_size, kernel_size=9, strides=1, dropout=0.2):
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