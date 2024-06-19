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
from metrics import eval_metrics
from dataset import random_split

def print_results(data, desc=['Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']):
    print('\t'.join(desc))
    print('\t'.join([f'{a:.3f}' if isinstance(a, float) else f'{a}' for a in data]))

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

class SLAMDatasetSeq(object):
    def __init__(self, seqlist, tokenizer, feature=None):
        self.seq_list = []
        self.label_list = []
        self.feature_list = []
        self.tokenizer = tokenizer
        self.feature = feature
        ind = 0
        for record in tqdm(seqlist):
            seq = str(record.seq)
            desc = record.id.split('|')
            name,label = desc[0],int(desc[1])
            if len(desc) == 3:
                pos,length = 0, 0
            else:
                pos,length = int(desc[3]),int(desc[4])
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
        return input_ids, attention_mask, torch.tensor(self.feature_list[index], dtype=torch.float), torch.tensor(self.label_list[index], dtype=torch.long)
    
    def __len__(self):
        return len(self.seq_list)
    
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

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, win_size, out_dim=64, kernel_size=3, strides=1, dropout=0.2):
        super(CNNEncoder, self).__init__()
        # if win_size < (kernel_size - 1) * 2:
        #     kernel_size = 7
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
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim*2, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        if bidirectional:
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

class SLAMNetSeq(nn.Module):
    """
    Parameter details.
    BERT_encoder: huggingface language model instance loaded by AutoModel function. e.g., 
    vocab_size: size of the dictionary of embeddings. [int]
    encoder_list: encoder used in combined model. ['cnn','lstm','plm', 'fea', 'gnn']. [list]
    PLM_dim: dimension of last hidden output for PLM.
    win_size: window size for modification-centered peptide. [int]
    embedding_dim: the size of each embedding vector.
    hidden_dim: hidden dimension for each encoder.
    out_dim: output dimension for each encoder.
    n_layers: number of BiLSTM layers.
    dropout: dropout rate.
    """
    def __init__(self, BERT_encoder, vocab_size, encoder_list=['cnn','lstm','fea'], win_size=51, embedding_dim=32, fea_dim=41, hidden_dim=64, out_dim=32, kernel_size=9, bidirectional=True, n_layers=1, dropout=0.2):
        super(SLAMNetSeq, self).__init__()
        dim_list = []
        self.encoder_list = encoder_list
        if 'cnn' in self.encoder_list:
            self.cnn_encoder = CNNEncoder(vocab_size, embed_dim=embedding_dim, hidden_dim=hidden_dim, win_size=win_size, out_dim=out_dim, kernel_size=kernel_size,dropout=dropout)
            dim_list.append(out_dim)
        if 'plm' in self.encoder_list:
            self.plm_encoder = PLMEncoder(BERT_encoder=BERT_encoder, out_dim=out_dim, PLM_dim=PLM_dim, kernel_size=kernel_size, dropout=dropout)
            dim_list.append(out_dim)
        if 'lstm' in self.encoder_list:
            self.lstm_encoder = BiLSTMEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
            dim_list.append(out_dim*2)
        if 'fea' in self.encoder_list:
            self.fea_encoder = FeatureEncoder(input_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim, win_size=win_size, dropout=dropout)
            dim_list.append(out_dim)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        combined_dim = sum(dim_list)
        self.decoder = MetaDecoder(combined_dim)

    def forward(self, input_ids, attention_mask, feature=None):
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
            
        logit = self.decoder(fuse_x)
        self.model_emd['fusion_out'] = detach(self.decoder.fusion_out)
        self.model_emd['attn_out'] = detach(self.decoder.attn_out)
        self.model_emd['final_out'] = detach(self.decoder.final_out)
        return nn.Sigmoid()(logit)

    def _extract_embedding(self):
        print(f'Extract embedding from', list(model_emd.keys()))
        return self.model_emd

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
        input_ids, attention_mask, feature, label = data
        feature = feature.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature)
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
        input_ids, attention_mask, feature, label = data
        feature = feature.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature)
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

def arg_parse():
    # argument parser
    parser = argparse.ArgumentParser()
    # directory and file settings
    root_dir = 'Datasets'
    parser.add_argument("--project_name", default='SLAM_seq', type=str,
                            help="Project name for saving model checkpoints and best model. Default:`SLAM_seq`.")
    parser.add_argument("--train", default=osp.join(root_dir, 'general_train.fa'), type=str,
                            help="Data directory. Default:`'Datasets/general_train.fa'`.")
    parser.add_argument("--test", default=osp.join(root_dir, 'general_test.fa'), type=str,
                            help="Data directory. Default:`'general_test.fa'`.")
    parser.add_argument("--model", default='Models/SLAM_wo_plm_and_structure', type=str,
                        help="Directory for model storage and logits. Default:`Models/SLAM_wo_plm_and_structure`.")
    parser.add_argument("--result", default='result', type=str,
                        help="Result directory for model training and evaluation. Default:`result`.")
    parser.add_argument("--PLM", default='Rostlab/prot_bert', type=str,
                        help="PLM directory. Default:`Rostlab/prot_bert`.")
    # Experiement settings
    parser.add_argument('--epoch', type=int, default=200, metavar='[Int]',
                        help='Number of training epochs. (default:200)')
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
    parser.add_argument('--lstm_nlayer', '-ln', type=int, default=1, metavar='[Int]',
                        help='Number of LSTM layer.(default:1).') 
    parser.add_argument('--dropout', '-dp', type=float, default=0.5, metavar='[Float]',
                        help='Dropout rate.(default:0.5).')                 
    parser.add_argument('--encoder', type=str, default='cnn,lstm,fea', metavar='[Str]',
                        help='Encoder list separated by comma chosen from cnn,lstm,fea,plm,gnn. (default:`cnn,lstm,fea`)')
    parser.add_argument('--seed', type=int, default=2024, metavar='[Int]',
                        help='Random seed. (default:2024)')
    parser.add_argument('--patience', type=int, default=20, metavar='[Int]',
                        help='Early stopping patience. (default:20)')
    return parser.parse_args()

if __name__=='__main__':
    save_model=True
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
    pretrained_model = args.PLM # '/mnt/f/data/pretrained_LM/prot_bert/' 
    local_files_only = True
    encoder_list = args.encoder.split(',') # ['cnn','lstm','fea']
    n_layers = args.lstm_nlayer
    dropout = args.dropout
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
    
    train_file = args.train
    seqlist = [record for record in SeqIO.parse(train_file, "fasta")]
    train_list, valid_list = random_split(seqlist, 0.2, seed=SEED):
    train_ds = SLAMDatasetSeq(train_file, tokenizer, feature=manual_fea)
    valid_ds = SLAMDatasetSeq(valid_list, tokenizer, feature=manual_fea)

    test_file = args.test
    test_ds = SLAMDatasetSeq(test_file, tokenizer, feature=manual_fea)
    window_size = test_ds.win_size
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=cpu,prefetch_factor=2)
    valid_loader = DataLoader(valid_ds,batch_size=batch_size,shuffle=True,num_workers=cpu, prefetch_factor=2)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=cpu, prefetch_factor=2)
    model = SLAMNetSeq(BERT_encoder=None, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,win_size=window_size,embedding_dim=32, fea_dim=41, hidden_dim=64, out_dim=32, n_layers=n_layers,dropout=dropout).to(device)
    # model.apply(weight_init)
    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print("Model Trainable Parameter: "+ str(params/1024/1024) + 'Mb' + "\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCELoss().to(device)
    result_list = []
    all_train_loss_list = []
    best_auc = 0
    best_epoch = 0
    patience = 0
    max_patience = args.patience

    desc=['Project', 'Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']
        for epoch in range(num_epochs):
        # Training
        start = time.perf_counter()
        train_step_loss, train_acc, train_loss, step = train_one_epoch(train_loader, model, device, optimizer, criterion)
        all_train_loss_list.extend(train_step_loss)
        end = time.perf_counter()
        print(f"Epoch {epoch+1} | {(end - start):.4f}s | Train | Loss: {train_loss: .6f}| Train acc: {train_acc:.4f}")
        start = time.perf_counter()
        valid_probs, valid_labels, valid_loss, valid_acc = test_binary(model, valid_loader, criterion, device)
        end = time.perf_counter()
        print(f"Epoch {epoch+1} | {(end - start):.4f}s | Valid | Valid loss: {valid_loss:.6f}| Valid acc: {valid_acc:.4f}")
        acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(valid_probs, valid_labels)
        result_info = [project+'_val', epoch, (tn+tp)/(tn+tp+fp+fn), th_, rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
        result_list.append(result_info)
        print_results(result_info, desc)
        if valid_loss > last_loss+0.1:
            patience += 1
        if patience > max_patience:
            break
        if valid_loss <= last_loss:
            last_loss = valid_loss
            best_auc = auc_
            best_acc = acc_
            best_epoch = epoch
            if save_model:
                save_path = osp.join(model_dir, f'best_{project}_model_epoch.pt')
                torch.save(model.state_dict(), save_path)
            start = time.perf_counter()
            test_probs, test_labels, test_loss, test_acc = test_binary(model, test_loader, criterion, device)
            end = time.perf_counter()
            print(f"Epoch {epoch+1} | {(end - start):.4f}s | Test | Test loss: {test_loss:.6f}| Test acc: {test_acc:.4f}")
            acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(test_probs, test_labels)
            result_info = [project+'_test', epoch, (tn+tp)/(tn+tp+fp+fn), th_, rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
            print_results(result_info, desc)
            best_test_probs = test_probs
            best_test_labels = test_labels
            best_result = result_info
    print('\nBest result:\n')
    print_results(best_result, desc)
    # Save training step loss
    loss_df = pd.DataFrame(all_train_loss_list)
    loss_df.columns = ['Loss']
    loss_df.to_csv(osp.join(result_dir, 'all_train_step_loss.txt'), sep='\t')
    
    # Save predicted logits and labels
    logit_df = pd.DataFrame([best_test_labels, best_test_probs]).transpose()
    logit_df.columns = ['Label', 'Logit']
    logit_df.to_csv(osp.join(result_dir, f"logits_results.txt"), sep='\t', index=False)

    epoch_df = pd.DataFrame(result_list)
    epoch_df.columns = desc
    epoch_df.to_csv(osp.join(result_dir, f'epoch_result.csv'), sep='\t', index=False)

    epoch_df = pd.DataFrame([best_result])
    epoch_df.columns = desc
    epoch_df.to_csv(osp.join(result_dir, f'best_result.csv'), sep='\t', index=False)