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
from model import *
from dataset import *

def print_results(data, desc=['Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']):
    print('\t'.join(desc))
    print('\t'.join([f'{a:.3f}' if isinstance(a, float) else f'{a}' for a in data]))

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
    # description='SLAM methods could enable accurate and efficient discovery for a newly-reported PTM, lysine β-hydroxybutyrylation (Kbhb).'
    # directory and file settings
    root_dir = 'Datasets'
    parser.add_argument("--project_name", default='SLAM_general', type=str,
                            help="Project name for saving model checkpoints and best model. Default:`SLAM_general`.")
    parser.add_argument("--train", default=osp.join(root_dir, 'general_train.fa'), type=str,
                            help="Data directory. Default:`'Datasets/general_train.fa'`.")
    parser.add_argument("--test", default=osp.join(root_dir, 'general_test.fa'), type=str,
                            help="Data directory. Default:`'Datasets/general_test.fa'`.")
    parser.add_argument("--model", default='Models/SLAM', type=str,
                        help="Directory for model storage and logits. Default:`Models/SLAM_combine`.")
    parser.add_argument("--result", default='result/SLAM', type=str,
                        help="Result directory for model training and evaluation. Default:`result/SLAM`.")
    parser.add_argument("--PLM", default='Rostlab/prot_bert/', type=str,
                        help="PLM directory. Default:`Rostlab/prot_bert/`.")
    parser.add_argument("--pdb_dir", default='Datasets/Structure/PDB', type=str,
                        help="PLM directory. Default:`Datasets/Structure/PDB`.")
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
    parser.add_argument('--gnn_layers', '-gl', type=int, default=5, metavar='[Int]',
                        help='Number of GNN layer.(default:5).') 
    parser.add_argument('--nneighbor', '-nb', type=int, default=32, metavar='[Int]',
                        help='Number of residue neighbors in protein graph.(default:32).') 
    parser.add_argument('--atom_type', '-at', type=str, default='CA', metavar='[Str]',
                        help='Atom type for construting the protein graph.(default:`CA`).') 
    parser.add_argument('--dropout', '-dp', type=float, default=0.5, metavar='[Float]',
                        help='Dropout rate.(default:0.5).')
    parser.add_argument('--encoder', type=str, default='cnn,lstm,fea,gnn', metavar='[Str]',
                        help='Encoder list separated by comma chosen from cnn,lstm,fea,plm,gnn. (default:`cnn,lstm,fea,gnn`)')
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
    pretrained_model = '/mnt/f/data/pretrained_LM/prot_bert/' # args.PLM
    local_files_only = True
    # Please set local_files_only into `True` if you are using Rostlab/prot_bert/
    # For local files: pretrained_LM/prot_bert/', please download pre-trained model from https://huggingface.co/Rostlab/prot_bert into this directory.
    encoder_list = args.encoder.split(',')
    n_layers = 1
    dropout = args.dropout
    patience = args.patience
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
    atom_type = args.atom_type # chosen from CA, CB, R, C, N, O
    gnn_layers = args.gnn_layers
    pdb_dir = args.pdb_dir

    train_file = args.train
    train_list_all = [record for record in SeqIO.parse(train_file, "fasta")]
    train_list, valid_list = random_split(train_list_all, 0.2, seed=SEED):
    train_ds = SLAMDataset(train_list, tokenizer, pdb_dir=pdb_dir, feature=manual_fea, nneighbor=nneighbor, atom_type=atom_type)
    valid_ds = SLAMDataset(valid_list, tokenizer, pdb_dir=pdb_dir, feature=manual_fea, nneighbor=nneighbor, atom_type=atom_type)
    
    test_file = args.test
    test_list = [record for record in SeqIO.parse(test_file, "fasta")]
    test_ds = SLAMDataset(test_list, tokenizer, pdb_dir=pdb_dir, feature=manual_fea, nneighbor=nneighbor, atom_type=atom_type)
    window_size = test_ds.win_size
    print(f"Training dataset: {len(train_ds)}   Valid dataset: {len(valid_ds)} |Testing dataset: {len(test_ds)}")

    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=cpu, collate_fn=graph_collate_fn, prefetch_factor=2)
    valid_loader = DataLoader(valid_ds,batch_size=batch_size,shuffle=True,num_workers=cpu, collate_fn=graph_collate_fn, prefetch_factor=2)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=cpu, collate_fn=graph_collate_fn, prefetch_factor=2)

    model = SLAMNet(BERT_encoder=BERT_encoder, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,PLM_dim=PLM_dim,win_size=window_size,embedding_dim=embedding_dim, fea_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim,node_dim=node_dim, edge_dim=edge_dim, gnn_layers=gnn_layers,n_layers=n_layers,dropout=dropout).to(device)
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
    last_loss = 100
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