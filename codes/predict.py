import torch_geometric
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from SLAM import *
from SLAM_seq import SLAMNetSeq
from hypara import *

matplotlib.use('agg')

def parse_pdb_chain(pdb_file, chain='A',pos=None, atom_type='CA', nneighbor=32, cal_cb=True):
    """
    ########## Process PDB file ##########
    """
    current_pos = -1000
    X = []
    current_aa = {} # N, CA, C, O, R
    first_aa_type = None
    first_aa_position = None
    with open(pdb_file, 'r') as pdb_f:
        for line in pdb_f:
            if line[21] == chain:
                if first_aa_type is None and line[0:4].strip() == "ATOM":
                    first_aa_type = line[17:20].strip()
                    first_aa_position = int(line[22:26].strip())
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
        closest_indices = sorted(np.argsort(distances)[:nneighbor])
        X = X[(closest_indices)]
    return X, first_aa_type, first_aa_position  # array shape: [Length, 6, 3] N, CA, C, O, R, CB

def get_graph_fea_chain(pdb_path, pos, chain='A', nneighbor=32, radius=10, atom_type='CA', cal_cb=True):
    X, first_aa_type, first_aa_position = parse_pdb_chain(pdb_path, chain=chain,pos=pos, atom_type=atom_type, nneighbor=nneighbor, cal_cb=cal_cb)
    X = torch.tensor(X).float()
    query_atom = X[:, atom_idx[atom_type]]
    edge_index = radius_graph(query_atom, r=radius, loop=False, max_num_neighbors=nneighbor, num_workers = 4)
    node, edge = get_geo_feat(X, edge_index)
    return Data(x=node, edge_index=edge_index, edge_attr=edge, name=os.path.basename(pdb_path).split('.')[0])

def _get_encoding(seq, feature=[BLOSUM62, BINA]):
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
    
def get_all_inputs(seq, pos, tokenizer, pdb_path):
    if pdb_path is not None:
        data = get_graph_fea_chain(pdb_path, pos, nneighbor=64, atom_type='CA', cal_cb=True)
    else:
        data = None
    fea = _get_encoding(seq)
    s = ''.join([token for token in re.sub(r"[UZOB*]", "X", seq.rstrip('*'))])
    max_len = len(s)
    encoded = tokenizer.encode_plus(seq, add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True,truncation=True, max_length=max_len, return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    return data, input_ids, attention_mask, torch.tensor(fea, dtype=torch.float)

def get_peptide(pos, window_size, seq, mirror=True):
    """Return peptide based on window_size. Missing residues are padded with X symbol (if mirror == False) or mirroring residues from the other side (if mirror == True)."""
    pos = pos-1
    half_window = int(window_size/2)
    start = pos - half_window
    left_padding = '' if start >= 0 else 'X' * abs(start)
    start = 0 if start < 0 else start
    end = pos + half_window + 1
    right_padding = 'X' * half_window
    end = len(seq) if end + 1 > len(seq) else end
    peptide_ = seq[start:end]
    if mirror:
        if left_padding == '' and right_padding == '':
            peptide = left_padding + peptide_ + right_padding
        elif left_padding == '' and right_padding != '': # mirror left
            peptide = left_padding + peptide_ + peptide_[:len(right_padding)][::-1]
        elif left_padding != '' and right_padding == '': # mirror right
            peptide = peptide_[::-1][:len(left_padding)] + peptide_ + right_padding
        else:
            peptide = None
    else:
        peptide = left_padding + peptide_ + right_padding
    if peptide is not None:
        peptide = peptide[:window_size]
        assert peptide[half_window] == 'K' and len(peptide) == window_size
        return peptide
    else:
        return None

def get_all_k(seqlist, window_size=51):
    peplist = []
    window_size = window_size
    half_window = window_size // 2
    for record in seqlist:
        name = record.id.split('|')[0]
        seq = str(record.seq)
        for m in re.finditer('K', seq):
            pos = m.start() + 1
            pep = get_peptide(pos, window_size, seq, mirror=False)
            if pep is not None:
                peplist.append([f'{name}|Pred|{pos}|{len(seq)}', pep])
    return peplist

def predict_engine(seq_path, pdb_path, threshold=0.06, species='general', chain='A', use_PLM=False):
    if pdb_path is not None and use_PLM: # Use PLM+GNN
        para = HyperParam_SLAM
    else:
        if pdb_path is None: # Only use PLM
        #     para = HyperParam_struct
        # else:
            para = HyperParam_seq # Not use PLM and GNN
    gpu = para.gpu
    # device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    n_layers = para.n_layers
    dropout = para.dropout
    embedding_dim = para.embedding_dim
    hidden_dim = para.hidden_dim
    out_dim = para.out_dim
    node_dim = para.node_dim
    edge_dim = para.edge_dim
    nneighbor = para.nneighbor
    atom_type = para.atom_type
    gnn_layers = para.gnn_layers
    encoder_list = para.encoder_list.split(',')
    fea_dim = para.fea_dim
    PLM_dim = para.PLM_dim
    window_size = para.window_size
    pretrained_model = para.pretrained_model
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False, use_fast=False)
    if pretrained_model is not None:
        BERT_encoder = AutoModel.from_pretrained(pretrained_model, local_files_only=True, output_attentions=False).to(device)
    else:
        BERT_encoder = None
    if para.model_dir.split('/')[-1] in ['SLAM']:
        model_file = osp.join(para.model_dir, f'{species}_model.pt')
        model = SLAMNet(BERT_encoder=BERT_encoder, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,PLM_dim=PLM_dim,win_size=window_size,embedding_dim=embedding_dim, fea_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim,node_dim=node_dim, edge_dim=edge_dim, gnn_layers=gnn_layers,n_layers=n_layers,dropout=dropout).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
    else: # 'SLAM_wo_plm_and_structure'
        model_file = osp.join(para.model_dir, f'{species}_seq_model.pt')
        model = SLAMNetSeq(BERT_encoder=None, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list,win_size=window_size,embedding_dim=32, fea_dim=41, hidden_dim=64, out_dim=32, n_layers=n_layers,dropout=dropout, kernel_size=3).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
    if pdb_path is not None:
        first_aa_position = int(parse_pdb_chain(pdb_path, chain=chain,pos=None, atom_type=atom_type, nneighbor=nneighbor, cal_cb=True)[-1])
        discrepancy = first_aa_position - 1
    else:
        discrepancy = 0
    seqlist = [record for record in SeqIO.parse(seq_path, "fasta")]
    peplist = get_all_k(seqlist, window_size=window_size)

    predictions = []
    model.eval()
    for desc, seq in peplist:
        seq = str(seq)
        tmp = desc.split('|')
        pos = int(tmp[2])
        g, input_ids, attention_mask, feature = get_all_inputs(seq,pos,tokenizer,pdb_path)
        feature = feature.unsqueeze(0).to(device)
        if g is not None:
            g = g.to(device)
            g.batch = torch.zeros(g.x.shape[0],dtype=torch.int64).to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if pdb_path is not None:
            pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, g_data=g)
        else:
            pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature)
        score = pred.squeeze().detach().item()
        if score > 0.24:
            confidence = 'Extremely high'
        elif 0.15 < score <= 0.24:
            confidence = 'High'
        elif 0.09 <= score < 0.15:
            confidence = 'Medium'
        elif threshold < score < 0.06:
            confidence = 'Low'
        else:
            confidence = 'Non-Kbhb'
        prediction = 'Yes' if score > threshold else 'No'
        seq = seq[20:31]
        result = [tmp[0], seq, pos+discrepancy, score, prediction, confidence]
        predictions.append(result)
    
    case = pd.DataFrame(predictions)
    case.columns = ['ID', 'Sequence', 'Position', 'Score', 'Prediction', 'Confidence']

    kbhb = case[case['Score']>threshold]
    return case, kbhb

def draw_scatter_ptm(case, savename='scatter.png', threshold=0.06):
    x_max = len(case['Position'])
    x_range = x_max - 0
    fig_width = max(10, (x_range / 10)*2)
    fig_height = 6 
    plt.figure(figsize=(fig_width, fig_height))
    palette = []
    for score in case['Score']:
        if score > 0.24:
            palette.append('#FA7070')
        elif 0.15 < score <= 0.24:
            palette.append('#F3CCF3')
        elif 0.09 <= score <= 0.15:
            palette.append('#F8F9D7')
        elif threshold <= score < 0.09:
            palette.append('#D2DAFF')
        else:
            palette.append('#EEEEEE')

    plt.scatter(x=case['Position'], y=case['Score'], s=100, c=palette, edgecolor='grey', linewidth=1.5)
    plt.axhline(y=threshold, color='#B06161', linestyle='--')  
    plt.xlabel('Position', fontsize=14, labelpad=10)
    plt.ylabel('Score', fontsize=14, labelpad=10)

    handles = [plt.Line2D([0], [0], color='#FA7070', lw=4, label='Extremely high (Sp = 0.95)'),
            plt.Line2D([0], [0], color='#F3CCF3', lw=4, label='High (Sp = 0.9)'),
            plt.Line2D([0], [0], color='#F8F9D7', lw=4, label='Medium (Sp = 0.85)'),
            plt.Line2D([0], [0], color='#D2DAFF', lw=4, label='Low (Sp = 0.8)'),
            plt.Line2D([0], [0], color='#EEEEEE', lw=4, label='Non-Kbhb')]
    plt.legend(handles=handles)
    plt.savefig(savename,dpi=600)

def draw_pie_chart_ptm(case, savename='pie_chart.png', threshold=0.06):
    extremely_high_confidence = sum(1 for score in case['Score'] if score > 0.24) # Sp 0.95
    high_confidence = sum(1 for score in case['Score'] if 0.15 < score <= 0.24) # Sp 0.9
    medium_confidence = sum(1 for score in case['Score'] if 0.09 < score <= 0.15) # Sp from 0.8 to 0.9
    low_confidence = sum(1 for score in case['Score'] if threshold < score <= 0.09) # Sp 0.7
    extremely_low_confidence = sum(1 for score in case['Score'] if score < threshold)

    sizes = [extremely_high_confidence, high_confidence, medium_confidence, low_confidence, extremely_low_confidence]
    # ind = [ind for ind,s in enumerate(sizes) if s > 0]
    colors = ['#FA7070', '#F3CCF3', '#F8F9D7', '#D2DAFF', '#EEEEEE']
    plt.figure()
    wedges, texts, autotexts = plt.pie(
        sizes, colors=colors, autopct='%1.1f%%', startangle=0, wedgeprops=dict(edgecolor='grey'))

    handles = [
        plt.Line2D([0], [0], color='#FA7070', lw=4, label='Extremely high (Sp = 0.95)'),
        plt.Line2D([0], [0], color='#F3CCF3', lw=4, label='High (Sp = 0.9)'),
        plt.Line2D([0], [0], color='#F8F9D7', lw=4, label='Medium (Sp = 0.85)'),
        plt.Line2D([0], [0], color='#D2DAFF', lw=4, label='Low (Sp = 0.8)'),
        plt.Line2D([0], [0], color='#EEEEEE', lw=4, label='Non-Kbhb')
    ]

    plt.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.01, 0.01), framealpha=1, facecolor='white', title_fontsize='10', fontsize='8')

    plt.tight_layout()
    plt.savefig(savename, dpi=600)

def draw_ptm(kbhb, savename='kbhb.png', threshold=0.4):
    x_max = len(kbhb['Position'])
    
    fig_width = 10
    fig_height = 6 
    plt.figure(figsize=(fig_width, fig_height))
    palette = []
    for score in kbhb['Score']:
        if score > 0.24:
            palette.append('#FA7070')
        elif 0.15 < score <= 0.24:
            palette.append('#F3CCF3')
        elif 0.09 <= score <= 0.15:
            palette.append('#F8F9D7')
        elif threshold <= score < 0.09:
            palette.append('#D2DAFF')
        else:
            palette.append('#EEEEEE')
    sns.barplot(x='Position', y='Score', data=kbhb, palette=palette,edgecolor='black', linewidth=1.5) # , width=0.5
    plt.axhline(y=threshold, color='#B06161', linestyle='--')
    plt.xlabel('Position',fontsize=14,labelpad=10)
    plt.ylabel('Score',fontsize=14,labelpad=10)
    handles = [plt.Line2D([0], [0], color='#FA7070', lw=4, label='Extremely high (Sp = 0.95)'),
            plt.Line2D([0], [0], color='#F3CCF3', lw=4, label='High (Sp = 0.9)'),
            plt.Line2D([0], [0], color='#F8F9D7', lw=4, label='Medium (Sp = 0.85)'),
            plt.Line2D([0], [0], color='#D2DAFF', lw=4, label='Low (Sp = 0.8)'),
            plt.Line2D([0], [0], color='#EEEEEE', lw=4, label='Non-Kbhb')]
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(savename,dpi=600)

if __name__=='__main__':
    chain = 'A'
    pdb_path = f'../case_study/5w49.pdb'
    pdb_path = None
    seq_path = f'../case_study/5w49.fa'
    species = 'general'
    use_PLM = True
    threshold = 0.06
    case, kbhb = predict_engine(seq_path, pdb_path, threshold=threshold, species=species, chain=chain, use_PLM=use_PLM)
    print(case)
    draw_ptm(kbhb, savename='../case_study/bar.png', threshold=threshold)
    draw_scatter_ptm(case, savename='../case_study/scatter.png', threshold=threshold)
    draw_pie_chart_ptm(case, savename='../case_study/pie_chart.png', threshold=threshold)