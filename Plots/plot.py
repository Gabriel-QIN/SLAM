import pandas as pd
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys 
sys.path.append("../codes")
from metrics import eval_metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE

def print_results(data, desc=['Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']):
    print('\t'.join(desc))
    print('\t'.join([f'{a:.3f}' if isinstance(a, float) else f'{a}' for a in data]))

def draw_tsne(test_labels, embedding, save_path, ptm_type='Kbhb'):
    
    X_embedded = TSNE(n_components=2).fit_transform(embedding)
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for val,color,ptm in [("0", '#03BECA',f'Non-{ptm_type} sites'), ("1", '#F77672',f'{ptm_type} sites')]:
        val = int(val)
        idx = np.where(test_labels == val)
        # idx = (test_labels == val).nonzero()
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],s=4,alpha=0.6, c=color, label=ptm)
    plt.legend(loc='upper right',prop={ 'size': 10},scatterpoints=1)
    plt.savefig(save_path,dpi=600) 

# emd_list = ['raw_emd','fea_in','bert_emd','plm_out','cnn_out','lstm_out','attn_out','final_out']


def draw_performance_cm(dat, save_path, cmap=plt.cm.Purples):
    # dat = np.array([line[7] for line in result_list]).reshape(4,4)
    categories =  ['General','Human', 'Mouse','False_smut']
    # randomly generated array 
    figure = plt.figure() 
    axes = figure.add_subplot(111) 
    # using the matshow() function  
    caxes = axes.matshow(dat, interpolation ='nearest', cmap=cmap, origin ='lower') 
    # figure.colorbar(caxes,ticks=[0, 0.2, 0.4, 0.6, 0.8, 1]) 
    # cbar = figure.colorbar(caxes)
    cmap = ccc
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = figure.colorbar(im)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    for i in range(len(dat)):
        for j in range(len(dat)):
            c = dat[i][j]
            if c > 0.001:
                plt.text(j, i, f"{c:.2f}", color='black', fontsize=10, va='center', ha='center')
                
    axes.set_xticklabels([' ']+categories, fontsize=12)
    axes.tick_params(axis='x', direction='out', pad=10)
    axes.set_yticklabels([' ']+categories, fontsize=12) 
    plt.ylabel('Models', fontsize=14, labelpad=10) # , fontweight='bold')
    plt.xlabel('Datasets', fontsize=14, labelpad=10) #, fontweight='bold')
    axes.xaxis.set_label_position('bottom') 
    axes.xaxis.tick_bottom()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)


def draw_AUROC(data, save_path, splist=['Human', 'Mouse','False smut', 'General']):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(6.4, 4.8), dpi=600)
    plt.title('Receiver Operating Characteristic', pad=20, fontsize=16)
    for ind, name in enumerate(splist):
        targets = data[ind][0]
        probs = data[ind][1]
        fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
        plt.plot(fpr, tpr, label = f'{name}: AUROC = {auc(fpr, tpr):.4f}',linewidth=2.0)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=14, labelpad=10)
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

def draw_AUPRC(data, save_path, splist=['Human', 'Mouse','False smut', 'General']):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(6.4, 4.8), dpi=600)
    plt.title('Precision Recall Curve', pad=20, fontsize=16)
    for ind, name in enumerate(splist):
        targets = data[ind][0]
        probs = data[ind][1]
        precision_1, recall_1, threshold_1 = precision_recall_curve(targets, probs)  # 计算Precision和Recall
        aupr_1 = auc(recall_1, precision_1)
        plt.plot(recall_1, precision_1, label = f'{name}: AUPRC = {aupr_1:.4f}',linewidth=2.0)
    plt.legend(loc = 'upper right')
    plt.plot([1, 0], linestyle='--', color='grey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', fontsize=14, labelpad=10)
    plt.xlabel('Recall', fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

def plot_confusion_matrix(cm, savepath, cmap=plt.cm.Blues, classes=['Non-Kbhb', 'Kbhb'], title='Confusion Matrix'):
    
    plt.figure(figsize=(6.4, 4.8), dpi=600)
    plt.grid(False)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='black', fontsize=15, va='center', ha='center')
            
#     norm = matplotlib.colors.Normalize(vmin=cm.min(), vmax=cm.max()*0.8)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18, pad=15)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90, fontsize=14)
    plt.yticks(xlocations, classes, fontsize=14)
    plt.ylabel('Actual label', fontsize=16, labelpad=10)
    plt.xlabel('Predict label', fontsize=16, labelpad=10)
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
#     plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savepath, dpi=600)

if __name__=='__main__':
    model_list = ['SLAM', 'CNN', 'GRU', 'LSTM', 'MLP','RF', 'XGboost']
    desc=['Project', 'Acc', 'Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']
    data = []
    for ML in model_list:
        df = pd.read_csv(f'logits/{ML}_logits_results.txt', sep='\t')
        df.columns = ['Label', 'Logit']
        test_labels = df['Label'].to_numpy()
        test_probs = df['Logit'].to_numpy()
        acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(test_probs, test_labels)
        result_info = [ML, (tn+tp)/(tn+tp+fp+fn), rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
        print_results(result_info, desc)
        data.append(np.array([test_labels, test_probs]))
    draw_AUROC(data, osp.join('logits', 'AUROC.png'), splist=model_list)
    draw_AUPRC(data, osp.join('logits', 'AUPRC.png'), splist=model_list)