#!/bin/python
# -*- coding:utf-8 -*- 

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, precision_recall_curve


def eval_metrics(probs, targets, cal_AUC=True):

    threshold_list = []
    for i in range(1, 100):
        threshold_list.append(i / 100.0)
    if cal_AUC:
        if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
            fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(),
                                             y_score=probs.detach().cpu().numpy())
        elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
            fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
        else:
            print('ERROR: probs or targets type is error.')
            raise TypeError
        auc_ = auc(x=fpr, y=tpr)
    else:
        auc_ = 0
    precision_1, recall_1, threshold_1 = precision_recall_curve(targets, probs)  # Precision and Recall
    aupr_1 = auc(recall_1, precision_1)
    threshold_best, rec_best, pre_best,F1_best, spe_best, mcc_best, pred_bi_best = 0, 0, 0, 0, 0, -1, None
    for threshold in threshold_list:
        acc, threshold, rec, pre,F1, spe, mcc, _, pred_bi, tn, fp, fn, tp = th_eval_metrics(threshold, probs, targets,cal_AUC=False)
        if mcc > mcc_best:
            acc, threshold_best, rec_best, pre_best, F1_best, spe_best, mcc_best, pred_bi_best, tn_, fp_, fn_, tp_ = acc, threshold, rec, pre,F1, spe, mcc, pred_bi, tn, fp, fn, tp

    return acc, threshold_best, rec_best, pre_best, F1_best, spe_best, mcc_best, auc_, pred_bi_best, aupr_1, tn_, fp_, fn_, tp_

def th_eval_metrics(threshold, probs, targets,cal_AUC=True):
    if isinstance(probs, torch.Tensor) and isinstance(targets,torch.Tensor):
        if cal_AUC:
            fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(), y_score=probs.detach().cpu().numpy())
            auc_ = auc(x=fpr, y=tpr)
        else:
            auc_ = 0
        pred_bi = targets.data.new(probs.shape).fill_(0)
        pred_bi[probs>threshold] = 1
        targets[targets==0] = 5
        targets[targets==1] = 10
        tn = torch.where((pred_bi+targets)==5)[0].shape[0]
        fp = torch.where((pred_bi+targets)==6)[0].shape[0]
        fn = torch.where((pred_bi+targets)==10)[0].shape[0]
        tp = torch.where((pred_bi+targets)==11)[0].shape[0]
        if tp>0:
            rec = tp / (tp + fn)
        else:
            rec = 0
        if tp > 0:
            pre = tp / (tp + fp)
        else:
            pre = 0
        if tn > 0:
            spe = tn / (tn + fp)
        else:
            spe = 0
        if rec+pre > 0:
            F1 = 2 * rec * pre / (rec + pre)
        else:
            F1 = 0
        acc = (tp + tn)/(tp + tn + fp + fn)
        mcc = (tp*tn-fp*fn)/torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
        if cal_AUC:
            fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs)
            auc_ = auc(x=fpr, y=tpr)
        else:
            auc_ = 0
        pred_bi = np.abs(np.ceil(probs - threshold))
        cm = confusion_matrix(targets, pred_bi)
        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
        if tp >0 :
            rec = tp / (tp + fn)
        else:
            rec = 1e-8
        if tp >0:
            pre = tp / (tp + fp)
        else:
            pre = 1e-8
        if tn>0:
            spe = tn / (tn + fp)
        else:
            spe = 1e-8
        acc = (tp + tn)/(tp + tn + fp + fn)
        mcc = matthews_corrcoef(targets, pred_bi)
        if rec + pre > 0:
            F1 = 2 * rec * pre / (rec + pre)
        else:
            F1 = 0
    else:
        print('ERROR: probs or targets type is error.')
        raise TypeError

    return acc, threshold, rec, pre, F1, spe, mcc, auc_, pred_bi, tn, fp, fn, tp


