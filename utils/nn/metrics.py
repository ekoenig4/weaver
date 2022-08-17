import numpy as np
import traceback
import sklearn.metrics as _m
from functools import partial
from ..logger import _logger
from torch_scatter import scatter_sum
import torch

# def _bkg_rejection(y_true, y_score, sig_eff):
#     fpr, tpr, _ = _m.roc_curve(y_true, y_score)
#     idx = next(idx for idx, v in enumerate(tpr) if v > sig_eff)
#     rej = 1. / fpr[idx]
#     return rej
#
#
# def bkg_rejection(y_true, y_score, sig_eff):
#     if y_score.ndim == 1:
#         return _bkg_rejection(y_true, y_score, sig_eff)
#     else:
#         num_classes = y_score.shape[1]
#         for i in range(num_classes):
#             for j in range(i + 1, num_classes):
#                 weights = np.logical_or(y_true == i, y_true == j)
#                 truth =


def roc_auc_score_ovo(y_true, y_score):
    if y_score.ndim == 1:
        return _m.roc_auc_score(y_true, y_score)
    else:
        num_classes = y_score.shape[1]
        result = np.zeros((num_classes, num_classes), dtype='float32')
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                weights = np.logical_or(y_true == i, y_true == j)
                truth = y_true == j
                score = y_score[:, j] / np.maximum(y_score[:, i] + y_score[:, j], 1e-6)
                result[i, j] = _m.roc_auc_score(truth, score, sample_weight=weights)
    return result


def confusion_matrix(y_true, y_score):
    if y_score.ndim == 1:
        y_pred = y_score > 0.5
    else:
        y_pred = y_score.argmax(1)
    return _m.confusion_matrix(y_true, y_pred, normalize='true')

def ndcg_score(input, target, batch):
    from .scatter_tools import scatter_sort
    ranks = torch.arange( len(batch) ).to(input.device)
    _, n_examples = batch.unique(return_counts=True)
    offset = n_examples.cumsum(dim=0) - n_examples[0]
    offset = torch.repeat_interleave(offset, n_examples).to(input.device)
    ranks = ranks - offset

    true_sort, _ = scatter_sort(target, batch, descending=True)
    _, argout = scatter_sort(input, batch, descending=True)
    pred_sort = target[argout]

    def calc_dcg(rel):
        num = 2**rel - 1
        den = torch.log2(ranks + 2)
        return scatter_sum(num/den, batch)

    dcg = calc_dcg(pred_sort)
    idcg = calc_dcg(true_sort)

    ndcg = dcg/idcg
    return ndcg.nanmean()

def relhitk_score(input, target, batch, k=1):
    from .scatter_tools import scatter_topk
    pred_k = scatter_topk(input, batch, k=k)[1]
    true_k1 = scatter_topk(target, batch, k=1)[1]
    relhitk = (target[pred_k] >= target[true_k1]).any(dim=-1).float().mean()
    return relhitk

def nhitk_score(input, target, batch, k=8):
    from .scatter_tools import scatter_topk
    pred_topk = scatter_topk(input, batch, k=k)[1]
    true_topk = scatter_topk(target, batch, k=k)[1]

    n_pred_topk = target[pred_topk].sum(dim=-1)
    n_true_topk = target[true_topk].sum(dim=-1)

    nhitk = (n_pred_topk >= n_true_topk).float().mean()
    return nhitk

def top_label_score(input, target, batch):
    from .scatter_tools import scatter_max
    truemax, _ = scatter_max(target, batch)
    _, predmax = scatter_max(input, batch)
    return (target[predmax]/truemax).nanmean()


_metric_dict = {
    'roc_auc_score': partial(_m.roc_auc_score, multi_class='ovo'),
    'roc_auc_score_matrix': roc_auc_score_ovo,
    'confusion_matrix': confusion_matrix,
    'ndcg': ndcg_score,
    'relhitk': relhitk_score,
    }


def _get_metric(metric):
    try:
        return _metric_dict[metric]
    except KeyError:
        return getattr(_m, metric)


def evaluate_metrics(y_true, y_score, eval_metrics=[]):
    results = {}
    for metric in eval_metrics:
        func = _get_metric(metric)
        try:
            results[metric] = func(y_true, y_score)
        except Exception as e:
            results[metric] = None
            _logger.error(str(e))
            _logger.debug(traceback.format_exc())
    return results