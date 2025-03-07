"""
Code taken and slightly modified from: https://github.com/BorgwardtLab/ggme/blob/main/src/metrics/utils.py

Utility functions and classes.
"""

from itertools import chain
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def pad_to_length(element: np.ndarray, length: int) -> np.ndarray:
    """
    Pad array to a predefined length by adding zeros.

    Args:
        element (np.ndarray): The array to pad.
        length (int): The length to pad the array to.

    Returns:
        np.ndarray: The padded array.
    """
    return np.pad(element, (0, length - len(element)), "constant")


def ensure_padded(X: List[np.ndarray], Y: Optional[List[np.ndarray]] = None):
    """
    Ensure that input arrays are padded to the same length.

    Args:
        X (list of np.ndarray): List of arrays to pad.
        Y (list of np.ndarray, optional): Additional list of arrays to pad. If provided, both X and Y will be padded to the length of the longest array in either list.

    Returns:
        tuple: A tuple containing the padded arrays. If Y is not provided, the second element of the tuple is None.

    (Warning) This only works because the function nx.degree_histogram(G) returns a vector of length max_degree,
    so we can simply pad by adding zeros to the end of the vector and know that it works. If you use a different
    histogram function, this would need to be updated! This function isn't necessary for the clustering coefficient
    and normalized Laplacian spectrum, since they are already the same size. However, if you intend to add your
    own histogram function, this function may need to be updated to do padding properly.
    """
    if Y is None:
        max_length = max(map(lambda a: len(a), X))
        return X, None
    else:
        max_length = max(map(lambda a: len(a), chain(X, Y)))

        X_padded = np.asarray([pad_to_length(el, max_length) for el in X])
        Y_padded = np.asarray([pad_to_length(el, max_length) for el in Y])

        return X_padded, Y_padded


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, cfg):
    model.eval()
    if cfg.model.method == 'nodeformer':
        out, _ = model(dataset.graph['node_feat'], dataset.graph['adjs'], cfg.model.tau)
    elif cfg.model.method == 'difformer':
        out = model(dataset.graph['node_feat'], dataset.graph['adjs'])
    else:
        out = model(dataset)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if cfg.dataset.name in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, cfg, result=None):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs_, x = dataset.graph['adjs'], dataset.graph['node_feat']
    adjs = []
    adjs.append(adjs_[0])
    for k in range(cfg.model.rb_order - 1):
        adjs.append(adjs_[k + 1])
    
    out, _ = model(x, adjs)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    
    if cfg.dataset.name in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out
