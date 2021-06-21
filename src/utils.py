import os
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score)
from torch_geometric.utils import sort_edge_index
from torch_geometric.data import Data


def pk_save(obj, file_path):
    return pickle.dump(obj, open(file_path, 'wb'))


def pk_load(file_path):
    if os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    else:
        return None


def load_csv_data(path):
    h_idx = 0
    t_idx = 1
    r_idx = 2
    df = pd.read_csv(path, header=0)
    cs = df.columns.tolist()
    df = df[cs[0:3]]

    es = (df[cs[h_idx]].append(df[cs[t_idx]])).unique().tolist()
    rs = df[cs[r_idx]].unique().tolist()

    return df, cs, es, rs


def classification_report(y, y_prob):
    ks = [1, 3, 5]
    pr_score_at_ks = []

    for k in ks:
        pr_at_k = []
        for i in range(y_prob.shape[0]):
            # forloop samples
            y_prob_index_topk = np.argsort(y_prob[i])[::-1][:k]
            inter = set(y_prob_index_topk) & set(y[i].nonzero()[0])
            pr_ith = len(inter) / k
            pr_at_k.append(pr_ith)
        pr_score_at_k = np.mean(pr_at_k)
        pr_score_at_ks.append(pr_score_at_k)

    # same result, for loop not required
    ids = y.sum(0).nonzero()[0]
    roc_auc = roc_auc_score(y[:, ids], y_prob[:, ids])
    pr_auc = average_precision_score(y[:, ids], y_prob[:, ids])

    ranks = []
    for i in range(y_prob.shape[0]):
        # rank all predictions
        idx = np.argsort(y_prob[i])[::-1]
        # Order labels by ranks. get highest ranked true prediction
        rank = np.where(y[i][idx] == 1.0)[0][0]
        ranks.append(rank + 1)

    return {
        'pr': pr_auc,
        'roc': roc_auc,
        'p@1': pr_score_at_ks[0],
        'p@3': pr_score_at_ks[1],
        'p@5': pr_score_at_ks[2],
        'mrr': np.mean(1./np.array(ranks))  # mean reciprical rank
    }


def overlap(e1, e2):
    r"""
    Looks for overlap between edge indexes e1 and e2. This is helpful for
    debugging
    """
    assert e1.size(0) == 2 and e2.size(0) == 2

    counter = 0
    e = torch.cat([e1, e2], 1)
    se, _ = sort_edge_index(e)
    for i in torch.arange(se.size(1) - 1):
        if (se[0, i] == se[0, i+1] and se[1, i] == se[1, i+1]):
            counter += 1
    return counter


def create_label(data, args, novel_eval=False):
    r"""
    Use edge_attr as labels (data.y) for computing loss and evaluation.
    Subsample edge_attr to use as inputs for training. This prevents using
    all of the expected output as an input to the network.
    """
    sub_edge_attr = subsample_edge_attr(data, args)

    if novel_eval:
        # Do not use input edges as labels. This makes it so that we only
        # use the newly discovered edges when evaluating
        data.y = data.edge_attr
        data.y[sub_edge_attr == 1] = 0
    else:
        data.y = data.edge_attr
    data.edge_attr = sub_edge_attr
    return data


def subsample_edge_attr(data, args):
    r"""
    Subsample args.edge_frac of data.edge_attr. It does this by randomly
    setting a fraction of the edge attributes to zero.
    """
    sz = data.edge_attr[data.edge_attr == 1].size()
    uniform_matr = torch.empty(sz).to(args.device).uniform_()
    sub_attr = (uniform_matr <= args.edge_frac).float()
    sub_edge_attr = torch.zeros(data.edge_attr.size()).to(args.device)
    sub_edge_attr[data.edge_attr == 1] = sub_attr
    return sub_edge_attr


def data_to_batch_list(data):
    r""" Split predictions across multiple GPUs.
    torch_geometric.DataParallel takes a list of Batch objects.
    Unfortunately, this means that the interface for all models needs to
    change to accomodate this.
    """
    num_devices = torch.cuda.device_count()
    target_edge_index = data.target_edge_index
    target_split = torch.split(target_edge_index,
                               target_edge_index.size(1) // num_devices,
                               dim=1)
    batch_list = []
    for target in target_split:
        batch_list += [
            Data(x=data.x,
                 edge_index=data.edge_index,
                 edge_attr=data.edge_attr,
                 target_edge_index=target)
        ]
    return batch_list


def build_eval(input_data, target_data):
    return Data(x=input_data.x,
                edge_index=input_data.edge_index,
                edge_attr=input_data.edge_attr,
                target_edge_index=target_data.edge_index,
                y=target_data.edge_attr)

