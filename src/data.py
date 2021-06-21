# from src.DDE.stream_dde import *
import random
import torch
import pandas as pd
from torch.utils import data
from torch_geometric.io.tu import Data

# def smiles2index(s):
#     t1 = bpe.process_line(s).split()
#     i1 = [words2idx[i] for i in t1]
#     return i1
#
#
# def index2multi_hot(i1):
#     v1 = torch.zeros(len(idx2word),)
#     v1[i1] = 1
#     return v1
#
#
# def smiles2vector(s1):
#     i1 = smiles2index(s1)
#     return index2multi_hot(i1)


class Dataset(data.Dataset):
    def __init__(self, df_index):
        self.df_index = df_index

    def __len__(self):
        return len(self.df_index)

    # index in specific torch data.Dataset (eg train, test, valid) to be translated to index in df
    def __getitem__(self, index):
        return self.df_index[index]


class Collator:
    pass


class DataCollator(Collator):

    def __init__(self, df_path, df_cols, df_drugs, df_edge_types, num_rels, share_neg=1):  # share neg is relative to data so if data had 50 edges 1 means 50 neg too
        self.df_path = df_path
        self.df_cols = df_cols  # relevant columns: drug1, drug2, edge type
        self.df_drugs = df_drugs
        self.df_edge_types = df_edge_types
        self.num_rels = num_rels
        self.share_neg = share_neg

    def __call__(self, dataset):
        index = dataset

        df = pd.read_csv(self.df_path)
        d1s = df.iloc[index][self.df_cols[0]].values.tolist()
        d2s = df.iloc[index][self.df_cols[1]].values.tolist()
        labels = self.df_edge_types.loc[index].values.tolist()  # need to use loc here since is not full dataframe

        # SMILES
        # # remove drugs for which we have no smiles
        # for i in reversed(range(len(labels))):
        #     if d1s[i] not in drugs or d2s[i] not in drugs:
        #         del[d1s[i]]
        #         del[d2s[i]]
        #         del[labels[i]]

        # node features x
        drugs = list(set(d1s + d2s))
        x = [self.df_drugs.index(d) for d in drugs]  # transformed to sparse tensor after data loading since loader does not support them
        # x = to_sparse_one_hot(x, len(self.df_drugs)).to_dense()
        # SMILES
        # x = torch.zeros(len(drugs), len(idx2word))
        # for i, drug in enumerate(drugs):
        #     x[i, :] = smiles2vector(self.dict_smiles[drug])

        edge_index = torch.zeros(2, len(labels)).long()
        for i in range(len(labels)):
            edge_index[0][i] = drugs.index(d1s[i])
            edge_index[1][i] = drugs.index(d2s[i])

        # random negative sampling
        # we assume we can add as many as requested and do not do additional sanity checks for now
        if self.share_neg > 0:
            neg_edge_index = torch.zeros(2, len(labels) * self.share_neg).long()
            neg_size = len(labels)*self.share_neg
            max_id = len(x) - 1
            i = 0
            while i < neg_size:
                i1, i2 = random.randint(0, max_id), random.randint(0, max_id)
                if self._ismember((i1, i2), neg_edge_index):
                    continue
                if (((df[self.df_cols[0]] == i1) & (df[self.df_cols[1]] == i2)) | ((df[self.df_cols[1]] == i1) & (df[self.df_cols[0]] == i2))).any():
                    continue
                # alternative: instead of loading df, check in batch only
                # if self._ismember((i1, i2), edge_index) \
                #         or self._ismember((i1, i2), neg_edge_index):
                #     continue
                neg_edge_index[0][i] = i1
                neg_edge_index[1][i] = i2
                i += 1
            # add also other directions
            neg_edge_index = torch.cat([neg_edge_index, torch.stack([neg_edge_index[1], neg_edge_index[0]], dim=0)], dim=-1)
            # if we wanted to add the negative edges to the KG (then increment also below second dim in idx and edge_attr):
            # edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)
            # labels = labels + [self.num_rels] * neg_size

        # here labels denote edge types still!
        labels = torch.LongTensor(labels)

        # create edge_attr based on orig relation types
        # we could do some collapsing here ie merging edges between some nodes and updating attributes = n-hot,
        #  but then we need to add new edge types for "merged edges" -- at least for RGCN -- so I didn't do this for now
        # for simplicity, we add a no-action dummy dimension -- then we can reuse this below as labels
        idx = torch.zeros(len(labels), self.num_rels + 2).long()  # +1 because we add dummy dimnension here for zeros, we remove it again; +1 because we want a no-relation label
        idx[:, 0:1] = labels.unsqueeze(-1) + 1
        edge_attr = torch.zeros(len(labels), self.num_rels + 2)
        edge_attr.scatter_(1, idx, 1)
        edge_attr = edge_attr[:, 1:]  # remove dummy dimension
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=-1)

        # just replicated parts of the edge_index etc. to make it undirected
        # where n is length of labels returned below
        graph = Data(x=x,
                     edge_index=edge_index,
                     edge_type=torch.cat([labels, labels], dim=-1),
                     edge_attr=edge_attr)

        # labels as one-hot
        if self.share_neg > 0:
            tasks = torch.cat([edge_index, neg_edge_index], dim=-1)
            neg_labels = torch.zeros(neg_size*2, self.num_rels + 1)
            neg_labels[:, self.num_rels] = 1
            labels = torch.cat([edge_attr, neg_labels], dim=0)  # reuse above edge attrs as labels one hot
        else:
            tasks = edge_index
            labels = edge_attr

        # for now we predict both directions though one (in first half of edge index) might be OK too
        # bin_labels = torch.zeros(len(labels)*2)
        # bin_labels[edge_type < self.num_rels] = 1
        return graph, tasks, labels

    def _ismember(self, edge, edge_index):
        src, trg = edge
        src_idx_in_eidx1 = (edge_index[0] == src).nonzero().squeeze(-1)
        src_idx_in_eidx2 = (edge_index[1] == src).nonzero().squeeze(-1)
        trgs = set(edge_index[1][src_idx_in_eidx1].tolist() + edge_index[0][src_idx_in_eidx2].tolist())
        return trg in trgs


# index is list of integers pointing to one-hot 1's
def to_sparse_one_hot(index, dim1):
    dim0 = len(index)
    i = torch.LongTensor([list(range(dim0)), index])
    v = torch.FloatTensor([1]*dim0)
    return torch.sparse.FloatTensor(i, v, torch.Size([dim0, dim1]))


#  basically second collator because within data loader sparse tensors are not supported
def pre_process(graph, dim1):
    graph.x = to_sparse_one_hot(graph.x, dim1)
    return graph


if __name__ == '__main__':
    pass
