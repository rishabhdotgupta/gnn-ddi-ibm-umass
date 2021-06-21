import sys
import os.path as osp
from itertools import repeat
import torch
import numpy as np
from sklearn import random_projection
from torch_sparse import coalesce
from torch_geometric.data import Data
from utils import build_eval
import csv

from preprocess.data import get_graph_size

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    print('If input x has nan or inf', np.isinf(out).any(), np.isnan(out).any())

    # for fast training, we discard one-hot encoding and use 32 dimension vector from gaussian distribution
    if prefix == 'ddi_constraint' or prefix == 'decagon':
        if name == 'allx':
            transformer = random_projection.GaussianRandomProjection(
                n_components=32)
            out = transformer.fit_transform(out)
    out = torch.FloatTensor(out)
    return out


def edge_index_from_dict(graph_dict, prefix):
    row, col = [], []
    edgetype = []
    type = 0 # TODO num of last ddis to extract rows, cols, edgeattrs below, don't need to do collapse for mixture
    # but need collapse method that mixes mixt attributes with existing if present (which is strange and should not BUT CHEC THIS -- otherwise we do not need it??)
    # also need to upate tyeps based on collapsed -> add map of merged edge attrs to new types

    # e = 0
    # for n, ts in graph_dict[86].items():
    #     for fake_type, edges in graph_dict.items():
    #         if fake_type == 86: continue
    #
    #         for t in ts:
    #             if t in edges[n] or n in edges[t]:
    #                 e += 1
    #                 print(n, t, "!"*30)
    # e is 314 here for deepddi as in test.py.

    me = len(graph_dict) - 1
    me = get_graph_size(graph_dict, subgraphidx=me)

    for fake_type, edges in graph_dict.items():
        for key, value in edges.items():
            row += repeat(key, len(value))
            col += value
            edgetype += repeat(type, len(value))
        type += 1
    print('edge type ', type)
    print('edgetype length ', len(edgetype))
    # i = edgetype[-me:]
    # for e in i:
    #     if e != 86:
    #         print("???",e)
    # s = sum(i)
    edge_attr = one_hot(torch.tensor(edgetype))
    if prefix == 'deepddi-ssp':
        edge_index, edge_attr = collapse(row, col, edge_attr)
        edge_index2, edge_attr2 = [], []
    else:
        ea2 = edge_attr[-me:, :]
        edge_index, edge_attr = collapse(row[:-me], col[:-me], edge_attr[:-me, :])
        edge_index2, edge_attr2 = collapse(row[-me:], col[-me:], ea2)
    # c = sum(sum(edge_attr2))
    # convert to Bool tensor ??
    # edge_attr = edge_attr > 0
    # return edge_index, edge_attr.float(), edge_index2, edge_attr2.float()
    return edge_index, edge_attr, edge_index2, edge_attr2


def read_ddi_data(folder, prefix, random_seed, data_ratio):
    # data_ratio = 100  # for ratio test purpose
    # random_seed = 3  # for analytis test purpose
    np.random.seed(random_seed)
    names = ['allx', 'graph']
    items = [read_file(folder, prefix, name) for name in names]
    x, graph = items

    print("gs:", get_graph_size(graph))
    # print("gs:", get_graph_size(graph, subgraphidx=86))

    edge_index, edge_attr, edge_index2, edge_attr2 = edge_index_from_dict(graph, prefix)

    # if not osp.exists("perm.idx.%d.npy" % random_seed):
    #     perm = np.random.permutation(edge_index.size(1))
    #     np.save(osp.join(folder, "perm.idx.%d.npy" % random_seed), perm)
    # else:
    #     perm = np.load(osp.join(folder, "perm.idx.%d.npy" % random_seed))

    # if not osp.exists(osp.join(folder, prefix+".perm.idx.seed_%d_ratio_%d.npy" % (random_seed, data_ratio))):
    perm = np.random.permutation(edge_index.size(1))
    perm = perm[:int(edge_index.size(1)*data_ratio/100)]
    np.save(osp.join(osp.join(folder, prefix+".perm.idx.seed_%d_ratio_%d.npy" %
                              (random_seed, data_ratio))), perm)
    # else:
    #     perm = np.load(osp.join(
    #         osp.join(folder, prefix+".perm.idx.seed_%d_ratio_%d.npy" % (random_seed, data_ratio))))

    perm = torch.LongTensor(perm)


    print('original stat', x.size(0))
    print('node count', x.size(0))
    print('edge count', edge_index.size(1))
    print('max in edge_index', max(np.unique(edge_index[1])))

    # reducing size of dataset with data_ratio
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    # edge_type = torch.Tensor(edge_type)[perm].long()

    print('now stat')
    print('node count', x.size(0))
    print('edge count', edge_index.size(1))
    print('max in edge_index', max(np.unique(edge_index[1])))

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, perm=perm)

    data2 = Data(x=x, edge_index=edge_index2,
                 edge_attr=edge_attr2) # TODO maybe later also include a permutation here
    # data.edge_type = edge_type

    return data, data2


def read_file_twosides(twosides='data/TWOSIDES/TWOSIDES.csv', sub=False):
    # Create dictionary with keys as edge types, node pairs as values
    relation_type = {}
    # Create dictionary with keys as nodes / drugs, index as values
    drugs = {}
    # Create dictionary with keys as edge types, index as values
    relation = {}

    count = 0               # Total lines + ignore the first row
    count_drugs = 0         # Total Unique drugs
    count_relation_type = 0 # Unique relation types

    # Import TWOSIDES Data
    with open(twosides, newline='') as csvfile:
        line = csv.reader(csvfile, delimiter=',')
        for row in line:
            # Update edge types
            if count != 0:
                if row[4] in relation_type:
                    relation_type[row[4]].append([row[0], row[2]])
                else:
                    relation_type[row[4]] = []
                    relation_type[row[4]].append([row[0], row[2]])
                # create index for the drugs if they dont have an index
                if row[0] not in drugs:
                    drugs[row[0]] = count_drugs
                    count_drugs += 1

                if row[2] not in drugs:
                    drugs[row[2]] = count_drugs
                    count_drugs += 1

                # create index for relation type if relation does not have one
                if row[4] not in relation:
                    relation[row[4]] = count_relation_type
                    count_relation_type += 1
            count = count + 1
    if sub == True:
        # num_edges: dict with key: relation type and value: # of edges
        num_edges = {}
        relation = {}
        new_relation_type = {}
        for key in relation_type:
            num_edges[key] = len(relation_type[key])

        # sort by values in dict num_edges
        sorted_edges = sorted(num_edges.items(), key=lambda item: item[1])
        # create new index and edges for subgraph relation type
        count_relation_type = 0
        # Total number of edges needs to be recounted
        new_count = 0
        for i in sorted_edges[4000:10000:60]:
            new_relation_type[i[0]] = relation_type[i[0]]
            relation[i[0]] = count_relation_type
            count_relation_type += 1
            new_count += i[1]
        relation_type = new_relation_type
        count = new_count
        print('subgraph created with ', count_relation_type, ' relation types')
        print('subgraph created with ', new_count, ' total edges')


    # Import synergy information
    count_synergy = 0
    synergy = 'data/TWOSIDES/synergism_new.txt'
    neg = 'data/TWOSIDES/deepddi_neg.txt'
    relation_type['Synergy'] = []

    # Initialize 'Synergy' as relation type
    relation['Synergy'] = count_relation_type
    count_relation_type += 1
    # add eges for the synergy data
    with open(synergy, newline='') as csvfile:
        line = csv.reader(csvfile)
        for row in line:
            relation_type['Synergy'].append([row[0].strip(), row[1].strip()])
            count_synergy += 1
    with open(neg, newline='') as csvfile:
        line = csv.reader(csvfile,delimiter=',')
        for row in line:
            relation_type['Synergy'].append([row[0].strip(), row[1].strip()])
            count_synergy += 1

    print('Imported TWOSIDES Data which contained ', count_drugs, ' drugs types')

    # edgetype will eventually contain relation type for the indexed edge
    edgetype = []
    edgetype += repeat(0, count + count_synergy )

    return relation_type, drugs, relation, edgetype, count_synergy


def read_twosides_data(folder, prefix, random_seed, data_ratio):
    [ddi_relations, index_drugs, index_relation, edgetype, me] = read_file_twosides(sub=True)
    n_drugs = max(index_drugs.values())+1

    edge_index_row = []
    edge_index_col = []

    count = 0
    # Transforming only TWOSIDES Data and Synergy Data (last relation type)
    for relation in ddi_relations:
        # orginally iterated through all possible combinations in decagon code,
        # that is not necessary.
        for edge in ddi_relations[relation]:
            d1 = index_drugs[edge[0]]
            d2 = index_drugs[edge[1]]

            # Add to Edge_Index - undirectional (not collapsed version)
            # Not necessary to have this if condition (repeated in collapse())
            if d1 < d2:
                edge_index_row.append(d1)
                edge_index_col.append(d2)
            else:
                edge_index_row.append(d2)
                edge_index_col.append(d1)

            # complete edge_type
            edgetype[count] = index_relation[relation]
            count += 1

    mat_x = np.zeros((n_drugs, n_drugs))
    for i in range(n_drugs):
        mat_x[i,i] = 1.
    x = torch.FloatTensor(mat_x.tolist())

    edge_attr = one_hot(torch.tensor(edgetype))
    edge_attr2 = edge_attr[-me:,:]
    edge_index, edge_attr = collapse(edge_index_row[:-me], edge_index_col[:-me], edge_attr[:-me,:])
    edge_index2, edge_attr2 = collapse(edge_index_row[-me:], edge_index_col[-me:], edge_attr2)
    print('Edge_attr and edge_index computed')

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data2 = Data(x=x, edge_index=edge_index2, edge_attr=edge_attr2)

    return data, data2


def collapse(row, col, edge_attr):
    # index = row*num_nodes + col
    dict1 = {}
    for i in range(len(row)):
        if row[i] < col[i]:
            ind = (row[i], col[i])
        else:
            ind = (col[i], row[i])
        if ind not in dict1:
            dict1[ind] = edge_attr[i]
        else:
            dict1[ind] += edge_attr[i]
    new_row = []
    new_col = []
    new_edge_attr = []
    for key in dict1.keys():
        new_row.append(key[0])
        new_col.append(key[1])
        dict1.get(key)[dict1.get(key) > 1] = 1.   # Get rid of overlap
        new_edge_attr.append(dict1.get(key))
    edge_index = torch.stack(
        [torch.LongTensor(new_row), torch.LongTensor(new_col)], dim=0)
    new_edge_attr = torch.stack(new_edge_attr, dim=0)
    return edge_index, new_edge_attr


# assume both are collapsed already
def collapse_datasets(data1, data2):
    ei1 = data1.edge_index
    ei2 = data2.edge_index

    dict1 = {}
    for i in range(ei1.shape[-1]):
        ind = (ei1[0][i], ei1[1][i])
        dict1[ind] = data1.edge_attr[i]

    for i in range(ei2.shape[-1]):
        row = ei2[0][i]
        col = ei2[1][i]
        if row < col:
            ind = (row, col)
        else:  # should not be case since was already collapsed
            ind = (col, row)
        if ind not in dict1:
            dict1[ind] = data2.edge_attr[i]
        else:
            dict1[ind] += data2.edge_attr[i]
    new_row = []
    new_col = []
    new_edge_attr = []
    for key in dict1.keys():
        new_row.append(key[0])
        new_col.append(key[1])
        new_edge_attr.append(dict1.get(key))
    edge_index = torch.stack(
        [torch.LongTensor(new_row), torch.LongTensor(new_col)], dim=0)
    new_edge_attr = torch.stack(new_edge_attr, dim=0)
    return Data(x=data1.x, edge_index=edge_index,
                edge_attr=new_edge_attr)


def one_hot(src, num_classes=None, dtype=None):
    r"""Converts labels into a one-hot format.
    Args:
        src (Tensor): The labels.
        num_classes (int or list, optional): The number of classes.
            (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.
    :rtype: :class:`Tensor`
    """

    src = src.to(torch.long)
    src = src.unsqueeze(-1) if src.dim() == 1 else src
    assert src.dim() == 2

    if num_classes is None:
        num_classes = src.max(dim=0)[0] + 1
    else:
        if torch.is_tensor(num_classes):
            num_classes = num_classes.tolist()

        num_classes = torch.tensor(
            repeat(num_classes, length=src.size(1)),
            dtype=torch.long,
            device=src.device)

    if src.size(1) > 1:
        zero = torch.tensor([0], device=src.device)
        src = src + torch.cat([zero, torch.cumsum(num_classes, 0)[:-1]])

    size = src.size(0), num_classes.sum()
    out = torch.zeros(size, dtype=dtype, device=src.device)
    out.scatter_(1, src, 1)
    return out


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def read_fb15k(folder, args):
    r""" FB15K provides a train/valid/test split. This function loads
    then into Data objects. It is tab separated and directed.
    """
    file_names = ['train.txt', 'valid.txt', 'test.txt']

    # gets ids across each file, so data has the same shapes
    sources, rels, sinks = set(), set(), set()
    for file_name in file_names:
        with open(folder + file_name, 'r') as fb_file:
            lines = [a.split('\t') for a in fb_file.readlines()]
            lines = [(a[0].strip(), a[1].strip(), a[2].strip()) for a in lines]
            # load in a fraction of the dataset
            lines = lines[:int(len(lines) * (args.data_ratio / 100))]

        sources |= set([a[0] for a in lines])
        rels |= set([a[1] for a in lines])
        sinks |= set([a[2] for a in lines])

    num_types = len(rels)
    nodes = list(set(sources | sinks))

    n_to_id = dict([(b, a) for (a, b) in enumerate(nodes)])
    rel_to_id = dict([(b, a) for (a, b) in enumerate(rels)])

    datasets = {}

    for file_name in file_names:
        with open(folder + file_name, 'r') as fb_file:
            lines = [a.split('\t') for a in fb_file.readlines()]
            lines = [(a[0].strip(), a[1].strip(), a[2].strip()) for a in lines]
            # load in a fraction of the dataset
            lines = lines[:int(len(lines) * (args.data_ratio / 100))]

        num_types = max(num_types, len(rels))

        edge_index = torch.zeros(2, len(lines)).long()
        edge_attr = torch.zeros(len(lines), len(rels))

        for i, line in enumerate(lines):
            source_id = n_to_id[line[0]]
            sink_id = n_to_id[line[2]]
            rel_id = rel_to_id[line[1]]
            edge = torch.LongTensor([source_id, sink_id])
            # if an edge occurs multiple times, they are merged by coalesce
            edge_index[:, i] = edge
            edge_attr[i][rel_id] = 1.0

        x = torch.diag(torch.ones(len(nodes))).float()

        edge_index, edge_attr = coalesce(edge_index, edge_attr, m=x.size(0),
                                         n=x.size(1), op='max')

        datasets[file_name] = Data(x=x, edge_index=edge_index.long(),
                                   edge_attr=edge_attr)

    train = datasets['train.txt']
    valid = datasets['valid.txt']
    test = datasets['test.txt']

    # print some stats for each dataset
    print('Loaded FB15K-237')
    print(' * train nodes: ', train.x.size(0))
    print(' * valid nodes: ', valid.x.size(0))
    print(' * test nodes: ', test.x.size(0))
    print(' * Number of relations: ', num_types)
    print(' * valid relations: ', valid.edge_attr.size(1))
    print(' * test relations: ', test.edge_attr.size(1))
    print(' * train node pairs: ', train.edge_index.size())
    print(' * train edge: ', train.edge_attr.sum())
    print(' * valid edge: ', valid.edge_attr.sum())
    print(' * test edge: ', test.edge_attr.sum())
    print(' * Max edge endpoint id: ', train.edge_index.max())

    valid = build_eval(train, valid)
    test = build_eval(train, test)
    train = build_eval(train, train)

    return (train, valid, test, None, num_types)
