import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from utils import build_eval
from utils_read import (read_ddi_data, read_twosides_data, read_fb15k)


def load_data(args):

    if args.data_dir == './data/deepddi-me':
        # no ssp for negative evidencee
        if (not args.ne_train and not args.new_edge) and args.use_ssp:
            read_func, prefix = read_ddi_data, 'deepddi-ssp'
        else:
            read_func, prefix = read_ddi_data, 'deepddi'
    elif args.data_dir == './data/TWOSIDES':
        read_func, prefix = read_twosides_data, 'twosides'
    elif args.data_dir == './data/fb15k-237':
        return read_fb15k('data/fb15k-237/', args)
    else:
        raise ValueError(f'Name of data directory {args.data_dir} not found')

    data, data2 = read_func(args.data_dir,
                            prefix,
                            random_seed=args.seed,
                            data_ratio=args.data_ratio)

    # # Consider negative evidence as all zeros not another edge type
    # data.edge_attr = data.edge_attr[:,:-1]
    # data2.edge_attr = data2.edge_attr[:, :-1]

    # num_types includes dummy type for "no ddi"
    # also, predict the n-hop vec so the latter especially is fine
    num_edges, num_types = data.edge_attr.size()
    train_size = int(num_edges * args.train_size)
    train_valid_size = train_size + int(num_edges * args.valid_size)

    train_data = Data(x=data.x,
                      edge_index=data.edge_index[:, :train_size],
                      edge_attr=data.edge_attr[:train_size, :])
    valid_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, train_size:train_valid_size],
        edge_attr=data.edge_attr[train_size:train_valid_size, :])
    test_data = Data(x=data.x,
                     edge_index=data.edge_index[:, train_valid_size:],
                     edge_attr=data.edge_attr[train_valid_size:, :])

    if prefix != 'deepddi-ssp' and args.ne_train == 0 and args.new_edge == 0:
        # remove dummy dim
        train_data.edge_attr = train_data.edge_attr[:, :-1]
        valid_data.edge_attr = valid_data.edge_attr[:, :-1]
        test_data.edge_attr = test_data.edge_attr[:, :-1]
        num_types -= 1

    if args.ne_train == 0 and args.new_edge == 0:
        # remove nodes that have no edges from our datasets.
        train_nodes = train_data.edge_index.contiguous().view(-1).unique()
        valid_nodes = valid_data.edge_index.contiguous().view(-1).unique()
        test_nodes = test_data.edge_index.contiguous().view(-1).unique()
        subset = torch.cat([train_nodes, valid_nodes, test_nodes]).unique()

        train_data = build_subgraph(subset, train_data)
        valid_data = build_subgraph(subset, valid_data)
        test_data = build_subgraph(subset, test_data)

    # add negative evidence as new edge in input graph.
    #if prefix != 'deepddi-ssp' and args.new_edge > 0:
    #    ne_edge_index = data2.edge_index
    #    ne_edge_attr = data2.edge_attr
    #    input_graph = train_data
    #    input_graph.edge_index = torch.cat([input_graph.edge_index,
    #                                        ne_edge_index],
    #                                       dim=1)
    #    input_graph.edge_attr = torch.cat([train_data.edge_attr, ne_edge_attr])
    #else:
    input_graph = train_data

    # make it so that the training graph is used as input for evaluation
    valid_data = build_eval(input_data=input_graph, target_data=valid_data)
    test_data = build_eval(input_data=input_graph, target_data=test_data)

    # input is initially the train data. We update the train data during
    # batchify depending on the settings used.
    train_data = build_eval(input_data=train_data, target_data=train_data)

    print(data2.edge_index)
    print(type(data2.edge_index))

    return train_data, valid_data, test_data, data2, num_types


def build_subgraph(subset, data):
    r""" Build subgraphs that only contain nodes that occur in each dataset.
    """
    edge_index, edge_attr = subgraph(subset, data.edge_index, data.edge_attr,
                                     relabel_nodes=True,
                                     num_nodes=data.x.size(0))
    data.x = data.x[subset]
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data


def data_to_undirected(data):
    r""" given some Data, make the edges undirected.
    """
    edge_index, edge_attr = undirect(data.edge_index, data.edge_attr)
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data


def undirect(edge_index, edge_attr):
    r"""
    Based off of torch_geometric.utils.to_undirected,
    but also accounts for edge_attr. Does not use torch_sparse.coalesce,
    so duplicates aren't removed here. Insteas, duplicates are removed
    when collapsing the dataset.
    """
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    return edge_index, edge_attr


def remove_ne(subgraph):
    r""" Remove negative evidence edges from the input graph.
    """
    input_ids = subgraph.edge_attr[:, -1] == 0
    subgraph.edge_index = subgraph.edge_index[:, input_ids]
    subgraph.edge_attr = subgraph.edge_attr[input_ids]
    return subgraph
