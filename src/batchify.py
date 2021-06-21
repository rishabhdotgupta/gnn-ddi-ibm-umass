import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling


class Batchify:
    r""" Batchify creates batches for the DDI graph data. It uses the
    run-time arguments to pick which batching functions to use. Negative
    evidence is optional to include.
    """
    def __init__(self, args, data, ne_data=None):
        self.args = args
        self.data = data
        self.ne_data = ne_data

    def batchify(self, batch_size):
        r""" Batching used for training with or without negative evidence
        """
        if self.args.ne_train > 0 or self.args.new_edge > 0:
            return Batchify.batchify_with_ne(self.data, self.ne_data,
                                             self.args, batch_size)
        else:
            return Batchify.batchify_without_ne(self.data, self.args,
                                                batch_size)

    def eval_batchify(self, batch_size):
        r""" Batching used for evaluations. This does not perform negative
        sampling.
        """
        batch_size = min(batch_size, self.data.edge_index.size(1))
        e_id = torch.arange(self.data.target_edge_index.size(1))
        b_ids = torch.split(e_id, batch_size)
        for b_id in b_ids:
            mask_size = self.data.target_edge_index.size(1)
            b_id_mask = Batchify.ids_to_mask(b_id, mask_size)
            target_edge_index = self.data.target_edge_index[:, b_id_mask]
            y = self.data.y[b_id_mask]
            batch = Data(x=self.data.x,
                         edge_index=self.data.edge_index,
                         edge_attr=self.data.edge_attr,
                         target_edge_index=target_edge_index,
                         y=y)
            yield batch

    @staticmethod
    def ids_to_mask(ids, size):
        mask = torch.zeros(size).bool()
        mask[ids] = 1
        return mask

    @staticmethod
    def batchify_with_ne(data, ne_data, args, batch_size):
        r""" yields a generator of batches from the input data. Performs
        negative sampling. It supports two cases: using negative evidnce
        as a replacement for negative samples (ne_train > 1) and using negative
        evidnce as edges in the input graph (new_edge > 1).
        """
        if batch_size > data.target_edge_index.size(1):
            batch_size = data.target_edge_index.size(1)
        e_id = torch.randperm(data.target_edge_index.size(1))
        b_ids = torch.split(e_id, batch_size)
        # split ne_data into the same number of groups as b_ids.
        ne_id = torch.randperm(ne_data.edge_index.size(1))
        ne_split_size = ne_id.size(0) // (e_id.size(0) // batch_size)
        ne_split_size = min(ne_split_size, batch_size)
        ne_b_ids = torch.split(ne_id, ne_split_size)

        for b_id, ne_b_id in zip(b_ids, ne_b_ids):
            b_id_mask = Batchify.ids_to_mask(b_id, data.target_edge_index.size(1))
            target_edge_index = data.target_edge_index[:, b_id_mask]
            y = data.y[b_id_mask]

            if args.ne_train > 0 and args.new_edge == 0:
                # add the negative evidence as a negative sample.
                # In this case, the input and target graphs are the same.
                # NE labels are all zero.
                ne_edge_index = ne_data.edge_index[:, ne_b_id]
                ne_edge_attr = torch.zeros(ne_data.edge_attr[ne_b_id].size())
                target_edge_index = torch.cat((target_edge_index,
                                               ne_edge_index),
                                              dim=1)
                y = torch.cat((y, ne_edge_attr))

                # add negative samples to complete the batch
                num_neg = batch_size - ne_edge_index.size(1)
                negative_targets = negative_sampling(data.edge_index,
                                                     num_nodes=data.x.size(0),
                                                     num_neg_samples=num_neg,
                                                     force_undirected=True)
                target_edge_index = torch.cat((target_edge_index,
                                               negative_targets),
                                              dim=1)
                # treat negative evidence as all zeros
                negative_labels = torch.zeros(negative_targets.size(1),
                                              y.size(1))
                y = torch.cat((y, negative_labels))

                input_mask = b_id_mask.bitwise_not()
                batch = Data(x=data.x,
                             edge_index=data.edge_index[:, input_mask],
                             edge_attr=data.edge_attr[input_mask],
                             target_edge_index=target_edge_index,
                             y=y)
            else:
                negative_targets = negative_sampling(data.edge_index,
                                                     num_nodes=data.x.size(0),
                                                     num_neg_samples=batch_size,
                                                     force_undirected=True)
                target_edge_index = torch.cat((target_edge_index,
                                               negative_targets),
                                              dim=1)
                # treat negative evidence as all zeros
                negative_labels = torch.zeros(negative_targets.size(1),
                                              y.size(1))
                y = torch.cat((y, negative_labels))
                input_mask = b_id_mask.bitwise_not()

                if args.new_edge > 0:
                    # add all negative evidence onto the input graph.
                    edge_index = data.edge_index[:, input_mask]
                    edge_attr = data.edge_attr[input_mask]
                    edge_index = torch.cat([edge_index, ne_data.edge_index],
                                           dim=1)
                    edge_attr = torch.cat([edge_attr, ne_data.edge_attr])

                batch = Data(x=data.x,
                             edge_index=edge_index,
                             edge_attr=edge_attr,
                             target_edge_index=target_edge_index,
                             y=y)
            yield batch

    @staticmethod
    def batchify_without_ne(data, args, batch_size):
        r""" Create batches without using negative evidence. It still
        performs negative sampling (when not using the MLP.)
        """
        print('not using negative evidence')
        batch_size = min(batch_size, data.edge_index.size(1))
        e_id = torch.randperm(data.target_edge_index.size(1))
        b_ids = torch.split(e_id, batch_size)
        for b_id in b_ids:
            b_id_mask = Batchify.ids_to_mask(b_id,
                                             data.target_edge_index.size(1))
            target_edge_index = data.target_edge_index[:, b_id_mask]
            y = data.y[b_id_mask]
            # safety check to not use negative samples when using the MLP
            if args.model_name != 'mlp':
                negative_targets = negative_sampling(data.edge_index,
                                                     num_nodes=data.x.size(0),
                                                     num_neg_samples=batch_size,
                                                     force_undirected=True)
                target_edge_index = torch.cat((target_edge_index,
                                               negative_targets),
                                              dim=1)
                # labels for negative samples are all zero
                negative_labels = torch.zeros(negative_targets.size(1),
                                              y.size(1))
                y = torch.cat((y, negative_labels))
            input_mask = b_id_mask.bitwise_not()
            batch = Data(x=data.x,
                         edge_index=data.edge_index[:, input_mask],
                         edge_attr=data.edge_attr[input_mask],
                         target_edge_index=target_edge_index,
                         y=y)
            yield batch

