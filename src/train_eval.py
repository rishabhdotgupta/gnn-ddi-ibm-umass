r""" models for evaluation of different models. (eval is below the decorator
at the very bottom)
"""


import torch
import torch.nn.functional as F
from utils import classification_report
from batchify import Batchify
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def train(data, ne_data, model, optimizer, args):
    model.train()
    optimizer.zero_grad()
    

    data = data.to(args.device)
    pred = model(data)
    data = data.to(args.device)
    if ne_data:
        ne_data = ne_data.to(args.device)

    total_loss = 0
    iters = 0
    batches = Batchify(args, data, ne_data)
    for batch in batches.batchify(args.batch_size):
        pred = model(batch.to(args.device))
        loss = F.binary_cross_entropy(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iters += 1

        # use reinforce
        samplers = ('sigmoid_sampler', 'relational_sampler')
        if args.sampling_name in samplers:
            optimizer.zero_grad()
            update_sampling(data, model, loss, args)

    loss = total_loss / iters

    return loss


def get_sample_grad(model, data):
    r""" Get gradient of subgraph logit wrt edge params
    """
    model.subgraph_logit.backward()
    return model.edge_params.grad


def update_sampling(data, model, loss, args):
    r""" update edge params using REINFORCE gradient estimator
    """
    model.edge_params.data -= args.lr * loss * get_sample_grad(model, data)


def plot_sample_data(eval_func):
    r""" Decorator for the evaluation function that traacks sampling data
    and saves plots
    """
    def plotting_wrapper(train_data, ne_data, data, model, args):
        if hasattr(model, 'edge_params'):
            model.average_frac_sampled.reset()
            model.average_sampled.reset()

        metrics = eval_func(train_data, ne_data, data, model, args)

        if hasattr(model, 'edge_params'):

            average_frac_samples = model.average_frac_sampled.average.detach()

            tick_size = 15

            # plot the average number of each edge type sampled
            plt.bar(range(model.model.out_dim), average_frac_samples)
            # plt.title('Average Fraction of Edge Type Sampled')
            plt.xlabel('Edge Type', fontsize=tick_size)
            plt.ylabel('Average Fraction Sampled', fontsize=tick_size)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            plt.savefig(f'average_fraction_sampled')
            plt.clf()

            # plot the average fraction of each edge type sampled
            plt.bar(range(model.model.out_dim),
                    model.average_sampled.average.detach())
            plt.xlabel('Edge Type', fontsize=tick_size)
            plt.ylabel('Average # Sampled', fontsize=tick_size)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            plt.savefig(f'average_sampled')
            plt.clf()

            # sort ddi types by frequency
            freqs = (train_data.edge_attr.sum(0)).cpu()
            indices = freqs.argsort(descending=True)

            # plot edge frequencies
            plt.bar(range(model.model.out_dim), freqs[indices])
            plt.xlabel('Edge Type', fontsize=tick_size)
            plt.ylabel('Frequency', fontsize=tick_size)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            plt.savefig('freqs')
            plt.clf()

            # plot edge frequencies
            plt.bar(range(model.model.out_dim), torch.log(freqs[indices]))
            plt.xlabel('Edge Type', fontsize=tick_size)
            plt.ylabel('Log Frequency', fontsize=tick_size)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            plt.savefig('log_freqs')
            plt.clf()

            # order samples and edge params in the same way.
            logits = model.edge_params.detach()[indices]
            probs = F.softmax(logits, dim=0)

            # plot sampling logits
            plt.bar(range(model.model.out_dim), logits)
            plt.xlabel('Edge Type', fontsize=tick_size)
            plt.ylabel('Sampling Logit', fontsize=tick_size)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            plt.savefig('logits')
            plt.clf()
            # freqs = valid_data.edge_attr.sum(0)
            plt.bar(range(model.model.out_dim), probs)
            plt.xlabel('Edge Type', fontsize=tick_size)
            plt.ylabel('Sampling Probability', fontsize=tick_size)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            plt.savefig('probs')
            plt.clf()
            # Sampling probabilities scaled by sampling frequency
            plt.bar(range(model.model.out_dim), freqs[indices] * probs)
            plt.xlabel('Edge Type', fontsize=tick_size)
            plt.ylabel('Scaled Frequency', fontsize=tick_size)
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            plt.savefig('scaled_freqs')
            plt.clf()
        return metrics
    return plotting_wrapper


@plot_sample_data
def eval(train_data, ne_data, data, model, args):
    model.eval()

    y_probs = []
    batches = Batchify(args, data)
    for batch in batches.eval_batchify(args.batch_size):
        y_prob = model(batch.to(args.device))
        y_probs.append(y_prob.detach().cpu())
    y_prob = torch.cat(y_probs)
    y_prob = y_prob.detach().cpu().numpy()
    y = data.y.detach().cpu().numpy()
    return classification_report(y, y_prob)
