import torch

from tensorboardX import SummaryWriter
import time

import os.path as osp
from train_eval import train, eval


class Experiment:
    def __init__(self, train_data, valid_data, test_data, ne_data, model,
                 model_save_path, optimizer, args):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.ne_data = ne_data
        self.model = model
        self.model_save_path = model_save_path
        self.optimizer = optimizer
        self.args = args

    def run(self):
        writer = SummaryWriter(self.model_save_path)

        best_pr = None
        best_pr_prog = {'cnt': 0, 'value': None}
        train_time = 0
        start = time.time()
        for epoch in range(1, self.args.epoch + 1):

            loss, metrics, epoch_time = self.run_epoch()
            train_time += epoch_time

            valid_roc, valid_pr = metrics['roc'], metrics['pr']
            if self.early_stop(metrics, best_pr_prog, writer, epoch):
                break

            writer.add_scalar('train/loss', loss, epoch - 1)
            writer.add_scalar('valid/roc', valid_roc, epoch - 1)
            writer.add_scalar('valid/pr', valid_pr, epoch - 1)

            print(('\nEpoch: [{:04d}/{:04d}],\n '
                   'Loss: {:.7f}, '
                   'Validation PR_AUC: {:.7f}, '
                   'Validation ROC_AUC: {:.7f}').format(epoch, self.args.epoch,
                                                        loss, valid_pr,
                                                        valid_roc))

            if best_pr is None or valid_pr >= best_pr:
                test_metrics, test_time = self.run_test()
                best_pr = valid_pr
                torch.save(self.model.state_dict(),
                           osp.join(self.model_save_path,
                                    self.args.model_name))
                for m, val in test_metrics.items():
                    writer.add_scalar('test/' + m, val, epoch - 1)

            s = ""
            for m, val in test_metrics.items():
                s += ' test/' + str(m) + " " + str(val)
            print(s)
        writer.close()

        self.timing(start, train_time, test_time, self.args.epoch)

    def run_epoch(self):
        r""" Train model on input data for a single epoch. Compute validation
        metrics.
        """
        train_start = time.time()
        loss = train(self.train_data, self.ne_data, self.model, self.optimizer,
                     self.args)
        # eval needs train and ne_data for plotting sampling data
        metrics = eval(self.train_data, self.ne_data, self.valid_data,
                       self.model, self.args)
        epoch_time = time.time() - train_start
        return loss, metrics, epoch_time

    def early_stop(self, metrics, best_pr_prog, writer, epoch):
        r""" Return True to stop when the pr progression is below the threshold
        for args.n_iter_no_change epochs.
        """
        _, valid_pr = metrics['roc'], metrics['pr']

        if best_pr_prog['value'] is None or valid_pr > best_pr_prog['value']:
            best_pr_prog['cnt'] = 0
            best_pr_prog['value'] = valid_pr

        if valid_pr < best_pr_prog['value']:
            best_pr_prog['cnt'] += 1
        if best_pr_prog['cnt'] > self.args.n_iter_no_change:
            writer.add_scalar('text/early_stop_epoch', epoch, 0)
            print('early stop')
            return True
        return False

    def run_test(self):
        test_start = time.time()
        test_metrics = eval(self.train_data, self.ne_data,
                            self.test_data, self.model, self.args)
        test_time = time.time() - test_start
        return test_metrics, test_time

    def timing(self, start, train_time, test_time, epoch):
        c = time.time() - start
        days = c // 86400
        hours = c // 3600 % 24
        minutes = c // 60 % 60
        seconds = c % 60
        print(f'Total Time: {days}d:{hours}h:{minutes}m:{seconds}s')
        avg_train_time = train_time / epoch
        print(f'Average Training Time per epoch: {avg_train_time}s')
        print(f'Inference Time: {test_time}s')
