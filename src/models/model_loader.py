import torch
from torch_geometric.nn import DataParallel

import os
import os.path as osp
from models.ddi_mlp import DDI_MLP
from models.ddi_nnconv import DDI_NNConv
from models.ddi_dedicom import DDI_dedicom
from models.ddi_distmult import DDI_distmult
from models.ddi_decagon import DDI_decagon
from models.ddi_decagon_prob import DDI_decagon_prob
from model_sampling.relational_sampler import RelationalSampler
from model_sampling.sigmoid_sampler import SigmoidSampler
from model_sampling.random_sampler import RandomSampler
from model_sampling.inverse_sampler import InverseSampler


class ModelLoader:
    r""" Pick and construct model and its save path given the command line
    and specified model arguments
    """
    def __init__(self, clargs, *args):
        self.clargs = clargs
        self.model_name = clargs.model_name
        self.sampling_name = clargs.sampling_name
        self.model_args = (*args, clargs)  # append clargs to model args

        self.models = {
            'mlp': DDI_MLP,
            'nnconv': DDI_NNConv,
            'dedicom': DDI_dedicom,
            'distmult': DDI_distmult,
            'decagon': DDI_decagon,
            'decagon_prob': DDI_decagon_prob
        }

        self.samplers = {
            'relational_sampler': RelationalSampler,
            'inverse_sampler': InverseSampler,
            'random_sampler': RandomSampler,
            'sigmoid_sampler': SigmoidSampler
        }

    def load(self):
        model = self.model_loader().to(self.clargs.device)
        save_path = self.model_paths(model)
        return model, save_path

    def model_loader(self):
        model = self.model_from_name()
        if self.clargs.sample:
            model = self.sampling_from_name(model)
        if torch.cuda.device_count() > 1 and not self.clargs.no_cuda:
            model = DataParallel(model)
        return model

    def model_from_name(self):
        if self.model_name in self.models:
            return self.models[self.model_name](*self.model_args)
        else:
            raise ValueError(f'Name of model {self.model_name} is invalid.')

    def sampling_from_name(self, model):
        sample_size = [7, 3]
        if self.sampling_name in self.samplers:
            return self.samplers[self.sampling_name](model, self.clargs,
                                                     size=sample_size)
        else:
            msg = f'Name of sampler {self.sampling_name} is invalid'
            raise ValueError(msg)

    def get_save_path(self):
        args = self.clargs
        f_name = 'GNN_data_{}_seed_{}_ratio_{}_ne_{}{}'.format(
            args.data_dir[args.data_dir.rindex("/") + 1:], args.seed,
            args.data_ratio, args.ne_train, args.ne_valid)
        return osp.join(osp.dirname(osp.realpath(__file__)),
                        '../saved_models/', f_name)

    def model_paths(self, model):
        model_save_path = self.get_save_path()
        if not osp.exists(model_save_path):
            os.makedirs(model_save_path)

        if self.clargs.resume:
            if self.clargs.resume_path is not None:
                resume_path = self.clargs.resume_path
            else:
                resume_path = model_save_path

            if osp.exists(os.path.join(resume_path, self.clargs.model_name)):
                print('loading pretrained model from ' + resume_path)
                model.load_state_dict(
                    torch.load(osp.join(resume_path, self.clargs.model_name)))
        return model_save_path
