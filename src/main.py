import torch
from torch import optim
import random
import numpy as np
from build_data import load_data
from argparser import base_args
from models.model_loader import ModelLoader
from experiment import Experiment


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = base_args()

    set_all_seeds(args.seed)

    train_data, valid_data, test_data, ne_data, num_types = load_data(args)

    loader = ModelLoader(args, train_data.x.size(1), num_types)
    model, model_save_path = loader.load()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('--- Run Info ---')
    for arg in vars(args):
        print(' * {}={}'.format(arg, getattr(args, arg)))

    exp = Experiment(train_data, valid_data, test_data, ne_data, model,
                     model_save_path, optimizer, args)
    exp.run()


if __name__ == '__main__':
    main()
