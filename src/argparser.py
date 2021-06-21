import argparse
import torch  # needed to set the device


def base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="forbid using cuda")

    parser.add_argument("--model_name",
                        default='mlp',
                        type=str,
                        help='lowercase name of model')

    # input and output dir
    data_dir_options = ['./data/deepdddi-me', './data/TWOSIDES',
                        './data/fb15k-237']
    parser.add_argument("--data_dir",
                        default='./data/deepddi-me',
                        type=str,
                        help=f"data directory, options: {data_dir_options}")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("--only_test", action="store_true", help="random seed")
    parser.add_argument("--topk", default=10, type=int, help="topk precision")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--resume_path",
                        type=str,
                        default=None,
                        help="saved model path")

    # parameters
    parser.add_argument("--epoch",
                        default=200,
                        type=int,
                        help="training epoch")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument(
        "--data_ratio",
        default=100,
        type=float,
        help=
        "percentage of edges to be considered, but we consider full neg. evidence if that argument is set"
    )
    parser.add_argument(
        "--n_iter_no_change",
        default=50,
        type=float,
        help="Maximum number of epochs to not meet ``tol`` improvement.")
    parser.add_argument("--train_size",
                        default=0.6,
                        type=float,
                        help="train_size")
    parser.add_argument("--valid_size",
                        default=0.2,
                        type=float,
                        help="validation size")
    parser.add_argument("--ne_train",
                        default=0,
                        type=int,
                        help="add negative evidence (if value > 0)")
    parser.add_argument("--ne_valid",
                        default=0,
                        type=int,
                        help="add negative evidence (if value > 0)")
    parser.add_argument("--ne_test",
                        default=1,
                        type=int,
                        help="add negative evidence (if value > 0)")
    parser.add_argument("--new_edge",
                        default=0,
                        type=float,
                        help="treat negative evidence as new edge type")
    parser.add_argument("--edge_frac",
                        default=1,
                        type=float,
                        help="fraction of edge attributes to keep during training.")
    parser.add_argument("--model_depth",
                        default=2,
                        type=int,
                        help="hidden dimension")
    parser.add_argument("--hidden_dim",
                        default=100,
                        type=int,
                        help="hidden dimension")
    parser.add_argument('--sample',
                        action='store_true',
                        help='Use sampler from src/model_sampling/')
    parser.add_argument('--sampling_name',
                        type=str,
                        help='lowercase name of sampling method to use')
    parser.add_argument('--activation',
                        default='softmax',
                        type=str,
                        help='activation function for message aggregation: sigmoid or softmax')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='Visualize results')
    parser.add_argument('--use_ssp',
                        default=1,
                        type=int,
                        help='use structural similarity profiles as drug features in place of one-hot encoding')
    parser.add_argument('--batch_size',
                        default=2000,
                        type=int,
                        help='batch size for train and eval')
    parser.add_argument('--basis',
                        default=2,
                        type=int,
                        help='basis for rgcn expansion')
    return parser


def which_device(args):
    if args.no_cuda:
        return torch.device('cpu')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def base_args():
    parser = base_parser()

    args = parser.parse_args()
    args.device = which_device(args)
    return args
