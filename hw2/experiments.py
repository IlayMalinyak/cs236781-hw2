import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchviz import make_dot
from IPython.display import display


from cs236781.train_results import FitResult

from .cnn import CNN, ResNet
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
}

OPTIMIZERS = {'SGD': torch.optim.SGD, 'ADAM': torch.optim.Adam}
LOSSES = {"cross entropy":torch.nn.CrossEntropyLoss}



def mlp_experiment(
    depth: int,
    width: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    dl_test: DataLoader,
    n_epochs: int,
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======
    
    n_layers = depth  # number of layers (not including output)
    hidden_dims = width  # number of output dimensions for each hidden layer
    activation = torch.nn.LeakyReLU(0.05)  # activation function to apply after each hidden layer
    out_activation = "softmax"  # activation function to apply at the output layer
    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    hp_arch= dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )
    
    lr, weight_decay, momentum = 0.001, 0.001, 0.001  # Arguments for SGD optimizer
    betas = (0.7, 0.99)
    # hp_optim = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)
    hp_optim = dict(lr=lr, weight_decay=weight_decay, betas=betas, loss_fn=loss_fn)


    model = BinaryClassifier(
    model=MLP(
        in_dim=2,
        dims=[*[hp_arch['hidden_dims'],]*hp_arch['n_layers'], 2],
        nonlins=[*[hp_arch['activation'],]*hp_arch['n_layers'], hp_arch['out_activation']]
    ),
    threshold=0.5,)
    loss_fn = hp_optim.pop('loss_fn')
    # optimizer = torch.optim.SGD(params=model.parameters(), **hp_optim)
    optimizer = torch.optim.Adam(params=model.parameters(), **hp_optim)
    trainer = ClassifierTrainer(model, loss_fn, optimizer)

    fit_result = trainer.fit(dl_train, dl_valid, num_epochs=n_epochs, print_every=0, verbose=False, early_stopping=3)
    optimal_thresh = select_roc_thresh(model, *dl_valid.dataset.tensors, plot=False)
        
    model.threshold = optimal_thresh
    epoch_result = trainer.test_epoch(dl_test)
    return model, optimal_thresh, fit_result.test_acc[-1], epoch_result.accuracy
    # ========================
    return model, thresh, valid_acc, test_acc


def cnn_experiment(
    run_name,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=None,
    lr=1e-3,
    reg=1e-3,
    # Model params
    filters_per_layer=[64],
    layers_per_block=2,
    pool_every=2,
    hidden_dims=[1024],
    model_type="cnn",
    # You can add extra configuration for your experiments here
    activation_type = "relu",
    activation_params =dict(),
    pooling_type = "max",
    pooling_params= dict(kernel_size=2),
    batchnorm=True,
    dropout=0.1,
    bottleneck=False,
    loss_fn = "cross entropy",
    optimizer = "SGD",
    hp_optim = dict(momentum=0.001),
    subset=False
):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)
    if subset:
        train_idx = list(range(0, subset))
        test_idx = list(range(0, subset//4))
        ds_train = torch.utils.data.Subset(ds_train, train_idx)
        ds_test = torch.utils.data.Subset(ds_test, test_idx)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: 
    L = layers_per_block
    K = filters_per_layer
    
    dl_train = DataLoader(ds_train, batch_size=bs_train)
    dl_test = DataLoader(ds_test, batch_size=bs_test)
    sample_shape = next(iter(dl_train))[0][0].shape
    print(f'sample shape-{sample_shape}')
    conv_channels = [elem for elem, count in zip(K, [L]*len(K)) for i in range(count)]
    net_params = dict(
        in_size=sample_shape, out_classes=10, channels=conv_channels,
        pool_every=pool_every, hidden_dims=hidden_dims,
        activation_type=activation_type, activation_params=activation_params,
        pooling_type=pooling_type, pooling_params=pooling_params,
        batchnorm=batchnorm, dropout=dropout,
        bottleneck=bottleneck
    )
    # print(net_params)
    model = ArgMaxClassifier(
    model=model_cls(**net_params)
    )
    
    # x = torch.randn(1, *sample_shape)
    # display(make_dot(model(x), params=dict(model.named_parameters())))

    
    print(model)
    optimizer = OPTIMIZERS[optimizer](params=model.parameters(),lr=lr, weight_decay=reg, **hp_optim)
    print('cfg', cfg)
    trainer = ClassifierTrainer(model, LOSSES[loss_fn](), optimizer, device)
    fit_res = trainer.fit(dl_train, dl_test, num_epochs=epochs, print_every=1, verbose=True, early_stopping=3)
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS236781 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
