import os
import sys
import yaml
import random
import argparse
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm

import utils
from models import models
from data import get_dataloader
from train import train, validation
from utils import convert_dict_to_tuple


def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    outdir = osp.join(config.outdir, config.exp_name)
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_loader, val_loader = get_dataloader.get_dataloaders(config)

    print("Loading model...")
    net = models.load_model(config)
    if config.num_gpu > 1:
        net = torch.nn.DataParallel(net)
    print("Done.")

    criterion, optimizer, scheduler = utils.get_training_parameters(config, net)
    train_epoch = tqdm(range(config.train.n_epoch), dynamic_ncols=True, desc='Epochs', position=0)

    # main process
    best_acc = 0.0
    for epoch in train_epoch:
        train(net, train_loader, criterion, optimizer, config, epoch)
        epoch_avg_acc = validation(net, val_loader, criterion, epoch)
        if epoch_avg_acc >= best_acc:
            utils.save_checkpoint(net, optimizer, scheduler, epoch, outdir)
            best_acc = epoch_avg_acc
        scheduler.step()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='Path to config file.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
