import os
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def save_checkpoint(model, optimizer, scheduler, epoch, outdir):
    """Saves checkpoint to disk"""
    filename = "model_{:04d}.pth".format(epoch)
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict([
        ('state_dict', weights),
        ('optimizer', optimizer.state_dict()),
        ('scheduler', scheduler.state_dict()),
        ('epoch', epoch),
    ])

    torch.save(state, filename)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_optimizer(config, net):
    lr = config.train.learning_rate

    print("Opt: ", config.train.optimizer)

    if config.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=config.train.momentum,
                                    weight_decay=config.train.weight_decay)
    else:
        raise Exception("Unknown type of optimizer: {}".format(config.train.optimizer))
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_schedule.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.n_epoch)
    elif config.train.lr_schedule.name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.train.lr_schedule.step_size,
                                                    gamma=config.train.lr_schedule.gamma)
    else:
        raise Exception("Unknown type of lr schedule: {}".format(config.train.lr_schedule))
    return scheduler


def get_training_parameters(config, net):
    criterion = torch.nn.CrossEntropyLoss().to('cuda')
    optimizer = get_optimizer(config, net)
    scheduler = get_scheduler(config, optimizer)
    return criterion, optimizer, scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self):
        return self.val, self.avg


def get_max_bbox(bboxes):
    bbox_sizes = [x[2] * x[3] for x in bboxes]
    max_bbox_index = np.argmax(bbox_sizes)
    return bboxes[max_bbox_index]
