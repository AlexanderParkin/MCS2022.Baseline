import torch
import numpy as np
from tqdm import tqdm

from utils import AverageMeter


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          config, epoch) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch (int): epoch number
    :return: None
    """
    model.train()

    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')

    train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

    for step, (x, y) in enumerate(train_iter):
        out = model(x.cuda().to(memory_format=torch.contiguous_format))
        loss = criterion(out, y.cuda())
        num_of_samples = x.shape[0]

        loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores = torch.softmax(out, dim=1).detach().cpu().numpy()
        predict = np.argmax(scores, axis=1)
        gt = y.detach().cpu().numpy()

        acc = np.mean(gt == predict)
        acc_stat.update(acc, num_of_samples)

        if step % config.train.freq_vis == 0 and not step == 0:
            acc_val, acc_avg = acc_stat()
            loss_val, loss_avg = loss_stat()
            print('Epoch: {}; step: {}; loss: {:.4f}; acc: {:.2f}'.format(epoch, step, loss_avg, acc_avg))

    acc_val, acc_avg = acc_stat()
    loss_val, loss_avg = loss_stat()
    print('Train process of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}'.format(epoch, loss_avg, acc_avg))


def validation(model: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               epoch) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: float: avg acc
     """
    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')

    with torch.no_grad():
        model.eval()
        val_iter = tqdm(val_loader, desc='Val', dynamic_ncols=True, position=2)

        for step, (x, y) in enumerate(val_iter):
            out = model(x.cuda().to(memory_format=torch.contiguous_format))
            loss = criterion(out, y.cuda())
            num_of_samples = x.shape[0]

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)

            scores = torch.softmax(out, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            acc = np.mean(gt == predict)
            acc_stat.update(acc, num_of_samples)

        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        print('Validation of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}'.format(epoch, loss_avg, acc_avg))
        return acc_avg
