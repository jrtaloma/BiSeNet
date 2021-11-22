#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.get_dataloader import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg



## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')




def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument('--state-checkpoint-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)


def set_model():
    net = model_factory[cfg.model_type](2)
    if not args.finetune_from is None:
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if cfg.use_sync_bn: net = set_syncbn(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_syncbn(net):
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train(state_ckpt, n_epochs=100):
    logger = logging.getLogger()
    is_dist = dist.is_initialized()

    ## dataset
    dl = get_data_loader(cfg, mode='train', distributed=is_dist)

    ## model
    net, criteria_pre, criteria_aux = set_model()

    ## state checkpoint
    if not state_ckpt is None:
        checkpoint = torch.load(state_ckpt)

    ## optimizer
    optim = set_optimizer(net)
    if not state_ckpt is None:
        optim.load_state_dict(checkpoint['optim'])

    ## meters
    if not state_ckpt is None:
        time_meter = checkpoint['time_meter']
        loss_meter = checkpoint['loss_meter']
        loss_pre_meter = checkpoint['loss_pre_meter']
        loss_aux_meters = checkpoint['loss_aux_meters']
    else:
        time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    if not state_ckpt is None:
        lr_schdr = checkpoint['lr_schdr']
    else:
        lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
            max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
            warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    if not state_ckpt is None:
        iteration = checkpoint['iteration']
    else:
        iteration = -1

    for it, (im, lb) in enumerate(dl,iteration+1):
        if it == cfg.max_iter:
            break

        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        logits, *logits_aux = net(im)
        loss_pre = criteria_pre(logits, lb)
        loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        loss.backward()
        optim.step()
        lr_schdr.step()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        ## print training log message
        if (it + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(
                it, cfg.max_iter, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)

        ## ending one epoch
        if (it + 1) % (cfg.max_iter//n_epochs) == 0:
            ## dump the model and the state
            epoch = int((it+1)/cfg.max_iter*n_epochs)
            model_pth = osp.join(cfg.respth, 'model_{}.pt'.format(epoch))
            state_pth = osp.join(cfg.respth, 'state_{}.pt'.format(epoch))
            model = net.state_dict()
            torch.save(model, model_pth)
            torch.save({
                'iteration': it,
                'optim': optim.state_dict(),
                'lr_schdr': lr_schdr,
                'time_meter': time_meter,
                'loss_meter': loss_meter,
                'loss_pre_meter': loss_pre_meter,
                'loss_aux_meters': loss_aux_meters
            }, state_pth)
            logger.info('\nsaved the model to {}'.format(model_pth))
            logger.info('\nsaved the state to {}'.format(state_pth))

    ## dump the final model
    model_pth = osp.join(cfg.respth, 'model_final.pt')
    model = net.state_dict()
    torch.save(model, model_pth)
    logger.info('\nsaved the model to {}'.format(model_pth))

    #logger.info('\nevaluating the final model')
    #torch.cuda.empty_cache()
    #heads, mious = eval_model(cfg, net.module)
    #logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

    return


def main():
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-train'.format(cfg.model_type), cfg.respth)
    state_ckpt = args.state_checkpoint_from
    train(state_ckpt)


if __name__ == "__main__":
    main()
