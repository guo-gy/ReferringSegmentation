import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
import wandb
from utils.dataset import RefDataset
from engine.engine import train, validate
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.gpu = 0
    args.rank = 0
    args.world_size = 1
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # wandb (Disabled locally to avoid requiring login during debugging)
    # wandb.init(job_type="training", mode="disabled")

    # build model
    model, param_list = build_segmenter(args)
    logger.info(model)
    model = model.cuda()
    # model = nn.parallel.DistributedDataParallel(model.cuda(), ...) # Removed DDP

    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()

    # build dataset
    train_data = RefDataset(lmdb_dir=args.train_lmdb,
                            mask_dir=args.mask_root,
                            dataset=args.dataset,
                            split=args.train_split,
                            mode='train',
                            input_size=args.input_size,
                            word_length=args.word_len)
    val_data = RefDataset(lmdb_dir=args.val_lmdb,
                          mask_dir=args.mask_root,
                          dataset=args.dataset,
                          split=args.val_split,
                          mode='val',
                          input_size=args.input_size,
                          word_length=args.word_len)

    # build dataloader (No DistributedSampler)
    init_fn = partial(worker_init_fn,
                      num_workers=0,  # Avoid Windows multiprocessing issues
                      rank=0,
                      seed=args.manual_seed)
    
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True, # DDP used sampler for shuffle
                                   num_workers=0,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   drop_last=True)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=False)

    best_IoU = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader (removed for single GPU)
        # train
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log,
              args)

        # evaluation
        iou, prec_dict = validate(val_loader, model, epoch_log, args)

        # save model
        lastname = os.path.join(args.output_dir, "last_model.pth")
        torch.save(
            {
                'epoch': epoch_log,
                'cur_iou': iou,
                'best_iou': best_IoU,
                'prec': prec_dict,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, lastname)
        if iou >= best_IoU:
            best_IoU = iou
            bestname = os.path.join(args.output_dir, "best_model.pth")
            shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)
