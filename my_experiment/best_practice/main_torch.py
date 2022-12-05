#!/usr/bin/env python

import argparse
from copy import deepcopy
from tqdm import tqdm
import os
import random
import shutil
import sys
import time
from typing import Callable, Tuple
import warnings
from enum import Enum
from pathlib import Path
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import torchvision.models as t_models

from utils import PlotMonitor, increment_path, logger, TqdmToLogger, ConfusionMatrix

model_names: list = sorted(name for name in models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(models.__dict__[name]))
model_names += sorted(name for name in t_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(t_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data',
                    metavar='DIR',
                    nargs='?',
                    default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=90,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size',
                    default=-1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')
parser.add_argument('--dummy',
                    action='store_true',
                    help="use fake data to benchmark")
parser.add_argument('--artifact-path',
                    default='run.log/artifacts',
                    type=str,
                    help='artifacts save to here')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    artifact_path = Path(args.artifact_path).resolve()
    artifact_path = increment_path(artifact_path, exist_ok=False)
    artifact_path.mkdir(parents=True, exist_ok=True)  # make dir
    args.artifact_path = artifact_path

    logger.set_file_dir(str(artifact_path))

    logger('--------user config:')
    for k, v in args.__dict__.items():
        if not k.startswith('_'):
            logger("%-30s: %-20s" % (k, getattr(args, k)))
    logger('--------------------')

    import platform
    s = f'🚀 Python-{platform.python_version()} torch-{torch.__version__} '
    space = ' ' * (len(s) + 1)
    for i, d in enumerate("0"):
        p = torch.cuda.get_device_properties(i)
        s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
    logger(s)

    if args.gpu is not None:
        logger("Use GPU: {} for training".format(args.gpu))

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    # load dataset
    train_dataset, val_dataset, class_num, class_list = get_dataset(args)
    args.class_num = class_num
    args.class_list = class_list

    # create model
    model: nn.Module
    if args.pretrained:
        logger("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logger("=> creating model '{}'".format(args.arch))
        if args.arch.startswith("my_"):
            model = models.__dict__[args.arch](num_classes=args.class_num)
        else:
            model = t_models.__dict__[args.arch](num_classes=args.class_num)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            # load model, checkpoint state_dict as FP32
            model.load_state_dict(checkpoint['model'].float().state_dict())
            logger("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger("=> no checkpoint found at '{}'".format(args.resume))

    # set model to appropriate device
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        logger('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int(
                    (args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Data loading code
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)

    monitors: dict[str, PlotMonitor] = {}

    def plot_hook(f, p):
        if f not in monitors.keys():
            monitors[f] = PlotMonitor(str(artifact_path))
        mon = monitors[f]
        for m in p.meters:
            if False == mon.check_line_existence(m.name):
                mon.add_line(m.name)
            v = m.val
            try:
                v = v.to("cpu")
                v = v.numpy()
            except:
                pass
            mon.add_data(m.name, v)
        mon.generate_graph(f'{args.arch}-{f}_result.jpg')

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args,
              plot_hook)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, plot_hook)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            # Save model
            ckpt = {
                'epoch': epoch + 1,
                'best_acc1': best_acc1,
                'args' : args,
                'model': deepcopy(de_parallel(model)).half(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'opt': vars(args),
                'date': time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
            }

            save_checkpoint(
                ckpt, is_best,
                str(args.artifact_path / f"{args.arch}_checkpoint.pth"))
            del ckpt


def train(train_loader, model, criterion, optimizer, epoch, device, args,
          display_hook: Callable):
    batch_time = AverageMeter('Time', ':.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, losses, top1, top5],
                             prefix='')

    # switch to train mode
    model.train()

    end = time.time()
    pbar = tqdm(enumerate(train_loader),
                total=len(train_loader),
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                dynamic_ncols=True)
    for i, (images, target) in pbar:
        # measure data loading time

        # move data to the same device as model
        images: torch.Tensor = images.to(device, non_blocking=True)
        target: torch.Tensor = target.to(device, non_blocking=True)

        # compute output
        output: torch.Tensor = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(
                ('%11s' * 2 + '%11s') %
                (f'{epoch}/{args.epochs - args.start_epoch - 1}', mem,
                 progress))
            # print(progress)
            display_hook(sys._getframe().f_code.co_name, progress)


def validate(val_loader, model, criterion, args, display_hook: Callable):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            pbar = tqdm(enumerate(loader),
                        total=len(loader),
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                        dynamic_ncols=True)
            for i, (images, target) in pbar:
                pbar.disable
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                pred, y = cal_prediction(output, target)
                confusion_matrix.process_batch(pred, y)

                if i % args.print_freq == 0:
                    pbar.set_description(('%11s') % progress)
                    display_hook(sys._getframe().f_code.co_name, progress)
            confusion_matrix.print()
            confusion_matrix.plot(save_dir=args.artifact_path, names=args.class_list)

    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader) + (
        args.distributed and
        (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
                             [losses, top1, top5],
                             prefix='Test: ')
    confusion_matrix = ConfusionMatrix(nc=args.class_num)
    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(
            val_loader.dataset)):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(
                len(val_loader.sampler) * args.world_size,
                len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def get_dataset(
        args
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, dict]:
    if args.dummy:
        logger("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000,
                                          transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000,
                                        transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'valid')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        # import torchvision
        # train_dataset = torchvision.datasets.CIFAR10(root="../../data",
        #                                        train=True,
        #                                        download=True,
        #                                        transform=transforms.Compose([
        #                                         transforms.RandomResizedCrop(222),
        #                                         transforms.RandomHorizontalFlip(),
        #                                         transforms.ToTensor(),
        #                                         normalize,
        #                                     ]))
        # val_dataset = torchvision.datasets.CIFAR10(root="../../data",
        #                                        train=False,
        #                                        download=True,
        #                                        transform=transforms.Compose([
        #                                         transforms.Resize(256),
        #                                         transforms.CenterCrop(224),
        #                                         transforms.ToTensor(),
        #                                         normalize,
        #                                     ]))
    class_num: int = 0
    class_list:list = []
    try:
        for c in train_dataset.classes:
            class_num += 1
            class_list.append(str(c))
            logger(f"class:\"{c}\" - id:{train_dataset.class_to_idx[c]}")
        logger(f"total class number: {class_num}")
    except AttributeError:  #'FakeData' object has no attribute 'classes'
        pass
    return (train_dataset, val_dataset, class_num, class_list)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def save_checkpoint(state, is_best: bool, filename: str = 'checkpoint.pth'):
    assert filename.endswith('.pth')
    torch.save(state, filename)
    if is_best:
        new_name = filename[:-4] + "_best" + filename[-4:]
        shutil.copyfile(filename, new_name)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count],
                             dtype=torch.float32,
                             device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters: list[AverageMeter] = meters
        self.prefix: str = prefix

    def __str__(self) -> str:
        entries = [self.prefix]
        entries += [str(meter) for meter in self.meters]
        return '  '.join(entries)

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cal_prediction(output: torch.Tensor,
                   target: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    y_label = target.cpu().view(-1).numpy()
    with torch.no_grad():
        _, pred = output.cpu().topk(1, 1, True, True)
        pred = pred.t().view(-1).numpy()
        return pred, y_label


if __name__ == '__main__':
    main()
