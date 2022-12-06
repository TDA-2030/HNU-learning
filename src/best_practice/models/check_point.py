
import shutil
import torch
import torch.nn as nn


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

