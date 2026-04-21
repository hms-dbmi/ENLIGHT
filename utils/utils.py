import math
import random

import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.warmup_epoch:
        lr = cfg.lr * epoch / cfg.warmup_epoch
    else:
        lr = cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg.warmup_epoch) / (cfg.train_epoch - cfg.warmup_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def set_seed(seed):
    # Set random seed for PyTorch
    torch.manual_seed(seed)

    # Set random seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for random module
    random.seed(seed)

    # Set random seed for CuDNN if available
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False