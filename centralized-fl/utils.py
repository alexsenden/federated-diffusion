import torch

from flwr.common import logger


def log_mem_info(device):
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024**2
    logger.log(20, mem_used_MB)
