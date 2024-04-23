import logging
from utils.util import setup_logger
from config.config_args import *
from processor.trainer import Trainer
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import numpy as np
import random
from torch.backends import cudnn


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        args.nodes = 1
        args.ngpus_per_node = len(args.gpu_ids)
        args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        #init_method=f'tcp://127.0.0.1:{args.port}',
        init_method=f'tcp://127.0.0.1:12361',
        world_size=world_size,
        rank=rank
    )


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(1 + rank)

    log_name = 'train_' + args.save_name
    setup_logger(logger_name=log_name, root=args.save_dir,
                 level=logging.INFO if rank in [-1, 0] else logging.WARN, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    Trainer(args, logger).run()
    cleanup()


def main():
    args = parser.parse_args()
    check_and_setup_parser(args)

    if args.ddp:
        mp.set_sharing_strategy('file_system')
        device_config(args)
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )

    else:
        log_name = 'train_' + args.save_name
        setup_logger(logger_name=log_name, root=args.save_dir, screen=True, tofile=True)
        logger = logging.getLogger(log_name)
        logger.info(str(args))

        args.rank = -1
        Trainer(args, logger).run(),


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

