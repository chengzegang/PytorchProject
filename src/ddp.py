import torch.distributed as dist
import os
from .trainer import Trainer

def init_process(rank, size, port, args, fn):
    backend = "nccl"
    if os.name == "nt":
        backend = "gloo"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args, rank, size)


def run(trainer_cfg, rank=0, size=1):
    trainer = Trainer(trainer_cfg)
    trainer.train()
        