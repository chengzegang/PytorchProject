import math
import numpy as np



def cosine_lr_lambda(shift: int, total_steps: int, steps: int) -> float:
    steps = steps + shift
    period = math.pi / total_steps
    lambda_ = (math.cos(steps * period + 1e-6) + 1) / 2
    return lambda_


def identity_lr_lambda(*args, **kwargs) -> float:
    return 1

def cosine_warmup_lr_lambda(shift: int, warmup_steps: int, total_steps: int, steps: int) -> float:
    steps = steps + shift
    if steps < warmup_steps:
        return steps / warmup_steps
    period = math.pi / (total_steps - warmup_steps)
    lambda_ = (math.cos((steps - warmup_steps) * period + 1e-6) + 1) / 2
    return lambda_


def cosine_anneal_lr_lambda(shift: int, batch_steps: int, steps: int) -> float:
    steps = steps + shift
    period = math.pi / batch_steps
    steps = steps % batch_steps
    lambda_ = (np.cos(steps * period + 1e-6) + 1) / 2
    return lambda_


def linear_cosine_anneal_lr_lambda(shift: int, total_steps: int, batch_steps: int, steps: int) -> float:
    steps = steps + shift
    period = math.pi / batch_steps
    in_batch_steps = steps % batch_steps
    lambda_ = ((total_steps - steps) / total_steps) * (np.cos(in_batch_steps * period) + 1) / 2
    return lambda_