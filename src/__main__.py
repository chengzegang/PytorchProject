import argparse
import torch
from . import ddp
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--dataset-dir', type=str, default='data', help='Path to the dataset directory')
    parser.add_argument('--batch-size', type=int, default=16, dest='batch_size')
    parser.add_argument('--num-workers', type=int, default=8, dest='num_workers')
    parser.add_argument('--max-epochs', type=int, default=10, dest='max_epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, dest='warmup_epochs')
    parser.add_argument('--lr', type=float, default=1e-4, dest='lr')
    parser.add_argument('--weight-decay', type=float, default=1e-5, dest='weight_decay')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95), dest='betas')
    parser.add_argument('--patch-size', type=int, default=32, dest='patch_size')
    parser.add_argument('--logdir', type=str, default='logs', dest='logdir')
    parser.add_argument('--image-size', type=int, default=224, dest='image_size')
    parser.add_argument('--hidden-size', type=int, default=768, dest='hidden_size')
    parser.add_argument('--num-encoder-layers', type=int, default=12, dest='num_encoder_layers')
    parser.add_argument('--num-decoder-layers', type=int, default=6, dest='num_decoder_layers')
    parser.add_argument('--num-heads', type=int, default=12, dest='num_heads')
    parser.add_argument('--intermediate-size', type=int, default=3072, dest='intermediate_size')
    parser.add_argument('--refresh-rate', type=int, default=100, dest='refresh_rate')
    parser.add_argument('--ddp', action='store_true', dest='ddp')
    parser.add_argument('--port', type=int, default=35532, dest='port')

    sys_argv = parser.parse_args()
    sys_argv = sys_argv.__dict__
    if sys_argv['ddp']:
        size = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            ddp.init_process, args=(size, sys_argv['port'], sys_argv, ddp.run), nprocs=size, join=True
        )
    else:
        ddp.run(sys_argv, rank=0, size=1)
        
    