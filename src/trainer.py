
import torch
from .utils import lr_lambda
from .nn import RectVit
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from typing import Tuple
import os
from . import datasets
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from functools import partial
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    
    
    def __init__(self, config: dict, **kwargs) -> None:
        self.config = config
        
        self._dataset = None
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._logdir = None
        self._dataset = None
        self._logger = None
        self.curr_epoch = None
        self._sampler = None
        self._steps_per_batch = None
        self.local_rank = 0
        self.world_size = 1
        self.global_steps = 0
        self.batch_steps = 0
        self.start_time = datetime.now()
        if dist.is_available() and dist.is_initialized():
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
    
    @property
    def device(self):
        return torch.device('cuda', self.local_rank)
    
    @property
    def model(self):
        if self._model is None:
            self._model = RectVit(**self.config)
            self._model.to(self.device)
            if dist.is_available() and dist.is_initialized():
                self._model = DistributedDataParallel(self._model, device_ids=[self.local_rank], output_device=self.local_rank)
        return self._model
    
    @property
    def logdir(self):
        if self._logdir is None and self.local_rank == 0:
            logdir = self.config["logdir"]
            Path(logdir).mkdir(parents=True, exist_ok=True)
            existing_versions = [
                version.split("_")[1]
                for version in os.listdir(logdir)
                if version.startswith("version_")
            ]
            my_version = (
                0
                if len(existing_versions) == 0
                else max([int(version) for version in existing_versions]) + 1
            )
            self._logdir = os.path.join(logdir, "version_" + str(my_version))
            Path(self._logdir).mkdir(parents=True, exist_ok=True)
        return self._logdir
    
    @property
    def steps_per_batch(self):
        if self._steps_per_batch is None:
            self._steps_per_batch = len(self.dataloader)
        return self._steps_per_batch

    @property
    def total_steps(self):
        return self.steps_per_batch * self.config['max_epochs']
    
    @property
    def logger(self):
        if self._logger is None and self.local_rank == 0:
            self._logger = SummaryWriter(self.logdir)
        return self._logger
    
    
    def init_optims(self) -> Tuple[Optimizer, _LRScheduler]:
        
        optimizer = AdamW(self.model.parameters(), lr=self.config['lr'], betas=self.config['betas'], weight_decay=self.config['weight_decay'])
        scheduler = LambdaLR(optimizer, partial(lr_lambda.cosine_warmup_lr_lambda, 0, self.config['warmup_epochs'] * self.steps_per_batch, self.total_steps))
        return optimizer, scheduler
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
    
    @property
    def sampler(self):
        if self._sampler is None and dist.is_available() and dist.is_initialized():
            self._sampler = DistributedSampler(self.dataset, shuffle=True)
        return self._sampler
    
    @property
    def dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=self.config['num_workers'],
        )
        
    @property
    def transform(self):
        transform_ = T.Compose([
            T.Resize(self.config['image_size']),
            T.CenterCrop(self.config['image_size']),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform_

    def inverse_normalize(self, x: torch.Tensor):
        x = TF.normalize(x,  [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
        x = x.clamp(0, 1)
        return x
        
        
    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = datasets.ImageFolder(self.config['dataset_dir'], transform=self.transform)
        return self._dataset
    
    
    def checkpoint(self):
        ckpt_dir = Path(self.logdir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch_{self.curr_epoch}.pth"
        torch.save(self.model.state_dict(), ckpt_path)
    
    
    def log(self, loss, X, Xh, curr_lr, **kwargs):
        if self.local_rank != 0:
            return
        self.logger.add_scalar('Train/Loss', loss.item(), self.global_steps)
        if self.batch_steps % self.config['refresh_rate'] == 0:
            x0 = X[0].detach().cpu()
            xh0 = Xh[0].detach().cpu()
            image = torch.cat([x0, xh0], dim=-1)
            image = self.inverse_normalize(image)
            self.logger.add_image('Train/Image', image, self.global_steps)

            elapsed_time = 'N/A'
            finish_time = 'N/A'
            if self.global_steps > 0:
                curr_time = datetime.now()
                elapsed_time = curr_time - self.start_time
                time_per_step = elapsed_time / self.global_steps
                total_time = time_per_step * self.total_steps
                finish_time = curr_time + total_time

                elapsed_time = str(elapsed_time).split('.')[0]
                finish_time = str(finish_time).split('.')[0]


            print(f'Epoch: {self.curr_epoch}, Batch: {self.batch_steps}, Loss: {loss.item():.4f}, LR: {curr_lr:.2e}, Time: {elapsed_time}, ETA: {finish_time}', flush=True)
    
    def train(self):
        self.model.train().to(self.device)

        optimizer, scheduler = self.init_optims()
        
        for ep in range(self.config['max_epochs']):
            self.curr_epoch = ep
            if dist.is_available() and dist.is_initialized():
                self.sampler.set_epoch(self.curr_epoch)
            
            for batch_idx, X in enumerate(self.dataloader):
                self.batch_steps = batch_idx
                X = X.to(self.device)
                Xh = self.model(X)
                optimizer.zero_grad()
                loss = F.mse_loss(X, Xh)
                loss.backward()
                optimizer.step()
                scheduler.step()
                curr_lr = scheduler.get_last_lr()[0]
                self.log(loss, X, Xh, curr_lr)
                self.global_steps += 1
            
            self.checkpoint()