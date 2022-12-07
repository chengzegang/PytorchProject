from __future__ import annotations
from typing import Tuple
from torch.utils.data import Dataset
from typing import Callable
from pathlib import Path
from PIL import Image
import os
class ImageFolder(Dataset):
    
    
    def __init__(self, folder: str, transform: Callable | None = None, target_transform: Callable = None, exts: Tuple[str] = ('jpeg', 'png', 'jpg')) -> None:
        super().__init__()
        self.folder = folder
        self.transform = transform
        self.target_transform = target_transform
        self._path = None
        self.exts = exts
    
    @property
    def path(self):
        if self._path is None:
            self._path = []
            for root, _, files in os.walk(self.folder):
                self._path.extend([Path(os.path.join(root, file)).absolute().resolve().as_posix() for file in files if file.lower().endswith(self.exts)])
        return self._path
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx: int):
        image = Image.open(self.path[idx]).convert('RGB')
        target = image
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target