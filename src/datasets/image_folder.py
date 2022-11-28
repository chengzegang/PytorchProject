from __future__ import annotations
from typing import Tuple
from torch.utils.data import Dataset
from typing import Callable
from pathlib import Path
from PIL import Image
import os
class ImageFolder(Dataset):
    
    
    def __init__(self, folder: str, transforms: Callable | None = None, exts: Tuple[str] = ('jpeg', 'png', 'jpg')) -> None:
        super().__init__()
        self.folder = folder
        self.transforms = transforms
        self._path = None
        self.exts = exts
    
    @property
    def path(self):
        if self._path is None:
            for root, _, files in os.walk(self.folder):
                self._path = [Path(os.path.join(root, file)).absolute().resolve().as_posix() for file in files if file.lower().endswith(self.exts)]
        return self._path
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx: int):
        image = Image.open(self.path[idx]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image