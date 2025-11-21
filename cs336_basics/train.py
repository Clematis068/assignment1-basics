import torch
import os
import numpy as np
from numpy.typing import NDArray
import typing
class Dataloader:
    def __init__(self, 
                 dataset: NDArray,
                 batch_size: int,
                 context_length: int,
                 device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        self.max_start_idx = len(dataset) - context_length # 看的是x，y还要顺延一位
        if self.max_start_idx < 0:
            raise ValueError("error idx < 0")
        
    def to_device_(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.device == 'cuda:0':
            return (
                x.pin_memory().to(device = self.device, non_blocking = False),
                y.pin_memory().to(device = self.device, non_blocking = False)
            )
        
        return x.to(device = self.device), y.to(device = self.device)
    
    def get_batch(self, ) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = torch.randint(0, self.max_start_idx, size = (self.batch_size,))
        start_idx = start_idx.tolist()
        x_list = [torch.from_numpy((self.dataset[i : i + self.context_length]).astype(np.int64)) for i in start_idx]
        y_list = [torch.from_numpy((self.dataset[i + 1 : i + self.context_length + 1]).astype(np.int64)) for i in start_idx]
        
        x_batch = torch.stack(x_list)
        y_batch = torch.stack(y_list)   

        return self.to_device_(x_batch, y_batch)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.get_batch()
            
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        'model_state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'iteration' : iteration
    }

    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']
    return iteration