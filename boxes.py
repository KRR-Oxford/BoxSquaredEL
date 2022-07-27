from dataclasses import dataclass
import torch

@dataclass
class Boxes:
    centers: torch.Tensor
    offsets: torch.Tensor
