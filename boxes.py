from dataclasses import dataclass
import torch


@dataclass
class Boxes:
    centers: torch.Tensor
    offsets: torch.Tensor

    def intersect(self, others):
        lower = torch.maximum(self.centers - self.offsets, others.centers - others.offsets)
        upper = torch.minimum(self.centers + self.offsets, others.centers + others.offsets)
        centers = (lower + upper) / 2
        offsets = torch.abs(upper - lower) / 2
        return Boxes(centers, offsets), lower, upper

    def translate(self, directions):
        return Boxes(self.centers + directions, self.offsets)

    def __getitem__(self, item):
        return Boxes(self.centers[item], self.offsets[item])
