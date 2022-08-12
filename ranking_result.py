from dataclasses import dataclass
import numpy as np

from typing import List


@dataclass
class RankingResult:
    top1: float
    top10: float
    top100: float
    ranks: List[int]
    auc: float

    def combine(self, other):
        return RankingResult(self.top1 + other.top1,
                             self.top10 + other.top10,
                             self.top100 + other.top100,
                             self.ranks + other.ranks,
                             0)  # The AUC needs to be recalculated after combining results

    def __str__(self):
        return f'top1: {self.top1 / len(self):.2f}, top10: {self.top10 / len(self):.2f}, ' \
               f'top100: {self.top100 / len(self):.2f}, mean: {np.mean(self.ranks):.2f}, median: {np.median(self.ranks):.2f}, ' \
               f'auc: {self.auc:.2f}'

    def __len__(self):
        return len(self.ranks)
