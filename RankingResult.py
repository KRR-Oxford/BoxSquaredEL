from dataclasses import dataclass

from typing import List


@dataclass
class RankingResult:
    top1: float
    top10: float
    top100: float
    ranks: List[int]
