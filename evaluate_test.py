from collections import Counter

import pandas as pd
import logging
import numpy as np
import torch
import math
from tqdm import trange

from utils.utils import get_device
from utils.emelpp_data_loader import load_data, load_valid_data

logging.basicConfig(level=logging.INFO)


def main():
    embeds = np.array([
        [3, 2.5, 2, 1.5],
        [2, 2, .5, .5],
        [5.5, 0, 1.5, 2],
        [4.5, 1, .5, .5],
        [5.5, -1, .5, 1],
    ])
    embedding_size = 2

    eval_data = np.array([
        [1, -1, 0],  # B < A t
        [2, -1, 0],  # C < A f
        [1, -1, 3],  # B < D f
        [3, -1, 0],  # D < A f
        [3, -1, 2],  # D < C t
        [1, -1, 2],  # B < C f
        [4, -1, 2],  # F < C t
    ])

    offsets = np.abs(embeds[:, embedding_size:])
    embeds = embeds[:, :embedding_size]

    c_embeds = embeds[eval_data[:, 0]]
    c_offsets = offsets[eval_data[:, 0]]
    d_embeds = embeds[eval_data[:, 2]]
    d_offsets = offsets[eval_data[:, 2]]

    euc = np.abs(c_embeds - d_embeds)
    results = euc + c_offsets - d_offsets
    results = np.clip(results, a_min=0, a_max=None)
    results = results.sum(axis=1)
    acc = (results == 0).sum().item() / eval_data.shape[0]
    print(acc)


if __name__ == '__main__':
    main()
