import json
import os

import numpy as np
import torch

from utils.utils import get_device

np.random.seed(100)

device = get_device()


def get_file_dir(dataset):
    return f'data/{dataset}/prediction'


def load_arrays(path):
    d = {}
    for file in os.listdir(path):
        arr = np.load(f'{path}/{file}')
        file = file.replace('.npy', '')
        d[file] = torch.from_numpy(arr).long()
    return d


def load_valid_data(dataset, classes):
    return load_valid_or_test_data(dataset, 'val')

def load_test_data(dataset, classes):
    return load_valid_or_test_data(dataset, 'test')


def load_valid_or_test_data(dataset, folder):
    path = f'{get_file_dir(dataset)}/{folder}'
    return load_arrays(path)


def load_data(dataset):
    folder = get_file_dir(dataset)
    data = load_arrays(f'{folder}/train')
    with open(f'{folder}/classes.json', 'r') as f:
        classes = json.load(f)
    with open(f'{folder}/relations.json', 'r') as f:
        relations = json.load(f)
    return data, classes, relations
