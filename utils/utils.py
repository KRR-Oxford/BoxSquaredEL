import torch
from joblib import Memory

cachedir = 'cache'
memory = Memory(cachedir, verbose=0)
printed_device = False


def get_device():
    global printed_device

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'
    if not printed_device:
        print(f'Using device: {device_name}')
        printed_device = True
    return device
