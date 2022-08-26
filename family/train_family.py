import torch
import torch.optim as optim

from model.BoxSquaredEL import BoxSquaredEL
from model.Elbe import Elbe
from utils.utils import get_device
from family_data import load_data
from tqdm import trange

torch.random.manual_seed(123)

data, classes, relations = load_data()
device = get_device()
# model = BoxSquaredEL(device, classes, len(relations), embedding_dim=2, batch=5, margin=0, neg_dist=.5,
#                      ranking_fn='l2', reg_factor=1, num_neg=0)
model = Elbe(device, classes, len(relations), embedding_dim=2, batch=5, margin1=0)
optimizer = optim.Adam(model.parameters(), lr=5e-2)
model = model.to(device)

model.train()
num_epochs = 300
pbar = trange(num_epochs)
for epoch in pbar:
    re = model(data)
    loss = sum(re)
    pbar.set_postfix({'loss': f'{loss.item():.2f}'})
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.save(f'family/out/{model.name}')
