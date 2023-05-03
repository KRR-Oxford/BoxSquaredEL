import json
import torch

import numpy as np

from utils.data_loader import DataLoader

dataset = 'ANATOMY'
data_loader = DataLoader.from_task('prediction')
data, classes, relations = data_loader.load_data(dataset)

data['role_chain'] = []
data['role_inclusion'] = []

owl_path = f'data/{dataset}/inferences/{dataset}_norm_full.owl'
with open(owl_path) as f:
    for line in f:
        if line.startswith('SubObjectPropertyOf'):
            line = line.strip()[20:-1]
            if line.startswith('ObjectPropertyChain'):
                line_chain = line.strip()[20:]
                line1 = line_chain.split(")")
                line10 = line1[0].split()
                if len(line10) == 0:
                    continue
                r1 = line10[0].strip()
                r2 = line10[1].strip()
                r3 = line1[1].strip()
                if r1 not in relations:
                    relations[r1] = len(relations)
                if r2 not in relations:
                    relations[r2] = len(relations)
                if r3 not in relations:
                    relations[r3] = len(relations)
                data['role_chain'].append((relations[r1], relations[r2], relations[r3]))
            else:
                it = line.split(' ')
                r1 = it[0]
                r2 = it[1]
                if r1 not in relations:
                    relations[r1] = len(relations)
                if r2 not in relations:
                    relations[r2] = len(relations)
                data['role_inclusion'].append((relations[r1], relations[r2]))

data['role_inclusion'] = torch.tensor(data['role_inclusion'], dtype=torch.long)
data['role_chain'] = torch.tensor(data['role_chain'], dtype=torch.long)

folder = f'data/{dataset}/prediction'
for key in ['role_inclusion', 'role_chain']:
    np.save(f'{folder}/train/{key}.npy', data[key])
with open(f'{folder}/relations.json', 'w+') as f:
    json.dump(relations, f, indent=2)

