import numpy as np
import torch
import random
import json

from utils.data_loader import DataLoader
from utils.utils import get_device, memory

device = get_device()


class InferencesDataLoader(DataLoader):

    def __init__(self):
        self.load_data = memory.cache(self.load_data)

    @staticmethod
    def get_file_dir(dataset):
        return f'data/{dataset}/inferences/'

    def load_val_data(self, dataset, classes):
        with open(self.get_file_dir(dataset) + 'val.json', 'r') as f:
            data = json.load(f)
        return {'nf1': torch.tensor([(classes[tup[0]], classes[tup[1]]) for tup in data])}

    def load_test_data(self, dataset, classes):
        with open(self.get_file_dir(dataset) + 'inferences.json', 'r') as f:
            data = json.load(f)
        return {'nf1': torch.tensor([(classes[tup[0]], classes[tup[1]]) for tup in data])}

    def load_data(self, dataset):
        filename = self.get_file_dir(dataset) + f'{dataset}_norm_full.owl'
        classes = {}
        relations = {}
        data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': [], 'role_inclusion': [], 'role_chain': []}
        with open(filename) as f:
            for line in f:
                if line.startswith('FunctionalObjectProperty'):
                    continue
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
                # Ignore SubClassOf()
                line = line.strip()[11:-1]
                if not line:
                    continue
                if line.startswith('ObjectIntersectionOf('):
                    # C and D SubClassOf E
                    it = line.split(' ')
                    c = it[0][21:]
                    d = it[1][:-1]
                    e = it[2]
                    if c not in classes:
                        classes[c] = len(classes)
                    if d not in classes:
                        classes[d] = len(classes)
                    if e not in classes:
                        classes[e] = len(classes)
                    form = 'nf2'
                    if e == 'owl:Nothing':
                        form = 'disjoint'
                    data[form].append((classes[c], classes[d], classes[e]))

                elif line.startswith('ObjectSomeValuesFrom('):
                    # R some C SubClassOf D
                    it = line.split(' ')
                    r = it[0][21:]
                    c = it[1][:-1]
                    d = it[2]
                    if c not in classes:
                        classes[c] = len(classes)
                    if d not in classes:
                        classes[d] = len(classes)
                    if r not in relations:
                        relations[r] = len(relations)
                    data['nf4'].append((relations[r], classes[c], classes[d]))
                elif line.find('ObjectSomeValuesFrom') != -1:
                    # C SubClassOf R some D
                    it = line.split(' ')
                    c = it[0]
                    r = it[1][21:]
                    d = it[2][:-1]
                    if c not in classes:
                        classes[c] = len(classes)
                    if d not in classes:
                        classes[d] = len(classes)
                    if r not in relations:
                        relations[r] = len(relations)
                    data['nf3'].append((classes[c], relations[r], classes[d]))
                else:
                    # C SubClassOf D
                    it = line.split(' ')
                    c = it[0]
                    d = it[1]
                    r = 'SubClassOf'
                    if r not in relations:
                        relations[r] = len(relations)
                    if c not in classes:
                        classes[c] = len(classes)
                    if d not in classes:
                        classes[d] = len(classes)
                    data['nf1'].append((classes[c], relations[r], classes[d]))

        if 'owl:Thing' not in classes:
            classes['owl:Thing'] = len(classes)

        class_ids = np.array(list(classes.values()))

        # # Add corrupted triples nf3
        # n_classes = len(classes)
        # data['nf3_neg'] = []
        # for c, r, d in data['nf3']:
        #     x = np.random.choice(prot_ids)
        #     while x == c:
        #         x = np.random.choice(prot_ids)
        #
        #     y = np.random.choice(prot_ids)
        #     while y == d:
        #         y = np.random.choice(prot_ids)
        #     data['nf3_neg'].append((c, r, x))
        #     data['nf3_neg'].append((y, r, d))

        data['nf1'] = torch.tensor(data['nf1'], dtype=torch.long)[:, [0, 2]]
        data['nf2'] = torch.tensor(data['nf2'], dtype=torch.long)
        data['nf3'] = torch.tensor(data['nf3'], dtype=torch.long)
        data['nf4'] = torch.tensor(data['nf4'], dtype=torch.long)
        data['disjoint'] = torch.tensor(data['disjoint'], dtype=torch.long)
        data['top'] = torch.tensor([classes['owl:Thing']], dtype=torch.long)
        data['role_inclusion'] = torch.tensor(data['role_inclusion'], dtype=torch.long)
        data['role_chain'] = torch.tensor(data['role_chain'], dtype=torch.long)
        data['nf3_neg'] = torch.tensor([], dtype=torch.long)
        data['prot_ids'] = class_ids

        random_state = np.random.get_state()
        for key, val in data.items():
            index = np.arange(len(data[key]))
            np.random.seed(100)
            np.random.shuffle(index)
            data[key] = val[index]
        np.random.set_state(random_state)
        return data, classes, relations
