import numpy as np
import torch

from utils.data_loader import DataLoader
from utils.utils import get_device, memory

device = get_device()


class OldDataLoader(DataLoader):

    def __init__(self):
        self.load_valid_or_test_data = memory.cache(self.load_valid_or_test_data)
        self.load_data = memory.cache(self.load_data)

    @staticmethod
    def get_file_start(dataset):
        return f'data/{dataset}/old/{dataset}'

    def load_val_data(self, dataset, classes):
        return self.load_valid_or_test_data(dataset, '_valid.txt', classes)

    def load_test_data(self, dataset, classes):
        return self.load_valid_or_test_data(dataset, '_test.txt', classes)

    # def load_inferences_data(dataset, classes):
    #     return load_valid_or_test_data(dataset, '_inferences.txt', classes)

    def load_valid_or_test_data(self, dataset, suffix, classes):
        data = []
        with open(self.get_file_start(dataset) + suffix, 'r') as f:
            for line in f:
                it = line.strip().split()
                id1 = it[0]
                id2 = it[1]
                if id1 not in classes or id2 not in classes:
                    continue
                data.append((classes[id1], classes[id2]))
        return {'nf1': torch.tensor(data)}

    def load_cls(self, train_data_file):
        train_subs = list()
        counter = 0
        with open(train_data_file, 'r') as f:
            for line in f:
                counter += 1
                it = line.strip().split()
                cls1 = it[0]
                cls2 = it[1]
                train_subs.append(cls1)
                train_subs.append(cls2)
        train_cls = list(set(train_subs))
        return train_cls, counter

    def get_all_sub_cls(self, dataset):
        train_file = self.get_file_start(dataset) + "_train.txt"
        va_file = self.get_file_start(dataset) + "_valid.txt"
        test_file = self.get_file_start(dataset) + "_test.txt"
        train_sub_cls, train_samples = self.load_cls(train_file)
        valid_sub_cls, valid_samples = self.load_cls(va_file)
        test_sub_cls, test_samples = self.load_cls(test_file)
        total_sub_cls = train_sub_cls + valid_sub_cls + test_sub_cls
        all_sub_cls = list(set(total_sub_cls))
        return all_sub_cls

    def load_data(self, dataset):
        filename = self.get_file_start(dataset) + '_latest_norm_mod.owl'
        classes = {}
        relations = {}
        data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': []}
        with open(filename) as f:
            for line in f:
                # Ignore SubObjectPropertyOf
                if line.startswith('SubObjectPropertyOf'):
                    continue
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

        prot_ids = []
        class_keys = list(classes.keys())
        for val in self.get_all_sub_cls(dataset):
            if val not in class_keys:
                cid = len(classes)
                classes[val] = cid
                prot_ids.append(cid)
            else:
                prot_ids.append(classes[val])
        prot_ids = np.array(prot_ids)

        # Add corrupted triples nf3
        n_classes = len(classes)
        data['nf3_neg0'] = []
        for c, r, d in data['nf3']:
            x = np.random.choice(prot_ids)
            while x == c:
                x = np.random.choice(prot_ids)

            y = np.random.choice(prot_ids)
            while y == d:
                y = np.random.choice(prot_ids)
            data['nf3_neg0'].append((c, r, x))
            data['nf3_neg0'].append((y, r, d))

        data['nf1'] = torch.tensor(data['nf1'], dtype=torch.long)[:, [0, 2]]
        data['nf2'] = torch.tensor(data['nf2'], dtype=torch.long)
        data['nf3'] = torch.tensor(data['nf3'], dtype=torch.long)
        data['nf4'] = torch.tensor(data['nf4'], dtype=torch.long)
        data['disjoint'] = torch.tensor(data['disjoint'], dtype=torch.long)
        data['top'] = torch.tensor([classes['owl:Thing']], dtype=torch.long)
        data['nf3_neg0'] = torch.tensor(data['nf3_neg0'], dtype=torch.long)
        data['prot_ids'] = prot_ids

        random_state = np.random.get_state()
        for key, val in data.items():
            index = np.arange(len(data[key]))
            np.random.seed(100)
            np.random.shuffle(index)
            data[key] = val[index]
        np.random.set_state(random_state)
        return data, classes, relations
