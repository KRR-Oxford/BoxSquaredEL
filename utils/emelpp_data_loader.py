import numpy as np
import torch

from utils.utils import get_device, memory

np.random.seed(100)

device = get_device()


def get_file_start(dataset):
    return f'data/{dataset}/EmELpp/{dataset}'


def load_valid_data(dataset, classes):
    return load_valid_or_test_data(dataset, '_valid.txt', classes)


def load_test_data(dataset, classes):
    return load_valid_or_test_data(dataset, '_test.txt', classes)


def load_inferences_data(dataset, classes):
    return load_valid_or_test_data(dataset, '_inferences.txt', classes)


@memory.cache
def load_valid_or_test_data(dataset, suffix, classes):
    data = []
    with open(get_file_start(dataset) + suffix, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = it[0]
            id2 = it[1]
            if id1 not in classes or id2 not in classes:
                continue
            data.append((classes[id1], classes[id2]))
    return data


def load_cls(train_data_file):
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


def get_all_sub_cls(dataset):
    train_file = get_file_start(dataset) + "_train.txt"
    va_file = get_file_start(dataset) + "_valid.txt"
    test_file = get_file_start(dataset) + "_test.txt"
    train_sub_cls, train_samples = load_cls(train_file)
    valid_sub_cls, valid_samples = load_cls(va_file)
    test_sub_cls, test_samples = load_cls(test_file)
    total_sub_cls = train_sub_cls + valid_sub_cls + test_sub_cls
    all_sub_cls = list(set(total_sub_cls))
    return all_sub_cls


@memory.cache
def load_data(dataset):
    filename = get_file_start(dataset) + '_latest_norm_mod.owl'
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
    for val in get_all_sub_cls(dataset):
        if val not in class_keys:
            cid = len(classes)
            classes[val] = cid
            prot_ids.append(cid)
        else:
            prot_ids.append(classes[val])
    prot_ids = np.array(prot_ids)

    # Add corrupted triples nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    for c, r, d in data['nf3']:
        x = np.random.choice(prot_ids)
        while x == c:
            x = np.random.choice(prot_ids)

        y = np.random.choice(prot_ids)
        while y == d:
            y = np.random.choice(prot_ids)
        data['nf3_neg'].append((c, r, x))
        data['nf3_neg'].append((y, r, d))

    data['nf1'] = torch.tensor(data['nf1'], dtype=torch.int32)[:, [0, 2]]
    data['nf2'] = torch.tensor(data['nf2'], dtype=torch.int32)
    data['nf3'] = torch.tensor(data['nf3'], dtype=torch.int32)
    data['nf4'] = torch.tensor(data['nf4'], dtype=torch.int32)
    data['disjoint'] = torch.tensor(data['disjoint'], dtype=torch.int32)
    data['top'] = torch.tensor([classes['owl:Thing']], dtype=torch.int32)
    data['nf3_neg'] = torch.tensor(data['nf3_neg'], dtype=torch.int32)
    data['prot_ids'] = prot_ids

    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(100)
        np.random.shuffle(index)
        data[key] = val[index]
    return data, classes, relations
