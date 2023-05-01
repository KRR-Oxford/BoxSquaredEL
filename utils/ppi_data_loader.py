import torch
import numpy as np
from utils.utils import memory
import random


def get_file_dir(dataset):
    return f'data/PPI/{dataset}'


@memory.cache
def load_protein_data(dataset, folder, proteins, relations):
    filename = f'{get_file_dir(dataset)}/{folder}/protein_links.txt'
    data = []
    rel = f'<http://interacts>'
    assert rel in relations
    with open(filename, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            if id1 not in proteins or id2 not in proteins:
                continue
            data.append((proteins[id1], relations[rel], proteins[id2]))
    return data


def is_protein(cls):
    return not cls.startswith('<http://purl.obolibrary.org/obo/GO_') and cls not in ['owl:Thing', 'owl:Nothing']


def contains_any_proteins(classes):
    return any([is_protein(cls) for cls in classes])


@memory.cache
def load_data(dataset):
    filename = f'{get_file_dir(dataset)}/train/{dataset}.owl'
    classes = {}
    proteins = {}
    relations = {}
    data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': [],
            'abox': {'role_assertions': [], 'concept_assertions': []}}
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

                if contains_any_proteins([c, d, e]):
                    continue
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

                if contains_any_proteins([c, d]):
                    continue
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

                if r not in relations:
                    relations[r] = len(relations)

                if is_protein(c):
                    if c not in proteins:
                        proteins[c] = len(proteins)
                    if is_protein(d):
                        assert r == '<http://interacts>'
                        if d not in proteins:
                            proteins[d] = len(proteins)
                        data['abox']['role_assertions'].append((relations[r], proteins[c], proteins[d]))
                        continue
                    else:
                        assert r == '<http://hasFunction>'
                        if d not in classes:
                            classes[d] = len(classes)
                        data['abox']['concept_assertions'].append((relations[r], classes[d], proteins[c]))
                        continue
                else:
                    assert not is_protein(d) and r != '<http://interacts>'

                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                data['nf3'].append((classes[c], relations[r], classes[d]))
            else:
                # C SubClassOf D
                it = line.split(' ')
                c = it[0]
                d = it[1]

                if contains_any_proteins([c, d]):
                    continue
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                data['nf1'].append((classes[c], classes[d]))

    # Check if TOP in classes and insert if it is not there
    if 'owl:Thing' not in classes:
        classes['owl:Thing'] = len(classes)
    if 'owl:Nothing' not in classes:
        classes['owl:Nothing'] = len(classes)

    assert not any([is_protein(cls) for cls in classes])

    data['nf1'] = torch.tensor(data['nf1'], dtype=torch.long)
    data['nf2'] = torch.tensor(data['nf2'], dtype=torch.long)
    data['nf3'] = torch.tensor(data['nf3'], dtype=torch.long)
    data['nf4'] = torch.tensor(data['nf4'], dtype=torch.long)
    data['disjoint'] = torch.tensor(data['disjoint'], dtype=torch.long)
    data['top'] = torch.tensor([classes['owl:Thing']], dtype=torch.long)
    data['nf3_neg'] = torch.tensor([], dtype=torch.long)
    data['abox']['role_assertions'] = torch.tensor(data['abox']['role_assertions'], dtype=torch.long)
    data['abox']['concept_assertions'] = torch.tensor(data['abox']['concept_assertions'], dtype=torch.long)
    data['class_ids'] = np.array(list(classes.values()))
    data['prot_ids'] = np.array(list(proteins.values()))

    random_state = np.random.get_state()
    np.random.seed(100)
    for key in data:
        if key == 'abox':
            data[key]['role_assertions'] = shuffle_tensor(data[key]['role_assertions'])
            data[key]['concept_assertions'] = shuffle_tensor(data[key]['concept_assertions'])
        else:
            data[key] = shuffle_tensor(data[key])
    np.random.set_state(random_state)
    return data, classes, proteins, relations


def shuffle_tensor(arr):
    index = np.arange(len(arr))
    np.random.shuffle(index)
    return arr[index]
