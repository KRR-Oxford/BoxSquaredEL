import torch
import numpy as np
from utils.utils import memory
import  random
np.random.seed(100)



'''load the normalized data(nf1, nf2, nf3, nf4)

Args: 
    filename: the normalized data, .owl format

Return:
    data: dictonary, nf1,nf2...data with triple or double class or relation index
    classes: dictonary, key is class name, value is according index
    relations: dictonary, key is relation name, value is according index
'''

@memory.cache
def load_data(filename):

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

    prot_ids = []
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            prot_ids.append(v)
    prot_ids = np.array(prot_ids)

    # Add at least one disjointness axiom if there is 0
    if len(data['disjoint']) == 0:
        nothing = classes['owl:Nothing']
        n_prots = len(prot_ids)
        for i in range(10):
            it = np.random.choice(n_prots, 2)
            if it[0] != it[1]:
                data['disjoint'].append(
                    (prot_ids[it[0]], prot_ids[it[1]], nothing))
                break

    # Add corrupted triples for nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    inter_ind = 0
    for k, v in relations.items():
        if k == '<http://interacts>':
            inter_ind = v
    for c, r, d in data['nf3']:
        if r != inter_ind:
            continue

        data['nf3_neg'].append((c, r, np.random.choice(prot_ids)))

        data['nf3_neg'].append((np.random.choice(prot_ids), r, d))


    data['nf1'] = torch.tensor(data['nf1'], dtype=torch.int32)
    data['nf2'] = torch.tensor(data['nf2'], dtype=torch.int32)
    data['nf3'] = torch.tensor(data['nf3'], dtype=torch.int32)
    data['nf4'] = torch.tensor(data['nf4'], dtype=torch.int32)
    data['disjoint'] = torch.tensor(data['disjoint'], dtype=torch.int32)
    data['top'] = torch.tensor([classes['owl:Thing']],  dtype=torch.int32)
    data['nf3_neg'] = torch.tensor(data['nf3_neg'], dtype=torch.int32)

    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(100)
        np.random.shuffle(index)
        data[key] = val[index]

    return data, classes, relations


    np.random.seed(100)

    '''load the normalized data(nf1, nf2, nf3, nf4)

    Args: 
        filename: the normalized data, .owl format

    Return:
        data: dictonary, nf1,nf2...data with triple or double class or relation index
        classes: dictonary, key is class name, value is according index
        relations: dictonary, key is relation name, value is according index
    '''

def load_data_ball(filename):

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

    prot_ids = []
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            prot_ids.append(v)
    prot_ids = np.array(prot_ids)

    # Add at least one disjointness axiom if there is 0
    if len(data['disjoint']) == 0:
        nothing = classes['owl:Nothing']
        n_prots = len(prot_ids)
        for i in range(10):
            it = np.random.choice(n_prots, 2)
            if it[0] != it[1]:
                data['disjoint'].append(
                    (prot_ids[it[0]], prot_ids[it[1]], nothing))
                break

    # Add corrupted triples for nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    inter_ind = 0
    for k, v in relations.items():
        if k == '<http://interacts>':
            inter_ind = v
    for c, r, d in data['nf3']:
        if r != inter_ind:
            continue

        data['nf3_neg'].append((c, r, np.random.choice(prot_ids)))

        data['nf3_neg'].append((np.random.choice(prot_ids), r, d))

    for i in data['nf2'][10000:]:
        a,b,c = i
        print(str(a)+" "+str(b)+" "+str(c))
        data['nf1'].append((c,a))
        data['nf1'].append((c, b))
    data['nf1'] = torch.tensor(data['nf1'], dtype=torch.int32)
    data['nf2'] = torch.tensor(data['nf2'], dtype=torch.int32)
    data['nf3'] = torch.tensor(data['nf3'], dtype=torch.int32)
    data['nf4'] = torch.tensor(data['nf4'], dtype=torch.int32)
    data['disjoint'] = torch.tensor(data['disjoint'], dtype=torch.int32)
    data['top'] = torch.tensor([classes['owl:Thing']], dtype=torch.int32)
    data['nf3_neg'] = torch.tensor(data['nf3_neg'], dtype=torch.int32)

    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(100)
        np.random.shuffle(index)
        data[key] = val[index]

    return data, classes, relations

'''load valid data

Args:
    valid_data_file: .txt file, one line means two interacted proteins
    classes: dictonary, key is class name, value is according index
    relations: dictonary, key is relation name, value is according index

Return value:
    data: classes[id1], relations[rel], classes[id2]
'''

def load_valid_data(valid_data_file, classes, relations):
    data = []
    rel = f'<http://interacts>'
    with open(valid_data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            data.append((classes[id1], relations[rel], classes[id2]))
    return data