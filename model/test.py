import numpy as np
from ELModel import ELModel
import torch
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
            # print(line)
            # print(line.strip()[11:-1])
            #print('##########################################################')
            line = line.strip()[11:-1]

            if not line:
                continue
            if line.startswith('ObjectIntersectionOf('):
                # C and D SubClassOf E
                #print(line)

                #SubClassOf(ObjectIntersectionOf(<http://4932.YPL266W> <http://purl.obolibrary.org/obo/GO_0008150>) <http://purl.obolibrary.org/obo/GO_0060049>)
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
    #iterate all classes and index of class
    for k, v in classes.items():

        #Todo: ??? filter something
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            prot_ids.append(v)
    prot_ids = np.array(prot_ids)

    # Add at least one disjointness axiom if there is 0
    if len(data['disjoint']) == 0:
        nothing = classes['owl:Nothing']
        n_prots = len(prot_ids)

        #randomlly choose 10 pairs and set them as disjoint
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

    data['nf1'] = torch.tensor(data['nf1'],dtype=torch.int32)
    data['nf2'] = torch.tensor(data['nf2'],dtype=torch.int32)
    data['nf3'] = torch.tensor(data['nf3'],dtype=torch.int32)
    data['nf4'] = torch.tensor(data['nf4'],dtype=torch.int32)
    data['disjoint'] = np.array(data['disjoint'])
    data['top'] = np.array([classes['owl:Thing'], ])
    data['nf3_neg'] = np.array(data['nf3_neg'])

    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(seed=100)
        np.random.shuffle(index)
        data[key] = val[index]

    return data, classes, relations
data,classes,relations = load_data('/home/pengx0a/BORG/EL embedding/el/data/data-train/yeast-classes-normalized.owl')

print(torch.__version__)