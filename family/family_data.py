import torch


def load_data():
    class_names = ['Father', 'Mother', 'Parent', 'Child', 'Male', 'Female']
    relation_names = ['hasChild', 'hasParent']

    classes = {c: i for i, c in enumerate(class_names)}
    relations = {r: i for i, r in enumerate(relation_names)}

    nf1 = [('Father', 'Male'), ('Mother', 'Female'), ('Mother', 'Parent'), ('Father', 'Parent')]
    nf2 = [('Male', 'Parent', 'Father'), ('Female', 'Parent', 'Mother')]
    nf3 = [('Parent', 'hasChild', 'Child'), ('Child', 'hasParent', 'Mother'), ('Child', 'hasParent', 'Father')]
    disjoint = [('Male', 'Female'), ('Parent', 'Child')]

    data = {
        'nf1': to_tensor(nf1, classes, relations),
        'nf2': to_tensor(nf2, classes, relations),
        'nf3': to_tensor(nf3, classes, relations, use_relations=True),
        'nf4': [],
        'nf3_neg0': [],
        'disjoint': to_tensor(disjoint, classes, relations),
    }
    return data, classes, relations


def to_tensor(data, classes, relations, use_relations=False):
    if use_relations:
        return torch.tensor([[classes[tup[0]], relations[tup[1]], classes[tup[2]]] for tup in data], dtype=torch.long)
    return torch.tensor([list(map(classes.get, tup)) for tup in data], dtype=torch.long)
