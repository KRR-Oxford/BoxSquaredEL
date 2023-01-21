import json
import os

import numpy as np

from utils.data_loader import DataLoader


class Split:
    nf1: np.ndarray = None
    nf2: np.ndarray = None
    nf3: np.ndarray = None
    nf4: np.ndarray = None

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(1, 5):
            file_path = f'{path}/nf{i}.npy'
            arr = self.__getattribute__(f'nf{i}')
            if arr is None:
                raise ValueError('Tried to save uninitialised split')
            np.save(file_path, arr)

    def get_all_classes(self):
        nf1_set = set(self.nf1.flatten().tolist())
        nf2_set = set(self.nf2.flatten().tolist())
        nf3_set = set(self.nf3[:, [0, 2]].flatten().tolist())
        nf4_set = set(self.nf4[:, [1, 2]].flatten().tolist())
        return nf1_set | nf2_set | nf3_set | nf4_set

    def get_all_relations(self):
        nf3_set = set(self.nf3[:, 1].flatten().tolist())
        nf4_set = set(self.nf4[:, 0].flatten().tolist())
        return nf3_set | nf4_set

    def remove_axioms_not_in_set(self, class_set, relations_set):
        self.remove_nf1_axioms(class_set)
        self.remove_nf2_axioms(class_set)
        self.remove_nf3_axioms(class_set, relations_set)
        self.remove_nf4_axioms(class_set, relations_set)

    def remove_nf1_axioms(self, class_set):
        mask = []
        for i in range(len(self.nf1)):
            tup = self.nf1[i].tolist()
            if tup[0] in class_set and tup[1] in class_set:
                mask.append(True)
            else:
                mask.append(False)
        self.nf1 = self.nf1[mask]

    def remove_nf2_axioms(self, class_set):
        mask = []
        for i in range(len(self.nf2)):
            tup = self.nf2[i].tolist()
            if tup[0] in class_set and tup[1] in class_set and tup[2] in class_set:
                mask.append(True)
            else:
                mask.append(False)
        self.nf2 = self.nf2[mask]

    def remove_nf3_axioms(self, class_set, relations_set):
        mask = []
        for i in range(len(self.nf3)):
            tup = self.nf3[i].tolist()
            if tup[0] in class_set and tup[2] in class_set and tup[1] in relations_set:
                mask.append(True)
            else:
                mask.append(False)
        self.nf3 = self.nf3[mask]

    def remove_nf4_axioms(self, class_set, relations_set):
        mask = []
        for i in range(len(self.nf4)):
            tup = self.nf4[i].tolist()
            if tup[0] in relations_set and tup[1] in class_set and tup[2] in class_set:
                mask.append(True)
            else:
                mask.append(False)
        self.nf4 = self.nf4[mask]


dataset = 'GALEN'
data_loader = DataLoader.from_task('inferences')
data, classes, relations = data_loader.load_data(dataset)

folder = f'data/{dataset}/prediction'
with open(f'{folder}/classes.json', 'w+') as f:
    json.dump(classes, f, indent=2)
with open(f'{folder}/relations.json', 'w+') as f:
    json.dump(relations, f, indent=2)

train_split = Split()
val_split = Split()
test_split = Split()

for i in range(1, 5):
    nf = f'nf{i}'
    num = data[nf].shape[0]
    num_train = int(0.8 * num)
    num_val = int(0.1 * num)
    # data is already shuffled by inference_data_loader
    train = data[nf][:num_train]
    val = data[nf][num_train:num_train + num_val]
    test = data[nf][num_train + num_val:]
    train_split.__setattr__(nf, train)
    val_split.__setattr__(nf, val)
    test_split.__setattr__(nf, test)

train_classes = train_split.get_all_classes()
train_rels = train_split.get_all_relations()
val_split.remove_axioms_not_in_set(train_classes, train_rels)
test_split.remove_axioms_not_in_set(train_classes, train_rels)

train_split.save(f'{folder}/train')
val_split.save(f'{folder}/val')
test_split.save(f'{folder}/test')

for key in ['disjoint', 'top', 'prot_ids']:
    np.save(f'{folder}/train/{key}.npy', data[key])
