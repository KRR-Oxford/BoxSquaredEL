import random
import json

from utils.data_loader import DataLoader

random.seed(100)

dataset = 'ANATOMY'
data_loader = DataLoader.from_task('inferences')
data, classes, relations = data_loader.load_data(dataset)

folder = f'data/{dataset}/inferences'

nf1_set = set([(l[0].item(), l[1].item()) for l in data['nf1']])
inference_data = []
with open(f'{folder}/inferences.owl', 'r') as f:
    for line in f:
        if not line.startswith('SubClassOf'):
            continue
        line = line.strip().replace('SubClassOf(', '').replace(')', '')
        class1, class2 = line.split(' ')
        if class1 not in classes or class2 not in classes:
            print('ERROR: encountered unknown class')
            continue
        if (classes[class1], classes[class2]) in nf1_set:
            continue
        inference_data.append((class1, class2))

random.shuffle(inference_data)
num_val = int(0.1 * len(inference_data))
val_data = inference_data[:num_val]
inference_data = inference_data[num_val:]

with open(f'{folder}/inferences.json', 'w+') as f:
    json.dump(inference_data, f)
with open(f'{folder}/val.json', 'w+') as f:
    json.dump(val_data, f)
