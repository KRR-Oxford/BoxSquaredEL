import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import matplotlib.ticker as plticker

from family_data import load_data
from model.loaded_models import LoadedModel

import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams.update({
    'font.size': 13,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


def plot_box_boxsqel(center, offset, color, text, text_pos):
    low = center - offset
    sizes = 2 * offset
    lower = low.tolist()
    sizes = sizes.tolist()
    rect = mpatches.Rectangle(lower, sizes[0], sizes[1], fill=False, edgecolor=color, linewidth=1.5)
    plt.gca().add_patch(rect)

    if text_pos == 'upper_right':
        coords = lower[0] + sizes[0] - 0.32, lower[1] + sizes[1] - 0.12
    elif text_pos == 'lower_right':
        coords = lower[0] + sizes[0] - 0.35, lower[1] + 0.04
    else:  # default is upper left
        coords = lower[0] + 0.03, lower[1] + sizes[1] - 0.12

    plt.text(coords[0], coords[1], text, color=color)


data, classes, relations = load_data()
device = 'cpu'
model = LoadedModel.from_name('boxsqel', f'out/boxsqel', 2, device, best=False)
class_boxes = model.get_boxes(model.class_embeds)

loc = plticker.MultipleLocator(base=0.5)
plt.figure(figsize=(6, 4.5))
plt.gca().yaxis.set_major_locator(loc)

r = '#ff1f5b'
g = '#00cd6c'
b = '#009ade'
p = '#af58ba'
y = '#ffc61e'
o = '#f28522'
colors = [b, o, g, r, p, y]
pos_dict = defaultdict(lambda: 'upper_left')
pos_dict['Parent'] = 'upper_right'
pos_dict['Female'] = 'lower_right'
for i, c in enumerate(['Parent', 'Male', 'Female', 'Father', 'Mother', 'Child']):
    center, offset = class_boxes.centers[classes[c]], class_boxes.offsets[classes[c]]
    plot_box_boxsqel(center, offset, colors[i], c, pos_dict[c])

plt.gca().autoscale()
plt.savefig('family_boxsqel.pdf', bbox_inches='tight')
plt.show()


model = LoadedModel.from_name('elbe', f'out/elbe', 2, device, best=False)
class_boxes = model.get_boxes(model.class_embeds)
plt.figure(figsize=(6, 4.5))
plt.gca().xaxis.set_major_locator(loc)
plt.gca().yaxis.set_major_locator(loc)

def plot_box_elbe(center, offset, color, text, text_pos):
    low = center - offset
    sizes = 2 * offset
    lower = low.tolist()
    sizes = sizes.tolist()
    rect = mpatches.Rectangle(lower, sizes[0], sizes[1], fill=False, edgecolor=color, linewidth=1.5)
    plt.gca().add_patch(rect)

    if text_pos == 'upper_right':
        coords = lower[0] + sizes[0] - 0.17, lower[1] + sizes[1] - 0.07
    elif text_pos == 'lower_right':
        coords = lower[0] + sizes[0] - 0.21, lower[1] + 0.04
    elif text_pos == 'upper_left':
        coords = lower[0] + 0.02, lower[1] + sizes[1] - 0.12
    elif isinstance(text_pos, tuple):
        coords = text_pos

    plt.text(coords[0], coords[1], text, color=color)


pos_dict['Father'] = (-0.15, 0.51)
pos_dict['Parent'] = (0.4, 0.5)
pos_dict['Mother'] = (-0.03, -0.16)
for i, c in enumerate(['Parent', 'Male', 'Female', 'Father', 'Mother', 'Child']):
    center, offset = class_boxes.centers[classes[c]], class_boxes.offsets[classes[c]]
    plot_box_elbe(center, offset, colors[i], c, pos_dict[c])

plt.gca().autoscale()
plt.savefig('family_elbe.pdf', bbox_inches='tight')
plt.show()
