import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

from road_damage_dataset import RoadDamageDataset
from utils import roaddamage_label_names

rcParams['figure.figsize'] = 14, 18
rcParams['figure.dpi'] = 240


dataset = RoadDamageDataset("RoadDamageDataset/All", split="trainval")

bboxes = []
labels = []
for i in range(400):
    img, bbox, label = dataset.get_example(i)
    if len(bbox) == 0:
        continue
    labels.append(label)
    bboxes.append(bbox)

bboxes = np.vstack(bboxes)
labels = np.concatenate(labels)

color = labels / labels.max()

label_names = [roaddamage_label_names[label] for label in labels]

H = bboxes[:, 2] - bboxes[:, 0]
W = bboxes[:, 3] - bboxes[:, 1]

fig, axes = plt.subplots(1)

axes.set_xlim([0, 610])
axes.set_ylim([0, 610])

axes.set_aspect(1)

axes.set_title("Distribution of bounding box sizes")
axes.set_xlabel("width")
axes.set_xlabel("height")

for label in np.unique(labels):
    axes.scatter(W[labels==label], H[labels==label], s=100,
                 marker=random.choice(['o', 'D', 'd', 'x', '+']),
                 label=roaddamage_label_names[label])
axes.legend()
plt.show()
