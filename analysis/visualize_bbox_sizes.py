import os

import random

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

from road_damage_dataset import RoadDamageDataset
from utils import roaddamage_label_names
from dataset_utils import load_labels_and_bboxes

rcParams['figure.figsize'] = 14, 18
rcParams['figure.dpi'] = 240

dataset_dir = os.path.join("RoadDamageDataset", "All")
dataset = RoadDamageDataset(dataset_dir, split="trainval")

indices = np.arange(len(dataset))
np.random.shuffle(indices)
N = 600

labels, bboxes = load_labels_and_bboxes(dataset, indices[:N])

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

uniques = np.unique(labels)
for i, label in enumerate(uniques):
    axes.scatter(W[labels==label], H[labels==label], s=100,
                 marker=Line2D.filled_markers[i % len(uniques)],
                 label=roaddamage_label_names[label])
axes.legend()
plt.show()
