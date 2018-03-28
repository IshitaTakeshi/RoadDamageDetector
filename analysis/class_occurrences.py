import os

import numpy as np
from matplotlib import pyplot as plt

from road_damage_dataset import RoadDamageDataset
from utils import roaddamage_label_names
from dataset_utils import load_labels_and_bboxes

dataset_dir = os.path.join("RoadDamageDataset", "All")
dataset = RoadDamageDataset(dataset_dir, split="trainval")

labels, bboxes = load_labels_and_bboxes(dataset)

bboxes = np.vstack(bboxes)
labels = np.concatenate(labels)

n, bins, patches = plt.hist(
    labels,
    bins=range(len(roaddamage_label_names) + 1),
    rwidth=0.8
)

positions = np.arange(len(roaddamage_label_names)) + 0.5
plt.ylabel("Occurrences")
plt.xticks(positions, roaddamage_label_names)
plt.show()
