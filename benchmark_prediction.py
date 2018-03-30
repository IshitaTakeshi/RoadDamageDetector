import timeit
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model', required=True)
parser.add_argument('--gpu', type=int, default=-1)

args = parser.parse_args()


def setup(pretrained_model, gpu):
    s = """
import sys

import numpy as np
import chainer
from chainer.dataset import concat_examples
from chainer.iterators import SerialIterator
from chainercv.links import SSD300

from utils import roaddamage_label_names
from road_damage_dataset import RoadDamageDataset

dataset = RoadDamageDataset("RoadDamageDataset/All")

model = SSD300(n_fg_class=len(roaddamage_label_names),
               pretrained_model='{0}')
N = 20

if {1} >= 0:
    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()

images = []
for i in range(N):
    image, bbox, label = dataset.get_example(i)
    images.append(image)
images = np.array(images)
"""
    return s.format(pretrained_model, gpu)


timer = timeit.Timer(
    'model.predict(images)',
    setup=setup(args.pretrained_model, args.gpu))
print(timer.timeit(number=200))
