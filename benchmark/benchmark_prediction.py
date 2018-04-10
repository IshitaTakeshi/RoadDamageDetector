import timeit
import argparse
import sys
import os

sys.path.append(os.getcwd())


def setup(base_network, pretrained_model, gpu, n_images):
    s = """
import sys

import numpy as np
import chainer
from chainer.dataset import concat_examples
from chainer.iterators import SerialIterator
from chainercv.links import SSD300

from utils import roaddamage_label_names
from road_damage_dataset import RoadDamageDataset
import ssd_resnet101

dataset = RoadDamageDataset("RoadDamageDataset/All")

base_network = '{0}'

if base_network == 'vgg16':
    model = SSD300(n_fg_class=len(roaddamage_label_names),
                   pretrained_model='{1}')
elif base_network == 'resnet101':
    model = ssd_resnet101.SSD224(n_fg_class=len(roaddamage_label_names),
                                 pretrained_model='{1}')
else:
    ValueError("Invalid model")

gpu_id = {2}
if gpu_id >= 0:
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu(gpu_id)

n_images = {3}

images = []
for i in range(n_images):
    image, bbox, label = dataset.get_example(i)
    images.append(image)
images = np.array(images)
"""
    return s.format(base_network, pretrained_model, gpu, n_images)


args = [
    ('vgg16', 'models/ssd300-vgg16-v0.2/model.npz', -1),
    ('vgg16', 'models/ssd300-vgg16-v0.2/model.npz', 0),
    ('resnet101', 'models/ssd224-resnet101-v0.1/model.npz', -1),
    ('resnet101', 'models/ssd224-resnet101-v0.1/model.npz', 0)
]

n_executions = 10

for n_images in range(10, 60, 10):
    print("Number of images     : {}".format(n_images))
    print("Number of executions : {}".format(n_executions))

    for base_network, pretrained_model, gpu in args:
        print("")
        print("Base network     : {}".format(base_network))
        print("Pretrained model : {}".format(pretrained_model))
        print("GPU ID           : {}".format(gpu))

        timer = timeit.Timer(
            'model.predict(images)',
            setup=setup(base_network, pretrained_model, gpu, n_images))
        t = timer.timeit(number=n_executions)

        print("{} [s]".format(t))
    print("")
