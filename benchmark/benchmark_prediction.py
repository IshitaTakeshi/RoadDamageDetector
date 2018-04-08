import timeit
import argparse


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

if {2} >= 0:
    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()

n_images = {3}

images = []
for i in range(n_images):
    image, bbox, label = dataset.get_example(i)
    images.append(image)
images = np.array(images)
"""
    return s.format(base_network, pretrained_model, gpu, n_images)


parser = argparse.ArgumentParser()
parser.add_argument('--base-network', choices=('vgg16', 'resnet101'),
                    default='vgg16', help='Base network')
parser.add_argument('--pretrained-model', required=True)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--n-images', type=int, default=2000)
parser.add_argument('--n-executions', type=int, default=1)

args = parser.parse_args()


timer = timeit.Timer(
    'model.predict(images)',
    setup=setup(args.base_network, args.pretrained_model, args.gpu,
                args.n_images))

print("{} [s]".format(timer.timeit(number=args.n_executions)))
