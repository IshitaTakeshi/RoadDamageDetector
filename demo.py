import argparse
from matplotlib import pyplot as plt
from matplotlib import rcParams

import chainer
from chainer.serializers import load_npz

from ssd_resnet101 import SSD224
from chainercv import utils
from chainercv.visualizations import vis_bbox

from utils import roaddamage_label_names


rcParams['figure.figsize'] = 16, 20
rcParams['figure.dpi'] = 240


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', required=True)
    parser.add_argument('image')
    args = parser.parse_args()

    model = SSD224(
        n_fg_class=len(roaddamage_label_names),
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=roaddamage_label_names)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
