import argparse
from matplotlib import pyplot as plt

import chainer
from chainer.serializers import load_npz

import chainercv
from chainercv import utils
from chainercv.visualizations import vis_bbox

import ssd_resnet101
from utils import roaddamage_label_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--base-network', choices=('vgg16', 'resnet101'),
                        default='vgg16', help='Base network')
    parser.add_argument('--pretrained-model', required=True)
    parser.add_argument('image')
    args = parser.parse_args()

    if args.base_network == 'vgg16':
        # pretrained_extractor is currently not available for this class
        model = chainercv.links.SSD300(
           n_fg_class=len(roaddamage_label_names),
           pretrained_model=args.pretrained_model)
    elif args.base_network == 'resnet101':
        model = ssd_resnet101.SSD224(
           n_fg_class=len(roaddamage_label_names),
           pretrained_extractor=args.pretrained_extractor,
           pretrained_model=args.pretrained_model)
    else:
        raise ValueError('Invalid base network')

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
