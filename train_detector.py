import os
import argparse
import copy
import warnings

import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainer.links.model.vision import resnet

import chainercv
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

import ssd_resnet101
from road_damage_dataset import RoadDamageDataset, roaddamage_label_names


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class MeanSubtraction(object):
    def __init__(self, mean):
        self.mean = mean.astype(np.float32)

    def __call__(self, in_data):
        img = in_data[0]
        img = img - self.mean
        return (img, *in_data[1:])


class ResNetPreparation(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, in_data):
        img = in_data[0]
        img = resnet.prepare(img, (self.size, self.size))
        return (img, *in_data[1:])


class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        bbox = np.array(bbox).astype(np.float32)

        if len(bbox) == 0:
            warnings.warn("No bounding box detected", RuntimeWarning)
            img = resize_with_random_interpolation(img, (self.size, self.size))
            mb_loc, mb_label = self.coder.encode(bbox, label)
            return img, mb_loc, mb_label

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        mb_loc, mb_label = self.coder.encode(bbox, label)
        return img, mb_loc, mb_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join("RoadDamageDataset", "All"))
    parser.add_argument('--batchsize', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--base-network', choices=('vgg16', 'resnet101'),
                        default='vgg16',
                        help='Base network')
    parser.add_argument('--pretrained-model', default=None,
                        help='Pretrained SSD model')
    parser.add_argument('--pretrained-extractor',
                        default='auto',
                        help='Pretrained CNN model to extract feature maps')
    parser.add_argument('--out', default='result-detection')
    parser.add_argument('--resume', default=None,
                        help='Initialize the trainer from given file')

    args = parser.parse_args()

    print("Data directory       : {}".format(args.data_dir))
    print("Batchsize            : {}".format(args.batchsize))
    print("GPU ID               : {}".format(args.gpu))
    print("Base network         : {}".format(args.base_network))
    print("Pretrained extractor : {}".format(args.pretrained_extractor))
    print("Pretrained model     : {}".format(args.pretrained_model))
    print("Output directory     : {}".format(args.out))
    print("Resume from          : {}".format(args.resume))


    if args.base_network == 'vgg16':
       # pretrained_extractor is currently not available for this class
        model = chainercv.links.SSD300(
           n_fg_class=len(roaddamage_label_names),
           pretrained_model=args.pretrained_model)
        preprocessing = MeanSubtraction(model.mean)
    elif args.base_network == 'resnet101':
        model = ssd_resnet101.SSD224(
           n_fg_class=len(roaddamage_label_names),
           pretrained_extractor=args.pretrained_extractor,
           pretrained_model=args.pretrained_model)
        preprocessing = ResNetPreparation(model.insize)
    else:
        raise ValueError('Invalid base network')

    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train = TransformDataset(
        RoadDamageDataset(args.data_dir, split='train'),
        Transform(model.coder, model.insize, model.mean)
    )

    train = TransformDataset(train, preprocessing)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    test = RoadDamageDataset(args.data_dir, split='val')
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # initial lr is set to 3e-4 by ExponentialShift
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=3e-4),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=roaddamage_label_names),
        trigger=(4000, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']),
        trigger=log_interval)

    # trainer.extend(extensions.ProgressBar())

    trainer.extend(extensions.snapshot(), trigger=(4000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(4000, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    print("setup finished")
    trainer.run()

    model.to_cpu()
    serializers.save_npz("model-detector.npz", model)


if __name__ == '__main__':
    main()
