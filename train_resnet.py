import argparse

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer import links as L
from ssd_resnet101 import ResNet101FineTuning
from road_damage_dataset import (roaddamage_label_names,
                                 RoadDamageClassificationDataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--loaderjob', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--resume', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', default='result-classification')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)

    args = parser.parse_args()

    resnet_fine_tuning = ResNet101FineTuning(
        n_class=len(roaddamage_label_names) + 1
    )

    model = L.Classifier(resnet_fine_tuning)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    # Load the datasets and mean file
    train = RoadDamageClassificationDataset(
            "RoadDamageDataset/All", split='train')
    val = RoadDamageClassificationDataset(
            "RoadDamageDataset/All", split='val')
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    model.to_cpu()
    serializers.save_npz(
        "model-resnet-extractor.npz",
        resnet_fine_tuning.base)

