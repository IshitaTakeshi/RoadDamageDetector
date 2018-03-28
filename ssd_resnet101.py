import chainer
from chainer.links import Linear, ResNet101Layers
from chainer import functions as F
from chainercv.links.model.ssd.ssd_vgg16 import (_load_npz, _imagenet_mean)
from chainercv.links.model.ssd import Multibox, SSD


class ResNet101FineTuning(chainer.Chain):
    def __init__(self, n_class, pretrained_model='auto'):
        super(ResNet101FineTuning, self).__init__()

        with self.init_scope():
            self.base = ResNet101Layers(pretrained_model)
            self.fc6 = Linear(2048, n_class)

    def __call__(self, x):
        activations = self.base(x, layers=["pool5"])
        h = activations["pool5"]
        return F.softmax(self.fc6(h))


class ResNet101Extractor(ResNet101Layers):
    insize = 224
    grids = (56, 28, 14, 7)

    def __init__(self, pretrained_model='auto'):
        super(ResNet101Extractor, self).__init__(pretrained_model)

    def __call__(self, x):
        layers = ["res2", "res3", "res4", "res5"]
        activations = super(ResNet101Extractor, self).__call__(x, layers)
        return [activations[layer] for layer in layers]


class SSD224(SSD):
    """Single Shot Multibox Detector with 224x224 inputs.

    This is a model of Single Shot Multibox Detector [#]_.
    This model uses :class:`ResNet101Extractor` as its feature extractor.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
       n_fg_class (int): The number of classes excluding the background.
       pretrained_model (str): The weight file to be loaded.
           The default value is :obj:`None`.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.
       pretrained_extractor (str): The `npz` weight file of `ResNet101Layers`.
           If this argument is specified as `auto`, it automatically loads and
           converts the caffemodel.
    """

    def __init__(self, n_fg_class=None,
                 pretrained_extractor='auto',
                 pretrained_model=None):
        super(SSD224, self).__init__(
            extractor=ResNet101Extractor(pretrained_extractor),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2, 3, 4), (2, 3, 4), (2, 3, 4), (2, 3, 4))),
            steps=(4, 8, 16, 32),
            sizes=(5, 15, 60, 120, 244),
            mean=_imagenet_mean)

        if pretrained_model:
            _load_npz(pretrained_model, self)
