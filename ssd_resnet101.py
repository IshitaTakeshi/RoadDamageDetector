import chainer
from chainer.links import ResNet101Layers
from chainer import functions as F
from chainercv.links.model.ssd.ssd_vgg16 import (_load_npz, _imagenet_mean)
from chainercv.links.model.ssd import Multibox, SSD


class ResNet101FineTuning(chainer.Chain):
    def __init__(self, n_fg_class, pretrained_model=None):
        super(ResNet101Extractor, self).__init__()
        with self.init_scope():
            # automatically load the caffemodel
            self.base = ResNet101Layers(pretrained_model='auto')
            self.fc6 = Linear(2048, n_fg_class + 1)

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
    """

    def __init__(self, n_fg_class=None, pretrained_model=None):

        super(SSD224, self).__init__(
            extractor=ResNet101Extractor(),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2,))),
                steps=(24, 48, 100, 200),
                sizes=(30, 60, 111, 162, 213),
                mean=_imagenet_mean)

        if pretrained_model:
            _load_npz(pretrained_model, self)
