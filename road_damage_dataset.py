import numpy as np
import os
import xml.etree.ElementTree as ET

import chainer
from chainercv.utils import read_image

from chainer.links.model.vision import resnet
from utils import roaddamage_label_names, generate_background_bbox


class RoadDamageDataset(chainer.dataset.DatasetMixin):

    """Bounding box dataset for RoadDamageDataset.

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`roadddamage_label_names`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
    """

    def __init__(self, data_dir, split='train'):

        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                "split must be either of 'train', 'traival', or 'val'"
            )

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')

            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.strip()
            label.append(roaddamage_label_names.index(name))
        bbox = np.array(bbox).astype(np.int32)
        label = np.array(label).astype(np.int32)

        # Load an image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)
        return img, bbox, label


class RoadDamageClassificationDataset(RoadDamageDataset):

    def __init__(self, data_dir, background_probability=None):
        """
        Generates images for road damage classification.
        This dataset returns :obj:`image, label`, a tuple of an image and its
        label. The image is basically of a damage part, but in a certain
        probability which can be specified by `background_probability`,
        a random background image is returned.

        Args:
            background_probability (float64): Probability to generate
            a background image.
            The default value is 1 / (number of damage categories + 1).
        """
        super(RoadDamageClassificationDataset, self).__init__(
            data_dir, split='trainval')

        self.background_probability = background_probability
        if background_probability is None:
            self.background_probability = 1 / (len(roaddamage_label_names) + 1)

    def get_example(self, i):
        image, bboxes, labels =\
            super(RoadDamageClassificationDataset, self).get_example(i)

        def generate_damage():
            index = np.random.randint(len(labels))

            label = labels[index]
            ymin, xmin, ymax, xmax = bboxes[index]
            damage = image[:, ymin:ymax, xmin:xmax]

            background = np.zeros(image.shape)
            background[:, ymin:ymax, xmin:xmax] = damage

            image = prepare(background)
            return image, label

        def generate_background():
            _, H, W = image.shape
            bbox = generate_background_bbox((H, W), (224, 224), bboxes)
            ymin, xmin, ymax, xmax = bbox
            image = prepare(image[:, ymin:ymax, xmin:xmax])
            label = len(roaddamage_label_names) + 1
            return image, label

        if np.random.rand() < self.background_probability:
            # generate_background
            try:
                return generate_background()
            except RuntimeError:
                # return damage if failed to generate background
                return generate_damage()

        return generate_damage()

