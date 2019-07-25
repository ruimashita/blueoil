import functools
import os.path
import glob
import numpy as np
from PIL import Image

from lmnet.datasets.base import SegmentationBase, StoragePathCustomizable
from lmnet.utils.random import shuffle, train_test_split


def get_image(filename, convert_rgb=True, ignore_class_idx=None):
    """Returns numpy array of an image"""
    image = Image.open(filename)
    #  sometime image data is gray.
    if convert_rgb:
        image = image.convert("RGB")
        image = np.array(image)
    else:
        image = image.split()[0]
        image = np.array(image)
        if ignore_class_idx is not None:
            # Replace ignore labelled class with enough large value
            image = np.where(image == ignore_class_idx, 255, image)
            image = np.where((image > ignore_class_idx) & (image != 255), image - 1, image)

    return image


class LenovoBase(SegmentationBase):
    """
    Base class for CamVid and the variant dataset formats.
    http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
    """
    def __init__(
            self,
            batch_size=10,
            *args,
            **kwargs
    ):

        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    extend_dir = ""
    ignore_class_idx = None

    @property
    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    @property
    def num_per_epoch(self):
        return len(self.files_and_annotations[0])

    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""

        if self.subset == "train":
            image_path = os.path.join(self.data_dir, "short/*.ARW")
            mask_path = os.path.join(self.data_dir, "long/*.ARW")
            image_files = sorted(glob.glob(image_path))
            label_files = sorted(glob.glob(mask_path))

        if self.subset == "validation":
            image_path = os.path.join(self.data_dir, "short/*.ARW")
            mask_path = os.path.join(self.data_dir, "long/*.ARW")
            image_files = sorted(glob.glob(image_path))
            label_files = sorted(glob.glob(mask_path))


        image_files, label_files = shuffle(image_files, label_files)
        print("files and annotations are ready")

        return image_files, label_files

    def __getitem__(self, i, type=None):
        image_files, label_files = self.files_and_annotations

        # image = get_image(image_files[i])
        # label = get_image(label_files[i], convert_rgb=False, ignore_class_idx=self.ignore_class_idx)
        image = image_files[i]
        label = label_files[i]

        return (image, label)

    def __len__(self):
        return self.num_per_epoch


class DSCRaw(LenovoBase):
    """CamVid
    Original CamVid dataset format.
    http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
    """

    classes = [
        "Red",
        "Green",
        "Blue",
        # "backgropund",
        # "person",
        # "unlabelled",  # it is not use.
    ]
    num_classes = len(classes)

    def __init__(
            self,
            batch_size=10,
            *args,
            **kwargs
    ):

        super().__init__(
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    @property
    def label_colors(self):
        background = [0, 0, 128]
        person = [128, 0, 0]


        label_colors = np.array([
            background,
            person
        ])

        return label_colors

    @property
    @functools.lru_cache(maxsize=None)
    def files_and_annotations(self):
        """Return all files and gt_boxes list."""

        if self.subset == "train":
            image_path = os.path.join(self.data_dir, "result_Sony/0003/train/*.npy")
            mask_path = os.path.join(self.data_dir, "result_Sony/0003/gt/*.npy")
            image_files = sorted(glob.glob(image_path))
            label_files = sorted(glob.glob(mask_path))

        if self.subset == "validation":
            image_path = os.path.join(self.data_dir, "result_Sony/0003/train/*.npy")
            mask_path = os.path.join(self.data_dir, "result_Sony/0003/gt/*.npy")
            image_files = sorted(glob.glob(image_path))
            label_files = sorted(glob.glob(mask_path))

            image_files = image_files[:100]
            label_files = label_files[:100]

        image_files, label_files = shuffle(image_files, label_files)
        print("files and annotations are ready")

        return image_files, label_files
