# -*- coding: utf-8 -*-
from easydict import EasyDict
import tensorflow as tf

from lmnet.common import Tasks
from lmnet.networks.object_detection.lm_fyolo import LMFYoloQuantize
from lmnet.datasets.open_images_v4 import OpenImagesV4BoundingBoxBase
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    ResizeWithGtBoxes,
    DivideBy255,
)
from lmnet.post_processor import (
    FormatYoloV2,
    ExcludeLowScoreBox,
    NMS,
)
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
    SSDRandomCrop,
)

from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LMFYoloQuantize


class FaceMouth(OpenImagesV4BoundingBoxBase):
    # extend_dir = "openimages/"
    extend_dir = "openimages_train_face_mouth/"


DATASET_CLASS = FaceMouth
_dataset_obj = DATASET_CLASS(subset="train", batch_size=1)

IMAGE_SIZE = [224, 224]
BATCH_SIZE = 32
DATA_FORMAT = "NHWC"
TASK = Tasks.OBJECT_DETECTION
CLASSES = _dataset_obj.classes

MAX_EPOCHS = 200
SAVE_STEPS = 1000
TEST_STEPS = 1000
SUMMARISE_STEPS = 1000
IS_DISTRIBUTION = False

# for debug
# IS_DEBUG = True
# SUMMARISE_STEPS = 1
# SUMMARISE_STEPS = 100
# TEST_STEPS = 10000
# SUMMARISE_STEPS = 100

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    ResizeWithGtBoxes(size=IMAGE_SIZE),
    DivideBy255()
])
anchors = [
    (1.25, 1.0), (1.0, 1.0), (1.0, 1.25), (1.5, 1.5),
]
score_threshold = 0.05
nms_iou_threshold = 0.5
nms_max_output_size = 100
POST_PROCESSOR = Sequence([
    FormatYoloV2(
        image_size=IMAGE_SIZE,
        classes=CLASSES,
        anchors=anchors,
        data_format=DATA_FORMAT,
    ),
    ExcludeLowScoreBox(threshold=score_threshold),
    NMS(iou_threshold=nms_iou_threshold, max_output_size=nms_max_output_size, classes=CLASSES,),
])

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
# In the origianl yolov2 Paper, with a starting learning rate of 10−3, dividing it by 10 at 60 and 90 epochs.
# Train data num per epoch is 16551
step_per_epoch = int(_dataset_obj.num_per_epoch / BATCH_SIZE)
NETWORK.LEARNING_RATE_KWARGS = {
        "values": [5e-4, 2e-2, 5e-3, 5e-4],
        "boundaries": [step_per_epoch * 5, step_per_epoch * 80, step_per_epoch * 160],
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.ANCHORS = anchors
NETWORK.OBJECT_SCALE = 5.0
NETWORK.NO_OBJECT_SCALE = 1.0
NETWORK.CLASS_SCALE = 1.0
NETWORK.COORDINATE_SCALE = 1.0
NETWORK.LOSS_IOU_THRESHOLD = 0.6
NETWORK.WEIGHT_DECAY_RATE = 0.0005
NETWORK.SCORE_THRESHOLD = score_threshold
NETWORK.NMS_IOU_THRESHOLD = nms_iou_threshold
NETWORK.NMS_MAX_OUTPUT_SIZE = nms_max_output_size
NETWORK.SEEN_THRESHOLD = _dataset_obj.num_per_epoch
# quantization
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2.0
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}
NETWORK.QUANTIZE_FIRST_CONVOLUTION = True
NETWORK.QUANTIZE_LAST_CONVOLUTION = False

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    FlipLeftRight(),
    SSDRandomCrop(min_crop_ratio=0.7),
    ResizeWithGtBoxes(size=IMAGE_SIZE),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
])
DATASET.ENABLE_PREFETCH = True
