# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from easydict import EasyDict
import tensorflow as tf

from lmnet.common import Tasks
from lmnet.networks.object_detection.lm_fyolo import LMFYoloV2_3Quantize
from lmnet.datasets.widerface import WiderFace
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    ResizeWithGtBoxes,
    PerImageStandardization,
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

NETWORK_CLASS = LMFYoloV2_3Quantize
DATASET_CLASS = WiderFace

IMAGE_SIZE = [160, 160]
BATCH_SIZE = 64
DATA_FORMAT = "NHWC"
TASK = Tasks.OBJECT_DETECTION
CLASSES = DATASET_CLASS.classes

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
    PerImageStandardization()
])
anchors = [
    (0.25, 0.25), (0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0),
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

dataset_obj = DATASET_CLASS(subset="train", batch_size=1)
step_per_epoch = int(dataset_obj.num_per_epoch / BATCH_SIZE)

init_lr = 0.0001
max_lr = 0.05
last_lr = 0.0005

values = [init_lr]
boundaries = []

warmup_epochs = 5

for i in range(1, warmup_epochs):
    values.append(max_lr * float(i) / warmup_epochs + init_lr)
    boundaries.append(step_per_epoch * i)

for i in range(warmup_epochs, MAX_EPOCHS):
    values.append(max_lr * (1 - float(i-warmup_epochs) / (MAX_EPOCHS-warmup_epochs)) + last_lr)
    boundaries.append(step_per_epoch * i)

NETWORK.LEARNING_RATE_KWARGS = {
        "values": values,
        "boundaries": boundaries,
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.ANCHORS = anchors
NETWORK.OBJECT_SCALE = 10.0
NETWORK.NO_OBJECT_SCALE = 1.0
NETWORK.CLASS_SCALE = 1.0
NETWORK.COORDINATE_SCALE = 2.0
NETWORK.LOSS_IOU_THRESHOLD = 0.6
NETWORK.WEIGHT_DECAY_RATE = 0
NETWORK.SCORE_THRESHOLD = score_threshold
NETWORK.NMS_IOU_THRESHOLD = nms_iou_threshold
NETWORK.NMS_MAX_OUTPUT_SIZE = nms_max_output_size
NETWORK.SEEN_THRESHOLD = dataset_obj.num_per_epoch
# quantization
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2.0
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}
NETWORK.QUANTIZE_FIRST_CONVOLUTION = False
NETWORK.QUANTIZE_LAST_CONVOLUTION = False

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    SSDRandomCrop(min_crop_ratio=0.7),
    ResizeWithGtBoxes(size=IMAGE_SIZE),
    FlipLeftRight(),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
])
DATASET.ENABLE_PREFETCH = True
