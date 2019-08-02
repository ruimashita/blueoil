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
from lmnet.networks.regression import LmnetV1, LmnetV1Quantize
from lmnet.datasets.donkey import Donkey
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    PerImageStandardization,
)
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    Hue,
    FlipLeftRight,
)
from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)


IS_DEBUG = False

NETWORK_CLASS = LmnetV1
DATASET_CLASS = Donkey

IMAGE_SIZE = [120, 160]
BATCH_SIZE = 64
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 100000
SAVE_STEPS = 10000
TEST_STEPS = 500
SUMMARISE_STEPS = 100
# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# distributed training
IS_DISTRIBUTION = False

# for debug
# MAX_STEPS = 10
# BATCH_SIZE = 31
# SAVE_STEPS = 2
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    PerImageStandardization(),
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.0001}

# NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
# NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
# NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
# step_per_epoch = int(50000 / BATCH_SIZE)
# NETWORK.LEARNING_RATE_KWARGS = {
#     "values": [0.01, 0.1, 0.01, 0.001, 0.0001],
#     "boundaries": [step_per_epoch, step_per_epoch * 50, step_per_epoch * 100, step_per_epoch * 198],
# }
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005
# NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
# NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
#     'bit': 2,
#     'max_value': 2
# }
# NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
# NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
    FlipLeftRight(),
])
DATASET.ENABLE_PREFETCH = True
