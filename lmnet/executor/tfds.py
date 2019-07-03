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
import os
import math

import click
import tensorflow as tf
import tensorflow_datasets as tfds

from lmnet.utils import executor, module_loader, config as config_util
from lmnet import environment
from lmnet.datasets.dataset_iterator import DatasetIterator






@click.command(context_settings=dict(help_option_names=['-h', '--help']))
def main():
    data_dir = "gs://lmfs-backend-us-west1/home/wakisaka/mnist"
    data_dir = "saved/tfrecord/mnist"
    data = tfds.load(name="mnist", split=["train", "test"], data_dir=data_dir)

    print(data)



    
if __name__ == '__main__':
    main()
