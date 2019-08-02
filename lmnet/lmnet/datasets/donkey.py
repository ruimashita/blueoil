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
import functools
from glob import glob
import os
import os.path
import json

import numpy as np
import PIL.Image

from lmnet.datasets.base import Base, StoragePathCustomizable
from lmnet.utils.random import train_test_split


class Donkey(StoragePathCustomizable, Base):
    extend_dir = "donkeycar/tub_1_19-07-24"
    
    """Abstract class of dataset for loading image files stored in a folder.

    structure like
        $DATA_DIR/extend_dir/cat/0001.jpg
        $DATA_DIR/extend_dir/cat/xxxa.jpeg
        $DATA_DIR/extend_dir/cat/yyyb.png
        $DATA_DIR/extend_dir/dog/123.jpg
        $DATA_DIR/extend_dir/dog/023.jpg
        $DATA_DIR/extend_dir/dog/wwww.jpg

    When child class has `validation_extend_dir`, the `validation` subset consists from the folders.
       $DATA_DIR/validation_extend_dir/cat/0001.jpg
       $DATA_DIR/validation_extend_dir/cat/xxxa.png
    """

    def __init__(
            self,
            is_shuffle=True,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.is_shuffle = is_shuffle

    @property
    @functools.lru_cache(maxsize=None)
    def classes(self):
        """Returns the classes list in the data set."""
        return ["angle", "throttle"]

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_per_epoch(self):
        return len(self.files)

    def _all_files(self):
        all_image_files = []

        json_files = glob(os.path.join(self.data_dir, 'record_*.json'))

        return json_files

    @property
    @functools.lru_cache(maxsize=None)
    def files(self):
        all_image_files = self._all_files()

        if self.validation_size > 0:
            train_image_files, test_image_files = train_test_split(
                all_image_files, test_size=self.validation_size, seed=1)

            if self.subset == "train":
                files = train_image_files
            else:
                files = test_image_files

            return files

        return all_image_files

    def get_image(self, filename):
        """Returns numpy array of an image"""
        image = PIL.Image.open(filename)

        #  sometime image data is gray.
        image = image.convert("RGB")

        image = np.array(image)

        return image

    def __getitem__(self, i, type=None):
        target_file = self.files[i]

        with open(target_file, 'r') as fp:
            json_data = json.load(fp)

        image_filename = os.path.join(self.data_dir, json_data["cam/image_array"])
        image = self.get_image(image_filename)

        angle = float(json_data['user/angle'])
        throttle = float(json_data["user/throttle"])

        label = np.array([angle, throttle])

        return (image, label)

    def __len__(self):
        return self.num_per_epoch
