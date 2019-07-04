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
import pandas as pd
import tensorflow as tf
# from tensorflow_datasets.core import tfrecords_reader
import tensorflow_datasets as tfds


from lmnet.utils import executor, module_loader, config as config_util
from lmnet import environment
from lmnet.datasets.dataset_iterator import DatasetIterator


# TODO(my_dataset): BibTeX citation
_CITATION = """
"""

# TODO(my_dataset):
_DESCRIPTION = """wakisaka dataset
"""


class MyDataset(tfds.core.GeneratorBasedBuilder):
    """TODO(my_dataset): Short description of my dataset."""

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "mask": tfds.features.Image(shape=(None, None, 1)),
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.

            # Homepage of the dataset for documentation
            urls=[],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(my_dataset): Downloads the data and defines the splits
        # dl_manager is a tfds.download.DownloadManager that can be used to
        # download and extract URLs

        base_path = dl_manager.manual_dir 

        print("manual_dir", dl_manager.manual_dir)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                # TODO(my_dataset): Tune the number of shards such that each shard
                # is < 4 GB.
                num_shards=10,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "txt": os.path.join(base_path, "train.txt"),
                    "base_path": base_path,
                },
            ),
        ]

    def _generate_examples(self, txt, base_path):
        """Yields examples."""

        df = pd.read_csv(
            txt,
            delim_whitespace=True,
            header=None,
            names=['image_files', 'label_files'],
        )

        image_files = df.image_files.tolist()
        label_files = df.label_files.tolist()

        image_files = [filename.replace("/SegNet/CamVid", base_path) for filename in image_files]
        label_files = [filename.replace("/SegNet/CamVid", base_path) for filename in label_files]

        for i in range(len(image_files)):
            image_file = image_files[i]
            mask_file = label_files[i]
            yield {
                "image": image_file,
                "mask": mask_file,
            }




def mnist():
    data_dir = "gs://lmfs-backend-us-west1/home/wakisaka/datasets"
    # data_dir = "saved/tfrecord/"

    builder = tfds.builder(name="mnist", data_dir=data_dir)
    info = builder.info

    builder.download_and_prepare(
        download_dir="saved/datasets/downloads/",
    )

    import ipdb; ipdb.set_trace()
    dataset = builder._as_dataset()
    print(info)
    print(dataset['train'])


def dataset_from_tfrecored():
    data_dir = "gs://lmfs-backend-us-west1/home/wakisaka/datasets"
    name = "my_dataset"
    version = "0.1.0"
    split = "train"
    # data_dir = "saved/datasets/"

    download_dir = "saved/datasets/downloads/"

    files = tf.io.gfile.glob(
        os.path.join(data_dir, name, version, "{}-{}.tfrecord*".format(name, split))
    )

    features = tfds.features.FeaturesDict({
        "image": tfds.features.Image(),
        "mask": tfds.features.Image(shape=(None, None, 1)),
    })

    serialized_info = features.get_serialized_info()
    file_format_adapter = tfds.core.file_format_adapter.TFRecordExampleAdapter(serialized_info)
    dataset = file_format_adapter.dataset_from_filename(files)

    dataset = dataset.map(
        features.decode_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        element = sess.run(next_element)
        images = element["image"]
        mask = element["mask"]

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
def main():

    # mnist()
    dataset_from_tfrecored()

    data_dir = "gs://lmfs-backend-us-west1/home/wakisaka/datasets"
    # data_dir = "saved/datasets/"

    download_dir = "saved/datasets/downloads/"


    builder = tfds.builder(name="my_dataset", data_dir=data_dir)
    builder.download_and_prepare(
        download_dir=download_dir,
    )
    info = builder.info
    dataset = builder.as_dataset()
    dataset = dataset['train']
    dataset = dataset.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        element = sess.run(next_element)
        print(element)
        images = element["image"]
        print(images.shape)
        mask = element["mask"]



if __name__ == '__main__':
    main()
