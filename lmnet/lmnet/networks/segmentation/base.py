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
import tensorflow as tf

from lmnet.networks.base import BaseNetwork
from lmnet.common import get_color_map


class Base(BaseNetwork):
    """base network for segmentation

    This base network is for segmentation.
    Each segmentation network class should extend this class.

    """

    def __init__(
            self,
            *args,
            label_colors=None,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        if label_colors is None:
            self.label_colors = get_color_map(self.num_classes)
        else:
            self.label_colors = label_colors

    def placeholderes(self):
        # shape = (self.batch_size, self.image_size[0], self.image_size[1], 3) \
        shape = (self.batch_size, self.image_size[0], self.image_size[1], 4) \
            if self.data_format == 'NHWC' else (self.batch_size, 3, self.image_size[0], self.image_size[1])
        print("shape:"+str(shape))
        images_placeholder = tf.placeholder(
            tf.float32,
            shape=shape,
            name="images_placeholder")
        labels_placeholder = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.image_size[0] * 2, self.image_size[1] * 2,3),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def inference(self, images, is_training):
        base = self.base(images, is_training)
        return tf.identity(base, name="output")


    def summary(self, output, labels=None):
        output_transposed = output if self.data_format == 'NHWC' else tf.transpose(output, perm=[0, 2, 3, 1])
        images = self.images if self.data_format == 'NHWC' else tf.transpose(self.images, perm=[0, 2, 3, 1])

        disp_rg = images[:,:,:,:2]
        disp_b = tf.expand_dims(images[:,:,:,3],axis=3)
        disp = tf.concat([disp_rg,disp_b],3)

        tf.summary.image("input", disp * 255)
        tf.summary.image("output", tf.cast(output * 255., tf.uint8))
        tf.summary.image("gt", tf.cast(labels * 255., tf.uint8))
        tf.summary.image("diff", (output - tf.cast(labels, tf.float32)) * 255)

        return

    def metrics(self, output, labels):
        output_transposed = output if self.data_format == 'NHWC' else tf.transpose(output, perm=[0, 2, 3, 1])

        results = {}
        updates = []
        with tf.name_scope('metrics_cals'):
            output_argmax = tf.argmax(output_transposed, axis=3)

            # accuracy, accuracy_update = tf.metrics.accuracy(labels, output_argmax)
            accuracy, accuracy_update = tf.metrics.accuracy(labels, output_transposed)
            results["accuracy"] = accuracy
            updates.append(accuracy_update)

            ious = []
            accs = []

            updates_op = tf.group(*updates)

            return results, updates_op


class SegnetBase(Base):
    """Base network class for LMSegnetV0 and LMSegnetV1.

    In loss function, multiply the ratio of the class frequency on the batch.
    The ratio is difference from `median  frequency  balancing` described in [SegNet](https://arxiv.org/pdf/1511.00561.pdf).
    """ # NOQA

    def __init__(
            self,
            weight_decay_rate=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.weight_decay_rate = weight_decay_rate

    def loss(self, output, labels):
        """Loss

        Args:
           output: Tensor of network output. shape is (batch_size, output_height, output_width, num_classes).
           labels: Tensor of grayscale imnage gt labels. shape is (batch_size, height, width).
        """
        if self.data_format == 'NCHW':
            output = tf.transpose(output, perm=[0, 2, 3, 1])
        with tf.name_scope("loss"):
            # calculate loss weights for each class.
            loss_weight = []
            all_size = tf.to_float(tf.reduce_prod(tf.shape(labels)))
            for class_index in range(self.num_classes):
                num_label = tf.reduce_sum(tf.to_float(tf.equal(labels, class_index)))
                weight = (all_size - num_label) / all_size
                loss_weight.append(weight)

            loss_weight = tf.Print(loss_weight, loss_weight, message="loss_weight:")

            reshape_output = tf.reshape(output, (-1, self.num_classes))
            # import pdb;pdb.set_trace()

            # label_flat = tf.reshape(labels, (-1, 1))
            # labels = tf.reshape(tf.one_hot(label_flat, depth=self.num_classes), (-1, self.num_classes))
            # softmax = tf.nn.softmax(reshape_output)
            G_loss = tf.reduce_mean(tf.abs(output - tf.cast(labels,tf.float32)))
            # cross_entropy = -tf.reduce_sum(
            #     (labels * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0))) * loss_weight,
            #     axis=[1]
            # )

            # cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
            # loss = cross_entropy_mean
            loss = G_loss
            if self.weight_decay_rate:
                weight_decay_loss = self._weight_decay_loss()
                tf.summary.scalar("weight_decay", weight_decay_loss)
                loss = loss + weight_decay_loss

            tf.summary.scalar("loss", loss)
            return loss

    def _weight_decay_loss(self):
        """L2 weight decay (regularization) loss."""
        losses = []
        print("apply l2 loss these variables")
        for var in tf.trainable_variables():

            # exclude batch norm variable
            if "kernel" in var.name:
                print(var.name)
                losses.append(tf.nn.l2_loss(var))

        return tf.add_n(losses) * self.weight_decay_rate
