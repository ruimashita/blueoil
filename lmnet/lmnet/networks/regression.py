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

import tensorflow as tf
from matplotlib import cm

from lmnet.blocks import lmnet_block
from lmnet.networks.base import BaseNetwork


class Base(BaseNetwork):
    """base network for classification

    This base network is for classification.
    Every classification's network class should extend this class.

    """

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

    def placeholderes(self):
        """Placeholders.

        Return placeholders.

        Returns:
            tf.placeholder: Placeholders.
        """

        shape = (self.batch_size, self.image_size[0], self.image_size[1], 3) \
            if self.data_format == 'NHWC' else (self.batch_size, 3, self.image_size[0], self.image_size[1])
        images_placeholder = tf.placeholder(
            tf.float32,
            shape=shape,
            name="images_placeholder")

        labels_placeholder = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.num_classes),
            name="labels_placeholder")

        return images_placeholder, labels_placeholder

    def inference(self, images, is_training):
        """inference.

        Params:
           images: images tensor. shape is (batch_num, height, width, channel)
        """
        base = self.base(images, is_training)
        self.output = tf.identity(base, name="output")
        return self.output

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

    def loss(self, output, labels):
        """loss.

        Params:
           output: softmaxed tensor from base. shape is (batch_num, num_classes)
           labels: onehot labels tensor. shape is (batch_num, num_classes)
        """

        with tf.name_scope("loss"):
            labels = tf.to_float(labels)
            cross_entropy = tf.reduce_sum(
                (labels - output)**2,
                axis=[1]
            )

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="mean")
            tf.summary.scalar("mse", cross_entropy_mean)
            loss = cross_entropy_mean

            if self.weight_decay_rate:
                weight_decay_loss = self._weight_decay_loss()
                tf.summary.scalar("weight_decay", weight_decay_loss)
                loss = loss + weight_decay_loss

            tf.summary.scalar("loss", loss)

            return loss

    def _heatmaps(self, target_feature_map):
        """Generate heatmap from target feature map.

        Args:
            target_feature_map (Tensor): Tensor to be generate heatmap. shape is [batch_size, h, w, num_classes].
        """
        assert target_feature_map.get_shape()[3].value == self.num_classes

        results = []

        # shpae: [batch_size, height, width, num_classes]
        heatmap = tf.image.resize_images(
            target_feature_map, [self.image_size[0], self.image_size[1]],
            method=tf.image.ResizeMethod.BICUBIC,
        )
        epsilon = 1e-10
        # standrization. all element are in the interval [0, 1].
        heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap) + epsilon)

        for i, class_name in enumerate(self.classes):
            class_heatmap = heatmap[:, :, :, i]
            indices = tf.to_int32(tf.round(class_heatmap * 255))
            color_map = cm.jet
            # Init color map for useing color lookup table(_lut).
            color_map._init()
            colors = tf.constant(color_map._lut[:, :3], dtype=tf.float32)
            # gather
            colored_class_heatmap = tf.gather(colors, indices)
            results.append(colored_class_heatmap)

        return results

    def summary(self, output, labels=None):
        super().summary(output, labels)

        images = self.images if self.data_format == 'NHWC' else tf.transpose(self.images, perm=[0, 2, 3, 1])

        tf.summary.image("input_images", images)

        if hasattr(self, "_heatmap_layer") and isinstance(self._heatmap_layer, tf.Tensor):
            heatmap_layer = self._heatmap_layer if self.data_format == 'NHWC' else tf.transpose(self._heatmap_layer,
                                                                                                perm=[0, 2, 3, 1])
            with tf.variable_scope('heatmap'):
                colored_class_heatmaps = self._heatmaps(heatmap_layer)
                for class_name, colored_class_heatmap in zip(self.classes, colored_class_heatmaps):
                    alpha = 0.1
                    overlap = alpha * images + colored_class_heatmap
                    tf.summary.image(class_name, overlap, max_outputs=1)

    def metrics(self, output, labels):
        """metrics.

        Params:
           softmax: probabilities applied softmax. shape is (batch_num, num_classes)
           labels: onehot labels tensor. shape is (batch_num, num_classes)
        """
        with tf.name_scope("metrics_calc"):
            labels = tf.to_float(labels)

            accuracy, update = tf.metrics.mean_squared_error(labels, output)

        metrics_dict = {"mse": accuracy}
        return metrics_dict, update


class LmnetV1(Base):
    """Lmnet v1 for classification.
    """
    version = 1.0

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu
        self.custom_getter = None

    def _get_lmnet_block(self, is_training, channels_data_format):
        return functools.partial(lmnet_block,
                                 activation=self.activation,
                                 custom_getter=self.custom_getter,
                                 is_training=is_training,
                                 is_debug=self.is_debug,
                                 use_bias=False,
                                 data_format=channels_data_format)

    def _space_to_depth(self, inputs=None, block_size=2, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])

        output = tf.space_to_depth(inputs, block_size=block_size, name=name)

        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        _lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        self.images = images

        x = _lmnet_block('conv1', images, 32, 3)
        x = self._space_to_depth(name='pool2', inputs=x)
        x = _lmnet_block('conv2', x, 32, 3)
        x = _lmnet_block('conv3', x, 32, 3)
        x = self._space_to_depth(name='pool4', inputs=x)
        x = _lmnet_block('conv4', x, 48, 3)
        x = _lmnet_block('conv5', x, 48, 3)
        x = self._space_to_depth(name='pool5', inputs=x)
        x = _lmnet_block('conv6', x, 64, 3)
        x = _lmnet_block('conv7', x, 64, 3)

        x = tf.layers.dropout(x, training=is_training)

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        x = tf.layers.conv2d(name='conv8',
                             inputs=x,
                             filters=self.num_classes,
                             kernel_size=1,
                             kernel_initializer=kernel_initializer,
                             activation=None,
                             use_bias=True,
                             data_format=channels_data_format)

        self._heatmap_layer = x

        h = x.get_shape()[1].value if self.data_format == 'NHWC' else x.get_shape()[2].value
        w = x.get_shape()[2].value if self.data_format == 'NHWC' else x.get_shape()[3].value
        x = tf.layers.average_pooling2d(name='pool7',
                                        inputs=x,
                                        pool_size=[h, w],
                                        padding='VALID',
                                        strides=1,
                                        data_format=channels_data_format)

        self.base_output = tf.reshape(x, [-1, self.num_classes], name='pool7_reshape')

        return self.base_output


class LmnetV1Quantize(LmnetV1):
    """Lmnet quantize network for classification, version 1.0

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater. See more at `lmnet.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater. See more at `lmnet.quantizations`.
        weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.
    """
    version = 1.0

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:

                if var.op.name.startswith("conv1/"):
                    return var

                if var.op.name.startswith("conv8/"):
                    return var

                return weight_quantization(var)
        return var
