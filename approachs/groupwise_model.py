# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from approachs.model import Model
from approachs.dnn_model import DNNModel


class GroupWise(DNNModel):
    def _build_net(self):
        self.y_ = tf.reshape(self.y, (-1, self.params.slate_size))
        batch_size = tf.shape(self.x)[0] // self.params.slate_size
        x_input = self._get_feature()                   # global feature
        # dnn
        group_size = self.params.group_size
        slate_size = self.params.slate_size
        input_x_bnf = tf.reshape(x_input, [-1, slate_size, x_input.shape[-1]])
        input_x_circle = tf.concat([input_x_bnf[:, :, :], input_x_bnf[:, :group_size-1, :]], axis = 1)
        self.logits = tf.zeros(shape=[batch_size, slate_size, 1], dtype = tf.float32)

        # layers = [1024, 512, 128, 1]
        layers = [128, 64, 1]
        activation = [tf.nn.relu for _ in range(len(layers)-1)] + [tf.nn.sigmoid]
        with tf.variable_scope('dnn'):
            for i in range(slate_size):
                inputs = input_x_circle[:, i: i + group_size, :]
                inputs = tf.reshape(inputs, [-1, inputs.shape[-2] * inputs.shape[-1]])
                for j, (dim, act) in enumerate(zip(layers, activation)):
                    inputs = tf.layers.dense(inputs=inputs,
                                              units=dim,
                                              activation=act, 
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params.l2_regu), 
                                              name='layer_'+str(j),
                                              reuse=tf.AUTO_REUSE)
                logits_inc = tf.reshape(inputs, (-1, group_size, 1))
                zeros_pre = tf.zeros([batch_size, i, 1], dtype = tf.float32)
                zeros_suf = tf.zeros([batch_size, slate_size - i + slate_size - group_size, 1], dtype = tf.float32)
                logits_inc = tf.concat([zeros_pre, logits_inc, zeros_suf], axis = 1)
                self.logits = self.logits + logits_inc[:, :slate_size, :] + logits_inc[:, slate_size:, :]
        
        score = tf.reshape(self.logits, (-1, slate_size))
        # loss
        logtis = tf.exp(tf.reshape(self.logits, (-1, slate_size)))
        logtis_s = logtis / tf.reduce_sum(logtis, 1, keep_dims=True)
        logtis_s = tf.clip_by_value(logtis_s, 1e-8, 1.0)
        label = self.y_
        label_s = label / tf.clip_by_value(tf.reduce_sum(label, 1, keep_dims=True), 1, 999)
        
        losses = -1 * tf.reduce_sum(label_s * tf.log(logtis_s), 1)
        loss = tf.reduce_mean(losses)
        
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss
        return score, loss

