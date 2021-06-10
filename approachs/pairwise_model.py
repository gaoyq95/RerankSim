# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from approachs.model import Model
from approachs.dnn_model import DNNModel


class PairWise(DNNModel):

    def _build_net(self):
        x_input = self._get_feature()                   # global feature
        # dnn
        layers = [128, 64, 1]
        if self.params.loss == 'logistic':
            activation = [tf.nn.relu for _ in range(len(layers)-1)] + [tf.nn.sigmoid]
        elif self.params.loss == 'hinge':
            activation = [tf.nn.relu for _ in range(len(layers)-1)] + [None]
        else:
            raise NotImplementedError('Not implement loss: %s' % (self.params.loss))
        
        with tf.variable_scope('dnn'):
            for i, (dim, act) in enumerate(zip(layers, activation)):
                x_input = tf.layers.dense(inputs=x_input,
                                          units=dim,
                                          activation=act, 
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params.l2_regu), 
                                          name='layer_'+str(i),
                                          reuse=tf.AUTO_REUSE)
        
        score = tf.reshape(x_input, (-1,), name='score')
        # loss
        slate_size = self.params.slate_size
        logits = x_input
        ground_truth = tf.reshape(self.y, (-1, slate_size))
        logits = tf.reshape(logits, (-1, slate_size))
        label_column = tf.tile(tf.reshape(ground_truth, (-1, 1, slate_size)), (1, slate_size, 1))
        label_row = tf.tile(tf.reshape(ground_truth, (-1, slate_size, 1)), (1, 1, slate_size))
        label_sub = tf.clip_by_value((label_row - label_column), -1., 1.)                       # [-1, 0, 1]
        logits_column = tf.tile(tf.reshape(logits, (-1, 1, slate_size)), (1, slate_size, 1))
        logits_row = tf.tile(tf.reshape(logits, (-1, slate_size, 1)), (1, 1, slate_size))
        logits_sub = logits_row - logits_column                                                  
        if self.params.loss == 'logistic':
            label = (label_sub + 1) * 0.5                                                       # [-1, 0, 1] --> [0, 0.5, 1]
            l2_loss = tf.losses.get_regularization_loss()
            loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=logits_sub, pos_weight=1.0))
            loss += l2_loss
        elif self.params.loss == 'hinge':
            label = label_sub                                                                   # [-1, 0, 1]
            l2_loss = tf.losses.get_regularization_loss()
            loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - label * logits_sub))
            loss += l2_loss
        else:
            raise NotImplementedError('Not implement loss: %s' % (self.params.loss))
        return score, loss

