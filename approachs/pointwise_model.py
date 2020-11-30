# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from approachs.model import Model
from approachs.dnn_model import DNNModel


class PointWise(DNNModel):
    
    def _build_net(self):
        x_input = self._get_feature()                   # global feature
        # dnn
        layers = [64, 32, 1]
        activation = [tf.nn.relu for _ in range(len(layers)-1)] + [None]
        
        with tf.variable_scope('dnn'):
            for i, (dim, act) in enumerate(zip(layers, activation)):
                x_input = tf.layers.dense(inputs=x_input,
                                          units=dim,
                                          activation=act, 
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params.l2_regu), 
                                          name='layer_'+str(i),
                                          reuse=tf.AUTO_REUSE)
        # loss
        if self.params.loss == 'mse':
            score = tf.reshape(x_input, (-1,), name='score')
            loss = tf.reduce_mean(tf.square(score - self.y))
        elif self.params.loss == 'ce':
            score = tf.reshape(tf.sigmoid(x_input), (-1,), name='score')
            loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(tf.reshape(self.y, (-1, 1)), tf.reshape(x_input, (-1, 1)), pos_weight=1.0))
        elif self.params.loss == 'hinge':
            score = tf.reshape(x_input, (-1,), name='score')
            
            loss = tf.reduce_mean(tf.losses.hinge_loss(tf.reshape(self.y, (-1, 1)), tf.reshape(x_input, (-1, 1))))
        else:
            raise NotImplementedError('Not implement loss: %s..' % (self.params.loss))
        
        # l2 norm
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss
        return score, loss