# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from approachs.model import Model
from approachs.dnn_model import DNNModel


class ListWise(DNNModel):

    def _build_net(self):
        x_input = self._get_feature()                   # global feature
        # dnn
        # layers = [1024, 512, 128, 1]
        layers = [128, 64, 1]
        activation = [tf.nn.relu for _ in range(len(layers)-1)] + [tf.nn.sigmoid]
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
        pv_size = self.params.slate_size
        batch_size = tf.shape(self.y)[0] // self.params.slate_size
        logits = x_input
        ground_truth = tf.reshape(self.y, (-1, pv_size))
        logits = tf.reshape(logits, (-1, pv_size))

        if self.params.loss == 'listNet':
            score_truth, score_logits = tf.exp(ground_truth), tf.exp(logits)
            prob_truth = self.listwise_prob(score_truth, pv_size, batch_size)
            prob_pred = self.listwise_prob(score_logits, pv_size, batch_size)

            # loss = -1 * tf.reduce_mean(tf.multiply(prob_truth, tf.log(tf.clip_by_value(prob_pred, 1e-5, 1.0))))
            loss = tf.reduce_mean(
                tf.reduce_sum(-1 * tf.multiply(prob_truth, tf.log(tf.clip_by_value(prob_pred, 1e-5, 1.0))), axis=[1, 2]))
        elif self.params.loss == 'listMLE':
            sorted_id = tf.reshape(tf.nn.top_k(ground_truth, k=pv_size, sorted=True).indices, (-1, 1))
            pv_id = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), 1), (1, pv_size)), (-1, 1))
            sorted_indices = tf.concat([pv_id, sorted_id], -1)

            logits = tf.reshape(tf.gather_nd(logits, sorted_indices), (-1, pv_size))

            score_logits = tf.exp(logits)
            score_sum = []
            for i in range(pv_size):
                score_sum.append(tf.reduce_sum(score_logits[:, i:], -1, keep_dims=True))
            score_sum = tf.concat(score_sum, axis=-1)

            likelihood = tf.reduce_sum(logits - tf.log(score_sum), -1)

            loss = -1 * tf.reduce_mean(likelihood)
        else:
            raise NotImplementedError('Not implement loss: %s' % (self.params.loss))

        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss
        return score, loss

    @staticmethod
    def listwise_prob(score, pv_size, batch_size):
        '''
        top-2 probability
        Args:
            score: A `Tensor` with shape [batch_size, slate_size]
            pv_size: slate size
            batch_size: batch size
        Return:
            A `Tensor` with shape [batch_size, slate_size, slate_size]
        '''
        score_first = tf.tile(tf.reshape(score, (-1, pv_size, 1)), (1, 1, pv_size))                             # (B, N, N)
        score_second = tf.tile(tf.reshape(score, (-1, 1, pv_size)), (1, pv_size, 1))                            # (B, N, N)
        sum_first = tf.expand_dims(tf.reduce_sum(score, axis=1, keep_dims=True), -1)                            # (B, 1, 1)
        sum_second = sum_first - score_first                                                                    # (B, N, N)
        prob = tf.multiply(tf.multiply(score_first, 1./sum_first), tf.multiply(score_second, 1./sum_second))    # (B, N, N)
        pairs = tf.ones([batch_size, pv_size, pv_size]) - tf.eye(pv_size, batch_shape=[batch_size])             # (B, N, N)
        pairs = tf.greater(pairs, 0.0)
        return tf.where(pairs, prob, tf.zeros([batch_size, pv_size, pv_size]))

