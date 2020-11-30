# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from approachs.model import Model
from utils.measure import gauc, pv_auc, ndcg
from approachs.rl_model import RLModel

class SLModel(RLModel):
    def _build_loss(self):
        self.lr = 1e-4
        self.gamma = 1
        discount = 1.0 / np.log2(np.arange(2, self.params.slate_size+2)).reshape((-1, self.params.slate_size))
        if self.params.loss == 'ce':
            prob_mask = self.act_probs_train_mask
            label_t = tf.reshape(tf.tile(self.item_label, (1, self.params.slate_size)), [-1, self.item_size])
            # label_mask = self.mask_in * label_t
            # label_mask = label_mask / tf.clip_by_value(tf.reduce_sum(label_mask, -1, keep_dims=True), 1.0, 100.0)
            label_mask = tf.nn.softmax(tf.add(tf.multiply(1.0 - self.mask_in, -1.0e9), label_t))
            ce = -1 * tf.reduce_mean(label_mask * tf.log(tf.clip_by_value(prob_mask, 1e-9, 1.0)), axis=-1)
            ce = tf.reshape(ce, (-1, self.params.slate_size))
            ce = tf.multiply(ce, discount)
            self.total_loss = tf.reduce_mean(tf.reduce_sum(ce, axis=1))
        elif self.params.loss == 'hinge':
            logtis = self.act_logits_train
            label_t = tf.reshape(tf.tile(self.item_label, (1, self.params.slate_size)), [-1, self.item_size])
            mask_1, mask_0 = label_t, 1 - label_t
            min_label_1 = tf.reduce_min(logtis + (1 - mask_1) * 1.0e9 + (1. - self.mask_in) * 1.0e9, -1)
            max_label_0 = tf.reduce_max(logtis + (1 - mask_0) * -1.0e9 + (1. - self.mask_in) * -1.0e9, -1)
            hg = tf.maximum(0.0, 1 - min_label_1 + max_label_0)
            hg = tf.reshape(hg, (-1, self.params.slate_size))
            hg = tf.multiply(hg, discount)
            self.total_loss = tf.reduce_mean(tf.reduce_sum(hg, axis=1))
        else:
            raise ValueError('No loss.')

        entropy = -tf.reduce_sum(self.act_probs_train_mask *
                                 tf.log(tf.clip_by_value(self.act_probs_train_mask, 1e-10, 1.0)), axis=-1)
        self.entropy = tf.reduce_mean(entropy)  # mean of entropy of pi(obs)

        # learning rate decay
        self.learning_rate = tf.train.exponential_decay(
                        learning_rate=self.params.learning_rate,
                        global_step=self.global_step,
                        decay_steps=self.params.decay_steps,
                        decay_rate=self.params.decay_rate,
                        staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

    def _build_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar(name="entropy", tensor=self.entropy, collections=['train'])
            tf.summary.scalar(name="total_loss", tensor=self.total_loss, collections=['train'])
            tf.summary.scalar(name='gauc', tensor=self.item_gauc, collections=['test'])
            tf.summary.scalar(name='ndcg', tensor=self.item_ndcg, collections=['test'])
        train_merged = tf.summary.merge_all('train')
        test_merged = tf.summary.merge_all('test')
        return train_merged, test_merged

    def train(self, item_input, item_label, raw_dec_input, act_mask):
        with self.graph.as_default():
            raw_dec_input = raw_dec_input.reshape([-1, self.params.feature_size_vis])
            act_mask = act_mask.reshape([-1, self.params.slate_size])
            
            _, total_loss, summary, step = self.sess.run(
                [self.train_op, self.total_loss, self.train_merged, self.global_step],
                feed_dict={
                    self.item_input: item_input,
                    self.item_label: item_label, 
                    self.raw_dec_input: raw_dec_input,
                    self.mask_in: act_mask,
                    self.train_phase: True
                })
            return total_loss, summary, step

    def evaluate(self, item_input, item_label, raw_dec_input, act_mask):
        with self.graph.as_default():
            raw_dec_input = raw_dec_input.reshape([-1, self.params.feature_size_vis])
            act_mask = act_mask.reshape([-1, self.params.slate_size])
            
            total_loss, summary, step = self.sess.run(
                [self.total_loss, self.train_merged, self.global_step],
                feed_dict={
                    self.item_input: item_input,
                    self.item_label: item_label, 
                    self.raw_dec_input: raw_dec_input,
                    self.mask_in: act_mask,
                    self.train_phase: True
                })
            return total_loss, summary, step



