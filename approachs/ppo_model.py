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


class PPOModel(RLModel):
    def _build_loss(self):
        self.clip_value = 0.1
        self.gamma = self.params.gamma

        with tf.variable_scope("train_input"):
            self.old_act_prob = tf.placeholder(dtype=tf.float32, shape=[None], name='old_act_prob')
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
            self.returns = tf.placeholder(dtype=tf.float32, shape=[None], name='returns')
            self.c_entropy = tf.placeholder(dtype=tf.float32, name='c_entropy')

        act_idx_one_hot = tf.one_hot(indices=self.actions, depth=self.item_size),
        cur_act_prob = tf.reduce_sum(self.act_probs_train_mask * act_idx_one_hot, axis=-1)
        ratios = tf.exp(tf.log(tf.clip_by_value(cur_act_prob, 1e-10, 1.0))
                        - tf.log(tf.clip_by_value(self.old_act_prob, 1e-10, 1.0)))
        self.ratio = ratios
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value,
                                          clip_value_max=1 + self.clip_value)
        loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
        self.loss_clip = -tf.reduce_mean(loss_clip)
        self.mean_gaes = -tf.reduce_mean(self.gaes)

        # construct computation graph for loss of entropy bonus
        entropy = -tf.reduce_sum(self.act_probs_train_mask *
                                 tf.log(tf.clip_by_value(self.act_probs_train_mask, 1e-10, 1.0)), axis=-1)
        self.entropy = tf.reduce_mean(entropy)  # mean of entropy of pi(obs)

        # construct computation graph for loss
        self.total_loss = self.loss_clip - self.c_entropy * self.entropy

        self.g = tf.reduce_mean(self.returns)

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
            tf.summary.scalar(name="policy_loss", tensor=self.loss_clip, collections=['train'])
            tf.summary.scalar(name="total_loss", tensor=self.total_loss, collections=['train'])
            tf.summary.scalar(name="returns", tensor=self.g, collections=['train'])
            tf.summary.scalar(name='gauc', tensor=self.item_gauc, collections=['test'])
            # tf.summary.scalar(name='pauc', tensor=self.item_pauc, collections=['test'])
            tf.summary.scalar(name='ndcg', tensor=self.item_ndcg, collections=['test'])
            tf.summary.histogram(name='ratio', values=self.ratio, collections=['train'])
            tf.summary.histogram(name='gaes', values=self.gaes, collections=['train'])
            tf.summary.scalar(name='mean_gaes', tensor=self.mean_gaes, collections=['train'])

        train_merged = tf.summary.merge_all('train')
        test_merged = tf.summary.merge_all('test')
        return train_merged, test_merged

    def train(self, item_input, raw_dec_input, old_act_prob, actions, rewards, act_mask, c_entropy):
        with self.graph.as_default():
            gaes, returns = self.get_gaes(rewards)
            raw_dec_input = raw_dec_input.reshape([-1, self.params.feature_size_vis])
            old_act_prob = old_act_prob.reshape([-1])
            actions = actions.reshape([-1])
            # train
            _, total_loss, mean_return, summary, step = self.sess.run(
                [self.train_op, self.total_loss, self.g, self.train_merged, self.global_step],
                feed_dict={self.item_input: item_input,
                            self.raw_dec_input: raw_dec_input,
                            self.old_act_prob: old_act_prob,
                            self.actions: actions,
                            self.mask_in_raw: act_mask.reshape([-1]),
                            self.gaes: gaes,
                            self.returns: returns,
                            self.c_entropy: c_entropy, 
                            self.train_phase: True
                            })
            return total_loss, mean_return, summary, step

    def evaluate(self, item_input, raw_dec_input, old_act_prob, actions, rewards, act_mask, c_entropy):
        with self.graph.as_default():
            gaes, returns = self.get_gaes(rewards)
            raw_dec_input = raw_dec_input.reshape([-1, self.params.feature_size_vis])
            old_act_prob = old_act_prob.reshape([-1])
            actions = actions.reshape([-1])
            # train
            total_loss, mean_return, summary, step = self.sess.run(
                [self.total_loss, self.g, self.train_merged, self.global_step],
                feed_dict={self.item_input: item_input,
                            self.raw_dec_input: raw_dec_input,
                            self.old_act_prob: old_act_prob,
                            self.actions: actions,
                            self.mask_in_raw: act_mask.reshape([-1]),
                            self.gaes: gaes,
                            self.returns: returns,
                            self.c_entropy: c_entropy, 
                            self.train_phase: True
                            })
            return total_loss, mean_return, summary, step

    def get_gaes(self, rewards):
        long_reward, returns = self.get_long_reward(rewards)
        gaes = np.reshape(long_reward,
                          [-1, self.params.rep_num, self.params.slate_size])
        gaes_std = gaes.std(axis=1, keepdims=True)
        gaes_std = np.where(gaes_std == 0, 1, gaes_std)
        gaes = (gaes - gaes.mean(axis=1, keepdims=True)) / gaes_std

        return gaes.reshape([-1]), returns.reshape([-1])
