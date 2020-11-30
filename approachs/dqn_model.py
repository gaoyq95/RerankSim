# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from approachs.model import Model
from approachs.rl_model import RLModel
from utils.measure import gauc, pv_auc, ndcg


class DQNModel(RLModel):

    def _build_graph(self):
        self.lstm_hidden_units = 32
        self.sample_val = 0.2

        with tf.variable_scope("input"):
            self.train_phase = tf.placeholder(tf.bool, name="training")
            self.sample_phase = tf.placeholder(tf.bool, name="sample_phase")            # True
            self.mask_in_raw = tf.placeholder(tf.float32, [None])                       # 选出的商品mask
            self.item_input = tf.placeholder(tf.float32, [None, self.params.slate_size, self.params.feature_size_vis])  #
            self.item_label = tf.placeholder(tf.float32, [None, self.params.slate_size])
            item_features = self.item_input

            self.item_size = self.params.slate_size
            self.mask_in = tf.reshape(self.mask_in_raw, [-1, self.item_size])                               # （B, pv_size）

            self.enc_input = tf.reshape(item_features, [-1, self.item_size, self.params.feature_size_vis])  # (B, pv_size, n)
            self.full_item_fts = self.enc_input
            self.pv_item_fts = tf.reshape(self.full_item_fts, (-1, self.full_item_fts.shape[-1]))           # (B * pv_size, n)

            self.raw_dec_input = tf.placeholder(tf.float32, [None, self.params.feature_size_vis])           # (B*pv_size, n)
            self.dec_input = self.raw_dec_input

        with tf.variable_scope("encoder"):
            # (B, pv_size, n) -> (B*pv_size, pv_size, n)
            enc_input_train = tf.reshape(tf.tile(self.enc_input, (1, self.params.slate_size, 1)), [-1, self.item_size, self.params.feature_size_vis])

            enc_input = tf.cond(self.train_phase, lambda: enc_input_train, lambda: self.enc_input)
            self.enc_outputs = self.get_dnn(enc_input, [32, 16], [tf.nn.relu, tf.nn.relu], "enc_dnn")       # (B*15, 15, n)

        with tf.variable_scope("encoder_state"):
            cell_dec = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_units)

        with tf.variable_scope("decoder"):
            # for training
            # (B, pv_size, n)
            dec_input = tf.reshape(self.dec_input, [-1, self.params.slate_size, self.params.feature_size_vis])
            zero_input = tf.zeros_like(dec_input[:, :1, :])                                                 # (B, 1, n)
            dec_input = tf.concat([zero_input, dec_input[:, :-1, :]], axis=1)                               # (B, pv_size, n)

            zero_state = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)
            new_dec_input = dec_input
            dec_outputs_train, _ = tf.nn.dynamic_rnn(cell_dec, inputs=new_dec_input, time_major=False, initial_state=zero_state)  # (B, 15, 64)
            dec_outputs_train = tf.reshape(dec_outputs_train, [-1, 1, self.lstm_hidden_units])              # (B, 1, 64)
            dec_outputs_train_tile = tf.tile(dec_outputs_train, [1, self.item_size, 1])                     # (B*15, 15, 64)

            x = tf.concat([self.enc_outputs, dec_outputs_train_tile], axis=-1)                              # (B*15, 15, n)
            # (B*15, 15)
            self.q_pred_train = tf.reshape(self.get_dnn(x, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"), [-1, self.item_size])

            self.q_pred_train_mask = tf.add(tf.multiply(1. - self.mask_in, -1.0e9), self.q_pred_train)

            # for predicting
            dec_input = tf.zeros([tf.shape(self.item_input)[0], self.params.feature_size_vis])              # (B, n)

            dec_states = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)
            mask_tmp = tf.ones([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)            # (B, 15)

            mask_list = []
            act_idx_list = []
            q_pred_one_list = []
            q_pred_all_list = []
            next_state_list = []
            scores_pred = tf.zeros([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)

            random_val = tf.random_uniform([], 0, 1)
            for k in range(self.params.slate_size):
                new_dec_input = dec_input

                dec_outputs, dec_states = cell_dec(new_dec_input, dec_states)
                mask_list.append(mask_tmp)

                dec_outputs_tile = tf.tile(tf.reshape(dec_outputs, [-1, 1, dec_outputs.shape[-1]]), [1, self.item_size, 1])

                x = tf.concat([self.enc_outputs, dec_outputs_tile], axis=-1)
                q_pred = tf.reshape(self.get_dnn(x, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"), [-1, self.item_size])  # （B,15）

                q_pred_mask = tf.add(tf.multiply(1. - mask_tmp, -1.0e9), q_pred)  # (B, 15)
                act_argmax = tf.reshape(tf.argmax(q_pred_mask, axis=-1), [-1])

                act_probs_mask_random = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), mask_tmp))
                act_random = tf.reshape(tf.multinomial(tf.log(act_probs_mask_random), num_samples=1), [-1])

                # act_det = tf.argmax(act_probs_mask, axis=1)
                # act_idx_out = tf.cond(self.sample_phase, lambda: act_stoc, lambda: act_det)
                act_idx_out = tf.cond(self.sample_phase, lambda: tf.cond(random_val < self.sample_val,
                                                                         lambda: act_random,
                                                                         lambda: act_argmax),
                                      lambda: act_argmax)

                tmp_range = tf.cast(tf.range(tf.shape(self.item_input)[0], dtype=tf.int32), tf.int64)
                idx_pair = tf.stack([tmp_range, act_idx_out], axis=1)
                indx_pair_argmax = tf.stack([tmp_range, act_argmax], axis=1)

                idx_one_hot = tf.one_hot(act_idx_out, self.item_size)

                mask_tmp = mask_tmp - idx_one_hot
                dec_input = tf.gather_nd(self.enc_input, idx_pair)  # 选出的商品 （B, 1）
                next_full_state = tf.gather_nd(self.full_item_fts, idx_pair)  # 选出的商品 （B, 1）
                # q_pred_one = tf.gather_nd(q_pred_mask, indx_pair)  # 每次选取的最大Q值（带探索的）
                q_pred_one = tf.gather_nd(q_pred_mask, indx_pair_argmax)  # 每次选取的最大Q值

                act_idx_list.append(act_idx_out)
                q_pred_one_list.append(q_pred_one)  # 当前步所选商品的q值
                q_pred_all_list.append(q_pred_mask)  # 当前步所有商品的q值
                next_state_list.append(next_full_state)  # 选出商品的特征

                scores_pred = scores_pred + tf.cast(idx_one_hot, dtype=tf.float32) * (1 - k * 0.03)

            self.mask_arr = tf.stack(mask_list, axis=1)
            self.act_idx_out = tf.stack(act_idx_list, axis=1)
            self.q_pred_one = tf.stack(q_pred_one_list, axis=1)
            self.q_pred_all = tf.stack(q_pred_all_list, axis=1)
            self.next_state_out = tf.reshape(tf.stack(next_state_list, axis=1), [-1, self.full_item_fts.shape[-1]])  # （B*15, n）

            self.rerank_predict = tf.identity(tf.reshape(scores_pred, [-1, 1]), 'rerank_predict')

        self.set_global_step(tf.train.create_global_step())
        self.set_saver(tf.train.Saver())
        with tf.variable_scope("loss"):
            self._build_loss()
            self.item_gauc = gauc(self.rerank_predict, tf.reshape(self.item_label, (-1, 1)), self.item_size)
            self.item_ndcg = ndcg(p=tf.reshape(self.rerank_predict, ((-1, self.item_size))),
                                  l=tf.reshape(self.item_label, (-1, self.item_size)),
                                  k=self.item_size)
        self.train_merged, self.test_merged = self._build_summary()

    def predict(self, item_input, item_label, sample_phase=True, train_phase=False):
        with self.graph.as_default():
            act_idx_out, q_pred_one, next_state_out, mask_arr, pv_item_fts, rerank_predict, \
            item_gauc, item_ndcg, test_summary = self.sess.run(
                [self.act_idx_out, self.q_pred_one, self.next_state_out, self.mask_arr,
                 self.pv_item_fts, self.rerank_predict, self.item_gauc, self.item_ndcg, self.test_merged],
                feed_dict={self.item_input: item_input,
                           self.train_phase: train_phase,
                           self.sample_phase: sample_phase,
                           self.item_label: item_label})
            return act_idx_out, q_pred_one, next_state_out, mask_arr, pv_item_fts, rerank_predict, \
                   item_gauc, item_ndcg, test_summary

    def _build_loss(self):
        with tf.variable_scope("train_input"):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.returns = tf.placeholder(dtype=tf.float32, shape=[None, self.params.slate_size], name='returns')

        act_idx_one_hot = tf.one_hot(indices=self.actions, depth=self.item_size),
        cur_q_pred = tf.reshape(tf.reduce_sum(self.q_pred_train_mask * act_idx_one_hot, axis=-1), [-1, self.params.slate_size])

        # construct computation graph for loss
        self.total_loss = tf.reduce_mean(tf.square(cur_q_pred - self.returns))

        self.mean_returns = tf.reduce_mean(self.returns[:, 0])

        # learning rate decay
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.params.learning_rate,
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
            tf.summary.scalar(name="total_loss", tensor=self.total_loss, collections=['train'])
            tf.summary.scalar(name="returns", tensor=self.mean_returns, collections=['train'])
            tf.summary.scalar(name='gauc', tensor=self.item_gauc, collections=['test'])
            tf.summary.scalar(name='ndcg', tensor=self.item_ndcg, collections=['test'])

        train_merged = tf.summary.merge_all('train')
        test_merged = tf.summary.merge_all('test')
        return train_merged, test_merged

    def train(self, item_input, raw_dec_input, old_q_pred, actions, rewards, act_mask, c_entropy):
        with self.graph.as_default():
            returns = self.get_returns(rewards, old_q_pred)  # (b, 15)
            raw_dec_input = raw_dec_input.reshape([-1, self.params.feature_size_vis])
            actions = actions.reshape([-1])

            # train
            _, total_loss, mean_return, summary, step = self.sess.run(
                [self.train_op, self.total_loss, self.mean_returns, self.train_merged, self.global_step],
                feed_dict={self.item_input: item_input,
                           self.raw_dec_input: raw_dec_input,
                           self.actions: actions,
                           self.mask_in_raw: act_mask.reshape([-1]),
                           self.returns: returns,
                           self.train_phase: True
                           })
            return total_loss, mean_return, summary, step

    def get_returns(self, rewards, old_q_pred):
        long_reward = np.zeros(rewards.shape)
        val = 0
        for i in reversed(range(self.params.slate_size)):
            long_reward[:, i] = self.params.gamma * val + rewards[:, i]
            val = long_reward[:, i]
        returns = long_reward[:, 0]

        long_reward = np.zeros(rewards.shape)
        old_q_pred = old_q_pred.reshape([-1, self.params.slate_size])
        for i in range(self.params.slate_size - 1):
            long_reward[:, i] = rewards[:, i] + self.params.gamma * old_q_pred[:, i + 1]
        return long_reward


# MontCarlo
class MontCarloModel(RLModel):

    def _build_graph(self):
        self.lstm_hidden_units = 32
        self.sample_val = 0.2

        with tf.variable_scope("input"):
            self.train_phase = tf.placeholder(tf.bool, name="training")
            self.sample_phase = tf.placeholder(tf.bool, name="sample_phase")
            self.mask_in_raw = tf.placeholder(tf.float32, [None])
            self.item_input = tf.placeholder(tf.float32, [None, self.params.slate_size, self.params.feature_size_vis])  #
            self.item_label = tf.placeholder(tf.float32, [None, self.params.slate_size])
            item_features = self.item_input

            self.item_size = self.params.slate_size
            self.mask_in = tf.reshape(self.mask_in_raw, [-1, self.item_size])  # （B, pv_size）

            self.enc_input = tf.reshape(item_features, [-1, self.item_size, self.params.feature_size_vis])  # (B, pv_size, n)
            self.full_item_fts = self.enc_input
            self.pv_item_fts = tf.reshape(self.full_item_fts, (-1, self.full_item_fts.shape[-1]))  # (B * pv_size, n)

            self.raw_dec_input = tf.placeholder(tf.float32, [None, self.params.feature_size_vis])  # (B*pv_size, n)
            self.dec_input = self.raw_dec_input

        with tf.variable_scope("encoder"):
            enc_input_train = tf.reshape(tf.tile(self.enc_input, (1, self.params.slate_size, 1)), [-1, self.item_size, self.params.feature_size_vis])

            enc_input = tf.cond(self.train_phase, lambda: enc_input_train, lambda: self.enc_input)
            self.enc_outputs = self.get_dnn(enc_input, [32, 16], [tf.nn.relu, tf.nn.relu], "enc_dnn")

        with tf.variable_scope("encoder_state"):
            cell_dec = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_units)

        with tf.variable_scope("decoder"):
            # for training
            dec_input = tf.reshape(self.dec_input, [-1, self.params.slate_size, self.params.feature_size_vis])
            zero_input = tf.zeros_like(dec_input[:, :1, :])  # (B, 1, n)
            dec_input = tf.concat([zero_input, dec_input[:, :-1, :]], axis=1)  # (B, pv_size, n)

            zero_state = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)
            new_dec_input = dec_input
            dec_outputs_train, _ = tf.nn.dynamic_rnn(cell_dec, inputs=new_dec_input, time_major=False, initial_state=zero_state)  # (B, 15, 64)
            dec_outputs_train = tf.reshape(dec_outputs_train, [-1, 1, self.lstm_hidden_units])  # (B, 1, 64)
            dec_outputs_train_tile = tf.tile(dec_outputs_train, [1, self.item_size, 1])  # (B*15, 15, 64)

            x = tf.concat([self.enc_outputs, dec_outputs_train_tile], axis=-1)  # (B*15, 15, n)
            # (B*15, 15)
            self.q_pred_train = tf.reshape(self.get_dnn(x, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"), [-1, self.item_size])

            self.q_pred_train_mask = tf.add(tf.multiply(1. - self.mask_in, -1.0e9), self.q_pred_train)

            # for predicting
            dec_input = tf.zeros([tf.shape(self.item_input)[0], self.params.feature_size_vis])  # (B, n)

            dec_states = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)
            mask_tmp = tf.ones([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)  # (B, 15)

            mask_list = []
            act_idx_list = []
            q_pred_one_list = []
            q_pred_all_list = []
            next_state_list = []
            scores_pred = tf.zeros([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)

            random_val = tf.random_uniform([], 0, 1)
            for k in range(self.params.slate_size):
                new_dec_input = dec_input

                dec_outputs, dec_states = cell_dec(new_dec_input, dec_states)
                mask_list.append(mask_tmp)

                dec_outputs_tile = tf.tile(tf.reshape(dec_outputs, [-1, 1, dec_outputs.shape[-1]]), [1, self.item_size, 1])

                x = tf.concat([self.enc_outputs, dec_outputs_tile], axis=-1)
                q_pred = tf.reshape(self.get_dnn(x, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"), [-1, self.item_size])  # （B,15）

                q_pred_mask = tf.add(tf.multiply(1. - mask_tmp, -1.0e9), q_pred)  # (B, 15)
                act_argmax = tf.reshape(tf.argmax(q_pred_mask, axis=-1), [-1])

                act_probs_mask_random = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), mask_tmp))
                act_random = tf.reshape(tf.multinomial(tf.log(act_probs_mask_random), num_samples=1), [-1])

                act_idx_out = tf.cond(self.sample_phase, lambda: tf.cond(random_val < self.sample_val,
                                                                         lambda: act_random,
                                                                         lambda: act_argmax),
                                      lambda: act_argmax)

                tmp_range = tf.cast(tf.range(tf.shape(self.item_input)[0], dtype=tf.int32), tf.int64)
                idx_pair = tf.stack([tmp_range, act_idx_out], axis=1)

                idx_one_hot = tf.one_hot(act_idx_out, self.item_size)

                mask_tmp = mask_tmp - idx_one_hot
                dec_input = tf.gather_nd(self.enc_input, idx_pair)
                next_full_state = tf.gather_nd(self.full_item_fts, idx_pair)
                q_pred_one = tf.gather_nd(q_pred_mask, idx_pair)

                act_idx_list.append(act_idx_out)
                q_pred_one_list.append(q_pred_one)
                q_pred_all_list.append(q_pred_mask)
                next_state_list.append(next_full_state)

                scores_pred = scores_pred + tf.cast(idx_one_hot, dtype=tf.float32) * (1 - k * 0.03)

            self.mask_arr = tf.stack(mask_list, axis=1)
            self.act_idx_out = tf.stack(act_idx_list, axis=1)
            self.q_pred_one = tf.stack(q_pred_one_list, axis=1)
            self.q_pred_all = tf.stack(q_pred_all_list, axis=1)
            self.next_state_out = tf.reshape(tf.stack(next_state_list, axis=1), [-1, self.full_item_fts.shape[-1]])  # （B*15, n）

            self.rerank_predict = tf.identity(tf.reshape(scores_pred, [-1, 1]), 'rerank_predict')

        self.set_global_step(tf.train.create_global_step())
        self.set_saver(tf.train.Saver())
        with tf.variable_scope("loss"):
            self._build_loss()
            self.item_gauc = gauc(self.rerank_predict, tf.reshape(self.item_label, (-1, 1)), self.item_size)
            self.item_ndcg = ndcg(p=tf.reshape(self.rerank_predict, ((-1, self.item_size))),
                                  l=tf.reshape(self.item_label, (-1, self.item_size)),
                                  k=self.item_size)
        self.train_merged, self.test_merged = self._build_summary()

    def predict(self, item_input, item_label, sample_phase=True, train_phase=False):
        with self.graph.as_default():
            act_idx_out, q_pred_one, next_state_out, mask_arr, pv_item_fts, rerank_predict, \
            item_gauc, item_ndcg, test_summary = self.sess.run(
                [self.act_idx_out, self.q_pred_one, self.next_state_out, self.mask_arr,
                 self.pv_item_fts, self.rerank_predict, self.item_gauc, self.item_ndcg, self.test_merged],
                feed_dict={self.item_input: item_input,
                           self.train_phase: train_phase,
                           self.sample_phase: sample_phase,
                           self.item_label: item_label})
            return act_idx_out, q_pred_one, next_state_out, mask_arr, pv_item_fts, rerank_predict, \
                   item_gauc, item_ndcg, test_summary

    def _build_loss(self):
        with tf.variable_scope("train_input"):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.returns = tf.placeholder(dtype=tf.float32, shape=[None, self.params.slate_size], name='returns')

        act_idx_one_hot = tf.one_hot(indices=self.actions, depth=self.item_size),
        cur_q_pred = tf.reshape(tf.reduce_sum(self.q_pred_train_mask * act_idx_one_hot, axis=-1), [-1, self.params.slate_size])

        # construct computation graph for loss
        self.total_loss = tf.reduce_mean(tf.square(cur_q_pred - self.returns))

        self.mean_returns = tf.reduce_mean(self.returns[:, 0])

        # learning rate decay
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.params.learning_rate,
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
            tf.summary.scalar(name="total_loss", tensor=self.total_loss, collections=['train'])
            tf.summary.scalar(name="returns", tensor=self.mean_returns, collections=['train'])
            tf.summary.scalar(name='gauc', tensor=self.item_gauc, collections=['test'])
            tf.summary.scalar(name='ndcg', tensor=self.item_ndcg, collections=['test'])

        train_merged = tf.summary.merge_all('train')
        test_merged = tf.summary.merge_all('test')
        return train_merged, test_merged

    def train(self, item_input, raw_dec_input, old_q_pred, actions, rewards, act_mask, c_entropy):
        with self.graph.as_default():
            returns = self.get_returns(rewards)  # (b, 15)
            raw_dec_input = raw_dec_input.reshape([-1, self.params.feature_size_vis])
            actions = actions.reshape([-1])

            # train
            _, total_loss, mean_return, summary, step = self.sess.run(
                [self.train_op, self.total_loss, self.mean_returns, self.train_merged, self.global_step],
                feed_dict={self.item_input: item_input,
                           self.raw_dec_input: raw_dec_input,
                           self.actions: actions,
                           self.mask_in_raw: act_mask.reshape([-1]),
                           self.returns: returns,
                           self.train_phase: True
                           })
            return total_loss, mean_return, summary, step

    def get_returns(self, rewards):
        long_reward, returns = self.get_long_reward(rewards)
        return long_reward


