# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from approachs.model import Model
from utils.measure import gauc, pv_auc, ndcg


class RLModel(Model):
    def __init__(self, params, model_path, model_name):
        self.params = params
        super(RLModel, self).__init__(model_path, model_name)

        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
    def _build_graph(self):
        self.lstm_hidden_units = 32
        self.sample_val = 0.2

        with tf.variable_scope("input"):
            self.train_phase = tf.placeholder(tf.bool, name="training")
            self.sample_phase = tf.placeholder(tf.bool, name="sample_phase")    # True
            self.mask_in_raw = tf.placeholder(tf.float32, [None])
            self.item_input = tf.placeholder(tf.float32, [None, self.params.slate_size, self.params.feature_size_vis])
            self.item_label = tf.placeholder(tf.float32, [None, self.params.slate_size])
            item_features = self.item_input

            self.item_size = self.params.slate_size
            self.mask_in = tf.reshape(self.mask_in_raw, [-1, self.item_size])

            self.enc_input = tf.reshape(item_features, [-1, self.item_size, self.params.feature_size_vis])
            self.full_item_fts = self.enc_input
            self.pv_item_fts = tf.reshape(self.full_item_fts, (-1, self.full_item_fts.shape[-1]))

            self.raw_dec_input = tf.placeholder(tf.float32, [None, self.params.feature_size_vis])
            self.dec_input = self.raw_dec_input

        with tf.variable_scope("encoder"):
            enc_input_train = tf.reshape(tf.tile(self.enc_input, (1, self.params.slate_size, 1)),
                                         [-1, self.item_size, self.params.feature_size_vis])
            enc_input = tf.cond(self.train_phase, lambda: enc_input_train, lambda: self.enc_input)
            self.enc_outputs = self.get_dnn(enc_input, [32, 16], [tf.nn.relu, tf.nn.relu], "enc_dnn")

        with tf.variable_scope("encoder_state"):
            cell_dec = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_units)
            
        with tf.variable_scope("decoder"):
            # for training
            dec_input = tf.reshape(self.dec_input, [-1, self.params.slate_size, self.params.feature_size_vis])
            zero_input = tf.zeros_like(dec_input[:, :1, :])
            dec_input = tf.concat([zero_input, dec_input[:, :-1, :]], axis=1)

            zero_state = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)
            new_dec_input = dec_input
            dec_outputs_train, _ = tf.nn.dynamic_rnn(cell_dec, inputs=new_dec_input, time_major=False,
                                                     initial_state=zero_state)
            dec_outputs_train = tf.reshape(dec_outputs_train, [-1, 1, self.lstm_hidden_units])
            dec_outputs_train_tile = tf.tile(dec_outputs_train, [1, self.item_size, 1])

            x = tf.concat([self.enc_outputs, dec_outputs_train_tile], axis=-1)
            self.act_logits_train = tf.reshape(self.get_dnn(x, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"), [-1, self.item_size])
            self.act_probs_train = tf.nn.softmax(self.act_logits_train)  
            self.act_probs_train_mask = tf.nn.softmax(tf.add(tf.multiply(1. - self.mask_in, -1.0e9), self.act_logits_train))

            # for predicting
            dec_input = tf.zeros([tf.shape(self.item_input)[0], self.params.feature_size_vis])

            dec_states = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)
            mask_tmp = tf.ones([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)

            mask_list = []
            act_idx_list = []
            act_probs_one_list = []
            act_probs_all_list = []
            next_state_list = []
            scores_pred = tf.zeros([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)

            random_val = tf.random_uniform([], 0, 1)
            for k in range(self.params.slate_size):
                new_dec_input = dec_input

                dec_outputs, dec_states = cell_dec(new_dec_input, dec_states)
                mask_list.append(mask_tmp)

                dec_outputs_tile = tf.tile(tf.reshape(dec_outputs, [-1, 1, dec_outputs.shape[-1]]),
                                           [1, self.item_size, 1])

                x = tf.concat([self.enc_outputs, dec_outputs_tile], axis=-1)
                act_logits_pred = tf.reshape(self.get_dnn(x, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"), 
                                             [-1, self.item_size])
                act_probs_mask = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), act_logits_pred))
                act_probs_mask_random = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), mask_tmp))

                act_random = tf.reshape(tf.multinomial(tf.log(act_probs_mask_random), num_samples=1), [-1])
                act_stoc = tf.reshape(tf.multinomial(tf.log(act_probs_mask), num_samples=1), [-1])
                # act_det = tf.argmax(act_probs_mask, axis=1)
                # act_idx_out = tf.cond(self.sample_phase, lambda: act_stoc, lambda: act_det)
                act_idx_out = tf.cond(self.sample_phase, lambda: tf.cond(random_val < self.sample_val, 
                                                                            lambda: act_random, 
                                                                            lambda: act_stoc), 
                                                         lambda: act_stoc)
                tmp_range = tf.cast(tf.range(tf.shape(self.item_input)[0], dtype=tf.int32), tf.int64)
                idx_pair = tf.stack([tmp_range, act_idx_out], axis=1)

                idx_one_hot = tf.one_hot(act_idx_out, self.item_size)

                mask_tmp = mask_tmp - idx_one_hot
                dec_input = tf.gather_nd(self.enc_input, idx_pair)
                next_full_state = tf.gather_nd(self.full_item_fts, idx_pair)
                act_probs_one = tf.gather_nd(act_probs_mask, idx_pair)

                act_idx_list.append(act_idx_out)
                act_probs_one_list.append(act_probs_one)
                act_probs_all_list.append(act_probs_mask)
                next_state_list.append(next_full_state)

                scores_pred = scores_pred + tf.cast(idx_one_hot, dtype=tf.float32) * (1 - k * 0.03)
            
            self.mask_arr = tf.stack(mask_list, axis=1)
            self.act_idx_out = tf.stack(act_idx_list, axis=1)
            self.act_probs_one = tf.stack(act_probs_one_list, axis=1)
            self.act_probs_all = tf.stack(act_probs_all_list, axis=1)
            self.next_state_out = tf.reshape(tf.stack(next_state_list, axis=1), [-1, self.full_item_fts.shape[-1]])

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

    def predict(self, item_input, item_label, sample_phase=False, train_phase=False):
        with self.graph.as_default():
            act_idx_out, act_probs_one, next_state_out, mask_arr, pv_item_fts, rerank_predict, \
                item_gauc, item_ndcg, test_summary = self.sess.run(
                [self.act_idx_out, self.act_probs_one, self.next_state_out, self.mask_arr, self.pv_item_fts, self.rerank_predict, \
                    self.item_gauc, self.item_ndcg, self.test_merged],  
                feed_dict={self.item_input: item_input, 
                            self.train_phase: train_phase,
                            self.sample_phase: sample_phase,
                            self.item_label: item_label})
            return act_idx_out, act_probs_one, next_state_out, mask_arr, pv_item_fts, rerank_predict, \
                item_gauc, item_ndcg, test_summary

    def rank(self, item_input, sample_phase=False, train_phase=False):
        with self.graph.as_default():
            act_idx_out = self.sess.run(self.act_idx_out,  
                feed_dict={self.item_input: item_input, 
                            self.train_phase: train_phase,
                            self.sample_phase: sample_phase})
            return act_idx_out

    def get_dnn(self, x, layer_nums, layer_acts, name="dnn"):
        input_ft = x
        assert len(layer_nums) == len(layer_acts)
        with tf.variable_scope(name):
            for i, layer_num in enumerate(layer_nums):
                input_ft = tf.contrib.layers.fully_connected(
                    inputs=input_ft,
                    num_outputs=layer_num,
                    scope='layer_%d' % i,
                    activation_fn=layer_acts[i],
                    reuse=tf.AUTO_REUSE)
        return input_ft
    
    def _build_summary(self):
        raise NotImplementedError
    
    def _build_loss(self):
        raise NotImplementedError

    def train(self, *args):
        raise NotImplementedError

    def get_long_reward(self, rewards):
        long_reward = np.zeros(rewards.shape)
        val = 0
        for i in reversed(range(self.params.slate_size)):
            long_reward[:, i] = self.params.gamma * val + rewards[:, i]
            val = long_reward[:, i]

        returns = long_reward[:, 0]
        return long_reward, returns