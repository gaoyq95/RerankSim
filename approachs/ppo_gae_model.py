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


class PPOGAEModel(RLModel):
    
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
        
        with tf.variable_scope("encoder_state_vf"):
            cell_dec_vf = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_units)
  
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

            if not self.params.vf_share:
                zero_state_vf = cell_dec_vf.zero_state(tf.shape(dec_input)[0], tf.float32)
                dec_outputs_train_vf, _ = tf.nn.dynamic_rnn(
                    cell_dec_vf, inputs=new_dec_input, time_major=False, initial_state=zero_state_vf)
            else:
                dec_outputs_train_vf = dec_outputs_train
            # dec_outputs_train_vf = dec_outputs_train
            self.vf = self.get_dnn(dec_outputs_train_vf, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn_vf") 

            # for predicting
            dec_input = tf.zeros([tf.shape(self.item_input)[0], self.params.feature_size_vis])

            dec_states = cell_dec.zero_state(tf.shape(dec_input)[0], tf.float32)

            if not self.params.vf_share:
                dec_states_vf = cell_dec_vf.zero_state(tf.shape(dec_input)[0], tf.float32)

            mask_tmp = tf.ones([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)

            mask_list = []
            act_idx_list = []
            act_probs_one_list = []
            act_probs_all_list = []
            next_state_list = []
            scores_pred = tf.zeros([tf.shape(self.item_input)[0], self.item_size], dtype=tf.float32)
            vf_list = []

            random_val = tf.random_uniform([], 0, 1)
            for k in range(self.params.slate_size):
                new_dec_input = dec_input

                dec_outputs, dec_states = cell_dec(new_dec_input, dec_states)
                mask_list.append(mask_tmp)

                dec_outputs_tile = tf.tile(tf.reshape(dec_outputs, [-1, 1, dec_outputs.shape[-1]]),
                                           [1, self.item_size, 1])

                x = tf.concat([self.enc_outputs, dec_outputs_tile], axis=-1)

                if not self.params.vf_share:
                    dec_outputs_vf, dec_states_vf = cell_dec_vf(new_dec_input, dec_states_vf)
                else:
                    dec_outputs_vf = dec_outputs
                # dec_outputs_vf = dec_outputs

                vf = self.get_dnn(dec_outputs_vf, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn_vf")
                vf = tf.reshape(vf, [-1])

                act_logits_pred = tf.reshape(self.get_dnn(x, [32, 16, 1], [tf.nn.relu, tf.nn.relu, None], "dec_dnn"), 
                                             [-1, self.item_size])
                act_probs_mask = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), act_logits_pred))
                act_probs_mask_random = tf.nn.softmax(tf.add(tf.multiply(1. - mask_tmp, -1.0e9), mask_tmp))

                act_random = tf.reshape(tf.multinomial(tf.log(act_probs_mask_random), num_samples=1), [-1])
                act_stoc = tf.reshape(tf.multinomial(tf.log(act_probs_mask), num_samples=1), [-1])
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
                vf_list.append(vf)

                scores_pred = scores_pred + tf.cast(idx_one_hot, dtype=tf.float32) * (1 - k * 0.03)
            
            self.mask_arr = tf.stack(mask_list, axis=1)
            self.act_idx_out = tf.stack(act_idx_list, axis=1)
            self.act_probs_one = tf.stack(act_probs_one_list, axis=1)
            self.act_probs_all = tf.stack(act_probs_all_list, axis=1)
            self.next_state_out = tf.reshape(tf.stack(next_state_list, axis=1), [-1, self.full_item_fts.shape[-1]])
            
            self.vf_out = tf.stack(vf_list, axis=1)

            self.rerank_predict = tf.identity(tf.reshape(scores_pred, [-1, 1]), 'rerank_predict')

        self.set_global_step(tf.train.create_global_step())
        self.set_saver(tf.train.Saver())
        with tf.variable_scope("loss"):
            self._build_loss()
            self.item_gauc = gauc(self.rerank_predict, tf.reshape(self.item_label, (-1, 1)), self.item_size)
            # self.item_pauc = pv_auc(self.rerank_predict, tf.reshape(self.item_label, (-1, 1)), tf.shape(self.item_label)[0], self.item_size)
            self.item_ndcg = ndcg(p=tf.reshape(self.rerank_predict, ((-1, self.item_size))), 
                                  l=tf.reshape(self.item_label, (-1, self.item_size)),
                                  k=self.item_size)
        self.train_merged, self.test_merged = self._build_summary()

    def predict(self, item_input, item_label, sample_phase=False, train_phase=False):
        with self.graph.as_default():
            act_idx_out, act_probs_one, next_state_out, mask_arr, pv_item_fts, rerank_predict, \
                vf, item_gauc, item_ndcg, test_summary = self.sess.run(
                [self.act_idx_out, self.act_probs_one, self.next_state_out, self.mask_arr, self.pv_item_fts, self.rerank_predict, \
                    self.vf_out, self.item_gauc, self.item_ndcg, self.test_merged],  
                feed_dict={self.item_input: item_input, 
                            self.train_phase: train_phase,
                            self.sample_phase: sample_phase,
                            self.item_label: item_label})
            return act_idx_out, act_probs_one, next_state_out, mask_arr, pv_item_fts, rerank_predict, \
                vf, item_gauc, item_ndcg, test_summary

    def _build_loss(self):
        self.clip_value = 0.1
        
        with tf.variable_scope("train_input"):
            self.old_act_prob = tf.placeholder(dtype=tf.float32, shape=[None], name='old_act_prob')
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
            self.c_entropy = tf.placeholder(dtype=tf.float32, name='c_entropy')
            self.returns = tf.placeholder(dtype=tf.float32, shape=[None, self.params.slate_size], name='returns')
            self.vf_old = tf.placeholder(dtype=tf.float32, shape=[None, self.params.slate_size], name='vf_old')

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

        # value function loss
        # value net loss
        vpred = self.vf
        clip_value_vf = self.params.clip_value_vf
        vpredclipped = self.vf_old + tf.clip_by_value(vpred - self.vf_old, -clip_value_vf, clip_value_vf)
        vf_losses1 = tf.square(vpred - self.returns)
        vf_losses2 = tf.square(vpredclipped - self.returns)
        self.loss_vf = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # construct computation graph for loss
        self.total_loss = self.loss_clip - self.c_entropy * self.entropy + self.params.c_vf * self.loss_vf

        self.g = tf.reduce_mean(self.returns[:, 0])

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

    def train(self, item_input, raw_dec_input, old_act_prob, actions, rewards, act_mask, c_entropy, values):
        with self.graph.as_default():
            gaes, returns = self.get_gaes_vf(rewards, values, self.params.gamma, self.params.lam)
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
                            self.gaes: gaes.reshape([-1]),
                            self.returns: returns,
                            self.c_entropy: c_entropy, 
                            self.train_phase: True, 
                            self.vf_old: values
                            })
            return total_loss, mean_return, summary, step

    def evaluate(self, item_input, raw_dec_input, old_act_prob, actions, rewards, act_mask, c_entropy, values):
        with self.graph.as_default():
            gaes, returns = self.get_gaes_vf(rewards, values, self.params.gamma, self.params.lam)
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
                            self.gaes: gaes.reshape([-1]),
                            self.returns: returns,
                            self.c_entropy: c_entropy, 
                            self.train_phase: True, 
                            self.vf_old: values
                            })
            return total_loss, mean_return, summary, step

    def get_gaes_vf(self, rewards, values, gamma, lam):
        '''
        Args:
            rewards: shape(BatchSize, num_step)
            values: shape(BatchSize, num_step)
            gamma: return discount
            lam: residual discount
        Returns:
            gaes: generalized advantage estimation
            returns: discounted return
        '''
        click_rewards = rewards
        click_values = values
        click_advs = np.zeros_like(click_rewards)
        click_lastgaelam = np.zeros(click_rewards.shape[0])
        for t in reversed(range(self.params.slate_size)):
            if t == self.params.slate_size - 1:
                nextnonterminal = 0.0
                click_nextvalues = np.zeros(click_rewards.shape[0])
            else:
                nextnonterminal = 1.0
                click_nextvalues = click_values[:, t + 1]
            click_delta = click_rewards[:, t] + gamma * click_nextvalues * nextnonterminal - click_values[:, t]
            click_advs[:, t] = click_lastgaelam = click_delta + gamma * lam * nextnonterminal * click_lastgaelam
        click_returns = click_advs + click_values
        gaes = click_advs
        returns = click_returns
        return gaes, returns
