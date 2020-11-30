# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from approachs.model import Model
from utils.measure import gauc, pv_auc, ndcg

class Discriminator(Model):
    def __init__(self, params, model_path, model_name):
        self.params = params
        super(Discriminator, self).__init__(model_path, model_name)

        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

    def _build_graph(self):
        
        with tf.variable_scope("input"):
            self.fts = tf.placeholder(tf.float32, [None, self.params.feature_size_vis], name='dense_feature')
            self.d_label = tf.placeholder(tf.float32, [None, 1], name='d_label')
            self.keep_prob = tf.placeholder(tf.float32, [])
            self.phase = tf.placeholder(tf.bool, name='phase')

            shop_feature = tf.reshape(self.fts, [-1, self.params.slate_size, self.params.feature_size_vis])

            inputph = shop_feature
            tensor_global_max = tf.reduce_max(inputph, axis=1, keepdims=True)  # (B, 1, d2)
            tensor_global_min = tf.reduce_min(inputph, axis=1, keepdims=True)  # (B, 1, d2)
            tensor_global_max_tile = tf.tile(tensor_global_max, [1, self.params.slate_size, 1])  # (B, 17, d2)
            tensor_global_min_tile = tf.tile(tensor_global_min, [1, self.params.slate_size, 1])  # (B, 17, d2)
            matrix_f_global = tf.where(tf.equal(tensor_global_max_tile, tensor_global_min_tile),
                                       tf.fill(tf.shape(inputph), 0.5),
                                       tf.div(tf.subtract(inputph, tensor_global_min_tile),
                                              tf.subtract(tensor_global_max_tile, tensor_global_min_tile)))

            tensor_global_mean = tf.divide(tf.reduce_sum(matrix_f_global, axis=1, keepdims=True),
                                           tf.cast(self.params.slate_size, dtype=tf.float32))  # (B, 1, d2)
            tensor_global_mean_tile = tf.tile(tensor_global_mean, [1, self.params.slate_size, 1])  # (B, 17, d2)

            tensor_global_sigma = tf.square(matrix_f_global - tensor_global_mean_tile)  # (B, 1, d2)
            
            new_shop_feature = tf.concat(
                [inputph, tensor_global_max_tile, tensor_global_min_tile, matrix_f_global, tensor_global_mean_tile,
                 tensor_global_sigma], axis=2)
            new_shop_feature = tf.reshape(new_shop_feature, [-1, new_shop_feature.shape[-1]])

        with tf.variable_scope("network"):

            dense_feature_normed = new_shop_feature
            fn = tf.nn.relu
            layer1 = tf.contrib.layers.fully_connected(
                inputs=dense_feature_normed,
                num_outputs=64,
                scope='layer1',
                activation_fn=fn)
            layer2 = layer1
            new_dense_feature = self.get_rnn_feature(layer2, self.phase, self.keep_prob)

            # new_feature = tf.concat([layer2, new_dense_feature], axis=-1) 
            new_feature = new_dense_feature

            with tf.variable_scope('d_predict'):
                layer3 = tf.contrib.layers.fully_connected(
                    inputs=new_feature,
                    num_outputs=32,
                    scope='layer3',
                    activation_fn=fn)
                layer4 = layer3
                output_layer = tf.contrib.layers.fully_connected(
                    inputs=layer4,
                    num_outputs=1,
                    scope='output',
                    activation_fn=None)

                self.d_logits = output_layer
                self.d_pred = tf.nn.sigmoid(output_layer)
                self.d_reward = -tf.log(1 - self.d_pred + 1e-8)

        with tf.variable_scope("loss"):
            y_ = self.d_label
            y = self.d_pred
            self.d_loss = -tf.reduce_mean(
                y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))

            self.entropy_loss = tf.reduce_mean(self.logit_bernoulli_entropy(self.d_logits))
            self.total_loss = self.d_loss - self.params.c_entropy_d * self.entropy_loss

        self.set_global_step(tf.train.create_global_step())
        self.set_saver(tf.train.Saver())

        # learning rate decay
        learning_rate = tf.train.exponential_decay(
                            learning_rate=self.params.learning_rate_d,
                            global_step=self.global_step,
                            decay_steps=self.params.decay_steps_d,
                            decay_rate=self.params.decay_rate_d,
                            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
        # for summary
        self.d_pv_auc = pv_auc(self.d_pred, self.d_label, tf.shape(shop_feature)[0], self.params.slate_size, bias=False)
        self.train_merged, self.test_merged = self._build_summary()

    def _build_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar(name="total_loss", tensor=self.total_loss, collections=['d_train', 'd_test'])
            tf.summary.scalar(name="d_loss", tensor=self.d_loss, collections=['d_train', 'd_test'])
            tf.summary.scalar(name="entropy", tensor=self.entropy_loss, collections=['d_train', 'd_test'])
            tf.summary.scalar(name="pv_auc", tensor=self.d_pv_auc, collections=['d_train', 'd_test'])
        train_merged = tf.summary.merge_all('d_train')
        test_merged = tf.summary.merge_all('d_test')
        return train_merged, test_merged

    def train(self, x_data, d_label):
        with self.graph.as_default():
            d_pred, d_pv_auc, total_loss, _, summary = self.sess.run(
                [self.d_pred, self.d_pv_auc, self.total_loss, self.train_op, self.train_merged],
                feed_dict={self.phase: True,
                        self.fts: x_data.reshape([-1, self.params.feature_size_vis]),
                        self.d_label: d_label.reshape(-1, 1),
                        self.keep_prob: self.params.keep_prob_d})
            return d_pred, d_pv_auc, total_loss, summary

    def test(self, x_data, d_label):
        with self.graph.as_default():
            d_pred, d_pv_auc, total_loss, summary = self.sess.run(
                [self.d_pred, self.d_pv_auc, self.total_loss, self.test_merged],
                feed_dict={self.phase: False,
                        self.fts: x_data.reshape([-1, self.params.feature_size_vis]),
                        self.d_label: d_label.reshape(-1, 1),
                        self.keep_prob: 1})
            return d_pred, d_pv_auc, total_loss, summary

    def get_pv_auc(self, x_data, d_label):
        with self.graph.as_default():
            return self.sess.run(self.d_pv_auc, feed_dict={self.phase: False,
                                                        self.fts: x_data.reshape([-1, self.params.feature_size_vis]),
                                                        self.d_label: d_label.reshape(-1, 1),
                                                       self.keep_prob: 1})

    def predict(self, x_data):
        with self.graph.as_default():
            return self.sess.run([self.d_pred, self.d_reward], feed_dict={self.phase: False,
                                                                        self.fts: x_data.reshape([-1, self.params.feature_size_vis]),
                                                                        self.keep_prob: 1})

    def get_rnn_feature(self, sft, phase, keep_prob):
        def get_a_cell(lstm_size, keep_prob):
            # lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            lstm = tf.nn.rnn_cell.GRUCell(num_units=lstm_size)
            return lstm
            # drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            # return drop
        layer_num = 1
        lstm_units = 32
        with tf.variable_scope("rnn"):
            sft = tf.reshape(sft, [-1, self.params.slate_size, sft.shape[-1]])
            # sft_normed = tf.contrib.layers.batch_norm(sft, center=True, scale=True, is_training=phase, scope='rnn_bn')
            sft_normed = sft
            mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(lstm_units, keep_prob) for _ in range(layer_num)])
            outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=sft_normed, time_major=False, dtype=tf.float32)
            h_state = tf.reshape(outputs, [-1, lstm_units])
        return h_state

    @staticmethod
    def logsigmoid(a):
        '''Equivalent to tf.log(tf.sigmoid(a))'''
        return -tf.nn.softplus(-a)

    """ Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
    def logit_bernoulli_entropy(self, logits):
        ent = (1.-tf.nn.sigmoid(logits))*logits - self.logsigmoid(logits)
        return ent

    def rank(self, user_feature, query, candidates):
        raise NotImplementedError