# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from approachs.model import Model
from utils.measure import gauc, pv_auc, ndcg


class LPModel(Model):
    def __init__(self, params, model_path, model_name):
        self.params = params
        super(LPModel, self).__init__(model_path, model_name)

        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

    def _build_graph(self):
        with tf.variable_scope("input"):
            self.fts = tf.placeholder(tf.float32, [None, self.params.feature_size_vis], name='dense_feature')
            self.ctr_label = tf.placeholder(tf.float32, [None, 1], name='ctr_label')
            self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            self.phase = tf.placeholder(tf.bool, name='phase')

            shop_feature = tf.reshape(self.fts, [-1, self.params.slate_size, self.params.feature_size_vis])
            # global feature
            inputph = shop_feature
            tensor_global_max = tf.reduce_max(inputph, axis=1, keep_dims=True)  # (B, 1, d2)
            tensor_global_min = tf.reduce_min(inputph, axis=1, keep_dims=True)  # (B, 1, d2)
            tensor_global_max_tile = tf.tile(tensor_global_max, [1, self.params.slate_size, 1])  # (B, 17, d2)
            tensor_global_min_tile = tf.tile(tensor_global_min, [1, self.params.slate_size, 1])  # (B, 17, d2)
            matrix_f_global = tf.where(tf.equal(tensor_global_max_tile, tensor_global_min_tile),
                                       tf.fill(tf.shape(inputph), 0.5),
                                       tf.div(tf.subtract(inputph, tensor_global_min_tile),
                                              tf.subtract(tensor_global_max_tile, tensor_global_min_tile)))

            tensor_global_mean = tf.divide(tf.reduce_sum(matrix_f_global, axis=1, keep_dims=True),
                                           tf.cast(self.params.slate_size, dtype=tf.float32))  # (B, 1, d2)
            tensor_global_mean_tile = tf.tile(tensor_global_mean, [1, self.params.slate_size, 1])  # (B, 17, d2)

            tensor_global_sigma = tf.square(matrix_f_global - tensor_global_mean_tile)  # (B, 1, d2)

            new_shop_feature = tf.concat(
                [inputph, tensor_global_max_tile, tensor_global_min_tile, matrix_f_global, tensor_global_mean_tile,
                 tensor_global_sigma], axis=2)
            new_shop_feature = tf.reshape(new_shop_feature, [-1, new_shop_feature.shape[-1]])

        with tf.variable_scope("network"):
            # dense_feature_normed = tf.contrib.layers.batch_norm(new_shop_feature,
            #                                                     center=True, scale=True,
            #                                                     is_training=self.phase,
            #                                                     scope='bn')
            dense_feature_normed = new_shop_feature
            fn = tf.nn.relu
            # fn = tf.nn.leaky_relu
            # fn = tf.nn.tanh
            layer1 = tf.contrib.layers.fully_connected(
                inputs=dense_feature_normed,
                num_outputs=64,
                scope='layer1',
                activation_fn=fn)
            # layer1 = tf.nn.dropout(layer1, self.keep_prob)
            # layer2 = tf.contrib.layers.fully_connected(
            #     inputs=layer1,
            #     num_outputs=64,
            #     scope='layer2',
            #     activation_fn=fn)
            # layer2 = tf.nn.dropout(layer2, self.keep_prob)
            layer2 = layer1
            new_dense_feature = self.get_rnn_feature(layer2, self.phase, self.keep_prob)

            new_feature = tf.concat([layer2, new_dense_feature], axis=-1) 

            layer3 = tf.contrib.layers.fully_connected(
                inputs=new_feature,
                num_outputs=32,
                scope='layer3',
                activation_fn=fn)
            # layer3 = tf.nn.dropout(layer3, self.keep_prob)
            # layer4 = tf.contrib.layers.fully_connected(
            #     inputs=layer3,
            #     num_outputs=32,
            #     scope='layer4',
            #     activation_fn=fn)
            # layer4 = tf.nn.dropout(layer4, self.keep_prob)
            layer4 = layer3
            output_layer = tf.contrib.layers.fully_connected(
                inputs=layer4,
                num_outputs=1,
                scope='output',
                activation_fn=tf.nn.sigmoid)
            self.ctr_pred = tf.identity(output_layer, 'ctr_pred')

        with tf.variable_scope("loss"):
            y_ = self.ctr_label
            y = self.ctr_pred
            self.ctr_loss = -1 * tf.reduce_mean(
                y_ * tf.log(tf.clip_by_value(y, 1e-5, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-5, 1.0)))
            self.loss = self.ctr_loss
            
            self.set_global_step(tf.train.create_global_step())
            
            # learning rate decay
            learning_rate = tf.train.exponential_decay(
                                learning_rate=self.params.learning_rate,
                                global_step=self.global_step,
                                decay_steps=self.params.decay_steps,
                                decay_rate=self.params.decay_rate,
                                staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            
            self.set_saver(tf.train.Saver())

            self.ctr_gauc = gauc(self.ctr_pred, self.ctr_label, self.params.slate_size)
            self.ctr_pauc = pv_auc(self.ctr_pred, self.ctr_label, tf.shape(shop_feature)[0], self.params.slate_size, bias=False)
            self.ctr_ndcg = ndcg(p=tf.reshape(self.ctr_pred, (-1, self.params.slate_size)),
                                 l=tf.reshape(self.ctr_label, (-1, self.params.slate_size)),
                                 k=self.params.slate_size)
        self.train_merged, self.test_merged = self._build_summary()
    
    def _build_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar(name="loss", tensor=self.loss, collections=['ctr_train', 'ctr_test'])
            # tf.summary.scalar(name="ctr_loss", tensor=self.ctr_loss, collections=['ctr_train', 'ctr_test'])
            tf.summary.scalar(name="ctr_gauc", tensor=self.ctr_gauc, collections=['ctr_train', 'ctr_test'])
            tf.summary.scalar(name="ctr_pauc", tensor=self.ctr_pauc, collections=['ctr_train', 'ctr_test'])
            tf.summary.scalar(name='ctr_ndcg', tensor=self.ctr_ndcg, collections=['ctr_train', 'ctr_test'])
        train_merged = tf.summary.merge_all('ctr_train')
        test_merged = tf.summary.merge_all('ctr_test')
        return train_merged, test_merged

    def train(self, x_data, ctr_label):
        with self.graph.as_default():
            _, ctr_pred, loss, ctr_gauc, ctr_pauc, ctr_ndcg, summary, global_step = self.sess.run(
                [self.train_op, self.ctr_pred, self.loss, self.ctr_gauc, self.ctr_pauc, self.ctr_ndcg, self.train_merged, self.global_step],
                feed_dict={self.phase: True,
                        self.fts: x_data.reshape((-1, x_data.shape[-1])),
                        self.ctr_label: ctr_label.reshape((-1, 1)),
                        self.keep_prob: self.params.keep_prob}
                )
            return ctr_pred, loss, ctr_gauc, ctr_pauc, ctr_ndcg, summary, global_step

    def evaluate(self, x_data, ctr_label):
        with self.graph.as_default():  
            ctr_pred, loss, ctr_gauc, ctr_pauc, ctr_ndcg, summary = self.sess.run(
                [self.ctr_pred, self.loss, self.ctr_gauc, self.ctr_pauc, self.ctr_ndcg, self.test_merged],
                feed_dict={self.phase: False,
                        self.fts: x_data.reshape((-1, x_data.shape[-1])),
                        self.ctr_label: ctr_label.reshape((-1, 1)),
                        self.keep_prob: 1.0}
                )
            return ctr_pred, loss, ctr_gauc, ctr_pauc, ctr_ndcg, summary

    def predict(self, x_data):
        with self.graph.as_default():
            input_shape = x_data.shape
            ctr_probs = self.sess.run(self.ctr_pred,
            feed_dict={self.phase: False,
                    self.fts: x_data.reshape((-1, x_data.shape[-1])),
                    self.keep_prob: 1.0})
            return ctr_probs.reshape(input_shape[:-1])

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
    
    def rank(self, user_feature, query, candidates):
        raise NotImplementedError


class Evaluator(object):
    def __init__(self, model_path):
        self.graph = tf.Graph()
        self.model_path = model_path
        with self.graph.as_default() as g:
            self.sess = tf.Session(graph=g)
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver = tf.train.import_meta_graph(meta_graph_or_file='{}.meta'.format(ckpt.model_checkpoint_path))
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                
                self.x = g.get_tensor_by_name('input/dense_feature:0')
                self.keep_prob = g.get_tensor_by_name('input/keep_prob:0')
                self.train_phase = g.get_tensor_by_name('input/phase:0')
                self.ctr_pred = g.get_tensor_by_name('network/ctr_pred:0')
                print('Load model:', ckpt.model_checkpoint_path)
            else:
                raise ValueError('No model file.')

    def predict(self, x):
        input_shape = x.shape
        with self.graph.as_default():
            ctr_pred = self.sess.run(self.ctr_pred, 
                feed_dict={self.x: x.reshape((-1, x.shape[-1])), self.train_phase:False, self.keep_prob: 1.0}
            )
        return ctr_pred.reshape(input_shape[:-1])

