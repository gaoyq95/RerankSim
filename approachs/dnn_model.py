# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from approachs.model import Model
from utils.measure import gauc, ndcg, pv_auc


class DNNModel(Model):
    def __init__(self, params, model_path, model_name):
        self.params = params
        super(DNNModel, self).__init__(model_path, model_name)
        
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
    
    def _build_graph(self):
        size = self.params.feature_size_vis
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, size), name='input')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, ), name='label')

        self.score, self.loss = self._build_net()
        
        self.set_global_step(tf.train.create_global_step())
        # learning rate decay
        self.learning_rate = tf.train.exponential_decay(
                        learning_rate=self.params.learning_rate,
                        global_step=self.global_step,
                        decay_steps=self.params.decay_steps,
                        decay_rate=self.params.decay_rate,
                        staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.set_saver(tf.train.Saver())

        self.gauc = gauc(p=self.score, l=self.y, n=self.params.slate_size)
        self.ndcg = ndcg(p=tf.reshape(self.score, (-1, self.params.slate_size)), 
                         l=tf.reshape(self.y, (-1, self.params.slate_size)), 
                         k=self.params.slate_size)
        self.pv_auc = pv_auc(p=self.score, 
                             l=self.y, 
                             n=tf.shape(self.x)[0] // self.params.slate_size, 
                             pv_size=self.params.slate_size, 
                             bias=True)
        self.train_merged, self.test_merged = self._build_summary()

    def _build_net(self):
        raise NotImplementedError

    def _build_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar(name="loss", tensor=self.loss, collections=['train', 'test'])
            tf.summary.scalar(name="gauc", tensor=self.gauc, collections=['train', 'test'])
            tf.summary.scalar(name="ndcg", tensor=self.ndcg, collections=['train', 'test'])
            tf.summary.scalar(name="pv_auc", tensor=self.pv_auc, collections=['train', 'test'])
        train_merged = tf.summary.merge_all('train')
        test_merged = tf.summary.merge_all('test')

        return train_merged, test_merged

    def _get_feature(self):
        with tf.variable_scope('feature'):
            inputph = tf.reshape(self.x, (-1, self.params.slate_size, self.x.shape[-1]))
            tensor_global_max = tf.reduce_max(inputph, axis=1, keep_dims=True)                      
            tensor_global_min = tf.reduce_min(inputph, axis=1, keep_dims=True)                      
            tensor_global_max_tile = tf.tile(tensor_global_max, [1, self.params.slate_size, 1])     
            tensor_global_min_tile = tf.tile(tensor_global_min, [1, self.params.slate_size, 1])     
            matrix_f_global = tf.where(tf.equal(tensor_global_max_tile, tensor_global_min_tile),
                                        tf.fill(tf.shape(inputph), 0.5),
                                        tf.div(tf.subtract(inputph, tensor_global_min_tile),
                                                tf.subtract(tensor_global_max_tile, tensor_global_min_tile)))

            tensor_global_mean = tf.divide(tf.reduce_sum(matrix_f_global, axis=1, keep_dims=True),
                                            tf.cast(self.params.slate_size, dtype=tf.float32))      
            tensor_global_mean_tile = tf.tile(tensor_global_mean, [1, self.params.slate_size, 1])   

            tensor_global_sigma = tf.square(matrix_f_global - tensor_global_mean_tile)              
            new_shop_feature = tf.concat(
                [inputph, tensor_global_max_tile, tensor_global_min_tile, matrix_f_global, tensor_global_mean_tile,
                    tensor_global_sigma], axis=2)
            return tf.reshape(new_shop_feature, [-1, new_shop_feature.shape[-1]])

    def train(self, samples, labels):
        with self.graph.as_default():
            if self.params.algo == 'GroupWise':
                # shuffle
                batch_size = samples.shape[0]
                batch_id = np.tile(np.arange(batch_size).reshape((-1, 1)), (1, self.params.slate_size)).reshape((-1,))
                shuffle_id = np.tile(np.random.permutation(self.params.slate_size), (batch_size))
                samples = samples[(batch_id, shuffle_id)].reshape((-1, samples.shape[-1]))
                labels = labels[(batch_id, shuffle_id)].reshape((-1, ))
            else:
                # reshape
                samples = samples.reshape((-1, samples.shape[-1]))
                labels = labels.reshape((-1, ))
            # train
            _, loss, gauc, ndcg, pv_auc, step, summary = self.sess.run(
                [self.opt, self.loss, self.gauc, self.ndcg, self.pv_auc, self.global_step, self.train_merged], 
                feed_dict={self.x: samples, self.y: labels})
            return loss, gauc, ndcg, pv_auc, step, summary
                
    def evaluate(self, samples, labels):
        with self.graph.as_default():
            # reshape
            samples = samples.reshape((-1, samples.shape[-1]))
            labels = labels.reshape((-1, ))
            loss, gauc, ndcg, pv_auc, step, summary = self.sess.run(
                [self.loss, self.gauc, self.ndcg, self.pv_auc, self.global_step, self.test_merged], 
                feed_dict={self.x: samples, self.y: labels})
            return loss, gauc, ndcg, pv_auc, step, summary

    def rank(self, samples):
        with self.graph.as_default():
            num = samples.shape[1]
            samples = samples.reshape((-1, samples.shape[-1]))
            prediction = self.sess.run(self.score, feed_dict={self.x: samples})
            prediction = prediction.reshape((-1, num))
            res = np.argsort(-1 * prediction, axis=1)
            return res