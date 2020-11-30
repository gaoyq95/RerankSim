# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import os
import tensorflow as tf


class Documents(object):
    def __init__(self, num_total, num_feature, num_visible, seed):
        self.num_total = num_total
        self.num_feature = num_feature
        self.num_visible = num_visible
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'documents')
        self.rng = np.random.RandomState(seed=seed)
        if not self.load_data():
            self.items = self._generate_feature()
            self.save_data()

    def _generate_feature(self):
        return self.rng.rand(self.num_total, self.num_feature)
    
    def save_data(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        np.save(os.path.join(self.path, 'documents.npy'), self.items)
        print('========Save documents.========')
    
    def load_data(self):
        try:
            self.items = np.load(os.path.join(self.path, 'documents.npy'))
            print('========Load documents.========')
            return True
        except Exception as identifier:
            print(identifier)
            return False

    def get_feature_all(self, indexes):
        return self.items[indexes]

    def get_feature_visible(self, indexes):
        return self.items[:, :self.num_visible][indexes]


class UserResponse(object):
    def __init__(self, doc_feature_size, slate_size, seed):
        self.doc_ft_size = doc_feature_size
        self.slate_size = slate_size
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'user')
        self.c1 = np.clip(-0.25 * np.log(np.arange(1, slate_size+1)) + 1, 0.6, 0.90)
        self._build()
        with self.graph.as_default() as g:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            if not self.restore_model():
                self.save_model()
    
    def save_model(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.saver.save(sess=self.sess, save_path=os.path.join(self.path, 'user-model'), global_step=0)
        print('========Save user model.========')

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
            print('========Restore user model.========')
            return True
        else:
            print('========No user model.========')
            return False

    def _build(self):
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            tf.set_random_seed(self.seed)
            self.x_input = tf.placeholder(dtype=tf.float32, shape=(None, self.doc_ft_size), name='input')
            self.x_input_reshape = tf.reshape(self.x_input, shape=(-1, self.slate_size, self.doc_ft_size))
            # response to items
            fc1 = tf.layers.dense(inputs=self.x_input, 
                                  units=1, 
                                  activation=tf.nn.sigmoid, 
                                  use_bias=True, 
                                  bias_initializer=tf.truncated_normal_initializer(),
                                  kernel_initializer=tf.truncated_normal_initializer(), 
                                  trainable=False)
            self.response2item = tf.identity(fc1, name='response2item')
            # response to sequence
            mean_pre = []
            for i in range(self.slate_size):
                if i == 0:
                    mean_pre.append(self.x_input_reshape[:, :1])
                else:
                    mean_pre.append(tf.reduce_mean(self.x_input_reshape[:, :i], axis=1, keep_dims=True))
            mean_feature = tf.concat(mean_pre, axis=1)
            # norm
            norm_mean = tf.sqrt(tf.reduce_sum(tf.square(mean_feature), axis=-1))
            norm_x = tf.sqrt(tf.reduce_sum(tf.square(self.x_input_reshape), axis=-1))
            # dot
            mul = tf.reduce_sum(tf.multiply(mean_feature, self.x_input_reshape), axis=-1)
            cos_dis = mul / (norm_mean * norm_x)
            self.response2sequence = 1 - cos_dis

    def response(self, inputs):
        inputs = inputs.reshape((-1, self.doc_ft_size))
        with self.graph.as_default() as g:
            response1, response2 = self.sess.run([self.response2item, self.response2sequence], feed_dict={self.x_input: inputs})
            response1 = response1.reshape((-1, self.slate_size))
            # 
            response2[:, 0] = np.max(response2, axis=-1)
            response1 = (response1 - np.min(response1, axis=-1, keepdims=True)) / (np.max(response1, axis=-1, keepdims=True) - np.min(response1, axis=-1, keepdims=True))
            response2 = (response2 - np.min(response2, axis=-1, keepdims=True)) / (np.max(response2, axis=-1, keepdims=True) - np.min(response2, axis=-1, keepdims=True))
            res = self.c1 * response1 + (1 - self.c1) * response2
            res = (res - 0.5) * 20
            res = 1 / (1 + np.exp(-1 * res))
            return res

    def response_to_items(self, inputs):
        with self.graph.as_default() as g:
            inputs = inputs.reshape((-1, self.doc_ft_size))
            response = self.sess.run(self.response2item, feed_dict={self.x_input: inputs})
            return response

    def feedback(self, response):
        click = np.array(self.rng.binomial(1, response), np.float32)
        return click

    def choose(self, inputs):
        return self.feedback(self.response(inputs))