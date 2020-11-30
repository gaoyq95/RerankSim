# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import dtypes
import numpy as np

from approachs.model import Model
from approachs.dnn_model import DNNModel


class SoftRank(DNNModel):
    def __init__(self, params, model_path, model_name):
        self.softRank_theta = 0.1
        super(SoftRank, self).__init__(params, model_path, model_name)
    
    def _build_net(self):
        
        self.list_index = tf.placeholder(dtype=tf.int32, shape=(None, self.params.slate_size), name='list_index')

        self.batch_index_bias = tf.placeholder(tf.int32, shape=[None])
        self.batch_expansion_mat = tf.placeholder(tf.float32, shape=[None, 1])
        self.batch_diag = tf.placeholder(tf.float32, shape=[None, self.params.slate_size, self.params.slate_size])

        x_input = self._get_feature()
        # dnn
        layers = [64, 32, 1]
        activation = [tf.nn.relu for _ in range(len(layers)-1)] + [None]
        with tf.variable_scope('dnn'):
            for i, (dim, act) in enumerate(zip(layers, activation)):
                x_input = tf.layers.dense(inputs=x_input,
                                          units=dim,
                                          activation=act, 
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params.l2_regu), 
                                          name='layer_'+str(i),
                                          reuse=tf.AUTO_REUSE)
            logits = x_input

            list_labels = tf.reshape(self.y, [-1, self.params.slate_size], name='list_labels')
            list_logits = tf.reshape(logits, [-1, self.params.slate_size], name='list_logits')
            rank_loss = self._soft_rank_loss(list_logits, list_labels, self.list_index)
            l2_loss = tf.losses.get_regularization_loss()
            loss = rank_loss + l2_loss
            return logits, loss

    def integral_Guaussian(self, mu, theta):
        a = -4.0 / math.sqrt(2.0 * math.pi) / theta
        exp_mu = tf.exp(a * mu)
        ig = tf.div(exp_mu, exp_mu + 1) * -1.0 + 1
        return ig

    def _soft_rank_loss(self, output, target_rels, target_indexs, name="softRank"):
        target_indexs = [tf.reshape(x, [-1]) for x in tf.split(target_indexs, self.params.slate_size, axis=1)]
        target_rels = [tf.reshape(x, [-1]) for x in tf.split(target_rels, self.params.slate_size, axis=1)]
        loss = None
        batch_size = tf.shape(target_rels[0])[0]
        theta = 0.1
        with tf.variable_scope(name):
            output = tf.nn.l2_normalize(output, 1)
            # compute pi_i_j
            tmp = tf.concat(axis=1, values=[self.batch_expansion_mat for _ in range(self.params.slate_size)])
            tmp_expand = tf.expand_dims(tmp, -2)
            output_expand = tf.expand_dims(output, -2)
            dif = tf.subtract(tf.matmul(tf.matrix_transpose(output_expand), tmp_expand),
                              tf.matmul(tf.matrix_transpose(tmp_expand), output_expand))
            # unpacked_pi = self.integral_Guaussian(dif, theta)
            unpacked_pi = tf.add(self.integral_Guaussian(dif, self.softRank_theta),
                                 self.batch_diag)  # make diag equal to 1.0
            # may need to unpack pi: pi_i_j is the probability that i is bigger than j
            pi = tf.unstack(unpacked_pi, None, 1)
            for i in range(self.params.slate_size):
                pi[i] = tf.unstack(pi[i], None, 1)
            # compute rank distribution p_j_r
            one_zeros = tf.matmul(self.batch_expansion_mat,
                                  tf.constant([1.0] + [0.0 for r in range(self.params.slate_size - 1)], tf.float32,
                                              [1, self.params.slate_size]))
            # initial_value = tf.unpack(one_zeros, None, 1)
            pr = [one_zeros for _ in range(self.params.slate_size)]  # [i][r][None]
            # debug_pr_1 = [one_zeros for _ in range(self.params.slate_size)] #[i][r][None]
            for i in range(self.params.slate_size):
                for j in range(self.params.slate_size):
                    # if i != j: #insert doc j
                    pr_1 = tf.pad(tf.stack(tf.unstack(pr[i], None, 1)[:-1], 1), [[0, 0], [1, 0]], mode='CONSTANT')
                    # debug_pr_1[i] = pr_1
                    # pr_1 = tf.concat(1, [self.batch_expansion_mat*0.0, tf.unpack(pr[i], None, 1)[:-1]])
                    factor = tf.tile(tf.expand_dims(pi[i][j], -1), [1, self.params.slate_size])
                    # print(factor.get_shape())
                    pr[i] = tf.add(tf.multiply(pr[i], factor),
                                   tf.multiply(pr_1, 1.0 - factor))
            # compute expected NDCG
            # compute Gmax
            Dr = tf.matmul(self.batch_expansion_mat,
                           tf.constant([1.0 / math.log(2.0 + r) for r in range(self.params.slate_size)], tf.float32,
                                       [1, self.params.slate_size]))
            gmaxs = []
            for i in range(self.params.slate_size):
                idx = target_indexs[i] + tf.to_int32(self.batch_index_bias)
                g = embedding_ops.embedding_lookup(target_rels, idx)
                gmaxs.append(g)
            _gmax = tf.exp(tf.stack(gmaxs, 1)) * (1.0 / math.log(2))
            Gmax = tf.reduce_sum(tf.multiply(Dr, _gmax), 1)
            # compute E(Dr)
            Edrs = []
            for i in range(self.params.slate_size):
                edr = tf.multiply(Dr, pr[i])
                Edrs.append(tf.reduce_sum(edr, 1))
            # compute g(j)
            g = tf.exp(tf.stack(target_rels, 1)) * (1.0 / math.log(2))
            dcg = tf.multiply(g, tf.stack(Edrs, 1))
            Edcg = tf.reduce_sum(dcg, 1)
            Ndcg = tf.div(Edcg, Gmax)
            # compute loss
            loss = (Ndcg * -1.0 + 1) * 10
        return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)  # , pi, pr, Ndcg]

    def train(self, samples, labels):
        with self.graph.as_default():
            assert samples.shape[0] == labels.shape[0]
            batch_size = samples.shape[0]            
            size = self.params.slate_size
            # feed
            index = np.array(
                [sorted(range(self.params.slate_size), key=lambda k:labels[i][k], reverse=True) for i in range(batch_size)]
            )
            batch_index_bias_v = np.array([i * self.params.slate_size for i in range(batch_size)])
            batch_expansion_mat_v = np.ones((batch_size, 1))
            batch_diag_v = np.array(
                [np.diag([0.5 for x in range(self.params.slate_size)]) for _ in range(batch_size)]
            )
            # reshape
            samples = samples.reshape((-1, samples.shape[-1]))
            labels = labels.reshape((-1, ))
            # train
            _, loss, gauc, ndcg, pv_auc, step, summary = self.sess.run(
                [self.opt, self.loss, self.gauc, self.ndcg, self.pv_auc, self.global_step, self.train_merged], 
                feed_dict={self.x: samples,
                            self.y: labels,
                            self.list_index:index,
                            self.batch_index_bias: batch_index_bias_v,
                            self.batch_expansion_mat: batch_expansion_mat_v,
                            self.batch_diag: batch_diag_v
                            })
            return loss, gauc, ndcg, pv_auc, step, summary

    def evaluate(self, samples, labels):
        with self.graph.as_default():
            batch_size = samples.shape[0]            
            # feed
            index = np.array(
                [sorted(range(self.params.slate_size), key=lambda k:labels[i][k], reverse=True) for i in range(batch_size)]
            )
            batch_index_bias_v = np.array([i * self.params.slate_size for i in range(batch_size)])
            batch_expansion_mat_v = np.ones((batch_size, 1))
            batch_diag_v = np.array(
                [np.diag([0.5 for x in range(self.params.slate_size)]) for _ in range(batch_size)]
            )
            # reshape
            samples = samples.reshape((-1, samples.shape[-1]))
            labels = labels.reshape((-1, ))
            # 
            loss, gauc, ndcg, pv_auc, step, summary = self.sess.run(
                [self.loss, self.gauc, self.ndcg, self.pv_auc, self.global_step, self.test_merged], 
                feed_dict={self.x: samples,
                            self.y: labels,
                            self.list_index:index,
                            self.batch_index_bias: batch_index_bias_v,
                            self.batch_expansion_mat: batch_expansion_mat_v,
                            self.batch_diag: batch_diag_v
                            })
            return loss, gauc, ndcg, pv_auc, step, summary
