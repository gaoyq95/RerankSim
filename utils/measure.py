# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math


def gauc(p, l, n):
    '''
    Args:
        p: predition
        l: label
        n: slate size
    Retruns:
        group auc.
    '''
    sj = tf.tile(tf.reshape(p, [-1, 1, n]), [1, n, 1])
    si = tf.tile(tf.reshape(p, [-1, n, 1]), [1, 1, n])
    cost = tf.where(tf.less(si, sj), tf.ones_like(si), tf.zeros_like(si))
    lj = tf.tile(tf.reshape(l, [-1, 1, n]), [1, n, 1])
    li = tf.tile(tf.reshape(l, [-1, n, 1]), [1, 1, n])
    ha = tf.where(tf.less(li, lj), tf.ones_like(li), tf.zeros_like(li))
    ha2 = tf.where(tf.equal(si, sj), tf.multiply(tf.ones_like(li), 0.5), tf.zeros_like(li))
    cnt = tf.reduce_sum(ha, [1, 2])
    ans1 = tf.reduce_sum(tf.multiply(ha, tf.add(cost, ha2)), [1, 2]) / cnt
    ans2 = tf.zeros_like(ans1)
    bs = tf.reduce_sum(tf.where(tf.equal(cnt, 0.0), tf.zeros_like(cnt), tf.ones_like(cnt)))
    return tf.where(tf.equal(bs, 0.0), 0.0, tf.reduce_sum(tf.where(tf.equal(cnt, 0.0), ans2, ans1)) / bs)

def pv_auc(p, l, n, pv_size, bias=True):
    '''
    Args:
        p: predition
        l: label
        n: batch size
        pv_size: slate size
        bias: bool, postion bias
    Returns:
        page auc.
    '''
    if bias:
        w = 1. / np.log2(np.arange(pv_size) + 2.)
    else:
        w = np.ones(pv_size)
    w = tf.tile(tf.reshape(w.astype(np.float32), [1, pv_size]), [n, 1])

    p = tf.reshape(p, [-1, pv_size])
    p = tf.multiply(p, w)
    p = tf.reduce_sum(p, axis=-1)
    
    l = tf.reshape(l, [-1, pv_size])
    l = tf.reduce_sum(l, axis=-1)
    # l = tf.where(tf.less(l, 1), tf.zeros_like(l), tf.ones_like(l))

    sj = tf.tile(tf.reshape(p, [-1, 1, n]), [1, n, 1])
    si = tf.tile(tf.reshape(p, [-1, n, 1]), [1, 1, n])
    cost = tf.where(tf.less(si, sj), tf.ones_like(si), tf.zeros_like(si))
    lj = tf.tile(tf.reshape(l, [-1, 1, n]), [1, n, 1])
    li = tf.tile(tf.reshape(l, [-1, n, 1]), [1, 1, n])
    ha = tf.where(tf.less(li, lj), tf.ones_like(li), tf.zeros_like(li))
    ha2 = tf.where(tf.equal(si, sj), tf.ones_like(li) / 2.0, tf.zeros_like(li))
    cnt = tf.reduce_sum(ha, [1, 2])
    ans1 = tf.reduce_sum(tf.multiply(ha, tf.add(cost, ha2)), [1, 2]) / cnt
    ans2 = tf.zeros_like(ans1)
    bs = tf.reduce_sum(tf.where(tf.equal(cnt, 0.0), tf.zeros_like(cnt), tf.ones_like(cnt))) 
    return tf.where(tf.equal(bs, 0.0), 0.0, tf.reduce_sum(tf.where(tf.equal(cnt, 0.0), ans2, ans1)) / bs)

def nd_indices(indices):
    indices.get_shape().assert_has_rank(2)
    batch_ids = tf.ones_like(indices) * tf.expand_dims(
        tf.range(tf.shape(input=indices)[0]), 1)
    return tf.stack([batch_ids, indices], axis=-1)

def avg(l1, l2):
    total = tf.reduce_sum(tf.where(tf.equal(l2, 0.0), tf.zeros_like(l1), tf.ones_like(l1)))
    ndcgsum = tf.reduce_sum(tf.where(tf.equal(l2, 0.0), tf.zeros_like(l1), tf.divide(l1, l2)))
    return tf.cond(tf.equal(total, 0.0), lambda:0.0, lambda:ndcgsum / total)

def dcg(l):
    return tf.reduce_sum(l / tf.log(2.0 + tf.cast(tf.range(tf.shape(input=l)[1]), dtype=tf.float32)), -1)

def ndcg(p, l, k):
    '''
    Args:
        p: prediction, shape(batch_size, slate_size)
        l: label, shape(batch_size, slate_size)
        k: top-k, int
    Returns:
        NDCG@k
    '''
    idx = tf.nn.top_k(p, k, sorted = True).indices
    idx = nd_indices(idx)
    labelList = tf.gather_nd(l, idx)
    idx = tf.nn.top_k(l, k, sorted = True).indices
    idx = nd_indices(idx)
    sortedLabelList = tf.gather_nd(l, idx)
    return avg(dcg(labelList), dcg(sortedLabelList))

def mean_average_precision(p, l):
    '''
    Args:
        p: prediction, shape(batch_size, slate_size)
        l: label, shape(batch_size, slate_size)
    Returns:
        Mean Average Precision(MAP)
    '''
    k = tf.shape(l)[1]
    idx = tf.nn.top_k(p, k, sorted=True).indices
    idx = nd_indices(idx)
    sorted_labels = tf.gather_nd(l, idx)
    sorted_relevance = tf.cast(
        tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    per_list_relevant_counts = tf.cumsum(sorted_relevance, axis=1)
    per_list_counts = tf.cumsum(tf.ones_like(sorted_relevance), axis=1)
    per_list_precisons = tf.where(tf.equal(per_list_counts, 0.0), 
                                  tf.zeros_like(per_list_counts), 
                                  tf.divide(per_list_relevant_counts, per_list_counts))
    
    total_precision = tf.reduce_sum(per_list_precisons * sorted_relevance, axis=1, keep_dims=True)
    total_relevance = tf.reduce_sum(sorted_relevance, axis=1, keep_dims=True)
    per_list_map = tf.where(tf.equal(total_relevance, 0.0), 
                            tf.zeros_like(total_relevance), 
                            tf.divide(total_precision, total_relevance))
    return tf.reduce_mean(per_list_map)
