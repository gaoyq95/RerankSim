# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import datetime
import numpy as np
import tensorflow as tf

from Env import UserResponse, Documents
from dataset import Dataset
from utils.io_utils import base_args
from approachs.lp_model import LPModel
from utils.io_utils import write_args 


def parse_args():
    parser = base_args()
    parser.add_argument('--algo', default='ctr_model', type=str, help='algorithm name')
    parser.add_argument('--epochs', default=200, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay_steps', default=500, type=int, help='learning rate decay steps')
    parser.add_argument('--decay_rate', default=0.99, type=float, help='learning rate decay rate')
    parser.add_argument('--keep_prob', default=0.8, type=float, help='dropout keep_prob')
    parser.add_argument('--l2_regu', default=0.0, type=float, help='l2 loss scale')
    parser.add_argument( '--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    FLAGS, _ = parser.parse_known_args()
    return FLAGS

def dist(v1, v2):
    s1 = np.sum(v1, axis=1)
    s1 = (s1-np.min(s1)) / (np.max(s1)-np.min(s1))
    s2 = np.sum(v2, axis=1)
    s2 = (s2-np.min(s2)) / (np.max(s2)-np.min(s2))
    m, var = np.mean(s1 - s2), np.var(s1 - s2)
    return m, var


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # write_args(args, './model_params.txt')
    # dataset
    doc = Documents(num_total=args.num_documents, 
                    num_feature=args.feature_size, 
                    num_visible=args.feature_size_vis, 
                    seed=100)
    train_set = Dataset(doc, 'train')
    val_set = Dataset(doc, 'validation')
    test_set = Dataset(doc, 'test')
    model = LPModel(args, os.path.join(args.checkpointDir, 'Evaluator', args.timestamp), 'ctr_model')

    with model.model_graph.as_default() as g: 
        sess = tf.Session(graph=g)
        model.set_sess(sess)

        path1, path2 = os.path.join(args.checkpointDir, 'Evaluator', args.timestamp, 'train'), \
                        os.path.join(args.checkpointDir, 'Evaluator', args.timestamp, 'test')
        if not os.path.isdir(path1):
            os.makedirs(path1)
        if not os.path.isdir(path2):
            os.makedirs(path2)
        train_writer = tf.summary.FileWriter(path1, g)
        test_writer = tf.summary.FileWriter(path2, g)
        sess.run(tf.global_variables_initializer())
        model.load_model()
 
        metrics_name = 'loss, ctr_gauc, ctr_pauc, ctr_ndcg'.split(', ')
        for e in range(args.epochs):
            print('Epoch: %d' % (e))
            while True:
                try:
                    _, x_data, y, ctr_user = train_set.read_e(args.batch_size)
                    res = model.train(x_data, y)
                    ctr_pred, step, summary = res[0], res[-1], res[-2]
                    mean, variance = dist(ctr_pred.reshape((-1, args.slate_size)), ctr_user)
                    summary_ = tf.Summary(value=[tf.Summary.Value(tag="summary/mean", simple_value=mean), 
                                                 tf.Summary.Value(tag="summary/variance", simple_value=variance)
                                                 ])
                    if step % 30 == 0:
                        print('step: %d'%(step), ', '.join([name+': '+str(value) for name, value in zip(metrics_name, res[1:])]))
                        train_writer.add_summary(summary, step)
                        train_writer.add_summary(summary_, step)
                    # validation
                    if step % 300 == 0:
                        metrics_value = [[] for _ in range(len(metrics_name))]
                        mean_a, variance_a = [], []
                        while True:
                            try:
                                _, x_data_e, y_e, ctr_user_e = val_set.read_e(1000)
                                res = model.evaluate(x_data_e, y_e)
                                ctr_pred = res[0]
                                for v, vl in zip(res[1:], metrics_value):
                                    vl.append(v)
                                mean, variance = dist(ctr_pred.reshape((-1, args.slate_size)), ctr_user_e)
                                mean_a.append(mean)
                                variance_a.append(variance)
                            except Exception as e:
                                print(e)
                                break
                        model.save_model()
                        summary_val = tf.Summary(value=[tf.Summary.Value(tag="summary/"+name, simple_value=np.mean(val)) 
                                                        for name, val in zip(metrics_name, metrics_value)])
                        summary_val_ = tf.Summary(value=[tf.Summary.Value(tag="summary/mean", simple_value=np.mean(mean_a)), 
                                                         tf.Summary.Value(tag="summary/variance", simple_value=np.mean(variance_a))
                                                         ])
                        test_writer.add_summary(summary_val, step)
                        test_writer.add_summary(summary_val_, step)
                        print('Validation:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name, metrics_value)]))
                except Exception as e:
                    print(e)
                    break
        # test
        metrics_value = [[] for _ in range(len(metrics_name))]
        mean_a, variance_a = [], []
        while True:
            try:
                _, x_data_t, y_t, ctr_user_t = test_set.read_e(1000)
                res = model.evaluate(x_data_t, y_t)
                ctr_pred = res[0]
                for v, vl in zip(res[1:], metrics_value):
                    vl.append(v)
                mean, variance = dist(ctr_pred.reshape((-1, args.slate_size)), ctr_user_t)
                mean_a.append(mean)
                variance_a.append(variance)
            except Exception as e:
                print(e)
                break
        print('Test:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name, metrics_value)]))
print('Done.')
