# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import datetime
import numpy as np
import tensorflow as tf
from approachs.pointwise_model import PointWise
from approachs.pairwise_model import PairWise
from approachs.listwise_model import ListWise
from approachs.groupwise_model import GroupWise
from approachs.softrank import SoftRank
from utils.io_utils import base_args
from dataset import Dataset
from Env import Documents
from utils.io_utils import write_args

def parse_args():
    parser = base_args()
    parser.add_argument('--algo', default='PointWise', type=str, help='PointWise, PairWise, ListWise, GroupWise, SoftRank')
    parser.add_argument('--epochs', default=30, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay_steps', default=1000, type=int, help='learning rate decay steps')
    parser.add_argument('--decay_rate', default=0.99, type=float, help='learning rate decay rate')
    parser.add_argument('--loss', default='mse', type=str, help='loss function')
    parser.add_argument('--l2_regu', default=0.0001, type=float, help='l2 loss scale')
    parser.add_argument('--group_size', default=10, type=int, help='GroupWise, group size.')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--evaluator_path', type=str, default='model_eval', help='evaluator ckpt dir')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


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
    # initialize model
    model_dict = {
        'PointWise': PointWise, 'PairWise': PairWise, 'ListWise': ListWise, 'SoftRank': SoftRank
    }
    if args.algo in ['PointWise', 'PairWise', 'ListWise', 'SoftRank']:
        model = model_dict[args.algo](args, os.path.join(args.checkpointDir, args.algo, args.loss, args.timestamp), args.algo)
    elif args.algo == 'GroupWise':
        model = GroupWise(args, os.path.join(args.checkpointDir, args.algo, str(args.group_size), args.timestamp), args.algo)
    else:
        raise NotImplementedError('')

    with model.model_graph.as_default() as g:                                           # rarank model graph
        sess = tf.Session(graph=g)
        model.set_sess(sess)
        
        if args.algo == 'GroupWise':
            path1, path2 = os.path.join(args.checkpointDir, args.algo, str(args.group_size), args.timestamp, 'train'), \
                            os.path.join(args.checkpointDir, args.algo, str(args.group_size), args.timestamp, 'test')
        else:
            path1, path2 = os.path.join(args.checkpointDir, args.algo, args.loss, args.timestamp, 'train'), \
                            os.path.join(args.checkpointDir, args.algo, args.loss, args.timestamp, 'test')
        train_writer = tf.summary.FileWriter(path1, g)
        test_writer = tf.summary.FileWriter(path2, g)
        if not os.path.isdir(path1):
            os.makedirs(path1)
        if not os.path.isdir(path2):
            os.makedirs(path2)
        sess.run(tf.global_variables_initializer())
        model.load_model()

        metrics_name = 'loss, gauc, ndcg, pv_auc'.split(', ')
        for e in range(args.epochs):
            print('Epoch: %d' % (e))
            while True:
                try:
                    x_data_id, x_data, y = train_set.read(args.batch_size)
                    res = model.train(x_data, y)
                    step, summary = res[-2], res[-1]
                    if step % 10 == 0:
                        print('step: %d'%(step), ', '.join([name+': '+str(value) for name, value in zip(metrics_name, res)]))
                        train_writer.add_summary(summary, step)
                    # validation
                    if step % 300 == 0:
                        # validation set
                        metrics_value = [[] for _ in range(len(metrics_name))]
                        while True:
                            try:
                                x_data_id_e, x_data_e, y_e = val_set.read(1000)
                                res = model.evaluate(x_data_e, y_e)
                                for v, vl in zip(res, metrics_value):
                                    vl.append(v)
                            except Exception as e:
                                print(e)
                                break
                        model.save_model()
                        summary_val = tf.Summary(value=[tf.Summary.Value(tag="summary/"+name, simple_value=np.mean(val)) 
                                                        for name, val in zip(metrics_name, metrics_value)])
                        test_writer.add_summary(summary_val, step)
                        print('Validation:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name, metrics_value)]))
                except Exception as e:
                    print(e)
                    break
    print('Done.')

