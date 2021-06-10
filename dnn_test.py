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
from Env import UserResponse, Documents
from dnn_train import parse_args
from approachs.lp_model import Evaluator
from utils.io_utils import write_args 
from test_model import TestModel


class TestDnnModel(TestModel):
    def _model_init(self, args):
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

        with model.model_graph.as_default() as g:  
            sess = tf.Session(graph=g)
            model.set_sess(sess)
            
            sess.run(tf.global_variables_initializer())
            model.load_model()
        return model
    
    def test_on_testset(self):
        with self.model.model_graph.as_default() as g: 
            metrics_name = 'loss, gauc, ndcg, pv_auc'.split(', ')
            # test set
            metrics_value = [[] for _ in range(len(metrics_name))]
            while True:
                try:
                    x_data_id, x_data, y = self.test_set.read(1000)
                    res = self.model.evaluate(x_data, y)
                    for v, vl in zip(res, metrics_value):
                        vl.append(v)
                except Exception as e:
                    print(e)
                    break
            print('Test:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name, metrics_value)]))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # write_args(args, './model_params.txt')
    
    t = TestDnnModel(args)
    t.test_on_testset()
    t.test_on_user()

    print('Done.')

