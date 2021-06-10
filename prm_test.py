# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import datetime
import numpy as np
import tensorflow as tf
from approachs.PRMModel import PRM
from utils.io_utils import base_args
from dataset import Dataset
from Env import UserResponse, Documents
from prm_train import parse_args
from approachs.lp_model import Evaluator
from utils.io_utils import write_args 
from test_model import TestModel


class TestPRMModel(TestModel):
    def _model_init(self, args):
        # initialize model
        model = PRM(params=args, model_path=os.path.join(args.checkpointDir, args.algo, args.loss), model_name=args.algo)
        with model.model_graph.as_default() as g:  
            sess = tf.Session(graph=g)
            model.set_sess(sess)
            sess.run(tf.global_variables_initializer())
            model.load_model()
        return model
    
    def test_on_testset(self):
        with self.model.model_graph.as_default() as g: 
            metrics_name = 'loss, gauc, ndcg'.split(', ')
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
    
    t = TestPRMModel(args)
    t.test_on_testset()
    t.test_on_user()

    print('Done.')

