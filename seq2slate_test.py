# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import numpy as np
import datetime
import tensorflow as tf
from Env import UserResponse, Documents
from dataset import Dataset
from utils.io_utils import base_args
from approachs.rl_sl_model import SLModel
from utils.io_utils import write_args 
from seq2slate_train import parse_args
from approachs.lp_model import Evaluator
from test_model import TestModel


class TestSeqModel(TestModel):
    def _model_init(self, args):
        # initialize model
        model = SLModel(args, os.path.join(args.checkpointDir, 'seq2seq', args.loss, args.timestamp), 'seq2slate')
        with model.model_graph.as_default() as g: 
            sess = tf.Session(graph=g)
            model.set_sess(sess)
            sess.run(tf.global_variables_initializer())
            model.load_model()
        return model
    
    def test_on_testset(self):
        with self.model.model_graph.as_default() as g: 
            # test set
            metrics_name = ['total_loss', 'gauc', 'ndcg']
            metrics_value = [[] for _ in range(len(metrics_name))]
            while True:
                try:
                    _, x_data, y = self.test_set.read(1000)
                    act_idx_out, act_probs_one, rl_outputs, mask_arr, _, _, gauc, ndcg, _ = self.model.predict(x_data, y)
                    total_loss, _, _ = self.model.evaluate(x_data, y, rl_outputs, mask_arr)
                    for v, vl in zip([total_loss, gauc, ndcg], metrics_value):
                        vl.append(v)
                except Exception as e:
                    print(e)
                    break
            print('Test:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name, metrics_value)]))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # write_args(args, './model_params.txt')
    
    t = TestSeqModel(args)
    t.test_on_testset()
    t.test_on_user()

    print('Done.')
