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
from approachs.ppo_gae_model import PPOGAEModel
from utils.io_utils import write_args 
from ppo_gae_train import parse_args
from approachs.lp_model import Evaluator
from test_model import TestModel


class TestPPOGAEModel(TestModel):
    def _model_init(self, args):
        # initialize model
        model = PPOGAEModel(args, os.path.join(args.checkpointDir, 'ppo_gae_rerank', args.timestamp), 'ppo')

        with model.model_graph.as_default() as g: 
            sess = tf.Session(graph=g)
            model.set_sess(sess)
            sess.run(tf.global_variables_initializer())
            model.load_model()
        return model
    
    def test_on_testset(self):
        with self.model.model_graph.as_default() as g: 
            metrics_name = ['total_loss', 'returns', 'gauc', 'ndcg']
            metrics_value = [[] for _ in range(len(metrics_name[1:]))]
            while True:
                try:
                    x_data_id, x_data, y = self.test_set.read(1000)
                    act_idx_out, act_probs_one, rl_outputs, mask_arr, _, _, values, gauc, ndcg, _ = self.model.predict(x_data, y)
                    rewards = self.evaluator.predict(rl_outputs).reshape((-1, self.args.slate_size))
                    _, mean_return = self.model.get_long_reward(rewards)
                    for v, vl in zip([mean_return, gauc, ndcg], metrics_value):
                        vl.append(v)
                except Exception as e:
                    print(e)
                    break
            print('Test:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name[1:], metrics_value)]))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    # write_args(args, './model_params.txt')
    
    t = TestPPOGAEModel(args)
    t.test_on_testset()
    t.test_on_user()

    print('Done.')
