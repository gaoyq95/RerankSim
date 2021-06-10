# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import datetime
import numpy as np
import tensorflow as tf

from dataset import Dataset
from Env import UserResponse, Documents
from approachs.lp_model import Evaluator


class TestModel(object):
    def __init__(self, args):
        self.args = args
        
        self.doc = Documents(num_total=args.num_documents, 
                        num_feature=args.feature_size, 
                        num_visible=args.feature_size_vis, 
                        seed=100)
        self.user = UserResponse(doc_feature_size=args.feature_size, 
                            slate_size=args.slate_size, 
                            seed=100)
        self.evaluator = Evaluator(model_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), args.evaluator_path))
        self.test_set = Dataset(self.doc, 'test')
        self.model = self._model_init(args)
    
    def _model_init(self, args):
        raise NotImplementedError

    def test_on_testset(self):
        raise NotImplementedError

    def test_on_user(self):
        with self.model.model_graph.as_default() as g: 
            user_res_all, eval_pred_all = [], []
            while True:
                try:
                    x_data_id, x_data, y = self.test_set.read(1000)
                    rank_id = self.model.rank(x_data)
                    batch_id = np.tile(np.arange(rank_id.shape[0]).reshape((-1, 1)), (1, rank_id.shape[1]))
                    x_data_id_rank = x_data_id[(batch_id, rank_id)]
                    # user
                    slate = self.doc.get_feature_all(x_data_id_rank.reshape((-1,)))
                    slate = slate.reshape((x_data_id_rank.shape[0], x_data_id_rank.shape[1], -1))
                    res = self.user.response(slate)
                    user_res_all.append(res)
                    # evaluator
                    slate_ = self.doc.get_feature_visible(x_data_id_rank.reshape((-1,)))
                    slate_ = slate_.reshape((x_data_id_rank.shape[0], x_data_id_rank.shape[1], -1))
                    pred = self.evaluator.predict(slate_)
                    eval_pred_all.append(pred)
                except Exception as e:
                    print(e)
                    break
            user_res_all = np.concatenate(user_res_all, axis=0)
            eval_pred_all = np.concatenate(eval_pred_all, axis=0)
            print('True score:', np.mean(np.sum(user_res_all, axis=1)))
            print('Eval score:', np.mean(np.sum(eval_pred_all, axis=1)))
