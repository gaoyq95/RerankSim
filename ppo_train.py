# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import numpy as np
import datetime
import tensorflow as tf
from Env import Documents
from dataset import Dataset
from utils.io_utils import base_args
from approachs.ppo_model import PPOModel
from approachs.lp_model import Evaluator
from utils.io_utils import write_args 


def parse_args():
    parser = base_args()
    parser.add_argument('--algo', default='rl-rerank', type=str, help='algorithm name')
    parser.add_argument('--epochs', default=3, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--rep_num', default=32, type=int, help='samples repeat number')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--gamma', default=1.0, type=float, help='discount rate')
    parser.add_argument('--c_entropy', default=0.001, type=float, help='entropy coefficient in loss')
    parser.add_argument('--update_steps', default=10, type=int, help='train times every batch')
    parser.add_argument('--decay_steps', default=3000, type=int, help='learning rate decay steps')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    # parser.add_argument('--l2_regu', default=0.0001, type=float, help='l2 loss scale')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--evaluator_path', type=str, default='model_eval', help='evaluator ckpt dir')
    parser.add_argument('--pai', type=bool, default=False, help='run on pai')
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
    model = PPOModel(args, os.path.join(args.checkpointDir, 'rl_rerank', args.timestamp), 'ppo')
    evaluator = Evaluator(model_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), args.evaluator_path))
    with model.model_graph.as_default() as g: 
        sess = tf.Session(graph=g)
        model.set_sess(sess)

        path1, path2 = os.path.join(args.checkpointDir, 'rl_rerank', args.timestamp, 'train'), \
                        os.path.join(args.checkpointDir, 'rl_rerank', args.timestamp, 'test')
        if not os.path.isdir(path1):
            os.makedirs(path1)
        if not os.path.isdir(path2):
            os.makedirs(path2)
        train_writer = tf.summary.FileWriter(path1, g)
        test_writer = tf.summary.FileWriter(path2, g)
        
        sess.run(tf.global_variables_initializer())
        model.load_model()

        metrics_name = ['total_loss', 'returns', 'gauc', 'ndcg']
        c_entropy = args.c_entropy
        for e in range(args.epochs):
            print('Epoch: %d' % (e))
            while True:
                try:
                    x_data_id, x_data, y = train_set.read(args.batch_size)
                    # tile
                    shape1 = x_data_id.shape
                    x_data_id = np.tile(x_data_id, (1, args.rep_num)).reshape((-1, shape1[1]))
                    shape2 = x_data.shape
                    x_data = np.tile(x_data, (1, args.rep_num, 1)).reshape((-1, shape2[1], shape2[2]))
                    shape3 = y.shape
                    y = np.tile(y, (1, args.rep_num)).reshape((-1, shape3[1])) 
                    
                    act_idx_out, act_probs_one, rl_outputs, mask_arr, _, _, gauc, ndcg, summary1 = model.predict(x_data, y)
                    rewards = evaluator.predict(rl_outputs).reshape((-1, args.slate_size))
                    
                    for _ in range(args.update_steps):
                        total_loss, mean_return, summary2, step = model.train(x_data, rl_outputs, act_probs_one, act_idx_out, rewards, mask_arr, c_entropy)
                    if ((not args.pai) and (step % (10 * int(args.update_steps)) == 0)) or \
                            (args.pai and ((step+1) % (10 * int(args.update_steps)) == 0)):
                        print('step: %d'%(step), ', '.join([name+': '+str(value) for name, value in zip(
                                                            metrics_name, [total_loss, mean_return, gauc, ndcg])]))
                        train_writer.add_summary(summary1, step)
                        train_writer.add_summary(summary2, step)
                    # validation
                    if ((not args.pai) and (step % (100 * int(args.update_steps)) == 0)) or \
                            (args.pai and ((step+1) % (100 * int(args.update_steps)) == 0)):
                        # validation set
                        metrics_value = [[] for _ in range(len(metrics_name[1:]))]
                        while True:
                            try:
                                x_data_id_e, x_data_e, y_e = val_set.read(1000)
                                act_idx_out, act_probs_one, rl_outputs, mask_arr, _, _, gauc, ndcg, _ = model.predict(x_data_e, y_e)
                                rewards = evaluator.predict(rl_outputs).reshape((-1, args.slate_size))
                                
                                _, mean_return = model.get_long_reward(rewards)
                                for v, vl in zip([mean_return, gauc, ndcg], metrics_value):
                                    vl.append(v)
                            except Exception as e:
                                print(e)
                                break
                        model.save_model()
                        summary_val = tf.Summary(value=[tf.Summary.Value(tag="summary/"+name, simple_value=np.mean(val)) 
                                                        for name, val in zip(metrics_name[1:], metrics_value)])
                        test_writer.add_summary(summary_val, step)
                        print('Validation:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name[1:], metrics_value)]))
                except Exception as e:
                    print(e)
                    break
    print('Done.')
