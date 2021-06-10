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
from approachs.discriminator import Discriminator
from utils.io_utils import write_args 


def parse_args():
    parser = base_args()
    parser.add_argument('--algo', default='rl-rerank-gd', type=str, help='algorithm name')
    parser.add_argument('--epochs', default=3, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--rep_num', default=32, type=int, help='samples repeat number')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--c_entropy', default=0.001, type=float, help='entropy coefficient in loss')
    parser.add_argument('--update_steps', default=10, type=int, help='train times every batch')
    parser.add_argument('--decay_steps', default=3000, type=int, help='learning rate decay steps')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    # parser.add_argument('--l2_regu', default=0.0001, type=float, help='l2 loss scale')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--evaluator_path', type=str, default='model_eval', help='evaluator ckpt dir')
    # discriminator
    parser.add_argument('--c_entropy_d', default=0.001, type=float, help=' of discriminator') 
    parser.add_argument('--keep_prob_d', default=0.8, type=float, help=' of discriminator')
    parser.add_argument('--learning_rate_d', default=1e-5, type=float, help=' of discriminator')
    parser.add_argument('--decay_steps_d', default=1000, type=int, help=' of discriminator')
    parser.add_argument('--decay_rate_d', default=0.99, type=float, help=' of discriminator')
    parser.add_argument('--c_rewards_d', default=0.005, type=float, help=' of discriminator')
    parser.add_argument('--update_rate_d', default=5, type=int, help=' of discriminator')
    parser.add_argument('--timestamp_d', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
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
    model = PPOModel(args, os.path.join(args.checkpointDir, 'rl_rerank_gd', args.timestamp), 'ppo')
    evaluator = Evaluator(model_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), args.evaluator_path))
    discriminator = Discriminator(args, os.path.join(args.checkpointDir, 'discriminator', args.timestamp_d), 'discriminator')
    
    with model.model_graph.as_default() as g: 
        sess = tf.Session(graph=g)
        model.set_sess(sess)

        path1, path2 = os.path.join(args.checkpointDir, 'rl_rerank_gd', args.timestamp, 'train'), \
                        os.path.join(args.checkpointDir, 'rl_rerank_gd', args.timestamp, 'test')
        if not os.path.isdir(path1):
            os.makedirs(path1)
        if not os.path.isdir(path2):
            os.makedirs(path2)
        train_writer = tf.summary.FileWriter(path1, g)
        test_writer = tf.summary.FileWriter(path2, g)
        
        sess.run(tf.global_variables_initializer())
        model.load_model()
    
    with discriminator.model_graph.as_default() as g_d: 
        sess_d = tf.Session(graph=g_d)
        discriminator.set_sess(sess_d)

        path1_d, path2_d = os.path.join(args.checkpointDir, 'discriminator', args.timestamp_d, 'train'), \
                            os.path.join(args.checkpointDir, 'discriminator', args.timestamp_d, 'test')
        if not os.path.isdir(path1_d):
            os.makedirs(path1_d)
        if not os.path.isdir(path2_d):
            os.makedirs(path2_d)
        train_writer_d = tf.summary.FileWriter(path1_d, g_d)
        test_writer_d = tf.summary.FileWriter(path2_d, g_d)

        sess_d.run(tf.global_variables_initializer())
        discriminator.load_model()

    metrics_name = ['total_loss', 'returns', 'gauc', 'ndcg']
    metrics_name_d = ['total_loss', 'pv_auc']
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
                
                act_idx_out, act_probs_one, rl_outputs, mask_arr, lp_x_data, _, gauc, ndcg, summary1 = model.predict(x_data, y)
                pred = evaluator.predict(rl_outputs).reshape((-1, args.slate_size))
                d_preds, d_rewards = discriminator.predict(rl_outputs)
                rewards = pred + d_rewards.reshape((-1, args.slate_size)) * args.c_rewards_d
                # train rl-rerank
                for _ in range(args.update_steps):
                    total_loss, mean_return, summary2, step = model.train(x_data, rl_outputs, act_probs_one, act_idx_out, rewards, mask_arr, args.c_entropy)
                
                if step % (10 * int(args.update_steps)) == 0:
                    print('ppo, step: %d'%(step), ', '.join([name+': '+str(value) for name, value in zip(
                                                        metrics_name, [total_loss, mean_return, gauc, ndcg])]))
                    train_writer.add_summary(summary1, step)
                    train_writer.add_summary(summary2, step)
                
                # train discriminator
                if step % (args.update_rate_d * int(args.update_steps)) == 0:
                    d_label = np.array([1] * lp_x_data.shape[0] + [0] * rl_outputs.shape[0])
                    d_x_data = np.concatenate([lp_x_data, rl_outputs], axis=0)
                    d_pred, d_pv_auc, d_total_loss, d_summary = discriminator.train(d_x_data, d_label)
                    print('dis, step: %d'%(step), ', '.join([name+': '+str(value) for name, value in zip(
                                                        metrics_name_d, [d_total_loss, d_pv_auc])]))
                    train_writer_d.add_summary(d_summary, step)
                
                # validation
                if step % (100 * int(args.update_steps)) == 0:
                    metrics_value = [[] for _ in range(len(metrics_name[1:]))]
                    metrics_value_d = [[] for _ in range(len(metrics_name_d))]
                    while True:
                        try:
                            x_data_id_e, x_data_e, y_e = val_set.read(1000)
                            # rl-rerank
                            act_idx_out, act_probs_one, rl_outputs, mask_arr, lp_x_data, _, gauc, ndcg, _ = model.predict(x_data_e, y_e)
                            pred = evaluator.predict(rl_outputs).reshape((-1, args.slate_size))
                            d_preds, d_rewards = discriminator.predict(rl_outputs)
                            rewards = pred + d_rewards.reshape((-1, args.slate_size)) * args.c_rewards_d
                            _, mean_return = model.get_long_reward(rewards)
                            for v, vl in zip([mean_return, gauc, ndcg], metrics_value):
                                vl.append(v)
                            # discriminator
                            d_label_test = np.array([1] * (x_data_e.shape[0] * args.slate_size) + [0] * (x_data_e.shape[0] * args.slate_size))
                            d_x_data_test = np.concatenate([lp_x_data, rl_outputs], axis=0)
                            _, d_pv_auc_test, d_total_loss_test, _ = discriminator.test(d_x_data_test, d_label_test)
                            for v, vl in zip([d_total_loss_test, d_pv_auc_test], metrics_value_d):
                                vl.append(v)
                        except Exception as e:
                            print(e)
                            break
                    model.save_model()
                    summary_val = tf.Summary(value=[tf.Summary.Value(tag="summary/"+name, simple_value=np.mean(val)) 
                                                    for name, val in zip(metrics_name[1:], metrics_value)])
                    test_writer.add_summary(summary_val, step)
                    print('Validation:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name[1:], metrics_value)]))
                    discriminator.save_model()
                    summary_val_d = tf.Summary(value=[tf.Summary.Value(tag="summary/"+name, simple_value=np.mean(val)) 
                                                        for name, val in zip(metrics_name_d, metrics_value_d)])
                    test_writer_d.add_summary(summary_val_d, step)
                    print('Validation_d:', ', '.join([name+': '+str(np.mean(val)) for name, val in zip(metrics_name_d, metrics_value_d)]))
            except Exception as e:
                print(e)
                break
print('Done.')


