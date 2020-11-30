# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import copy
import os
import tensorflow as tf
from Env import UserResponse, Documents
from utils.io_utils import base_args

DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')


class LogDataset(object):
    def __init__(self, document, user, num_candidate, slate_size, train_size, val_size, test_size, path, seed):
        self.doc = document
        self.user = user
        self.num_candidate = num_candidate
        self.slate_size = slate_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.path = path
        self.rng = np.random.RandomState(seed=seed)

    def log_data(self):
        print('Log data...')
        x_data_id = []
        x_data = []
        x_data_user = []
        total = self.train_size + self.val_size + self.test_size 
        for i in range(total):
            if (i+1) % 10000 == 0: print(i+1, '/', total)
            doc_ids = self.rng.permutation(self.doc.num_total)[:self.num_candidate]
            x_data_id.append(doc_ids)
            x_data.append(self.doc.get_feature_visible(doc_ids))
            x_data_user.append(self.doc.get_feature_all(doc_ids))
        x_data_id, x_data, x_data_user = map(np.array, [x_data_id, x_data, x_data_user])
        
        score = self.rank_random(x_data)
        sort_id = np.argsort(-1 * score, axis=1)
        sort_id = sort_id[:, :self.slate_size]
        batch_id = np.tile(np.arange(sort_id.shape[0]).reshape((-1, 1)), (1, sort_id.shape[1]))
        items_id, items, items_user = map(lambda x: x[(batch_id, sort_id)], [x_data_id, x_data, x_data_user] )

        res = self.user.response(items_user)
        click = self.user.feedback(res)
        self.items_id = items_id
        self.items = items
        self.click = click
        self.res = res
        print('Log data: done.')

    def split_data(self):
        # # shuffle
        shuffle_id = self.rng.permutation(self.items.shape[0])
        self.items_id, self.items, self.click, self.res = map(lambda x: x[shuffle_id], [self.items_id, self.items, self.click, self.res])
        # self.train_set = {'docs_id': self.items_id[:self.train_size], 
        #                   'docs':    self.items[:self.train_size], 
        #                   'click':   self.click[:self.train_size], 
        #                   'res':     self.res[:self.train_size] }
        # self.val_set = {'docs_id': self.items_id[self.train_size : self.train_size + self.val_size], 
        #                 'docs':    self.items[self.train_size : self.train_size + self.val_size], 
        #                 'click':   self.click[self.train_size : self.train_size + self.val_size], 
        #                 'res':     self.res[self.train_size : self.train_size + self.val_size] }
        # self.test_set = {'docs_id': self.items_id[-self.test_size:], 
        #                  'docs':    self.items[-self.test_size:], 
        #                  'click':   self.click[-self.test_size:], 
        #                  'res':     self.res[-self.test_size:] }       
        # # split
        self.train_set = {'docs_id': self.items_id[:self.train_size], 
                          'click':   self.click[:self.train_size], 
                          'res':     self.res[:self.train_size] }
        self.val_set = {'docs_id': self.items_id[self.train_size : self.train_size + self.val_size], 
                        'click':   self.click[self.train_size : self.train_size + self.val_size], 
                        'res':     self.res[self.train_size : self.train_size + self.val_size] }
        self.test_set = {'docs_id': self.items_id[-self.test_size:], 
                         'click':   self.click[-self.test_size:], 
                         'res':     self.res[-self.test_size:] }    

    def rank_random(self, x_data):
        return self.rng.rand(x_data.shape[0], x_data.shape[1])

    def save_data(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        for fn, value in zip(['train-set.pk', 'validation-set.pk', 'test-set.pk'], 
                             [self.train_set, self.val_set, self.test_set]):
            with open(os.path.join(self.path, fn), 'wb') as f:
                pickle.dump(value, f)
                print('save %s.' % (fn))


class Dataset(object):
    def __init__(self, documents, dataset):
        self.documents = documents
        self.dataset = dataset
        self.path = DATASET_PATH
        if self.dataset == 'train':
            self.data = self.restore_data(os.path.join(self.path, 'train-set.pk'))
        elif self.dataset == 'validation':
            self.data = self.restore_data(os.path.join(self.path, 'validation-set.pk'))
        elif self.dataset == 'test':
            self.data = self.restore_data(os.path.join(self.path, 'test-set.pk'))
        else:
            raise ValueError('Not dataset: %s.' % (self.dataset))
        # self.docs_id, self.docs, self.click, self.res = self.data['docs_id'], self.data['docs'], self.data['click'], self.data['res']
        self.docs_id, self.click, self.res = self.data['docs_id'], self.data['click'], self.data['res']
        self.reset()

    def reset(self):
        assert (self.size > 0), 'dataset is empty.'
        if self.dataset == 'train':
            self.id_shuffle = np.random.permutation(self.size)
        elif self.dataset == 'validation':
            self.id_shuffle = np.arange(self.size)
        elif self.dataset == 'test':
            self.id_shuffle = np.arange(self.size)

    def restore_data(self, file_name):
        try:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                print('========Load %s.========' % (file_name))
            return data
        except Exception as identifier:
            print(identifier)
            return None

    def read(self, batch_size):
        '''
        Args:
            batch_size: batch size
        Returns:
            docs_id: documents indices, shape: (batch_size, slate_size(15))
            docs: documents feature, shape: (batch_size, slate_size(15), feature_size(30))
            label: click(1) or not(0), shape: (batch_size, slate_size(15))
        '''
        if self.id_shuffle.size >= batch_size:
            id_batch = self.id_shuffle[:batch_size]
            self.id_shuffle = self.id_shuffle[batch_size:]
            docs_id, label = map(lambda x: x[id_batch], [self.docs_id, self.click])
            docs = self.documents.get_feature_visible(docs_id.reshape((-1,))).reshape((docs_id.shape[0], docs_id.shape[1], -1))
            return docs_id, docs, label
        else:
            self.reset()
            raise ValueError('Dataset-%s is exhausted.'%(self.dataset))

    def read_e(self, batch_size):
        if self.id_shuffle.size >= batch_size:
            id_batch = self.id_shuffle[:batch_size]
            self.id_shuffle = self.id_shuffle[batch_size:]
            docs_id, label, res = map(lambda x: x[id_batch], [self.docs_id, self.click, self.res])
            docs = self.documents.get_feature_visible(docs_id.reshape((-1,))).reshape((docs_id.shape[0], docs_id.shape[1], -1))
            return docs_id, docs, label, res
        else:
            self.reset()
            raise ValueError('Dataset is exhausted.')

    @property
    def size(self):
        assert self.docs_id.shape[0] == self.click.shape[0]
        return self.docs_id.shape[0]


if __name__ == "__main__":
    FLAGS, _ = base_args().parse_known_args()

    # log data
    doc = Documents(num_total=FLAGS.num_documents, 
                    num_feature=FLAGS.feature_size, 
                    num_visible=FLAGS.feature_size_vis, 
                    seed=100)
    user = UserResponse(doc_feature_size=FLAGS.feature_size, 
                        slate_size=FLAGS.slate_size, 
                        seed=100)
    data_logger = LogDataset(document=doc, 
                             user=user, 
                             num_candidate=FLAGS.num_candidate, 
                             slate_size=FLAGS.slate_size,
                             train_size=int(3e5), 
                             val_size=int(1e4), 
                             test_size=int(5e4), 
                             path=DATASET_PATH,
                             seed=100)
    data_logger.log_data()
    data_logger.split_data()
    data_logger.save_data()

    # # read data
    # doc = Documents(num_total=FLAGS.num_documents, 
    #                 num_feature=FLAGS.feature_size, 
    #                 num_visible=FLAGS.feature_size_vis, 
    #                 seed=100)
    # train_set = Dataset(doc, 'train')
    # val_set = Dataset(doc, 'validation')
    # test_set = Dataset(doc, 'test')

    # print(train_set.size)
    # print(np.mean(train_set.res, axis=0))

    # print(val_set.size)
    # print(np.mean(val_set.res, axis=0))

    # print(test_set.size)
    # print(np.mean(test_set.res, axis=0))
    