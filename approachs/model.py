# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import os
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Model(object):
    def __init__(self, model_path, model_name):
        '''
        Args:
            model_path: model dirname
            model_name: model name
            ips: Inverse propensity weights, numpy.ndarray, shape(1, slate size).
                 Default is None.
        '''
        self.model_path = model_path
        self.model_name = model_name
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()
    
    @abc.abstractmethod
    def _build_graph(self):
        'build tensorlfow graph.'

    @abc.abstractmethod
    def rank(self, user_feature, query, candidates):
        '''rank candidate documents.
        Args:
            user_feature: users features, numpy.ndarray, shape(batch_size, user_feature_size)
            query: queries, numpy.ndarray, shape(batch_size, query_size)
            candidates: documents to be ranked, 
                        numpy.ndarray, shape:(batch_size, candidate_set_size, documents_feature_size)
        Returns: 
        A list of slates, where each slate is an integer array of size
            slate_size, where each element is an index into the set of
            current_documents presented
        '''

    def set_sess(self, session):
        self.sess = session

    def set_saver(self, saver):
        self.saver = saver
    
    def set_global_step(self, global_step):
        self.global_step = global_step
    
    @property
    def model_graph(self):
        return self.graph
    
    @property
    def model_session(self):
        return self.sess

    def load_model(self):
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
                print('Restore model:', ckpt.model_checkpoint_path)

    def save_model(self):
        with self.graph.as_default():
            self.saver.save(sess=self.sess, 
                            save_path=os.path.join(self.model_path, self.model_name), 
                            global_step=self.global_step)
            print('Save model.')
    
