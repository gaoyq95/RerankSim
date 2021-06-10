# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np


NUM_DOCS = 1000                     # documents number
NUM_CANDIDATE = 15                  # candidate documents number
SLATE_SIZE = 15                     # slate size
FEATURE_SIZE = 30                   # document feature size
FEATURE_SIZE_VIS = 30               # visible document feature size


def base_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_documents', type=int, default=NUM_DOCS, help='documents number')
    parser.add_argument('--num_candidate', type=int, default=NUM_CANDIDATE, help='candidate set size')
    parser.add_argument('--slate_size', type=int, default=SLATE_SIZE, help='slate size')
    parser.add_argument('--feature_size', type=int, default=FEATURE_SIZE, help='document featurem size')
    parser.add_argument('--feature_size_vis', type=int, default=FEATURE_SIZE_VIS, help='visible document feature size')

    parser.add_argument('--checkpointDir', type=str, default='./ckpt/', help='checkpointDir')
    return parser


def write_args(args, fn):
    with open(fn, 'a', encoding='utf-8') as f:
        f.write(str(args))
        f.write('\n')