# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from .model import Model
from utils.measure import pv_auc, gauc, ndcg, mean_average_precision


def separate_by_pv(x, y):
    """
    :param x: [B,20]
    :param y: [B,20]
    :return:
    """
    y_max = tf.reduce_max(y, axis=1)  # B
    _ctr_index = tf.where(tf.less(y_max, 1.8), tf.ones_like(y_max), tf.zeros_like(y_max))  # B
    _lp_index = tf.where(tf.less(y_max, 1.8), tf.zeros_like(y_max), tf.ones_like(y_max))  # B

    ctr_index = tf.reshape(tf.where(tf.less(0.8, _ctr_index)), [-1, 1])  # 这里需要设置1
    lp_index = tf.reshape(tf.where(tf.less(0.8, _lp_index)), [-1, 1])

    ctr_x = tf.reshape(tf.gather_nd(x, ctr_index), [-1, ])
    lp_x = tf.reshape(tf.gather_nd(x, lp_index), [-1, ])

    ctr_y = tf.reshape(tf.gather_nd(y, ctr_index), [-1, ])
    lp_y = tf.reshape(tf.gather_nd(y, lp_index), [-1, ])

    ctr_num = tf.shape(ctr_y)[0]
    lp_num = tf.shape(lp_y)[0]

    return ctr_x, ctr_y, ctr_num, lp_x, lp_y, lp_num


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, l2_regu, eps=1e-6, **kwargs):
        self.eps = eps
        self._kernel_regularizer = tf.keras.regularizers.l2(l2_regu)
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', regularizer=self._kernel_regularizer, shape=input_shape[-1:], initializer=tf.keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=tf.keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        # 论文中的公式6
        self.temper = np.sqrt(d_model)
        self.dropout = tf.keras.layers.Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])  # (None, None, None)
        if mask is not None:
            mmask = tf.keras.layers.Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = tf.keras.layers.Add()([attn, mmask])
        attn = tf.keras.layers.Activation('softmax')(attn)  # (None, None, None)
        attn = self.dropout(attn)  # (None, None, None)
        output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1]))([attn, v])  # (None, None, 64)
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster;
    # mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, l2_regu, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self._kernel_regularizer = tf.keras.regularizers.l2(l2_regu)

        self.qs_layer = tf.keras.layers.Dense(n_head * d_k, kernel_regularizer=self._kernel_regularizer, use_bias=False)
        self.ks_layer = tf.keras.layers.Dense(n_head * d_k, kernel_regularizer=self._kernel_regularizer, use_bias=False)
        self.vs_layer = tf.keras.layers.Dense(n_head * d_v, kernel_regularizer=self._kernel_regularizer, use_bias=False)
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(l2_regu=l2_regu) if use_norm else None
        # 假设一个batch的样本， 该层的批输入形状（batch_size, seq_len, feature_len）
        # TimeDistributed层的作用就是把Dense层应用到这 seq_len 个具体的向量上，对每一个feature进行了一个Dense操作
        self.w_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model, kernel_regularizer=self._kernel_regularizer))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]  (None, 20, 64)
        ks = self.ks_layer(k)  # (None, 20, 64)
        vs = self.vs_layer(v)  # (None, 20, 64)

        def reshape1(x):
            s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
            x = tf.reshape(x, [s[0], s[1], n_head, d_k])  # [batch_size, len_q, n_head,  d_k]
            x = tf.transpose(x, [2, 0, 1, 3])  # [n_head，batch_size, len_q, d_k]
            x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
            return x

        qs = tf.keras.layers.Lambda(reshape1)(qs)  # (None, None, 64)
        ks = tf.keras.layers.Lambda(reshape1)(ks)  # (None, None, 64)
        vs = tf.keras.layers.Lambda(reshape1)(vs)  # (None, None, 64)

        if mask is not None:
            mask = tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, n_head, 0))(mask)
        head, attn = self.attention(qs, ks, vs, mask=mask)

        def reshape2(x):
            s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
            x = tf.reshape(x, [n_head, -1, s[1], s[2]])  # [n_head, batch_size, len_v, d_v]
            x = tf.transpose(x, [1, 2, 0, 3])  # [batch_size, len_v, n_head, d_v]
            x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
            return x

        head = tf.keras.layers.Lambda(reshape2)(head)  # (None, None, 64)

        outputs = self.w_o(head)  # (None, None, 64)
        outputs = tf.keras.layers.Dropout(self.dropout)(outputs)  # (None, None, 64)

        if not self.layer_norm: return outputs, attn

        outputs = tf.keras.layers.Add()([outputs, q])  # 残差项 (None, 20, 64)
        return self.layer_norm(outputs), attn  # (None, 20, 64)


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1, l2_regu=0.001):
        _kernel_regularizer = tf.keras.regularizers.l2(l2_regu)
        self.w_1 = tf.keras.layers.Conv1D(d_inner_hid, 1, kernel_regularizer=_kernel_regularizer, activation='relu')
        self.w_2 = tf.keras.layers.Conv1D(d_hid, 1, kernel_regularizer=_kernel_regularizer)
        self.layer_norm = LayerNormalization(l2_regu=l2_regu)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)  # (None, 20, 128)
        output = self.w_2(output)  # (None, 20, 64)
        output = self.dropout(output)  # (None, 20, 64)
        output = tf.keras.layers.Add()([output, x])  # (None, 20, 64)
        return self.layer_norm(output)  # (None, 20, 64)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, l2_regu=0.001):
        # Transformer encoder
        self._kernel_regularizer = tf.keras.regularizers.l2(l2_regu)
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, l2_regu=l2_regu)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout, l2_regu=l2_regu)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, layers=2, dropout=0.1, l2_regu=0.001):

        self._kernel_regularizer = tf.keras.regularizers.l2(l2_regu)
        self.emb_dropout = tf.keras.layers.Dropout(dropout)  # dropout
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout, l2_regu) for _ in range(layers)]

    def __call__(self, x, return_att=False, mask=None, active_layers=999):
        x = self.emb_dropout(x)  # (None, 20, 64)
        if return_att: atts = []
        for enc_layer in self.layers[:active_layers]:  # 这里只会有两层
            x, att = enc_layer(x, mask)
            if return_att:
                atts.append(att)
        return (x, atts) if return_att else x


# get position
def get_pos(batch_size, seq_len):
    outputs = np.zeros((batch_size, seq_len), dtype=np.int32)
    i = 0
    for i in range(batch_size):
        outputs[i] = np.arange(seq_len, dtype=np.int32)
        i += 1
    return outputs


class PRM(Model):
    def __init__(self, params, model_path, model_name):
        self.params = params
        super(PRM, self).__init__(model_path, model_name)

    def _build_graph(self):
        self.label_threshold = 2
        self.slate_size = self.params.slate_size

        self.item_feature_size = self.params.feature_size# item feature size
        # self.user_feature_size = self.params.feature_size_u  # user feature size
        # self.query_feature_size = self.params.feature_size_q  # query feature size

        self.learning_rate = self.params.learning_rate
        # self.epochs = self.params.epochs
        self.batch_size = self.params.batch_size

        # seq_len=20, d_feature=42, d_model=64, d_inner_hid=128,  n_head=1, d_k=64, d_v=64, layers=2, dropout=0.1
        self.seq_len = self.params.slate_size
        self.d_feature = self.item_feature_size
        self.d_model = 64
        self.d_inner_hid = 128
        self.n_head = 1
        self.d_k = 64
        self.d_v = 64
        self.layers = 2
        self.dropout = 0.3
        self.seed = 1
        self._kernel_regularizer = tf.keras.regularizers.l2(self.params.l2_regu)

        self.set_global_seeds(self.seed)
        with tf.variable_scope("PRM"):
            self.v_input = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, self.d_feature], name="v_input")  #
            self.pos_input = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name="pos_input")
            self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len], name="label_input")
            assert self.v_input.name == "PRM/v_input:0"
            assert self.pos_input.name == "PRM/pos_input:0"
            assert self.labels.name == "PRM/label_input:0"

            # input layer
            self.d0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.d_model, kernel_regularizer=self._kernel_regularizer))(self.v_input)  # [batch_size, 20, 64]
            self.encoder = Encoder(self.d_model, self.d_inner_hid, self.n_head, self.d_k, self.d_v, self.layers, self.dropout, self.params.l2_regu)

            self.pos_embedding = tf.keras.layers.Embedding(self.seq_len, self.d_model)  # pos embedding
            self.p0 = self.pos_embedding(self.pos_input)
            self.combine_input = tf.keras.layers.Add()([self.d0, self.p0])  # 公式4；
            # encoder layer
            self.enc_output = self.encoder(self.combine_input, active_layers=999)
            # Output layer
            self.time_score_dense1 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.d_model, kernel_regularizer=self._kernel_regularizer, activation='tanh'))(self.enc_output)
            self.time_score_dense2 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1, kernel_regularizer=self._kernel_regularizer))(self.time_score_dense1)
            self.flat = tf.keras.layers.Flatten()(self.time_score_dense2)
            self.score_output = tf.keras.layers.Activation(activation='softmax')(self.flat)  # (B,N)
            self.score = tf.identity(self.score_output, name="score")
            assert self.score.name == "PRM/score:0"

        pv_size = self.params.slate_size
        # loss
        self.processed_labels = self.labels / (tf.reduce_sum(self.labels, axis=1, keep_dims=True) + 1e-5)  # yi/(sum(y))
        prob = self.processed_labels * tf.log(tf.clip_by_value(self.score_output, 1e-5, 1.0))  # (B,N)

        self.loss = tf.reduce_mean(-tf.reduce_sum(prob, axis=1))
        self.l2_loss = tf.losses.get_regularization_loss()
        self.loss += self.l2_loss

        # opt
        self.set_global_step(tf.train.create_global_step())
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
        # saver
        self.set_saver(tf.train.Saver())

        y = tf.reshape(self.labels, (-1,))
        y_lp = tf.where(tf.less(y, 1.5), tf.zeros_like(y), tf.ones_like(y))
        y_click = tf.where(tf.less(y, 0.5), tf.zeros_like(y), tf.ones_like(y))

        self.gauc_lp = gauc(p=self.score, l=y_lp, n=pv_size)
        self.gauc_click = gauc(p=self.score, l=y_click, n=pv_size)
        self.ndcg_lp = ndcg(p=tf.reshape(self.score, (-1, pv_size)), l=tf.reshape(y_lp, (-1, pv_size)), k=pv_size)
        self.ndcg_click = ndcg(p=tf.reshape(self.score, (-1, pv_size)), l=tf.reshape(y_click, (-1, pv_size)), k=pv_size)
        self.map_lp = mean_average_precision(p=tf.reshape(self.score, (-1, pv_size)), l=tf.reshape(y_lp, (-1, pv_size)))
        self.map_click = mean_average_precision(p=tf.reshape(self.score, (-1, pv_size)), l=tf.reshape(y_click, (-1, pv_size)))

        # summary
        self.train_merged, self.test_merged = self._build_summary()

    def _build_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar(name="loss", tensor=self.loss, collections=['train', 'test'])
            tf.summary.scalar(name="gauc_lp", tensor=self.gauc_lp, collections=['train', 'test'])
            tf.summary.scalar(name="gauc_click", tensor=self.gauc_click, collections=['train', 'test'])
            tf.summary.scalar(name="ndcg_lp", tensor=self.ndcg_lp, collections=['train', 'test'])
            tf.summary.scalar(name="ndcg_click", tensor=self.ndcg_click, collections=['train', 'test'])
            tf.summary.scalar(name="map_lp", tensor=self.map_lp, collections=['train', 'test'])
            tf.summary.scalar(name="map_click", tensor=self.map_click, collections=['train', 'test'])

        train_merged = tf.summary.merge_all('train')
        test_merged = tf.summary.merge_all('test')
        return train_merged, test_merged

    def train(self, slates=None, labels=None):  # users, queries, slates, labels
        """
        :param users: (batch_size, 1)
        :param queries: (batch_size, 1)
        :param slates: (batch_size, 20, self.item_feature_size)
        :param labels: (batch_size, 20)
        :return:
        """
        v_input = slates.reshape([-1, self.slate_size, self.item_feature_size])
        labels_input = labels.reshape([-1, self.slate_size])
        pos_input = get_pos(v_input.shape[0], self.slate_size)

        with self.graph.as_default():
            _, loss, gauc_lp, gauc_click, ndcg_lp_v, ndcg_click_v, map_lp_v, map_click_v, step, summary = self.sess.run(
                [self.opt, self.loss, self.gauc_lp, self.gauc_click, self.ndcg_lp, self.ndcg_click,
                 self.map_lp, self.map_click, self.global_step, self.train_merged],
                feed_dict={self.v_input: v_input,
                           self.pos_input: pos_input,
                           self.labels: labels_input,
                           tf.keras.backend.learning_phase(): True})

        # return loss, gauc_lp, gauc_click, ndcg_lp_v, ndcg_click_v, map_lp_v, map_click_v, step, summary
        return loss, gauc_click, ndcg_click_v, step, summary


    def evaluate(self, slates=None, labels=None):
        """
        :param users:
        :param queries:
        :param slates:
        :param labels:
        :return:
        """
        v_input = slates.reshape([-1, self.slate_size, self.item_feature_size])
        labels_input = labels.reshape([-1, self.slate_size])
        pos_input = get_pos(v_input.shape[0], self.slate_size)

        with self.graph.as_default():
            p0, loss, gauc_lp, gauc_click, ndcg_lp_v, ndcg_click_v, map_lp_v, map_click_v, step, summary = self.sess.run(
                [self.p0, self.loss, self.gauc_lp, self.gauc_click, self.ndcg_lp, self.ndcg_click,
                 self.map_lp, self.map_click, self.global_step, self.train_merged],
                feed_dict={self.v_input: v_input,
                           self.pos_input: pos_input,
                           self.labels: labels_input,
                           tf.keras.backend.learning_phase(): False})

        # return p0, loss, gauc_lp, gauc_click, ndcg_lp_v, ndcg_click_v, map_lp_v, map_click_v, step, summary
        return loss, gauc_click, ndcg_click_v, step, summary

    def rank(self, candidates):
        """
        :param users:       (None, 20)
        :param candidates:  (None, 20, self.feature_size)
        :param queries:     (None, 22)
        :return:
        """
        v_input = candidates.reshape([-1, self.slate_size, self.item_feature_size])
        pos_input = get_pos(v_input.shape[0], self.slate_size)

        with self.graph.as_default():
            # predict
            prediction = self.sess.run(self.score, feed_dict={self.v_input: v_input,
                                                              self.pos_input: pos_input,
                                                              tf.keras.backend.learning_phase(): False})  # predict model
        # print(prediction)
        prediction = prediction.reshape((-1, self.slate_size))
        res = np.argsort(-1 * prediction, axis=1)
        return res

    def set_global_seeds(self, seed):
        tf.set_random_seed(seed)
