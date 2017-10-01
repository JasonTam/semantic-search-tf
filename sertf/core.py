import pandas as pd
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import itertools as it
import pickle
import os

from tensorflow.contrib.tensorboard.plugins import projector

from time import time

LOG_DIR = '/tmp/tensorboard-logs/semantic/'


def get_win(seq, win_size=4):
    ii = np.random.randint(0, max(0, len(seq) - win_size) + 1)
    return seq[ii:ii + win_size]


class Model(object):
    def __init__(self,
                 vocab,
                 n_entities,
                 word_emb_size=64,
                 entity_emb_size=16,
                 n_negs_per_pos=10,
                 l2_emb=1e-2,
                 l2_map=1e-2,
                 batch_size=1024,
                 opt=tf.train.AdamOptimizer(
                     learning_rate=0.001, beta1=0.9, beta2=0.999),
                 ):

        self.vocab = vocab
        self.n_entities = n_entities

        self.word_emb_size = word_emb_size
        self.entity_emb_size = entity_emb_size
        self.n_negs_per_pos = n_negs_per_pos
        self.l2_emb = l2_emb
        self.l2_map = l2_map
        self.batch_size = batch_size

        self.opt = opt
        self.ph_d = {}

        self.loss = self.build_loss_graph()
        self.train_op = self.get_train_op()

    def build_loss_graph(self):
        with tf.variable_scope('reg'):
            reg_emb = tf.contrib.layers.l2_regularizer(self.l2_emb)
            reg_map = tf.contrib.layers.l2_regularizer(self.l2_map)

        with tf.variable_scope('emb'):
            word_embs = tf.get_variable(
                name='word',
                shape=(len(self.vocab), self.word_emb_size),
                initializer=None,  # use default glorot
                regularizer=reg_emb
            )
            entity_embs = tf.get_variable(
                name='item',
                shape=(self.n_entities, self.entity_emb_size),
                initializer=None,  # use default glorot
                regularizer=reg_emb
            )

        with tf.variable_scope('ph'):
            ngram_ph = tf.sparse_placeholder(tf.int32)
            pos_entity_ph = tf.placeholder(
                tf.int32, shape=[self.batch_size, 1])
            neg_entities_ph = tf.placeholder(
                tf.int32, shape=[self.batch_size, self.n_negs_per_pos])
            self.ph_d['ngram'] = ngram_ph
            self.ph_d['pos_entity'] = pos_entity_ph
            self.ph_d['neg_entities'] = neg_entities_ph

        with tf.variable_scope('looked'):
            agg_looked_word_emb = tf.nn.embedding_lookup_sparse(
                word_embs, ngram_ph, None, combiner='mean')
            pos_looked_entity_emb = tf.nn.embedding_lookup(
                entity_embs, pos_entity_ph)
            neg_looked_entities_emb = tf.nn.embedding_lookup(
                entity_embs, neg_entities_ph)

        f = tf.contrib.layers.fully_connected(
            inputs=agg_looked_word_emb,
            num_outputs=self.entity_emb_size,
            activation_fn=tf.nn.tanh,
            # use default xavier/glorot init
            weights_regularizer=reg_map,
            scope='map',
        )

        pos_score = tf.sigmoid(
            tf.reduce_sum(
                tf.multiply(f, tf.squeeze(pos_looked_entity_emb)),
                axis=-1, keep_dims=False),
            name='pos_score')
        neg_scores = tf.sigmoid(
            tf.reduce_sum(
                tf.multiply(
                    tf.reshape(
                        tf.tile(f, tf.constant([self.n_negs_per_pos, 1])),
                        shape=[self.batch_size,
                               self.n_negs_per_pos,
                               self.entity_emb_size]),
                    neg_looked_entities_emb),
                axis=-1, keep_dims=False),
            name='neg_scores')

        loss = tf.reduce_mean(
            tf.log(pos_score) + tf.reduce_sum(tf.log(1. - neg_scores), axis=-1),
            name='loss_mnce'
        )

        loss_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss_tot = loss + loss_reg

        return loss_tot

    def get_train_op(self):
        global_step = tf.get_variable(
            'global_step', shape=[], trainable=False,
            initializer=tf.constant_initializer(0))

        train_op = self.opt.minimize(self.loss, global_step=global_step)
        return train_op


def win_gen(data_words_enc, entity_codes, n_entities, n_negs_per_pos,
                 ph_d,
                 batch_size):
    # note: actually we should be uniformly sampling over entities
    # rather than documents
    while True:
        inds = np.random.permutation(np.arange(len(data_words_enc)))

        for ii in range(0, len(inds) - batch_size + 1, batch_size):
            batch_inds = inds[ii:ii + batch_size]

            batch_pos_entity_codes = entity_codes[batch_inds, None]
            batch_neg_entity_codes = np.random.randint(
                0, n_entities, size=[batch_size, n_negs_per_pos])

            batch_words = data_words_enc.iloc[batch_inds].map(get_win)
            batch_words_rows, batch_words_cols = zip(*it.chain(
                *([(row, col) for col in cols]
                  for row, cols in enumerate(batch_words))))

            # Note: we are supposed to grab a window of words here
            #     instead of the entire doc
            batch_words_sptv = tf.SparseTensorValue(
                np.array(batch_words_rows)[:, None],
                np.array(batch_words_cols, dtype='int32'),
                dense_shape=[batch_size])

            feed_d = {
                ph_d['ngram']: batch_words_sptv,
                ph_d['pos_entity']: batch_pos_entity_codes,
                ph_d['neg_entities']: batch_neg_entity_codes,
            }

            yield feed_d
