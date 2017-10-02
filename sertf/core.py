import pandas as pd
import numpy as np
import tensorflow as tf
import itertools as it
from typing import Sized, Dict, Sequence
from tensorflow.contrib.tensorboard.plugins import projector


LOG_DIR = '/tmp/tensorboard-logs/semantic/'


def get_win(seq: Sequence, win_size: int=4) -> Sequence:
    """ Extracts a contiguous window of data from a sequence

    Parameters
    ----------
    seq
        Sequence to extract from
    win_size
        Window size

    Returns
    -------
    win_seq
        Windowed Sequence
    """
    ii = np.random.randint(0, max(0, len(seq) - win_size) + 1)
    win_seq = seq[ii:ii + win_size]
    return win_seq


class Model(object):
    def __init__(self,
                 vocab: Sized,
                 n_entities: int,
                 word_emb_size: int=64,
                 entity_emb_size: int=16,
                 n_negs_per_pos: int=10,
                 l2_emb: float=1e-2,
                 l2_map: float=1e-2,
                 batch_size: int=1024,
                 opt: tf.train.Optimizer=tf.train.AdamOptimizer(
                     learning_rate=0.001, beta1=0.9, beta2=0.999),
                 ):
        """
        Model for training word and entity embeddings for entity retrieval as
        formulated by [1]_


        Parameters
        ----------
        vocab
            Word vocabulary
        n_entities
            Number of entities in catalog
        word_emb_size
            Size of word embeddings
        entity_emb_size
            Size of entity embeddings
        n_negs_per_pos
            Number of negatives to sample per positive
        l2_emb
            L2 regularization scale for embeddings parameters
        l2_map
            L2 regularization scale for mapping layer weights
        batch_size
            Feed batch size
        opt
            Training optimizer

        References
        ----------
        .. [1] Van Gysel, Christophe, Maarten de Rijke, and Evangelos Kanoulas.
           "Learning latent vector spaces for product search." Proceedings of
           the 25th ACM International on Conference on Information and
           Knowledge Management. ACM, 2016.

        """

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
        self.embs_d = {}

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
            self.embs_d['word'] = word_embs
            self.embs_d['entity'] = entity_embs

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
            tf.log(pos_score) +
            tf.reduce_sum(tf.log(1. - neg_scores),axis=-1),
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


def win_gen(data_words_enc: pd.DataFrame,
            entity_codes: np.array, n_entities: int,
            n_negs_per_pos: int,
            ph_d: Dict[str, tf.Tensor],
            batch_size: int):
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
