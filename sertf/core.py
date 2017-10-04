import pandas as pd
import numpy as np
import tensorflow as tf
import itertools as it
from typing import Sized, Dict, Sequence, Generator, Any


LOG_DIR = '/tmp/tensorboard-logs/semantic/'

BATCH_DIM = None


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

        self.opt = opt

        # Convenience Dict Accessors
        self.ph_d = {}
        self.emb_d = {}
        self.reg_d = {}
        self.w2e_xfm = None

        self.init_embs()
        self.init_phs()
        self.build_w2e_graph()
        self.loss_op = self.get_loss_op()
        self.train_op = self.get_train_op()

        self.forward = self.forward_positive

    def init_embs(self):
        """ Initialize embedding tensors
        """
        with tf.variable_scope('reg'):
            self.reg_d['emb'] = tf.contrib.layers.l2_regularizer(self.l2_emb)
        with tf.variable_scope('emb'):
            self.emb_d['word'] = tf.get_variable(
                name='word',
                shape=(len(self.vocab), self.word_emb_size),
                initializer=None,  # use default glorot
                regularizer=self.reg_d['emb']
            )
            self.emb_d['entity'] = tf.get_variable(
                name='item',
                shape=(self.n_entities, self.entity_emb_size),
                initializer=None,  # use default glorot
                regularizer=self.reg_d['emb']
            )

    def init_phs(self):
        """ Initialize placeholder tensors
        """
        with tf.variable_scope('ph'):
            self.ph_d['ngram'] = tf.sparse_placeholder(tf.int32)
            self.ph_d['pos_entity'] = tf.placeholder(
                tf.int32, shape=[BATCH_DIM, 1])
            self.ph_d['neg_entities'] = tf.placeholder(
                tf.int32, shape=[BATCH_DIM, self.n_negs_per_pos])

    def build_w2e_graph(self):
        """ Graph to transform words to entity-space
        """
        with tf.variable_scope('looked'):
            agg_looked_word_emb = tf.nn.embedding_lookup_sparse(
                self.emb_d['word'], self.ph_d['ngram'], None, combiner='mean')
        with tf.variable_scope('reg'):
            reg_map = tf.contrib.layers.l2_regularizer(self.l2_map)
        with tf.variable_scope('xfmer'):
            self.w2e_xfm = tf.contrib.layers.fully_connected(
                inputs=agg_looked_word_emb,
                num_outputs=self.entity_emb_size,
                activation_fn=tf.nn.tanh,
                # use default xavier/glorot init
                weights_regularizer=reg_map,
                scope='map',
            )

    def forward_positive(self):
        with tf.variable_scope('looked'):
            pos_looked_entity_emb = tf.nn.embedding_lookup(
                self.emb_d['entity'], self.ph_d['pos_entity'])

        pos_score = tf.sigmoid(tf.reduce_sum(
            tf.multiply(self.w2e_xfm, tf.squeeze(pos_looked_entity_emb)),
            axis=-1, keep_dims=False),
            name='pos_score')

        return pos_score

    def forward_negatives(self):
        with tf.variable_scope('looked'):
            neg_looked_entities_emb = tf.nn.embedding_lookup(
                self.emb_d['entity'], self.ph_d['neg_entities'])

        neg_scores = tf.sigmoid(tf.reduce_sum(
            tf.multiply(tf.reshape(
                tf.tile(self.w2e_xfm, tf.constant([self.n_negs_per_pos, 1])),
                shape=[-1,  # also batch dim
                       self.n_negs_per_pos,
                       self.entity_emb_size]),
                neg_looked_entities_emb),
            axis=-1, keep_dims=False),
            name='neg_scores')

        return neg_scores

    def get_loss_op(self):
        pos_score = self.forward_positive()
        neg_scores = self.forward_negatives()
        loss_contrast = tf.reduce_mean(
            tf.log(pos_score) +
            tf.reduce_sum(tf.log(1. - neg_scores), axis=-1),
            name='loss_mnce'
        )
        loss_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss_tot = tf.add(loss_contrast, loss_reg, name='loss_tot')
        return loss_tot

    def get_train_op(self):
        global_step = tf.get_variable(
            'global_step', shape=[], trainable=False,
            initializer=tf.constant_initializer(0))

        train_op = self.opt.minimize(self.loss_op, global_step=global_step)
        return train_op


def win_gen(data_words_enc: pd.DataFrame,
            entity_codes: np.array, n_entities: int,
            n_negs_per_pos: int,
            ph_d: Dict[str, tf.Tensor],
            batch_size: int) -> Generator[Dict[str, Any]]:
    # TODO: actually we should be uniformly sampling over entities
    #   rather than documents
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
