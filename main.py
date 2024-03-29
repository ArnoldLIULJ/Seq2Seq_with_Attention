#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from sklearn.utils import shuffle
from seq2seq_Att import Seq2seq_Attention
from tests.utils import CustomTestCase
from tensorlayer.cost import cross_entropy_seq


class Model_SEQ2SEQ_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.batch_size = 16

        cls.vocab_size = 20
        cls.embedding_size = 32
        cls.dec_seq_length = 5
        cls.trainX = np.random.randint(20, size=(100, 6))
        cls.trainY = np.random.randint(20, size=(100, cls.dec_seq_length + 1))
        cls.trainY[:, 0] = 0  # start_token == 0

        # Parameters
        cls.src_len = len(cls.trainX)
        cls.tgt_len = len(cls.trainY)

        assert cls.src_len == cls.tgt_len

        cls.num_epochs = 500
        cls.n_step = cls.src_len // cls.batch_size

    @classmethod
    def tearDownClass(cls):
        pass

    def test_basic_simpleSeq2Seq(self):

        model_ = Seq2seq_Attention(
            hidden_size=128,
            cell = tf.keras.layers.SimpleRNNCell,
            embedding_layer=tl.layers.Embedding(vocabulary_size=self.vocab_size, embedding_size=self.embedding_size),
            method = 'concat')
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        for epoch in range(self.num_epochs):
            model_.train()
            trainX, trainY = shuffle(self.trainX, self.trainY)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX[20:,:], targets=trainY[20:,:], batch_size=self.batch_size,
                                                    shuffle=False), total=self.n_step,
                             desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):
                dec_seq = Y[:, :-1]
                target_seq = Y[:, 1:]

                with tf.GradientTape() as tape:
                    ## compute outputs
                    output = model_(inputs=[X, dec_seq])
                    # print(output)
                    output = tf.reshape(output, [-1, self.vocab_size])

                    loss = cross_entropy_seq(logits=output, target_seqs=target_seq)
                    grad = tape.gradient(loss, model_.trainable_weights)
                    optimizer.apply_gradients(zip(grad, model_.trainable_weights))

                total_loss += loss
                n_iter += 1

            model_.eval()
            test_sample = trainX[0:2, :].tolist()

            top_n = 1
            for i in range(top_n):
                prediction = model_([test_sample], seq_length=self.dec_seq_length, sos=0)
                print("Prediction: >>>>>  ", prediction, "\n Target: >>>>>  ", trainY[0:2, 1:], "\n\n")

            # printing average loss after every epoch
            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, self.num_epochs, total_loss / n_iter))


if __name__ == '__main__':
    unittest.main()
