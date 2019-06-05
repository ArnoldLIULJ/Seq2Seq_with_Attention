#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer





class Encoder(Layer):
    def __init__(self, hidden_size, cell, embedding_layer, name=None):
        super(Encoder, self).__init__(name)
        self.cell = cell(hidden_size)
        self.hidden_size = hidden_size
        self.embedding_layer = embedding_layer
        self.build((None, None, self.embedding_layer.embedding_size))
        self._built = True
        
    def build(self, inputs_shape):
        self.cell.build(input_shape=tuple(inputs_shape))
        self._built = True
        if self._trainable_weights is None:
            self._trainable_weights = list()
        
        for var in self.cell.trainable_variables:
            self._trainable_weights.append(var)

    def forward(self, src_seq, initial_state=None):
        
        states = initial_state if initial_state is not None else self.cell.get_initial_state(src_seq)
        encoding_hidden_states = list()
        total_steps = src_seq.get_shape().as_list()[1]
        for time_step in range(total_steps):
            if not isinstance(states, list):
                states = [states]
            output, states = self.cell.call(src_seq[:,time_step,:], states, training=self.is_train)
            encoding_hidden_states.append(states[0])
        return output, encoding_hidden_states, states[0]
        




class Decoder_Attention(Layer):
    def __init__(self, hidden_size, cell, embedding_layer, method, name = None):
        super(Decoder_Attention, self).__init__(name)
        self.cell = cell(hidden_size)
        self.hidden_size = hidden_size
        self.embedding_layer = embedding_layer
        self.method = method
        self.build((None, hidden_size+self.embedding_layer.embedding_size))
        self._built = True
        
        
    def build(self, inputs_shape):
        self.cell.build(input_shape=tuple(inputs_shape))
        self._built = True
        if self.method is "concat":
            self.W = self._get_weights("W", shape=(2*self.hidden_size, self.hidden_size))
            self.V = self._get_weights("V", shape=(self.hidden_size, 1))
        elif self.method is "general":
            self.W = self._get_weights("W", shape=(self.hidden_size, self.hidden_size))
        if self._trainable_weights is None:
            self._trainable_weights = list()
        
        for var in self.cell.trainable_variables:
            self._trainable_weights.append(var)

    def score(self, encoding_hidden, hidden, method):
        # encoding = [B, T, H]
        # hidden = [B, H]
        


        # combined = [B,T,2H]
        if method is "concat":
            # hidden = [B,H]->[B,1,H]->[B,T,H]
            hidden = tf.expand_dims(hidden, 1)
            hidden = tf.tile(hidden, [1, encoding_hidden.shape[1],1])
            # combined = [B,T,2H]
            combined = tf.concat([hidden, encoding_hidden], 2)
            combined = tf.cast(combined, tf.float32)
            score = tf.tensordot(combined, self.W, axes=[[2], [0]]) # score = [B,T,H]
            score = tf.nn.tanh(score) # score = [B,T,H]
            score = tf.tensordot(self.V, score, axes=[[0], [2]]) # score = [1,B,T]
            score = tf.squeeze(score, axis=0) # score = [B,T]
            
        elif method is "dot":
            # hidden = [B,H]->[B,H,1]
            hidden = tf.expand_dims(hidden, 2)
            score = tf.matmul(encoding_hidden, hidden)
            score = tf.squeeze(score, axis=2)
        elif method is "general":
            # hidden = [B,H]->[B,H,1]
            score = tf.matmul(hidden, self.W)
            score = tf.expand_dims(score, 2)
            score = tf.matmul(encoding_hidden, score)
            score = tf.squeeze(score, axis=2)
            
        score = tf.nn.softmax(score, axis=-1) # score = [B,T]
        return score

    def forward(self, dec_seq, enc_hiddens, last_hidden, method):
        # dec_seq = [B, T], enc_hiddens = [B, T, H], last_hidden = [B, H]
        # dec_out = [B, T, V]
        
        total_steps = dec_seq.get_shape().as_list()[1]
        states = last_hidden
        cell_outputs = list()
        for time_step in range(total_steps):
            attention_weights = self.score(enc_hiddens, last_hidden, method) 
            attention_weights = tf.expand_dims(attention_weights, 1) #[B, 1, T]
            context = tf.matmul(attention_weights, enc_hiddens) #[B, 1, H]
            context = tf.squeeze(context, 1) #[B, H]
            inputs = tf.concat([dec_seq[:,time_step,:], context], 1)
            if not isinstance(states, list):
                states = [states]
            cell_output, states = self.cell.call(inputs, states, training=self.is_train)
            cell_outputs.append(cell_output)
            last_hidden = states[0]

        cell_outputs = tf.convert_to_tensor(cell_outputs)
        cell_outputs = tf.transpose(cell_outputs, perm=[1,0,2])
        return cell_outputs



enc_seq = np.random.randint(0,high=10, size=(16,8))
dec_seq = np.random.randint(0,high=10, size=(16,5))
encoding_hidden = np.random.rand(16,5,10)
hidden = np.random.rand(16,10)
encoding_hidden = tl.layers.Input([16, 5, 10])
hidden = tl.layers.Input([16,10])
# dec_seq = tl.layers.Input([16,5], dtype=tf.int32)
# enc_seq = tl.layers.Input([16,8], dtype=tf.int32)
# attention = Attention(10)




class Seq2seq_Attention(Model):
    def __init__(self, hidden_size, embedding_layer, cell, method, name=None):
        super(Seq2seq_Attention, self).__init__(name)
        self.enc_layer = Encoder(hidden_size, cell, embedding_layer)
        self.dec_layer = Decoder_Attention(hidden_size, cell, embedding_layer, method=method)
        self.embedding_layer = embedding_layer
        self.dense_layer = tl.layers.Dense(n_units=self.embedding_layer.vocabulary_size, in_channels=hidden_size)
        self.method = method
    
    def forward(self, inputs):
        src_seq, dec_seq = inputs[0], inputs[1]
        dec_seq = self.embedding_layer(dec_seq)
        src_seq = self.embedding_layer(src_seq)
        enc_output, encoding_hidden_states, last_hidden_states = self.enc_layer(src_seq)
        encoding_hidden_states = tf.convert_to_tensor(encoding_hidden_states)
        encoding_hidden_states = tf.transpose(encoding_hidden_states, perm=[1,0,2])
        last_hidden_states = tf.convert_to_tensor(last_hidden_states)
        dec_output = self.dec_layer(dec_seq, encoding_hidden_states, last_hidden_states, method=self.method)
        batch_size = dec_output.shape[0]
        dec_output = tf.reshape(dec_output, [-1, dec_output.shape[-1]])
        dec_output = self.dense_layer(dec_output)
        dec_output = tf.reshape(dec_output, [batch_size, -1, dec_output.shape[-1]])
        return dec_output


# model = Seq2seq_Attention(hidden_size=10,
#     cell = tf.keras.layers.SimpleRNNCell,
#     embedding_layer=tl.layers.Embedding(vocabulary_size=40, embedding_size=20))

# model.train()
# model([enc_seq, dec_seq])


