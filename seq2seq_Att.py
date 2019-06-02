#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer


# class Attention(Layer):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.build(tuple())
#         self._built = True

#     def build(self, inputs_shape):
#         self.W = self._get_weights("W", shape=(2*self.hidden_size, self.hidden_size))

#     def forward(self, encoding_hidden, hidden):
#         # encoding = [B, T, H]
#         # hidden = [B,H]->[B,1,H]->[B,T,H]
        
#         hidden = tf.expand_dims(hidden, 1)
#         hidden = tf.tile(hidden, [1, encoding_hidden.shape[1],1])
        
#         # combined = [B,T,2H]
#         combined = tf.concat([hidden, encoding_hidden], 2)
        
#         combined = tf.cast(combined, tf.float32)
#         e = tf.tensordot(combined, self.W, axes=[[2], [0]]) # e = [B,T,H]
#         # a = [B,T,H]->[B,1,H]->[B,1,T]
#         a = tf.nn.softmax(e, axis=-1)
#         a = tf.reduce_sum(a, 1, keepdims=True)
#         a = tf.matmul(a, tf.transpose(e, perm=[0,2,1]))
#         a = tf.squeeze(a, axis=1) # [B, T]
#         a = tf.expand_dims(a, 1) # [B,1,T]
#         a = tf.matmul(a, e) # [B, 1, H]
#         return a

#     def __repr__(self):
#         return "attention layer"


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
        print("done")
        return output, encoding_hidden_states, states[0]
        




class Decoder(Layer):
    def __init__(self, hidden_size, cell, embedding_layer, name = None):
        super(Decoder, self).__init__(name)
        self.cell = cell(hidden_size)
        self.hidden_size = hidden_size
        self.embedding_layer = embedding_layer
        #self.att_layer = Attention(hidden_size)
        self.build((None, hidden_size+self.embedding_layer.embedding_size))
        self._built = True
        
        
    def build(self, inputs_shape):
        #self.att_layer.build(tuple())
        self.cell.build(input_shape=tuple(inputs_shape))
        self._built = True
        self.W = self._get_weights("W", shape=(2*self.hidden_size, self.hidden_size))
        
        if self._trainable_weights is None:
            self._trainable_weights = list()
        
        for var in self.cell.trainable_variables:
            self._trainable_weights.append(var)

    def score(self, encoding_hidden, hidden):
        # encoding = [B, T, H]
        # hidden = [B,H]->[B,1,H]->[B,T,H]
        
        hidden = tf.expand_dims(hidden, 1)
        hidden = tf.tile(hidden, [1, encoding_hidden.shape[1],1])
        
        # combined = [B,T,2H]
        combined = tf.concat([hidden, encoding_hidden], 2)
        
        combined = tf.cast(combined, tf.float32)
        e = tf.tensordot(combined, self.W, axes=[[2], [0]]) # e = [B,T,H]
        # a = [B,T,H]->[B,1,H]->[B,1,T]
        a = tf.nn.softmax(e, axis=-1)
        a = tf.reduce_sum(a, 1, keepdims=True)
        a = tf.matmul(a, tf.transpose(e, perm=[0,2,1]))
        a = tf.squeeze(a, axis=1) # [B, T]
        a = tf.expand_dims(a, 1) # [B,1,T]
        a = tf.matmul(a, e) # [B, 1, H]
        return a

    def forward(self, dec_seq, enc_hiddens, last_hidden):
        # dec_seq = [B, T], enc_hiddens = [B, T, H], last_hidden = [B, H]
        # dec_out = [B, T, V]
        
        total_steps = dec_seq.get_shape().as_list()[1]
        states = last_hidden
        for time_step in range(total_steps):
            context = self.score(enc_hiddens, last_hidden) # [B, 1, H]
            context = tf.squeeze(context, 1)
            inputs = tf.concat([dec_seq[:,time_step,:], context], 1)
            if not isinstance(states, list):
                states = [states]
            cell_output, states = self.cell.call(inputs, states, training=self.is_train)
            last_hidden = states[0]
        return cell_output



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
    def __init__(self, hidden_size, embedding_layer, cell, name=None):
        super(Seq2seq_Attention, self).__init__(name)
        self.enc_layer = Encoder(hidden_size, cell, embedding_layer)
        self.dec_layer = Decoder(hidden_size, cell, embedding_layer)
        self.embedding_layer = embedding_layer
    
    def forward(self, inputs):
        src_seq, dec_seq = inputs[0], inputs[1]
        dec_seq = self.embedding_layer(dec_seq)
        src_seq = self.embedding_layer(src_seq)
        enc_output, encoding_hidden_states, last_hidden_states = self.enc_layer(src_seq)
        encoding_hidden_states = tf.convert_to_tensor(encoding_hidden_states)
        encoding_hidden_states = tf.transpose(encoding_hidden_states, perm=[1,0,2])
        last_hidden_states = tf.convert_to_tensor(last_hidden_states)
        dec_output = self.dec_layer(dec_seq, encoding_hidden_states, last_hidden_states)
        return dec_output


# model = Seq2seq_Attention(hidden_size=10,
#     cell = tf.keras.layers.SimpleRNNCell,
#     embedding_layer=tl.layers.Embedding(vocabulary_size=40, embedding_size=20))

# model.train()
# model([enc_seq, dec_seq])


