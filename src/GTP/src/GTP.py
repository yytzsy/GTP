# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf
from tqdm import tqdm
from params import Params
from layers import *
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
import numpy as np
import cPickle as pickle
from params import Params
import numpy as np



class GTP(object):

    def __init__(self, word_emb_init, is_training = True):
        # Build the computational graph when initializing
        self.is_training = is_training
        self.graph = tf.Graph()

        self.video = tf.placeholder(tf.float32, [Params.batch_size, Params.max_video_len, Params.dim_image])   #(batch_size, timestep, dim)
        self.sentence_index = tf.placeholder(tf.int32, [Params.batch_size, Params.max_sentence_len])            #(batch_size, timestep)
        self.video_w_len = tf.placeholder(tf.int32, [Params.batch_size,])      #(batch_size,)
        self.sentence_w_len = tf.placeholder(tf.int32, [Params.batch_size,])     #(batch_size,)
        self.indices = tf.placeholder(tf.float32, [Params.max_choose_len+1, Params.batch_size, Params.max_video_len+1])  #(5+1, batch_size, timestep+1)



        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(initial_value = word_emb_init, name='Wemb')
            sentence_emb = []
            for i in xrange(Params.max_sentence_len):
                sentence_emb.append(tf.nn.embedding_lookup(self.Wemb, self.sentence_index[:,i]))
            sentence_emb = tf.stack(sentence_emb)
            self.sentence = tf.transpose(sentence_emb,[1,0,2])

        self.params = get_attn_params(Params.attn_size, initializer = tf.contrib.layers.xavier_initializer)

        # video and sentence encoding
        self.video_encoding = bidirectional_GRU(self.video,
                                self.video_w_len,
                                cell_fn = SRUCell if Params.SRU else GRUCell,
                                layers = Params.num_layers,
                                scope = "video_encoding",
                                output = 0,
                                is_training = self.is_training)

        self.sentence_encoding = bidirectional_GRU(self.sentence,
                                self.sentence_w_len,
                                cell_fn = SRUCell if Params.SRU else GRUCell,
                                layers = Params.num_layers,
                                scope = "sentence_encoding",
                                output = 0,
                                is_training = self.is_training)


        self.cell_f = apply_dropout(GRUCell(Params.attn_size*2), is_training = self.is_training)

        self.sentence_video_interaction()
        self.similarity_graph_construction()
        self.graph_convolution()
        self.bidirectional_readout()
        self.get_sentence_state()

        if self.is_training:
            self.current_position_list = tf.placeholder(tf.int32,[Params.batch_size,Params.max_choose_len+1])
            self.outputs_train()
            self.loss_function()
        else:
            self.inference_f_state = tf.placeholder(tf.float32, [Params.batch_size, Params.attn_size*2])
            self.current_position = tf.placeholder(tf.int32,[Params.batch_size])
            self.inference_step = tf.placeholder(tf.int32)
            self.outputs_inference()

        total_params()

    def sentence_video_interaction(self):
        weights_interaction = ([self.params["W_v"],self.params["W_s"]],self.params["v_interaction"]), self.params["W_fuse"], self.params["b_fuse"]
        self.sentece_video_interaction_fts, self.sentence_video_attention \
        = attend_sentence_with_video_fts(self.sentence_encoding, self.video_encoding, self.sentence_w_len, weights_interaction)



    def similarity_graph_construction(self):
        self.G_pre, self.G, self.G_softmax, self.G_softmax_I = construct_noWg_graph(self.sentece_video_interaction_fts,self.video_w_len,scope = "similarity_graph_construction")


    def graph_convolution(self):
        params = [self.params["W_c1"],self.params["W_c2"],self.params["W_c3"],self.params["W_c4"]]
        inputs = self.sentece_video_interaction_fts
        for i in range(Params.num_gcn):
            inputs = GCN(self.G_softmax_I,inputs,params[i],scope = "graph_convolution_"+str(i),is_training = self.is_training)
        self.video_gcn_represent = inputs


    def bidirectional_readout(self):
        self.final_bidirectional_outputs = bidirectional_GRU(self.video_gcn_represent,
                                    self.video_w_len,
                                    cell_fn = SRUCell if Params.SRU else GRUCell,
                                    # layers = Params.num_layers, # or 1? not specified in the original paper
                                    scope = "bidirectional_readout",
                                    output = 0,
                                    is_training = self.is_training)



    def get_sentence_state(self):
        with tf.variable_scope("sentence_self_encoding"):
            self.sentence_state,self.sentence_attn = avg_sentence_pooling(self.sentence_encoding, units = Params.attn_size, memory_len = self.sentence_w_len, scope = "sentence_pooling")



    def outputs_train(self):
        params = ([self.params["W_h_P"],self.params["W_h_a"]],self.params["v"])
        self.probability_f = sequence_pointer_net(self.final_bidirectional_outputs, self.video_w_len, self.current_position_list,\
                                                    self.sentence_state, self.cell_f, params) # scope = "sequence_pointer_network"


    def outputs_inference(self):
        params = ([self.params["W_h_P"],self.params["W_h_a"]],self.params["v"])
        self.probability_f_inference, self.output_f_state, self.output_cond = \
        inference_sequence_pointer_net(self.inference_step,self.final_bidirectional_outputs, self.video_w_len, self.current_position,\
                                       self.sentence_state, self.cell_f, self.inference_f_state, params) #scope = "sequence_pointer_network_inference"



    def loss_function(self):
        with tf.variable_scope("loss"):
            self.choose_loss_f = Params.alpha_choose * loss_choose(self.probability_f,self.indices)
            tv = tf.trainable_variables()
            self.regular_loss = Params.alpha_regular * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
            self.loss =  self.choose_loss_f + self.regular_loss


def debug():
    word_emb_init = np.array(np.load(open(Params.word_fts_path)).tolist(),np.float32)
    model = new_RGCN_sequence(word_emb_init,is_training = False)
    print("Built model")


# debug()