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



class GTP_P(object):

    def __init__(self, word_emb_init, is_training = True):
        # Build the computational graph when initializing
        self.is_training = is_training
        self.graph = tf.Graph()

        self.video = tf.placeholder(tf.float32, [Params.batch_size, Params.max_video_len, Params.dim_image])   #(batch_size, timestep, dim)
        self.sentence_index = tf.placeholder(tf.int32, [Params.batch_size, Params.max_sentence_len])            #(batch_size, timestep)
        self.video_w_len = tf.placeholder(tf.int32, [Params.batch_size,])      #(batch_size,)
        self.sentence_w_len = tf.placeholder(tf.int32, [Params.batch_size,])     #(batch_size,)
        self.indices = tf.placeholder(tf.float32, [Params.batch_size, Params.max_video_len])



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


        self.sentence_video_interaction()
        self.similarity_graph_construction()
        self.graph_convolution()
        self.classify_output()

        if self.is_training:
            self.loss_function()
        total_params()



    def sentence_video_interaction(self):
        weights_interaction = ([self.params["W_v"],self.params["W_s"]],self.params["v_interaction"]), self.params["W_fuse"], self.params["b_fuse"]
        self.sentece_video_interaction_fts, self.sentence_video_attention \
        = attend_sentence_with_video_fts(self.sentence_encoding, self.video_encoding, self.sentence_w_len, weights_interaction)


    def similarity_graph_construction(self):
        self.G_pre, self.G, self.G_softmax, self.G_softmax_I = construct_noWg_graph(self.sentece_video_interaction_fts,self.video_w_len,scope = "similarity_graph_construction")


    def graph_convolution(self):
        params = [self.params["W_c1"],self.params["W_c2"],self.params["W_c3"]]
        inputs = self.sentece_video_interaction_fts
        for i in range(Params.num_gcn):
            inputs = GCN(self.G_softmax_I,inputs,params[i],scope = "graph_convolution_"+str(i),is_training = self.is_training)
        self.video_gcn_represent = inputs


    def classify_output(self):
        self.probability = outputs_probability_classify(self.video_gcn_represent,self.video_w_len,self.params["W_classify"])


    def loss_function(self):
        
        with tf.variable_scope("loss"):

            self.relevance_loss = Params.alpha_relevance * cross_entropy(self.probability, self.indices, self.video_w_len)

            tv = tf.trainable_variables()
            self.regular_loss = Params.alpha_regular * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

            self.loss =  self.relevance_loss + self.regular_loss



def debug():
    word_emb_init = np.array(np.load(open(Params.word_fts_path)).tolist(),np.float32)
    model = new_RGCN_choose(word_emb_init,is_training = False)
    print("Built model")


# debug()