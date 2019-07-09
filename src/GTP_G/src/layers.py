# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell
from params import Params
from zoneout import ZoneoutWrapper
from tensorflow.contrib.layers import batch_norm


'''
attention weights from https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
W_u^Q.shape:    (2 * attn_size, attn_size)
W_u^P.shape:    (2 * attn_size, attn_size)
W_v^P.shape:    (attn_size, attn_size)
W_g.shape:      (4 * attn_size, 4 * attn_size)
W_h^P.shape:    (2 * attn_size, attn_size)
W_v^Phat.shape: (2 * attn_size, attn_size)
W_h^a.shape:    (2 * attn_size, attn_size)
W_v^Q.shape:    (attn_size, attn_size)
'''

def get_attn_params(attn_size,initializer = tf.truncated_normal_initializer):
    '''
    Args:
        attn_size: the size of attention specified in https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
        initializer: the author of the original paper used gaussian initialization however I found xavier converge faster

    Returns:
        params: A collection of parameters used throughout the layers
    '''
    with tf.variable_scope("attention_weights"):
        params = {
                "W_h_P":tf.get_variable("W_h_P",dtype = tf.float32, shape = (4 * attn_size, attn_size), initializer = initializer()),
                "W_h_a":tf.get_variable("W_h_a",dtype = tf.float32, shape = (2 * attn_size, attn_size), initializer = initializer()),
                "v":tf.get_variable("v",dtype = tf.float32, shape = (attn_size), initializer =initializer()),

                }
        return params





def attend_sentence_with_video_fts(sentence_fts, video_fts, sentence_len, params, scope = 'sentence_video_attention'):
    with tf.variable_scope(scope):
        Weights, W_fuse, b_fuse = params
        attention_list = []
        attended_fts_list = []
        for i in range(Params.max_video_len):
            with tf.variable_scope(scope):
                if i > 0: tf.get_variable_scope().reuse_variables()
                video_slice = tf.reshape(tf.slice(video_fts,begin=[0,i,0],size=[-1,1,-1]),[Params.batch_size,-1])
                attention_inputs = [sentence_fts, video_slice]
                attention_weights = attention(attention_inputs, Params.attn_size, Weights, scope = "attention", memory_len = sentence_len)
                extended_attention_weights = tf.expand_dims(attention_weights, -1)
                attended_fts = tf.reduce_sum(extended_attention_weights * sentence_fts, 1)
                attention_list.append(attention_weights)
                attended_fts_list.append(attended_fts)

        attended_sentence_fts = tf.transpose(tf.stack(attended_fts_list),[1,0,2])
        concat_video_sentence_fts = tf.concat([video_fts,attended_sentence_fts],-1)
        fused_video_sentence_fts = tf.nn.relu(tf.matmul(tf.reshape(concat_video_sentence_fts,[-1,4*Params.attn_size]),W_fuse)+b_fuse)
        fused_video_sentence_fts = tf.reshape(fused_video_sentence_fts,[Params.batch_size,Params.max_video_len,-1])
        video_sentence_attentions = tf.transpose(tf.stack(attention_list),[1,0,2])

        return fused_video_sentence_fts, video_sentence_attentions





def mask_attn_score_extend_matrix(score, memory_sequence_length, score_mask_value = -1e8):
    # score (b,max_len,max_len) (20,210,210)
    # memory_sequence_length (b) (20)
    score_mask = tf.cast(tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1]),tf.int32) #(b,max_len) (20,210)
    score_mask = tf.tile(tf.expand_dims(score_mask,2),[1,1,1]) #(b,max_len,1) (20,210,1)
    score_mask_transpose = tf.transpose(score_mask,[0,2,1]) #(b,1,max_len) (20,1,210)
    score_mask_matrix = tf.matmul(score_mask,score_mask_transpose) #(b,max_len,max_len) (20,210,210)
    score_mask_matrix = tf.cast(score_mask_matrix,tf.bool)
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask_matrix, score, score_mask_values)


    
def mask_graph(inputs,video_len,score_mask_value = 0.0):
    score_mask = tf.cast(tf.sequence_mask(video_len, maxlen=inputs.shape[1]),tf.int32)
    score_mask = tf.tile(tf.expand_dims(score_mask,2),[1,1,1]) #(b,max_len,1) (20,210,1)
    score_mask_transpose = tf.transpose(score_mask,[0,2,1]) #(b,1,max_len) (20,1,210)
    score_mask_matrix = tf.matmul(score_mask,score_mask_transpose) #(b,max_len,max_len) (20,210,210)
    score_mask_matrix = tf.cast(score_mask_matrix,tf.bool)
    score_mask_values = score_mask_value * tf.ones_like(inputs)
    return tf.where(score_mask_matrix, inputs, score_mask_values)



def mask_att_score_extend_matrix_normalize(score, memory_sequence_length):

    score_mask = tf.cast(tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1]),tf.int32) #(b,max_len) (20,210)
    score_mask = tf.tile(tf.expand_dims(score_mask,2),[1,1,1]) #(b,max_len,1) (20,210,1)
    score_mask_transpose = tf.transpose(score_mask,[0,2,1]) #(b,1,max_len) (20,1,210)
    score_mask_matrix = tf.matmul(score_mask,score_mask_transpose) #(b,max_len,max_len) (20,210,210)
    score_mask_matrix = tf.cast(score_mask_matrix,tf.bool)
    score_mask_values = 0.0 * tf.ones_like(score)
    transfer_score = tf.where(score_mask_matrix, score, score_mask_values)

    score_mask_values_row_sum = tf.reduce_sum(transfer_score,2) #(20,210)
    score_mask_values_row_sum_extend = tf.tile(tf.expand_dims(score_mask_values_row_sum,2),[1,1,Params.max_video_len]) #(20,210,210)
    score_normalize = tf.div(transfer_score,score_mask_values_row_sum_extend+1e-8)

    score_mask_values = -1e8 * tf.ones_like(score_normalize)
    score_normalize_softmax_pre = tf.where(score_mask_matrix, score_normalize, score_mask_values)
    score_normalize_softmax = mask_graph(tf.nn.softmax(score_normalize_softmax_pre * Params.gcn_softmax_scale, dim=2),memory_sequence_length)

    idx = tf.convert_to_tensor(np.eye(Params.max_video_len,dtype =np.float32))
    score_normalize_softmax_I_pre = tf.nn.softmax(score_normalize_softmax_pre * Params.gcn_softmax_scale, dim=2)+idx
    score_normalize_softmax_I = mask_graph(score_normalize_softmax_I_pre,memory_sequence_length)

    return score_normalize,score_normalize_softmax,score_normalize_softmax_I



def encoding(word, char, word_embeddings, char_embeddings, scope = "embedding"):
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
        return word_encoding, char_encoding



def apply_dropout(inputs, size = None, is_training = True):
    '''
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    '''
    if Params.dropout is None and Params.zoneout is None:
        return inputs
    if Params.zoneout is not None:
        return ZoneoutWrapper(inputs, state_zoneout_prob= Params.zoneout, is_training = is_training)
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(inputs,
                                            output_keep_prob = 1 - Params.dropout,
                                            # variational_recurrent = True,
                                            # input_size = size,
                                            dtype = tf.float32)
    else:
        return inputs





def bidirectional_GRU(inputs, inputs_len, cell = None, cell_fn = tf.contrib.rnn.GRUCell, units = Params.attn_size, layers = 1, scope = "Bidirectional_GRU", output = 0, is_training = True, reuse = None):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    '''
    with tf.variable_scope(scope, reuse = reuse):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
                inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))

            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
                cell_bw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
            else:
                cell_fw, cell_bw = [apply_dropout(cell_fn(units), size = inputs.shape[-1], is_training = is_training) for _ in range(2)]
                
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                        sequence_length = inputs_len,
                                                        dtype=tf.float32)
        if output == 0:
            return tf.concat(outputs, 2)
        elif output == 1:
            return tf.reshape(tf.concat(states,1),(Params.batch_size, shapes[1], 2*units))






def sequence_pointer_net(video, video_len, forward_position_list, sentence_state, cell_f, params, scope = None):
    
    weights_p = params
        
    video_shape = np.shape(video)
    zeros_video_padding = tf.zeros([video_shape[0],1,video_shape[2]],tf.float32)
    expand_video = tf.concat([video,zeros_video_padding],1)
    inputs = [expand_video,sentence_state]
    
    p_logits_list = []
    for i in range(Params.max_choose_len+1):
        with tf.variable_scope('sequence_attention'):
            if i > 0: tf.get_variable_scope().reuse_variables()

            if i == 0:
                f_state = sentence_state
                f_inputs = inputs
                f_direction = 'start'
            else:
                f_direction = 'forward'

            with tf.variable_scope('forward'):
                p_logits = attention_with_padding(f_inputs, Params.attn_size, weights_p, memory_len = video_len, current_position = forward_position_list[:,i], direction = f_direction, scope = "attention_f")
                p_logits_list.append(p_logits)
                scores = tf.expand_dims(p_logits, -1)
                attention_pool = tf.reduce_sum(scores * expand_video,1)
                _, f_state = cell_f(attention_pool, f_state)
                f_inputs = [expand_video, f_state]


    return tf.stack(p_logits_list)




def inference_sequence_pointer_net(i,video, video_len, forward_position, sentence_state, cell_f, \
                                     inference_f_state, params, scope = None):

    weights_p = params

    video_shape = np.shape(video)
    zeros_video_padding = tf.zeros([video_shape[0],1,video_shape[2]],tf.float32)
    expand_video = tf.concat([video,zeros_video_padding],1)

    with tf.variable_scope('sequence_attention'):

        def true_proc():
            f_inputs = [expand_video, inference_f_state]
            f_state = inference_f_state
            return f_inputs,f_state

        def false_proc():
            f_inputs = [expand_video, sentence_state]
            f_state = sentence_state
            return f_inputs,f_state


        f_inputs,f_state = tf.cond(tf.greater(i,tf.constant(0)), true_fn = true_proc, false_fn = false_proc)


        with tf.variable_scope('forward'):
            p_logits = attention_with_padding(f_inputs, Params.attn_size, weights_p, memory_len = video_len, current_position = forward_position, direction = 'forward', scope = "attention_f")
            scores = tf.expand_dims(p_logits, -1)
            attention_pool = tf.reduce_sum(scores * expand_video,1)
            _, f_o_state = cell_f(attention_pool, f_state)


    return p_logits,f_o_state,tf.greater(i,tf.constant(0))





def attention_rnn(inputs, inputs_len, units, attn_cell, bidirection = True, scope = "gated_attention_rnn", is_training = True):
    with tf.variable_scope(scope):
        if bidirection:
            outputs = bidirectional_GRU(inputs,
                                        inputs_len,
                                        cell = attn_cell,
                                        scope = scope + "_bidirectional",
                                        output = 0,
                                        is_training = is_training)
        else:
            outputs, _ = tf.nn.dynamic_rnn(attn_cell, inputs,
                                            sequence_length = inputs_len,
                                            dtype=tf.float32)
        return outputs




def sentence_pooling(memory, units, weights, memory_len = None, scope = "sentence_pooling"):
    with tf.variable_scope(scope):
        shapes = memory.get_shape().as_list()
        V_r = tf.get_variable("sentence_param", shape = (Params.max_sentence_len, units), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
        inputs_ = [memory, V_r]
        attn = attention(inputs_, units, weights, memory_len = memory_len, scope = "sentence_attention_pooling")
        attn = tf.expand_dims(attn, -1)
        return tf.reduce_sum(attn * memory, 1)




def avg_sentence_pooling(memory, units, memory_len = None, scope = "sentence_pooling"):
    with tf.variable_scope(scope):
        avg_attn = tf.cast(tf.cast(tf.sequence_mask(memory_len, maxlen=Params.max_sentence_len),tf.int32),tf.float32)
        avg_attn = tf.div(avg_attn, tf.cast(tf.tile(tf.expand_dims(memory_len,1),[1,Params.max_sentence_len]),tf.float32) )
        attn = tf.expand_dims(avg_attn, -1)
        return tf.reduce_sum(attn * memory, 1), avg_attn




def gated_attention(memory, inputs, states, units, params, self_matching = False, memory_len = None, scope="gated_attention"):
    with tf.variable_scope(scope):
        weights, W_g = params
        inputs_ = [memory, inputs]
        states = tf.reshape(states,(Params.batch_size,Params.attn_size))
        if not self_matching:
            inputs_.append(states)
        scores = attention(inputs_, units, weights, memory_len = memory_len)
        save_attention = scores
        scores = tf.expand_dims(scores,-1)
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs,attention_pool),axis = 1)
        g_t = tf.sigmoid(tf.matmul(inputs,W_g))
        return g_t * inputs




def mask_attn_score(score, memory_sequence_length, score_mask_value = -1e8):
    score_mask = tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)



def mask_attn_score_with_padding(score, memory_sequence_length, score_mask_value = -1e8):
    score_mask_pre = tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1]-1)
    score_mask_padding = tf.ones([score.shape[0],1],tf.int32)
    score_mask_padding = tf.cast(score_mask_padding,tf.bool)
    score_mask = tf.concat([score_mask_pre,score_mask_padding],1)
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)




def mask_attn_score_with_padding_with_position(score, memory_sequence_length, current_sequence_position, direction, score_mask_value = -1e8):
    
    score_mask_all = tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1]-1)

    if direction == 'forward':
        current_mask_pre = tf.sequence_mask(current_sequence_position+1, maxlen=score.shape[1]-1)
        current_mask_transfer = 1 - tf.cast(current_mask_pre,tf.int32)
    elif direction == 'backward':
        current_mask_pre = tf.sequence_mask(current_sequence_position, maxlen=score.shape[1]-1)
        current_mask_transfer = tf.cast(current_mask_pre,tf.int32)
    elif direction == 'start':
        current_mask_pre = tf.sequence_mask(score.shape[1]-1,maxlen = score.shape[1]-1) # all true
        current_mask_transfer = tf.cast(current_mask_pre,tf.int32)

    score_mask_pre = tf.cast(tf.cast(score_mask_all,tf.int32) * current_mask_transfer,tf.bool)

    score_mask_padding = tf.ones([score.shape[0],1],tf.int32)
    score_mask_padding = tf.cast(score_mask_padding,tf.bool)
    score_mask = tf.concat([score_mask_pre,score_mask_padding],1)

    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)



def attention(inputs, units, weights, scope = "attention", memory_len = None, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp,w) in enumerate(zip(inputs,weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],Params.attn_size], initializer = tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is Params.batch_size:
                outputs = tf.reshape(outputs, (shapes[0],1,-1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0],-1))
            outputs_.append(outputs)
        outputs = sum(outputs_)
        if Params.bias:
            b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            outputs += b
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores) # all attention output is softmaxed now






def attention_with_padding(inputs, units, weights, scope, memory_len = None, current_position = None, direction = None, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp,w) in enumerate(zip(inputs,weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],Params.attn_size], initializer = tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is Params.batch_size:
                outputs = tf.reshape(outputs, (shapes[0],1,-1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0],-1))
            outputs_.append(outputs)
        outputs = sum(outputs_)
        if Params.bias:
            b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            outputs += b
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score_with_padding_with_position(scores, memory_len, current_position, direction)
        return tf.nn.softmax(scores) # all attention output is softmaxed now






def loss_choose(output_f,target_f):
    choose_probability = target_f * tf.log(output_f + 1e-8)
    loss_choose = -tf.reduce_sum(choose_probability)
    return loss_choose





def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))


def LayerNormalization(inputs, epsilon = 1e-5, scope = None):
 
    """ Computer LN given an input tensor. We get in an input of shape
    [N X D] and with LN we compute the mean and var for each individual
    training point across all it's hidden dimensions rather than across
    the training batch as we do in BN. This gives us a mean and var of shape
    [N X 1].
    """
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    with tf.variable_scope(scope + 'LN'):
        scale = tf.get_variable('alpha',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('beta',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
 
    return LN



def BatchNormalization(inputs,is_training,scope = None):

    with tf.variable_scope(scope + 'BN') as BN:
        # BN_output = tf.cond(is_training,
        #     lambda: batch_norm(
        #         inputs, is_training=True, center=True,
        #         scale=True, activation_fn=tf.nn.relu,
        #         updates_collections=None, scope=BN),
        #     lambda: batch_norm(
        #         inputs, is_training=False, center=True,
        #         scale=True, activation_fn=tf.nn.relu,
        #         updates_collections=None, scope=BN, reuse=True))
        def true_proc():
            return batch_norm(
                inputs, is_training=True, center=True,
                scale=True, activation_fn=tf.nn.relu,
                updates_collections=None, scope=BN)

        def false_proc():
            return batch_norm(
                inputs, is_training=False, center=True,
                scale=True, activation_fn=tf.nn.relu,
                updates_collections=None, scope=BN, reuse=True)


        BN_output = tf.cond(tf.cast(is_training,tf.bool), true_fn = true_proc, false_fn = false_proc)

        return BN_output
