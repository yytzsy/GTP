#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
from collections import defaultdict
#from tf.nn import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import rnn_cell
#from tensorflow.models.rnn import rnn, rnn_cell
from keras.preprocessing import sequence
import unicodedata
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
import logging
import string
from GTP_P import GTP_P
import random
import cPickle as pkl
from params import Params
import math
from metric import *





#########################  Global path will used ###############
finetune = False
if finetune:
    MODEL = ''

optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
            "adam":tf.train.AdamOptimizer,
            "gradientdescent":tf.train.GradientDescentOptimizer,
            "adagrad":tf.train.AdagradOptimizer}


def make_prepare_path(task):


    sub_dir = 'GTP_P'

    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    log_file_name = 'screen_output_'+task+'_'+str(time_stamp)+'_'+sub_dir+'.log'
    fh = logging.FileHandler(filename=log_file_name, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    logging.root.addHandler(fh)

    model_save_dir = Params.pre_model_save_dir+sub_dir
    result_save_dir = Params.pre_result_save_dir+sub_dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    if not os.path.exists(Params.words_path):
        os.makedirs(Params.words_path)

    return sub_dir, logging, model_save_dir, result_save_dir


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract a CNN features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--task', dest='task',
                        help='train or test',
                        default='train', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_word_embedding(word_embedding_path,wordtoix_path,ixtoword_path,extracted_word_fts_init_path):
    print('loading word features ...')
    word_fts_dict = np.load(open(word_embedding_path)).tolist()
    wordtoix = np.load(open(wordtoix_path)).tolist()
    ixtoword = np.load(open(ixtoword_path)).tolist()
    word_num = len(wordtoix)
    extract_word_fts = np.random.uniform(-3,3,[word_num,300]) 
    count = 0
    for index in range(word_num):
        if ixtoword[index] in word_fts_dict:
            extract_word_fts[index] = word_fts_dict[ ixtoword[index] ]
            count = count + 1
    np.save(extracted_word_fts_init_path,extract_word_fts)


def get_video_data_HL(video_data_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.array(List)


def get_video_data_jukin(video_data_path_train, video_data_path_val):
    title = []
    video_list_train = get_video_data_HL(video_data_path_train) # get h5 file list
    video_list_val = get_video_data_HL(video_data_path_val)
    for ele in video_list_train:
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['video_name']
        batch_title = batch_data['sentence']
        for i in xrange(len(batch_fname)):
            title.append(batch_title[i])
    for ele in video_list_val:
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['video_name']
        batch_title = batch_data['sentence']
        for i in xrange(len(batch_fname)):
            title.append(batch_title[i])
    title = np.array(title)
    video_caption_data = pd.DataFrame({'Description':title})
    return video_caption_data, video_list_train, video_list_val


def preProBuildWordVocab(logging,sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    logging.info('preprocessing word counts and creating vocab based on word count threshold {:d}'.format(word_count_threshold))
    word_counts = {} # count the word number
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1  # if w is not in word_counts, will insert {w:0} into the dict

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    logging.info('filtered words from {:d} to {:d}'.format(len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


def softmax(pre_inMatrix,video_len):
    [m,n] = np.shape(pre_inMatrix)
    inMatrix = np.zeros([m,n],dtype = np.float32)-1e10
    for id_m in range(0,m):
        for id_n in range(0,video_len[id_m]):
            inMatrix[id_m,id_n] = pre_inMatrix[id_m,id_n]*4.0
    outMatrix = np.zeros([m,n],dtype = np.float32)+0.0
    for id_m in range(0,m):
        for id_n in range(0,n):
            outMatrix[id_m,id_n] = math.exp(inMatrix[id_m,id_n])
        soft_sum = np.sum(outMatrix[id_m])
        outMatrix[id_m] = outMatrix[id_m] * 1.0 / soft_sum
    return outMatrix



def choose_commmon_person(person_choose_label_list):
    final_choose_indice = []
    [batch_size,person_num,max_video_len] = np.shape(person_choose_label_list)
    for i in range(batch_size):
        person_choose = [[] for k in range(person_num)]
        for person_id in range(person_num):
            person_choose[person_id] = np.argwhere(person_choose_label_list[i][person_id] > 0)
        common_id = -1
        max_iou = 0.0
        for person_id in range(person_num):
            current_result = person_choose[person_id]
            iou_list = []
            for gap in range(person_num-1):
                other_result = person_choose[np.mod(person_id+gap+1,person_num)]
                iou_list.append(get_iou(current_result,other_result))
            if max_iou < np.mean(iou_list):
                max_iou = np.mean(iou_list)
                common_id = person_id
        final_choose_indice.append(person_choose_label_list[i][common_id])
    return np.array(final_choose_indice)


def train(sub_dir, logging, model_save_dir, result_save_dir):

    meta_data, train_data, val_data = get_video_data_jukin(Params.video_data_path_train, Params.video_data_path_val)
    captions = meta_data['Description'].values
    for c in string.punctuation:
        captions = map(lambda x: x.replace(c, ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(logging, captions, word_count_threshold=1)
    np.save(Params.ixtoword_path, ixtoword)
    np.save(Params.wordtoix_path, wordtoix)
    if os.path.exists(Params.word_fts_path):
        word_emb_init = np.array(np.load(open(Params.word_fts_path)).tolist(),np.float32)
    else:
        get_word_embedding(Params.word_embedding_path,Params.wordtoix_path,Params.ixtoword_path,Params.word_fts_path)
        word_emb_init = np.array(np.load(open(Params.word_fts_path)).tolist(),np.float32)



    model = GTP_P(word_emb_init,is_training = True)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)
    optimizer = optimizer_factory[Params.optimizer](**Params.opt_arg[Params.optimizer])
    if Params.clip:
        gvs = optimizer.compute_gradients(model.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
    else:
        train_op = optimizer.minimize(model.loss)

    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=100)
    tf.initialize_all_variables().run()

    with tf.device("/cpu:0"):
        if finetune:
            saver.restore(sess, MODEL)


    ############################################# start training ####################################################

    tStart_total = time.time()
    for epoch in range(Params.n_epochs):

        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        tStart_epoch = time.time()
        loss_epoch_f = np.zeros(len(train_data)) # each item in loss_epoch record the loss of this h5 file

        for current_batch_file_idx in xrange(len(train_data)):

            logging.info("current_batch_file_idx = {:d}".format(current_batch_file_idx))
            tStart = time.time()
            current_batch = h5py.File(train_data[current_batch_file_idx],'r')

            logging.info(train_data[current_batch_file_idx])

            # processing sentence
            current_captions_tmp = current_batch['sentence']
            current_captions = []
            for ind in range(Params.batch_size):
                current_captions.append(current_captions_tmp[ind])
            current_captions = np.array(current_captions)
            for ind in range(Params.batch_size):
                for c in string.punctuation: 
                    current_captions[ind] = current_captions[ind].replace(c,'')
            for i in range(Params.batch_size):
                current_captions[i] = current_captions[i].strip()
                if current_captions[i] == '':
                    current_captions[i] = '.'
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=Params.max_sentence_len-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_length = np.array( map(lambda x: (x != 0).sum(), current_caption_matrix )) # save the sentence length of this batch

            # processing video
            current_video_feats =  current_batch['video_data']
            current_video_length = np.zeros(Params.batch_size,dtype = np.int32)
            for i in range(Params.batch_size):
                current_video_length[i] = int(np.sum(current_batch['video_label'][i]))

            new_current_indice = choose_commmon_person(current_batch['choose_label'])
            _, loss_val, relevance_loss, regular_loss, probability = sess.run(
                    [train_op, model.loss, model.relevance_loss, model.regular_loss, model.probability],
                    feed_dict={
                        model.video: current_video_feats,
                        model.sentence_index: current_caption_matrix,
                        model.indices: new_current_indice,
                        model.video_w_len: current_video_length,
                        model.sentence_w_len: current_caption_length
                        })
            loss_epoch_f[current_batch_file_idx] = relevance_loss
            logging.info("loss = {:f} relevance_loss = {:f} regular_loss = {:f}".format(loss_val,relevance_loss,regular_loss))

        logging.info("Epoch: {:d} done.".format(epoch))
        tStop_epoch = time.time()
        logging.info('Current Epoch Mean loss forward: {:f}'.format(np.mean(loss_epoch_f)))
        logging.info('Epoch Time Cost: {:f} s'.format(round(tStop_epoch - tStart_epoch,2)))


        #################################################### test ################################################################################################
        if np.mod(epoch, Params.iter_test) == 0 or epoch == Params.n_epochs -1:
            logging.info('Epoch {:d} is done. Saving the model ...'.format(epoch))
            with tf.device("/cpu:0"):
                saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=epoch)


    logging.info("Finally, saving the model ...")
    with tf.device("/cpu:0"):
        saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=epoch)
    tStop_total = time.time()
    logging.info("Total Time Cost: {:f} s".format(round(tStop_total - tStart_total,2)))





def test(model_save_dir, result_save_dir):

    meta_data, train_data, val_data = get_video_data_jukin(Params.video_data_path_train, Params.video_data_path_val)
    wordtoix = (np.load(Params.wordtoix_path)).tolist()
    word_emb_init = np.array(np.load(open(Params.word_fts_path)).tolist(),np.float32)

    model = GTP_P(word_emb_init,is_training = False)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)
    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=100)
        latest_checkpoint = tf.train.latest_checkpoint(model_save_dir)
        tmp = latest_checkpoint.split('model-')
        all_epoch = int(tmp[1])
        logging.info(str(all_epoch))


    for epoch in range(all_epoch+1):

        with tf.device("/cpu:0"):
            saver.restore(sess, tmp[0]+'model-'+str(epoch))

        if os.path.exists(result_save_dir+'/'+str(epoch)+'.pkl'):
            continue


        result = []

        for current_batch_file_idx in xrange(len(val_data)):

            current_batch = h5py.File(val_data[current_batch_file_idx],'r')

            # processing sentence
            current_captions_tmp = current_batch['sentence']
            current_captions = []
            for ind in range(Params.batch_size):
                current_captions.append(current_captions_tmp[ind])
            current_captions = np.array(current_captions)
            for ind in range(Params.batch_size):
                for c in string.punctuation: 
                    current_captions[ind] = current_captions[ind].replace(c,'')
            for i in range(Params.batch_size):
                current_captions[i] = current_captions[i].strip()
                if current_captions[i] == '':
                    current_captions[i] = '.'
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=Params.max_sentence_len-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_length = np.array( map(lambda x: (x != 0).sum(), current_caption_matrix )) # save the sentence length of this batch

            current_video_feats =  current_batch['video_data']
            current_video_length = np.zeros(Params.batch_size,dtype = np.int32)
            for i in range(Params.batch_size):
                current_video_length[i] = int(np.sum(current_batch['video_label'][i]))


            batch_probability = sess.run([model.probability],
                    feed_dict={
                        model.video: current_video_feats,
                        model.sentence_index: current_caption_matrix,
                        model.video_w_len: current_video_length,
                        model.sentence_w_len: current_caption_length
                        })


            batch_probability = batch_probability[0]
            for i in range(Params.batch_size):
                one_probability = batch_probability[i]
                one_probability = one_probability[:current_video_length[i]]
                sort_index = np.argsort(one_probability)[::-1]
                final_choose_result = sort_index[:5]
                result.append([current_batch['video_name'][i],\
                               current_batch['seg'][i],\
                               current_batch['sentence'][i],\
                               current_batch['choose_label'][i],\
                               one_probability,\
                               final_choose_result]
                               )
                logging.info(str(final_choose_result))
                logging.info(str(np.where(current_batch['choose_label'][i][3] > 0)))
        pkl.dump(result,open(result_save_dir+'/'+str(epoch)+'.pkl','wb'))


        logging.info('***************************************************************')
        analysis_iou(result,epoch,logging)
        analysis_prf(result,epoch,logging)
        logging.info('***************************************************************')




if __name__ == '__main__':

    args = parse_args()
    sub_dir, logging, model_save_dir, result_save_dir = make_prepare_path(args.task)

    if args.task == 'train':
        with tf.device('/cpu:0'):
            train(sub_dir, logging, model_save_dir, result_save_dir)
    if args.task == 'test':
        with tf.device('/cpu:0'):
            test(model_save_dir, result_save_dir)

