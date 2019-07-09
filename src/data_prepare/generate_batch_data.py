import numpy as np
import os, json, h5py, math, pdb, glob
import unicodedata
import cPickle as pkl
import sys


max_len = 210
c3d_per_f = 8
c3d_fts_dim = 500
batch_size = 20

data_save_dir = '../../data/'
train_json_path = '../../dataset/activitynet/train.json'
test_json_path = '../../dataset/activitynet/val_merge.json'
train_json = json.load(open(train_json_path))
test_json = json.load(open(test_json_path))
video_info = pkl.load(open('../../dataset/activitynet/video_info.pkl','r'))
feature_path_c3d = '/mnt/data/yuanyitian/ActivityNet/download/fts/C3D/sub_activitynet_v1-3.c3d.hdf5'
all_c3d_fts = h5py.File(feature_path_c3d)

train_anno_file = open('../../dataset/annotated_thumbnail/anno_train.txt','r')
test_anno_file = open('../../dataset/annotated_thumbnail/anno_test.txt','r')
val_anno_file = open('../../dataset/annotated_thumbnail/anno_val.txt','r')
train_lines = train_anno_file.readlines()
test_lines = test_anno_file.readlines()
val_lines = val_anno_file.readlines()

remain_file = open('./remain.txt','w')


def get_annoSegFts(video_name,seg,c3d_per_f):
    video_data = np.zeros([max_len,c3d_fts_dim])+0.0
    video_label = np.zeros(max_len)+0.0
    video_full_fts = np.array(all_c3d_fts[video_name]['c3d_features'])
    [video_fps,video_total_frames,_,_] = video_info[video_name]
    if video_name in train_json:
        video_duration = train_json[video_name]['duration']
    else:
        video_duration = test_json[video_name]['duration']
    if video_duration <= 0:
        return None,None
    if len(video_full_fts) <= 0:
        return None,None
    left_frame = int(seg[0] * 1.0 / video_duration * video_total_frames)
    right_frame =  int(seg[1] * 1.0 / video_duration * video_total_frames)
    start = left_frame
    jj = 0
    while int(start+video_fps) < right_frame and int(start+video_fps) < int(video_total_frames):
        current_left = start
        current_right = min(start+ int(2*video_fps),video_total_frames)
        left_idx = int(current_left*1.0 / c3d_per_f)
        right_idx = int(current_right*1.0 / c3d_per_f)
        try:
            seg_fts = np.mean(video_full_fts[left_idx:right_idx,:],axis=0)
        except:
            print 'find error'
            seg_fts = np.zeros([c3d_fts_dim])+0.0
        else:
            pass
        video_data[jj,:] = seg_fts
        video_label[jj] = 1.0
        jj+=1
        start = current_right
    video_data = np.array(video_data)
    return video_data,video_label


def save_batch_h5py(data_save_dir,split_type,batch_id,video_name_list,seg_list,sentence_list,video_data_list,video_label_list,choose_label_list,mean_choose_label_list):
    if not os.path.exists(data_save_dir+split_type):
        os.makedirs(data_save_dir+split_type)
    h5py_batch = h5py.File(data_save_dir+split_type+'/'+split_type+str(batch_id)+'.h5','w')
    h5py_batch['video_name'] = video_name_list
    h5py_batch['seg'] = seg_list
    h5py_batch['sentence'] = sentence_list
    h5py_batch['video_data'] = np.array(video_data_list)
    h5py_batch['video_label'] = np.array(video_label_list)
    h5py_batch['choose_label'] = np.array(choose_label_list)
    h5py_batch['mean_choose_label'] = np.array(mean_choose_label_list)


#video_name, seg, video_data, choose_label_list, mean_choose_label, sentence
def generate_h5py(lines,split_type):
    print split_type
    video_name_list = []
    seg_list = []
    sentence_list = []
    video_data_list = []
    video_label_list = []
    choose_label_list = []
    mean_choose_label_list = []

    batch_id = 0
    current_line_num = 0
    for line in lines:
        content = line.split('\t')
        video_name = content[0]

        seg = eval(content[1])
        if seg[1] - seg[0] < 20.0:
            current_line_num += 1
            continue

        sentence = content[2]
        sentence_len = len(sentence.split(' '))
        if sentence_len > 35:
            current_line_num += 1
            continue

        video_data,video_label = get_annoSegFts(video_name,seg,c3d_per_f)
        if video_data is None and video_label is None:
            continue

        tmp = eval(content[3])
        mean_choose_label = np.zeros(max_len)+0.0
        choose_label = np.zeros([len(tmp),max_len],dtype=np.float)+0.0
        person_id = 0
        for item in tmp:
            for ii in item:
                mean_choose_label[ii-1]+=1
                choose_label[person_id][ii-1] = 1.0
            person_id+=1
        if len(tmp) == 0:
            continue
        mean_choose_label = mean_choose_label * 1.0 / len(tmp)

        video_name_list.append(video_name)
        seg_list.append(seg)
        sentence_list.append(sentence)
        video_data_list.append(video_data)
        video_label_list.append(video_label)
        choose_label_list.append(choose_label)
        mean_choose_label_list.append(mean_choose_label)

        if len(video_name_list) == batch_size or current_line_num == len(lines)-1:
            save_batch_h5py(data_save_dir,split_type,batch_id,video_name_list,seg_list,sentence_list,video_data_list,video_label_list,choose_label_list,mean_choose_label_list)
            video_name_list = []
            seg_list = []
            sentence_list = []
            video_data_list = []
            video_label_list = []
            choose_label_list = []
            mean_choose_label_list = []
            batch_id+=1

        current_line_num += 1
        remain_file.write(line)




def getlist(feature_folder_name,split):
    list_path = os.path.join(feature_folder_name,split+'/')
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        if split == 'train':
            if ele[-6:-3] == '247' or ele[-6:-3] == '352' or ele[-6:-3] == '295' or ele[-6:-3] == '317' or ele[-6:-3] == '357': #bad data, reason to be found
                continue
        f.write(ele+'\n')

generate_h5py(train_lines,'train')
generate_h5py(test_lines,'test')
generate_h5py(val_lines,'val')
getlist(os.path.abspath(data_save_dir),'train')
getlist(os.path.abspath(data_save_dir),'test')
getlist(os.path.abspath(data_save_dir),'val')


