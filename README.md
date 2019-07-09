# GTP
Code for the paper: "Sentence Specified Dynamic Video Thumbnail Generation", implemented with tensorflow.

## Dataset

* The annoated sentence specified dynamic video thumbnail dataset is provided in the 'dataset' folder. The concrete description of the dataset is provided in the 'dataset_description.txt'. We also provide the description here.

* The 'dataset' folder contains the annotated dataset for sentence specified dynamic video thumbnail generation, as well as other auxiliary data.

### ./dataset/glove.840B.300d_dict.npy
* Glove word embeddings in our work, the original features can be downloaded at: https://nlp.stanford.edu/projects/glove/.

### ./dataset/activitynet: ActivityNet Captions data
* (1) train.json: Video caption training annotations from the ActivityNet Captions dataset
* (2) val_merge.json: Video caption validation annotations from the ActivityNet Captions dataset. In the AcitivtyNet Captions dataset, there are two validation set (val_1.json and val_2.json), we merge the two validation sets into one as the val_merge.json.
* (3) video_info.pkl: It is a dictionary which contains video information in ActivityNet.
 	video_info = pkl.load(open(video_info.pkl,'r'))
	video_name_list = video_info.keys()
	one_video_name = video_name_list[0]
	[video_fps,video_total_frames,_,_] = video_info[one_video_name]





